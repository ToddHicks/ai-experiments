'''
This version will work on optimizing the model so ties net no change in score.
The previous version has a situation where long games net a positive score, but not a winning strategy.
So at some point, the computer will not care about losing. Losing should always have a penalty.
I've stepped away from how the Q_model typically uses next max q, and instead use the value by percentage against gammma and current.
'''
import argparse
import logging
import os
import random
import sys
import threading
import time

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from sqlalchemy import Column, Float, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

app = Flask(__name__)
CORS(app, origins=["https://ai-experiments-connect4-ui.onrender.com"])

# trying to ensure logs are flushed.
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# We already take steps to reward longer games, so I wanted to ensure a quick loss is punished.
alpha = 0.7 # Learns over experiences.

gamma = 0.3 # Value on future rewards
epsilon = 0.10 # Randomness (was .20)
controlled_random = 0.5 # Percent of randomness is limited to 2,4th best options. (2,3,4)

games = {}
games_lock = threading.Lock()
game_count = 0
debug = True

DATABASE_URL = os.getenv("DATABASE_INTERNAL_URL")
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

Base = declarative_base()

class QTable(Base):
    __tablename__ = 'q_table_connect4-5'
    state = Column(String, primary_key=True)
    action0 = Column(Float, default=0.0, nullable=False)
    action1 = Column(Float, default=0.0, nullable=False)
    action2 = Column(Float, default=0.0, nullable=False)
    action3 = Column(Float, default=0.0, nullable=False)
    action4 = Column(Float, default=0.0, nullable=False)
    action5 = Column(Float, default=0.0, nullable=False)
    action6 = Column(Float, default=0.0, nullable=False)
    reward = Column(Float, default=0.0, nullable=False)

class GameStats(Base):
    __tablename__ = 'game_stats_connect4-5'
    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(String, nullable=False, unique=True)
    turns_played = Column(Integer, nullable=False, default=0)
    winner = Column(Integer, nullable=False)

Base.metadata.create_all(engine)

class Connect4QLearning:
    def __init__(self):
        self.board = np.zeros((6, 7), dtype=int)
        self.state_action_pairs = []
        self.turns_played = 0
        self.last_active = time.time()

@app.before_request
def handle_options():
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        headers = response.headers

        headers['Access-Control-Allow-Origin'] = 'https://ai-experiments-connect4-ui.onrender.com'
        headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS, DELETE'
        headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'

        return response

def get_state(board):
    return str(tuple(map(int, board.flatten())))

def get_q_value(state, action):
    session = Session()
    q_row = session.query(QTable).filter_by(state=state).first()
    session.close()
    # print(f"State: {state}, Action: {action}, Q-Value: {q_row}")
    if q_row:
        return getattr(q_row, f'action{action}')

    return random.uniform(-0.1, 0.1)

def update_q_table(state, action, reward, next_state, turns_played):
    session = Session()
    next_q_row = session.query(QTable).filter_by(state=next_state).first()

    if reward < 1: # We don't want to discourage winning quickly, so if it was a win condition. Skip this.
        reward = reward * ((42-turns_played)/35) 
    #max_next_q = max([getattr(next_q_row, f'action{i}', 0.0) for i in range(7)]) if next_q_row else 0.0
    # next max of 0 is slowing down learning. However, if we use this we may be ignoring potential rewards by assuming
    # untried options will be bad options.
    q_values = [getattr(next_q_row, f'action{i}', 0.0) for i in range(7)] if next_q_row else []
    non_zero_q_values = [q for q in q_values if q != 0.0]
    max_next_q = max(non_zero_q_values) if non_zero_q_values else reward

    q_row = session.query(QTable).filter_by(state=state).first()
    if not q_row:
        q_row = QTable(state=state)
        session.add(q_row)

    current_q = getattr(q_row, f'action{action}', 0.0) or 0.0
    if max_next_q == 0: 
        max_next_q = reward
    new_q = current_q * (1-alpha) + alpha * (reward * (1-gamma) + gamma * max_next_q)
    print(f'reward: {reward}, max_next_q: {max_next_q}, current_q: {current_q}, new_q: {new_q}')
    setattr(q_row, f'action{action}', new_q)

    session.commit()
    session.close()

def record_game_stats(game_id, turns_played, winner):
    game_stats = GameStats(game_id=str(game_id), turns_played=turns_played, winner=winner)
    session = Session()
    session.add(game_stats)
    session.commit()
    session.close()

def choose_action(game):
    state = get_state(game.board)
    available_moves = [col for col in range(7) if game.board[0][col] == 0]
    is_exploration = random.uniform(0, 1) < epsilon
    if not is_exploration:
        # I think we can do this more efficient, right now it queries the table for each action, yet it has all the actions on one query.
        # This will require some reorganization, holding off making this change right now.
        q_values = {action: get_q_value(state, action) for action in available_moves}
        max_q = max(q_values.values(), default=float('-inf'))
        best_moves = [action for action, q in q_values.items() if q == max_q]
        choice = random.choice(best_moves)
        print(f'Turn: {game.turns_played} Choice: {choice} \nLogic: {q_values} \nState: {state}')
        return choice
    else:
        controlled_exploration = random.uniform(0, 1) < controlled_random
        if controlled_exploration:
            # Get Q-values for each available move
            q_values = {action: get_q_value(state, action) for action in available_moves}
            # Sort by Q-value in descending order
            sorted_moves = sorted(q_values.items(), key=lambda x: x[1], reverse=True)
            # Get the 2nd, 3rd, and 4th highest moves (indices 1 to 3 after sorting)
            top_moves = [move[0] for move in sorted_moves[1:4]]
            # Randomly select one of these moves
            choice = random.choice(top_moves)
            print(f"2nd-4th highest moves: {top_moves}, Chosen move: {choice}")
            return choice
        else: 
            choice = random.choice(available_moves)
            print(f'Turn: {game.turns_played} Random Choice: {choice}')
            return choice

def drop_piece(board, col, piece):
    for row in range(5, -1, -1):
        if board[row][col] == 0:
            board[row][col] = piece
            return True
    return False

def check_winner(board):
    for row in range(6):
        for col in range(7):
            if board[row][col] == 0:
                continue
            if col + 3 < 7 and np.all(board[row, col:col+4] == board[row][col]):
                return board[row][col]
            if row + 3 < 6 and np.all(board[row:row+4, col] == board[row][col]):
                return board[row][col]
            if row + 3 < 6 and col + 3 < 7 and np.all([board[row+i][col+i] == board[row][col] for i in range(4)]):
                return board[row][col]
            if row + 3 < 6 and col - 3 >= 0 and np.all([board[row+i][col-i] == board[row][col] for i in range(4)]):
                return board[row][col]
    if not np.any(board == 0):
        return 0
    return None

@app.route('/new_connect4', methods=['POST'])
def create_game():
    global game_count, games
    with games_lock:
        game_id = random.getrandbits(53)
        games[game_id] = Connect4QLearning()
        board = games[game_id].board.tolist()
        game_count =+ 1
        return jsonify({"message": "New Connect 4 game created!", "game_id": game_id, "board": board})

@app.route('/random', methods=['POST'])
def set_random():
    data = request.json
    if not data:
        return jsonify({"error": "Invalid input"}), 422
    global epsilon, controlled_random
    try:
        epsilon = int(data['epsilon'])
        controlled_random = int(data['controlled_random'])
    except Exception as e:
        print(f'Error: {e}')
        return jsonify({"error": "Invalid input"}), 422
    
@app.route('/random', methods=['GET'])
def get_random():
   return jsonify({
        "epsiolon": epsilon,
        "controlled_random": controlled_random
    }) 

@app.route('/act_connect4', methods=['POST'])
def take_turn():
    data = request.json
    if not data:
        return jsonify({"error": "Invalid input"}), 422

    game_id = data['game_id']
    game = games.get(game_id)
    if not game:
        return jsonify({"error": "Game not found"}), 404

    move = data['action']
    if move < 0 or move > 6 or not drop_piece(game.board, move, -1):
        return jsonify({"error": "Invalid move"}), 400

    # Get current state and record player's move
    state = get_state(game.board)
    game.state_action_pairs.append((state, move))
    game.turns_played += 1

    # Check if player won
    winner = check_winner(game.board)
    if winner is not None:
        reward = -1  # Player win = bad for AI
        for state, action in game.state_action_pairs:
            update_q_table(state, action, reward, state, game.turns_played)
        record_game_stats(game_id, game.turns_played, int(winner))
        with games_lock:
            del games[game_id]
        return jsonify({"message": "Game over!", "board": game.board.tolist(), "winner": int(winner)})

    # AI takes a turn
    ai_move = choose_action(game)
    drop_piece(game.board, ai_move, 1)
    next_state = get_state(game.board)
    game.turns_played += 1
    game.state_action_pairs.append((state, ai_move))  # Use current state for AI's action

    # Check if AI won
    winner = check_winner(game.board)
    if winner is not None:
        reward = 1

        # Iterate through all state-action pairs
        for i, (state, action) in enumerate(game.state_action_pairs):
            #given_reward =  reward * ((i+1) / len(game.state_action_pairs)) This would be a scaled reward based on distance from choice.
            given_reward = reward
            update_q_table(state, action, given_reward, next_state, game.turns_played)
        #for state, action in game.state_action_pairs:
        #    update_q_table(state, action, reward, next_state, game.turns_played)
        record_game_stats(game_id, game.turns_played, int(winner))
        with games_lock:
            del games[game_id]
        return jsonify({"message": "Game over!", "board": game.board.tolist(), "winner": int(winner)})

    return jsonify({"message": "Turn successful", "board": game.board.tolist(), "winner": None})


@app.route('/stats', methods=['GET'])
def get_stats():
    """
    Method will return the recent game stats from PostgreSQL.
    """
    session = Session()
    # Total games played
    total_games = session.query(GameStats).count()

    # Get the last 10 games ordered by most recent
    last_10_games = session.query(GameStats.winner).order_by(GameStats.id.desc()).limit(10).all()

    # Extract winners from the last 10 games
    last_10_winners = [game.winner for game in last_10_games]

    # Count the results
    wins = last_10_winners.count(1)
    losses = last_10_winners.count(-1)
    ties = last_10_winners.count(0)
    session.close()

    return jsonify({
        "games_played": total_games,
        "last_10_wins": wins,
        "last_10_losses": losses,
        "last_10_ties": ties
    })

def cleanup_games():
    while True:
        time.sleep(60)  # Run every 60 seconds
        with games_lock:
            for game_id in list(games.keys()):
                print(f'Checking activity for: {game_id}')
                if time.time() - games[game_id].last_active > 600:  # 10 minutes of inactivity
                    del games[game_id]
                    print(f"Game {game_id} removed due to inactivity.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    debug = args.debug
    threading.Thread(target=cleanup_games, daemon=True).start()
    port = int(os.getenv("PORT", 5001))
    app.run(host='0.0.0.0', port=port)
    #app.run(host='127.0.0.1', port=5010, debug=True)
