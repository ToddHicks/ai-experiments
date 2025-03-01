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

alpha = 0.2 # Learns slowly over experiences.
gamma = 0.9 # Value on future rewards
epsilon = 0.05 # Randomness
turn_bonus = 0.5 # Bonus for longer turns (Eventually lower this to 0.25 or lower.)

games = {}
games_lock = threading.Lock()
game_count = 0
debug = True

DATABASE_URL = os.getenv("DATABASE_INTERNAL_URL")
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

Base = declarative_base()

class QTable(Base):
    __tablename__ = 'q_table_connect4'
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
    __tablename__ = 'game_stats_connect4'
    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(String)
    turns_played = Column(Integer)
    winner = Column(Integer)

Base.metadata.create_all(engine)

class Connect4QLearning:
    def __init__(self):
        self.board = np.zeros((6, 7), dtype=int)
        self.state_action_pairs = []
        self.turns_played = 0
        self.last_active = time.time()

def get_state(board):
    return str(tuple(map(int, board.flatten())))

def get_q_value(state, action):
    session = Session()
    q_row = session.query(QTable).filter_by(state=state).first()
    session.close()
    # print(f"State: {state}, Action: {action}, Q-Value: {q_row}")
    if q_row:
        return getattr(q_row, f'action{action}', random.uniform(-0.1, 0.1))

    return random.uniform(-0.1, 0.1)

def update_q_table(state, action, reward, next_state, turns_played):
    session = Session()
    next_q_row = session.query(QTable).filter_by(state=next_state).first()
    max_next_q = max([getattr(next_q_row, f'action{i}', 0.0) for i in range(7)]) if next_q_row else 0.0

    q_row = session.query(QTable).filter_by(state=state).first()
    if not q_row:
        q_row = QTable(state=state)
        session.add(q_row)

    current_q = getattr(q_row, f'action{action}', 0.0) or 0.0
    reward += (turns_played / 42) * turn_bonus
    #new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
    new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_next_q)
    # Ensure new_q stays in range.
    new_q = max(min(new_q, 1), -1)
    # print(f'reward: {reward}, max_next_q: {max_next_q}, current_q: {current_q}, new_q: {new_q}')
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
        q_values = {action: get_q_value(state, action) for action in available_moves}
        max_q = max(q_values.values(), default=float('-inf'))
        best_moves = [action for action, q in q_values.items() if q == max_q]
        choice = random.choice(best_moves)
        # print(f'Turn: {game.turns_played} Choice: {choice} Logic: {q_values}')
        return choice
    else:
        choice = random.choice(available_moves)
        # print(f'Turn: {game.turns_played} Random Choice: {choice}')
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

    game.turns_played += 1
    state = get_state(game.board)
    game.state_action_pairs.append((state, move))

    winner = check_winner(game.board)

    if winner is None:
        ai_move = choose_action(game)
        drop_piece(game.board, ai_move, 1)
        game.turns_played += 1
        next_state = get_state(game.board)
        game.state_action_pairs.append((next_state, ai_move))
        winner = check_winner(game.board)

    board = game.board.tolist()

    if winner is not None:
        reward = 1 if winner == 1 else -1
        for state, action in game.state_action_pairs:
            update_q_table(state, action, reward, state, game.turns_played)
        record_game_stats(game_id, game.turns_played, int(winner))
        with games_lock:
            del games[game_id]
        return jsonify({"message": "Game over!", "board": board, "winner": int(winner)})

    return jsonify({"message": "Turn successful", "board": board, "winner": winner})

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
