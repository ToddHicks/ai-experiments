import os

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["https://ai-experiments-ui.onrender.com"])

import argparse
import random
import sys
import threading
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
# For postgres storage.
from sqlalchemy import (JSON, Column, Float, Integer, String, Text,
                        create_engine)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import declarative_base, sessionmaker

"""
This is my first attempt at 2 things.
1. Creating a service in python.
2. Creating a learning algorithm.
With minimal testing both show signs of working.
"""
alpha=0.3
gamma=0.9
epsilon=0.00

scaler = StandardScaler()
model = SGDRegressor(max_iter=1000, tol=1e-3)
game_count = 0
debug = True

# Initialize DB connection
# Pulled from the host.
DATABASE_URL = os.getenv("DATABASE_INTERNAL_URL")

Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(engine)  # Create table if it doesn't exist

class QTable(Base):
    __tablename__ = 'q_table'
    
    state = Column(String, primary_key=True)  # Board state as the primary key
    action_0 = Column(Float, default=0.0)
    action_1 = Column(Float, default=0.0)
    action_2 = Column(Float, default=0.0)
    action_3 = Column(Float, default=0.0)
    action_4 = Column(Float, default=0.0)
    action_5 = Column(Float, default=0.0)
    action_6 = Column(Float, default=0.0)
    action_7 = Column(Float, default=0.0)
    action_8 = Column(Float, default=0.0)
    reward = Column(Float, default=0.0)


class GameStats(Base):
    __tablename__ = "game_stats"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    game = Column(Integer, nullable=False)
    turns = Column(Integer, nullable=False)
    winner = Column(Integer, nullable=False)  # -1 (player), 1 (AI), 0 (tie)

# this map will be stored as games[id] = game_object
# usually, games[id].board will be most common access.
games = {}
# locks for threading
games_lock = threading.Lock()
q_lock = threading.Lock()
stats_lock = threading.Lock()

class TicTacToeQLearning:
    """
    Q-Learning based Tic-Tac-Toe AI.
    Uses reinforcement learning to improve its decision-making over time.
    """
    def __init__(self):
        """
        Initializes the TicTacToe Game.
        :param game_id: The ID associated with this game.
        """
        self.board = np.zeros((3, 3), dtype=int)
        self.state_action_pairs = []
        self.turns_played = 0
        self.last_active = time.time()
        
def get_state(board):
    """
    Returns a string representation of the board state.
    """
    return str(tuple(board.flatten()))
    
def get_q_value(state, action):
    """
    Retrieves the Q-value for a given state-action pair from PostgreSQL.
    If the state-action pair is unknown, returns a small random value.
    """
    session = SessionLocal()
    q_entry = session.query(QTable).filter_by(state=state).first()
    
    if q_entry:
        session.close()
        return getattr(q_entry, f"action_{action}")  # Fetch the Q-value for the given action
    
    session.close()
    return random.uniform(-0.1, 0.1)  # Return small random value for unknown states

def choose_action(game):
    """
    Chooses an action using an epsilon-greedy strategy.
    """
    global debug
    state = get_state(game.board)
    available_moves = [i for i in range(9) if game.board.flatten()[i] == 0]
    
    is_exploration = random.uniform(0, 1) < epsilon
    
    q_values = {action: get_q_value(state, action) for action in available_moves}
    max_q = max(q_values.values(), default=float('-inf'))
    best_moves = [action for action, q in q_values.items() if q == max_q]
    
    if is_exploration:
        action = random.choice(available_moves)
        decision_type = "Exploration"
    else:
        action = random.choice(best_moves)
        decision_type = "Best choice"
    
    if debug:
        print(f"AI chose move {action} ({decision_type})")
        print(f"Q-values: {q_values}")
        print(f"Best moves: {best_moves} with Q-value: {max_q}")
        print(f"Chosen action matches max Q-value: {action in best_moves}")
    
    return action

def update_q_table(state, action, reward, next_state, turns_played):
    """
    Updates the Q-table in PostgreSQL using the Q-learning formula.
    """
    session = SessionLocal()
    
    # Fetch the current Q-values
    q_entry = session.query(QTable).filter_by(state=state).first()
    
    # Compute Q-value update
    max_next_q = max([get_q_value(next_state, a) for a in range(9)], default=0)
    q_old = get_q_value(state, action)
    reward *= 2
    reward += turns_played * 0.01  # Reward for prolonging the game
    q_update = (1 - alpha) * q_old + alpha * (reward + gamma * max_next_q)
    
    if q_entry:
        setattr(q_entry, f"action_{action}", q_update)  # Update specific action value
        q_entry.reward = reward  # Update reward
    else:
        # Insert a new row if the state doesn't exist
        new_entry = QTable(
            state=state,
            **{f"action_{i}": random.uniform(-0.1, 0.1) for i in range(9)},  # Initialize Q-values
            reward=reward
        )
        setattr(new_entry, f"action_{action}", q_update)
        session.add(new_entry)
    
    session.commit()
    session.close()

def load_game_stats():
    """
    Loads game statistics from the PostgreSQL database.
    """
    global game_count
    session = SessionLocal()
    game_count = session.query(GameStats).count()
    session.close()

def update_game_stats(game_count, game, winner, filename="game_stats.csv"):
    session = SessionLocal()
    new_entry = GameStats(game=game_count, turns=game.turns_played, winner=winner)
    session.add(new_entry)
    session.commit()
    session.close()

def check_winner(board):
    """
    Checks for a winner in the Tic-Tac-Toe game.
    """
    for line in [board[i, :] for i in range(3)] + [board[:, i] for i in range(3)] + [board.diagonal(), np.fliplr(board).diagonal()]:
        if all(line == 1): 
            print('Computer Wins, You Stink!')
            return 1
        if all(line == -1): 
            print('Player Wins!')
            return -1
    if 0 not in board:
        print("It's a tie!")
        return 0
    return None

### End point calls here.
@app.route('/stats', methods=['GET'])
def get_stats():
    """
    Method will return the recent game stats from PostgreSQL.
    """
    session = SessionLocal()
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

@app.route('/new', methods=['POST'])
def create_game():
    """
    Creates a new game
    """
    global game_count, games, games_lock
    with games_lock:
        print('Creating new game!')
        game_id=random.getrandbits(53)
        games[game_id] = TicTacToeQLearning()
        game_count += 1
        board = str(games[game_id].board)
        return jsonify({"message": "New game created!", "game_id": game_id, "board": board})

@app.route('/act', methods=['POST'])
def take_turn():
    """
    Player takes a turn, then the computer does. The players action is passed in as
    {
        game_id: 'some number',
        action: [0:8]
    }
    """
    global game_stats, games 
    print('Taking turn!')
    data = request.json  # Get JSON data from request
    print(data)
    if not data:
        return jsonify({"error": "Invalid input"}), 422
    
    game_id = data['game_id']
    # Should check that the game exists.
    game = games[game_id]
    game.last_active = time.time()

    move = data['action']
    # Should check the action is valid (0-8)

    # Players turn
    prev_state = get_state(game.board)
    game.board[move // 3, move % 3] = -1
    game.state_action_pairs.append((prev_state, move))
    game.turns_played += 1
    winner = check_winner(game.board)
    # Computers turn
    if (winner == None):
        prev_state = get_state(game.board)
        move = choose_action(game)
        game.board[move // 3, move % 3] = 1
        game.state_action_pairs.append((prev_state, move))
        game.turns_played += 1
        winner = check_winner(game.board)
    next_state = get_state(game.board)
    board = game.board

    if winner is not None:
        for state, action in game.state_action_pairs:
            update_q_table(state, action, winner, next_state, game.turns_played)
        update_game_stats(game_count, game, winner)
        with games_lock:
            del games[game_id]
    return jsonify({"message": "Turn Successful", "board": str(board), "winner": winner})

@app.route('/matrix', methods=['POST'])
def get_matrix_choices():
    global games 
    data = request.json  # Get JSON data from request
    print(data)
    if not data:
        return jsonify({"error": "Invalid input"}), 422
    
    game_id = data['game_id']
    game = games[game_id]
    available_moves = [i for i in range(9) if game.board.flatten()[i] == 0]
    q_values = {action: get_q_value(get_state(game.board), action) for action in available_moves}
    max_q = max(q_values.values(), default=float('-inf'))
    best_moves = [action for action, q in q_values.items() if q == max_q]
    return jsonify({"q_values": q_values, "max_q": max_q, "best_moves": best_moves})

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
    load_game_stats()
    debug = args.debug
    threading.Thread(target=cleanup_games, daemon=True).start()
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
    #app.run(host='127.0.0.1', port=5009, debug=True)  # Runs locally on port 5000