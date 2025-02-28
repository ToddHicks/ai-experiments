import argparse
import random
import sys
import threading
import time

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

alpha = 0.9
gamma = 0.8 
epsilon = 0.1

q_table = pd.DataFrame(columns=[str(i) for i in range(7)] + ['reward'])
game_stats = pd.DataFrame(columns=['game', 'turns', 'winner'])

games = {}
games_lock = threading.Lock()
q_lock = threading.Lock()
stats_lock = threading.Lock()
game_count = 0
debug = True

class Connect4QLearning:
    def __init__(self):
        self.board = np.zeros((6, 7), dtype=int)
        self.state_action_pairs = []
        self.turns_played = 0
        self.last_active = time.time()

def get_state(board):
    return str(tuple(board.flatten()))

def get_q_value(state, action):
    global q_table
    if state in q_table.index:
        return q_table.loc[state, str(action)]
    return random.uniform(-0.1, 0.1)

def update_q_table(state, action, reward, next_state, turns_played):
    global q_table
    with q_lock:
        if state not in q_table.index:
            q_table.loc[state] = [0.0] * 7 + [0.0]
        next_q_values = q_table.loc[next_state, [str(i) for i in range(7)]] if next_state in q_table.index else [0.0] * 7
        max_next_q = max(next_q_values)
        reward += turns_played * 0.01  # Reward for prolonging the game
        q_table.loc[state, str(action)] += alpha * (reward + gamma * max_next_q - q_table.loc[state, str(action)])

def choose_action(game):
    state = get_state(game.board)
    available_moves = [col for col in range(7) if game.board[0][col] == 0]
    is_exploration = random.uniform(0, 1) < epsilon

    q_values = {action: get_q_value(state, action) for action in available_moves}
    max_q = max(q_values.values(), default=float('-inf'))
    best_moves = [action for action, q in q_values.items() if q == max_q]

    if is_exploration:
        return random.choice(available_moves)
    else:
        return random.choice(best_moves)

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

def save_q_table(filename="q_table_connect4.csv"):
    with q_lock:
        q_table.to_csv(filename)

def load_q_table(filename="q_table_connect4.csv"):
    global q_table
    try:
        q_table = pd.read_csv(filename, index_col=0)
    except FileNotFoundError:
        q_table = pd.DataFrame(columns=[str(i) for i in range(7)] + ['reward'])

def update_game_stats(game_count, game, winner):
    global game_stats
    with stats_lock:
        new_row = pd.DataFrame([{'game': game_count, 'turns': game.turns_played, 'winner': winner}])
        game_stats = pd.concat([game_stats, new_row], ignore_index=True)

def save_game_stats(filename="game_stats_connect4.csv"):
    with stats_lock:
        game_stats.to_csv(filename)

def load_game_stats(filename="game_stats_connect4.csv"):
    global game_stats, game_count
    try:
        game_stats = pd.read_csv(filename, index_col=0)
        game_count = len(game_stats)
    except FileNotFoundError:
        game_stats = pd.DataFrame(columns=['game', 'turns', 'winner'])
        game_count = 0

@app.route('/new_connect4', methods=['POST'])
def create_game():
    global game_count, games
    with games_lock:
        game_id = random.getrandbits(53)
        games[game_id] = Connect4QLearning()
        board = games[game_id].board.tolist()
        return jsonify({"message": "New Connect 4 game created!", "game_id": game_id, "board": board})

@app.route('/act_connect4', methods=['POST'])
def take_turn():
    global game_stats, games
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
            update_q_table(state, action, reward, next_state if winner == 1 else state, game.turns_played)
            save_game_stats()
        save_q_table()
        with games_lock:
            game_count =+ 1
            update_game_stats(game_count, game, winner)
            del games[game_id]
        return jsonify({"message": "Game over!", "board": board, "winner": int(winner)})

    return jsonify({"message": "Turn successful", "board": board, "winner": winner})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    debug = args.debug
    load_game_stats()
    load_q_table()
    app.run(host='127.0.0.1', port=5010, debug=True)
