from flask import Flask, jsonify, request

app = Flask(__name__)

import argparse
import random
import sys
import threading
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

"""
This is my first attempt at 2 things.
1. Creating a service in python.
2. Creating a learning algorithm.
With minimal testing both show signs of working.
"""
alpha=1.0
gamma=0.9 
epsilon=0.01

q_table = pd.DataFrame(columns=[str(i) for i in range(9)] + ['reward'])
game_stats = pd.DataFrame(columns=['game', 'turns', 'winner'])
scaler = StandardScaler()
model = SGDRegressor(max_iter=1000, tol=1e-3)
game_count = 0
debug = True


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
    Retrieves the Q-value for a given state-action pair.
    If state-action is unknown, returns a small random value.
    """
    global q_table
    if state in q_table.index:
        return q_table.loc[state, str(action)]
    return random.uniform(-0.1, 0.1)

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
    Updates the Q-table using the Q-learning formula and incentivizes longer games.
    """
    global q_table
    with q_lock:
        max_next_q = max([get_q_value(next_state, a) for a in range(9)], default=0)
        q_old = get_q_value(state, action)
        reward += turns_played * 0.1  # Reward for prolonging the game
        q_update = (1 - alpha) * q_old + alpha * (reward + gamma * max_next_q)
        
        if state not in q_table.index:
            new_row = pd.DataFrame([[random.uniform(-0.1, 0.1) for _ in range(9)] + [reward]], index=[state], columns=q_table.columns)
            q_table = pd.concat([q_table, new_row])
        q_table.loc[state, str(action)] = q_update
        q_table.loc[state, 'reward'] = reward

def save_q_table(filename="q_table.csv"):
    """
    Saves the Q-table to a CSV file.
    """
    with q_lock:
        q_table.to_csv(filename)

def load_q_table(filename="q_table.csv"):
    """
    Loads the Q-table from a CSV file.
    """
    global q_table
    try:
        q_table = pd.read_csv(filename, index_col=0)
    except FileNotFoundError:
        q_table = pd.DataFrame(columns=[str(i) for i in range(9)] + ['reward'])

def save_game_stats(filename="game_stats.csv"):
    """
    Saves the game statistics to a CSV file.
    """
    global game_stats
    with stats_lock:
        game_stats.to_csv(filename, index=False)

def load_game_stats(filename="game_stats.csv"):
    """
    Loads the game statistics from a CSV file.
    """
    global game_stats, game_count
    try:
        game_stats = pd.read_csv(filename)
        game_count = len(game_stats)
    except FileNotFoundError:
        game_stats = pd.DataFrame(columns=['game', 'turns', 'winner'])
        game_count = 0


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
@app.route('/new', methods=['POST'])
def create_game():
    """
    Creates a new game
    """
    global game_count, games, games_lock
    with games_lock:
        print('Creating new game!')
        game_id=random.getrandbits(128)
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
    # Computers turn
    prev_state = get_state(game.board)
    move = choose_action(game)
    game.board[move // 3, move % 3] = 1
    game.state_action_pairs.append((prev_state, move))
    next_state = get_state(game.board)
    game.turns_played += 1
    board = game.board

    winner = check_winner(game.board)
    if winner is not None:
        for state, action in game.state_action_pairs:
            update_q_table(state, action, winner, next_state, game.turns_played)
        save_q_table()
        with stats_lock:
            new_row = pd.DataFrame([{'game': game_count, 'turns': game.turns_played, 'winner': winner}])
            game_stats = pd.concat([game_stats, new_row], ignore_index=True)
        save_game_stats()
        with games_lock:
            del games[game_id]
    return jsonify({"message": "Turn Successful", "board": str(board), "winner": winner})

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
    
    load_q_table()
    load_game_stats()
    debug = args.debug
    threading.Thread(target=cleanup_games, daemon=True).start()
    app.run(host='127.0.0.1', port=5009, debug=True)  # Runs locally on port 5000