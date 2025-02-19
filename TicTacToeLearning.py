import argparse
import random
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


class TicTacToeQLearning:
    """
    Q-Learning based Tic-Tac-Toe AI.
    Uses reinforcement learning to improve its decision-making over time.
    """
    def __init__(self, alpha=1.0, gamma=0.9, epsilon=0.01):
        """
        Initializes the Q-learning model.
        :param alpha: Learning rate
        :param gamma: Discount factor
        :param epsilon: Exploration probability
        """
        self.q_table = pd.DataFrame(columns=[str(i) for i in range(9)] + ['reward'])
        self.game_stats = pd.DataFrame(columns=['game', 'turns', 'winner'])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.scaler = StandardScaler()
        self.model = SGDRegressor(max_iter=1000, tol=1e-3)
        self.game_count = 0
        self.debug = False
        
    def get_state(self, board):
        """
        Returns a string representation of the board state.
        """
        return str(tuple(board.flatten()))
    
    def get_q_value(self, state, action):
        """
        Retrieves the Q-value for a given state-action pair.
        If state-action is unknown, returns a small random value.
        """
        if state in self.q_table.index:
            return self.q_table.loc[state, str(action)]
        return random.uniform(-0.1, 0.1)
    
    def choose_action(self, board):
        """
        Chooses an action using an epsilon-greedy strategy.
        """
        state = self.get_state(board)
        available_moves = [i for i in range(9) if board.flatten()[i] == 0]
        
        is_exploration = random.uniform(0, 1) < self.epsilon
        
        q_values = {action: self.get_q_value(state, action) for action in available_moves}
        max_q = max(q_values.values(), default=float('-inf'))
        best_moves = [action for action, q in q_values.items() if q == max_q]
        
        if is_exploration:
            action = random.choice(available_moves)
            decision_type = "Exploration"
        else:
            action = random.choice(best_moves)
            decision_type = "Best choice"
        
        if self.debug:
            print(f"AI chose move {action} ({decision_type})")
            print(f"Q-values: {q_values}")
            print(f"Best moves: {best_moves} with Q-value: {max_q}")
            print(f"Chosen action matches max Q-value: {action in best_moves}")
        
        return action
    
    def update_q_table(self, state, action, reward, next_state, turns_played):
        """
        Updates the Q-table using the Q-learning formula and incentivizes longer games.
        """
        max_next_q = max([self.get_q_value(next_state, a) for a in range(9)], default=0)
        q_old = self.get_q_value(state, action)
        reward += turns_played * 0.1  # Reward for prolonging the game
        q_update = (1 - self.alpha) * q_old + self.alpha * (reward + self.gamma * max_next_q)
        
        if state not in self.q_table.index:
            new_row = pd.DataFrame([[random.uniform(-0.1, 0.1) for _ in range(9)] + [reward]], index=[state], columns=self.q_table.columns)
            self.q_table = pd.concat([self.q_table, new_row])
        self.q_table.loc[state, str(action)] = q_update
        self.q_table.loc[state, 'reward'] = reward
    
    def save_q_table(self, filename="q_table.csv"):
        """
        Saves the Q-table to a CSV file.
        """
        self.q_table.to_csv(filename)
    
    def load_q_table(self, filename="q_table.csv"):
        """
        Loads the Q-table from a CSV file.
        """
        try:
            self.q_table = pd.read_csv(filename, index_col=0)
        except FileNotFoundError:
            self.q_table = pd.DataFrame(columns=[str(i) for i in range(9)] + ['reward'])
    
    def save_game_stats(self, filename="game_stats.csv"):
        """
        Saves the game statistics to a CSV file.
        """
        self.game_stats.to_csv(filename, index=False)
    
    def load_game_stats(self, filename="game_stats.csv"):
        """
        Loads the game statistics from a CSV file.
        """
        try:
            self.game_stats = pd.read_csv(filename)
            self.game_count = len(self.game_stats)
        except FileNotFoundError:
            self.game_stats = pd.DataFrame(columns=['game', 'turns', 'winner'])
            self.game_count = 0


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    ai = TicTacToeQLearning()
    ai.load_q_table()
    ai.load_game_stats()
    ai.debug = args.debug
    
    while True:
        board = np.zeros((3, 3), dtype=int)
        print("Let's play! You are 'O' (-1). AI is 'X' (1). Enter a move (0-8), or QUIT to quit:")
        
        state_action_pairs = []
        turn = -1
        turns_played = 0
        ai.game_count += 1
        
        while True:
            print("Selection Grid:\n", np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
            print("Game Board:\n", board)
            prev_state = ai.get_state(board)
            
            try: 
                if turn == -1:
                    selection = input("Your move: ")
                    if selection.upper() == 'QUIT':
                        print('Quitting game, thanks for playing!')
                        sys.exit()
                    move = int(selection)
                    if move < 0 or move > 8 or board.flatten()[move] != 0:
                        print("Invalid move, try again.")
                        continue
                    board[move // 3, move % 3] = -1
                else:
                    move = ai.choose_action(board)
                    board[move // 3, move % 3] = 1
                
                next_state = ai.get_state(board)
                state_action_pairs.append((prev_state, move))
                turns_played += 1
                winner = check_winner(board)
                if winner is not None:
                    for state, action in state_action_pairs:
                        ai.update_q_table(state, action, winner, next_state, turns_played)
                    ai.save_q_table()
                    new_row = pd.DataFrame([{'game': ai.game_count, 'turns': turns_played, 'winner': winner}])
                    ai.game_stats = pd.concat([ai.game_stats, new_row], ignore_index=True)
                    ai.save_game_stats()
                    break
                turn *= -1
            except SystemExit:
                raise # Quit
            except Exception as e:
                print("Invalid input. Please enter a number between 0 and 8, or type 'QUIT' to quit:")
