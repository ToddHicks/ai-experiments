import argparse
import sys

import numpy as np
import requests


def print_board(board):
    print("Selection Grid:")
    print(np.array([[i for i in range(7)]]))
    print("Game Board:")
    print(np.array(board))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    base = 'https://ai-experiments-connect4-server.onrender.com'
    #base = "http://127.0.0.1:5010"
    
    while True:
        response = requests.post(f"{base}/new_connect4").json()
        if args.debug:
            print(response)
        game_id = response['game_id']
        board = response['board']
        print("Let's play Connect 4! You are 'O' (-1). AI is 'X' (1). Enter a move (0-6), or QUIT to quit:")
            
        while True:
            print_board(board)
            
            try: 
                selection = input("Your move: ")
                if selection.upper() == 'QUIT':
                    print('Quitting game, thanks for playing!')
                    sys.exit()
                move = int(selection)
                if move < 0 or move > 6:
                    print("Invalid move, try again.")
                    continue
                body = {"game_id": game_id, "action": move}
                if args.debug:
                    print(body)
                response = requests.post(f"{base}/act_connect4", json=body)
                if args.debug:
                    print(response)
                response = response.json()
                if args.debug: 
                    print(response)
                board = response['board']
                winner = response['winner']
                if winner is not None:
                    print_board(board)
                    if winner == 1: 
                        print('Computer Wins, You Stink!')
                    elif winner == -1: 
                        print('Player Wins!')
                    else:
                        print("It's a tie!")
                    break
            except SystemExit:
                raise # Quit
            #except Exception as e:
            #    print(e)
            #    print("Invalid input. Please enter a number between 0 and 6, or type 'QUIT' to quit:")
