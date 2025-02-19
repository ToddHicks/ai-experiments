import argparse
import sys

import numpy as np
import requests

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    while True:
        response = requests.post("http://127.0.0.1:5009/new").json()
        if args.debug:
            print(response)
        game_id = response['game_id'] # Will error if it doesn't exist.
        board = response['board']
        print("Let's play! You are 'O' (-1). AI is 'X' (1). Enter a move (0-8), or QUIT to quit:")
            
        while True:
            print("Selection Grid:\n", np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
            print("Game Board:\n", board)
            
            try: 
                selection = input("Your move: ")
                if selection.upper() == 'QUIT':
                    print('Quitting game, thanks for playing!')
                    sys.exit()
                move = int(selection)
                if move < 0 or move > 8:
                    print("Invalid move, try again.")
                    continue
                body = {"game_id": game_id, "action": move}
                if args.debug:
                    print(body)
                response = requests.post("http://127.0.0.1:5009/act", json=body)
                if args.debug:
                    print(response)
                response = response.json()
                if args.debug: 
                    print(response)
                board = response['board']
                winner = response['winner']
                if winner is not None:
                    if winner == 1: 
                        print('Computer Wins, You Stink!')
                    if winner == -1: 
                        print('Player Wins!')
                    else:
                        print("It's a tie!")
                    break
            except SystemExit:
                raise # Quit
            except Exception as e:
                print(e)
                print("Invalid input. Please enter a number between 0 and 8, or type 'QUIT' to quit:")