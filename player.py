
from board_logic_and_mcts.Gomoku_Board import GBoard
from board_logic_and_mcts.Pente_Board import PenteBoard
from board_logic_and_mcts.MCTS import MCTS
from board_logic_and_mcts.Node import Node
import time

# Rules: "Gomoku" or "Pente"
# Board_size: 15 or 50

class Player:
    def __init__(self, rules, board_size):          # returns a Board Object, 
        self.rules = rules                          # board logic and move making are implemented in the board object
        self.board_size = board_size
        self.board_class = None
        if (rules == "Gomoku"):
            self.board_class = GBoard
        else:
            self.board_class = PBoard

# player_symbol is always 1

    def play(self, board: list, turn_number, last_opponent_move: tuple, player_symbol=1):
        game = self.board_class(size)
        game.board = board
        opponent_move_x, opponent_move_y = last_opponent_move
        game.makeMove(opponent_move_x, opponent_move_y, player_symbol)
        print("Opponent move registered.")
        print("Now planning next move...")
        
        # Call MCTS algorithm to determine next move
        root = Node(game, None) #Current game after player makes move
        mcts = MCTS(root, player_symbol)
        start = time.time()
        best_node = mcts.best_move()
        col, row = best_node.board.last_move
        end = time.time()
        if mcts.gameOver:
            print("[TEAM NAME] chose column : ", col + 1, " row : ", row + 1)
            print(f"Time taken: {end - start:.2f}")
            
#            game.makeMove(col, row, ai)        // THIS METHOD MAKeS MOVE
        print("[TEAM NAME] chose column : ", col + 1, " row : ", row + 1)
        print(f"Time taken: {end - start:.2f}")
        return row, col     #row is y, col is x