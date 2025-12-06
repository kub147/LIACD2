import torch
import os
import random

# Project imports
from board_logic_and_mcts.Gomoku_Board import GBoard
from board_logic_and_mcts.Pente_Board import PBoard as PenteBoard
from board_logic_and_mcts.MCTS_Neural import MCTS_Neural
from board_logic_and_mcts.Node import Node
from CNN.nn_model import PolicyValueNet


class Player:
    def __init__(self, rules, board_size):
        self.rules = rules.lower()
        self.board_size = board_size

        # Select board logic based on rules
        if "gomoku" in self.rules:
            self.board_class = GBoard
        elif "pente" in self.rules:
            self.board_class = PenteBoard
        else:
            self.board_class = GBoard

        # --- MODEL LOADING ---
        # The evaluation server does not have a GPU, so 'cpu' will be selected automatically.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the architecture used during training (channels=128)
        self.model = PolicyValueNet(board_size=board_size, channels=128)

        # Path to the trained weights
        model_path = os.path.join("CNN", "alphazero_policy_v_best.pth")

        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                print(f"[Player] Error loading model: {e}")
                self.model = None
        else:
            # print(f"[Player] Warning: Model file not found at {model_path}")
            self.model = None

    def play(self, board, turn_number, last_opponent_move):
        # Optimization: If the board is empty, play center immediately
        is_empty = not any(cell != 0 for row in board for cell in row)
        if is_empty:
            center = self.board_size // 2
            return center, center

        # 1. Reconstruct game state
        game = self.board_class(self.board_size)
        game.board = board

        # According to project rules: 1 is always 'me', 2 is 'opponent'
        me = 1

        # 2. Create MCTS root
        root = Node(game, None)

        # 3. Configure Neural MCTS
        # Simulation limit set to 250 to stay safely within the 5s time limit on CPU
        mcts = MCTS_Neural(root, me, self.model, self.device, simulation_limit=250, timeout=4.0)

        # 4. Get best move
        best_node = mcts.best_move()

        if best_node and best_node.board and best_node.board.last_move:
            col, row = best_node.board.last_move
            # Return (row, col) as required by the project specification
            return row, col
        else:
            # Fallback: Random move if something goes wrong
            possible_moves = [(r, c) for r in range(self.board_size) for c in range(self.board_size) if
                              board[r][c] == 0]
            if possible_moves:
                return random.choice(possible_moves)
            return 0, 0