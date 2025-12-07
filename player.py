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

        # --- 1. GAME & MODEL SELECTION ---
        # Select the appropriate board logic and model weights based on the rules.
        if "gomoku" in self.rules:
            self.board_class = GBoard
            model_filename = "CNN/final_gomoku_model.pth"  # Model trained for Gomoku
        elif "pente" in self.rules:
            self.board_class = PenteBoard
            model_filename = "pente_model.pth"  # Model trained for Pente
        else:
            # Fallback to Gomoku settings
            self.board_class = GBoard
            model_filename = "alphazero_policy_v_best.pth"

        # --- 2. MODEL LOADING ---
        # The evaluation server does not have a GPU, so 'cpu' will be selected automatically.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the architecture (same for both games: 2 channels, 128 filters)
        self.model = PolicyValueNet(board_size=board_size, channels=128)

        # Construct the full path to the weight file
        model_path = os.path.join("CNN", model_filename)

        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                print(f"[Player] Error loading model {model_filename}: {e}")
                self.model = None
        else:
            # Model file missing - MCTS will run in random rollout mode (fallback)
            self.model = None

    def play(self, board, turn_number, last_opponent_move):
        # Optimization: If the board is empty, play center immediately to save time.
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
        # Simulation limit set to 400 to balance strength and speed within 5s limit
        # Timeout buffer set to 4.0s
        mcts = MCTS_Neural(root, me, self.model, self.device, simulation_limit=400, timeout=4.0)

        # 4. Get best move
        best_node = mcts.best_move()

        if best_node and best_node.board and best_node.board.last_move:
            col, row = best_node.board.last_move
            # Return (row, col) as required by the project specification
            return row, col
        else:
            # Fallback: Random move if something goes wrong or no legal moves found
            possible_moves = [(r, c) for r in range(self.board_size) for c in range(self.board_size) if
                              board[r][c] == 0]
            if possible_moves:
                return random.choice(possible_moves)
            return 0, 0