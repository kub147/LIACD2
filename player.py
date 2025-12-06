import torch
import os
import time
import random

# Importy z Twojego projektu
from board_logic_and_mcts.Gomoku_Board import GBoard
from board_logic_and_mcts.Pente_Board import PBoard as PenteBoard
from board_logic_and_mcts.MCTS_Neural import MCTS_Neural
from board_logic_and_mcts.Node import Node

# --- FIX: Używamy nn_model, bo na nim był robiony trening (train_policy.py) ---
from CNN.nn_model import PolicyValueNet


class Player:
    def __init__(self, rules, board_size):
        self.rules = rules.lower()
        self.board_size = board_size

        if "gomoku" in self.rules:
            self.board_class = GBoard
        elif "pente" in self.rules:
            self.board_class = PenteBoard
        else:
            self.board_class = GBoard

        # --- ŁADOWANIE MODELU ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # FIX: Zmiana klasy na PolicyValueNet i channels na 128 (zgodnie z train_policy.py)
        self.model = PolicyValueNet(board_size=board_size, channels=128)

        # Ścieżka do wag
        model_path = os.path.join("CNN", "alphazero_policy_v_best.pth")

        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                # print(f"[Player] Model loaded successfully.")
            except Exception as e:
                print(f"[Player] Error loading model: {e}")  # Zostawiamy printa do debugu
                self.model = None
        else:
            self.model = None

    def play(self, board, turn_number, last_opponent_move):
        # --- FIX: Optymalizacja pustej planszy (natychmiastowy ruch) ---
        # Sprawdzamy czy plansza jest pusta (suma 1 i 2 wynosi 0)
        is_empty = not any(cell != 0 for row in board for cell in row)
        if is_empty:
            center = self.board_size // 2
            return center, center


        # 1. Odtworzenie stanu gry
        game = self.board_class(self.board_size)
        game.board = board

        me = 1

        # 2. Tworzymy korzeń drzewa
        root = Node(game, None)

        # 3. Konfiguracja MCTS
        mcts = MCTS_Neural(root, me, self.model, self.device, simulation_limit=250, timeout=4.0)
        # 4. Decyzja
        best_node = mcts.best_move()

        if best_node and best_node.board and best_node.board.last_move:
            col, row = best_node.board.last_move
            return row, col
        else:
            # Fallback
            import random
            possible_moves = [(r, c) for r in range(self.board_size) for c in range(self.board_size) if
                              board[r][c] == 0]
            if possible_moves:
                return random.choice(possible_moves)
            return 0, 0