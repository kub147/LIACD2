import torch
import numpy as np
import time
import random

from board_logic_and_mcts.MCTS import MCTS
from board_logic_and_mcts.Node import Node


class MCTS_Neural(MCTS):
    def __init__(self, initial_state, current_player, model, device, simulation_limit=800, timeout=5):
        # Do not pass timeout to parent MCTS as it doesn't support it
        super().__init__(initial_state, current_player, simulation_limit)
        self.timeout = timeout

        self.model = model
        self.device = device

        if self.model:
            self.model.eval()

    def evaluate_leaf(self, node):
        """
        Evaluates the board state using the Neural Network instead of random rollouts.
        Returns a value from -1 (loss) to 1 (win) from the perspective of the player to move.
        """
        board_np = np.array(node.board.board)

        # Determine current player for the network input (Channel 0 = Current, Channel 1 = Opponent)
        flat_board = board_np.flatten()
        stones = np.count_nonzero(flat_board)
        # Assuming Black (1) starts. Even stones = Player 1's turn.
        current_p = 1 if stones % 2 == 0 else 2

        plane_me = (board_np == current_p).astype(np.float32)
        plane_opp = (board_np == (3 - current_p)).astype(np.float32)

        input_tensor = np.stack([plane_me, plane_opp])
        input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            _, value_pred = self.model(input_tensor)

        value = value_pred.item()
        return value, current_p

    def backpropagation_neural(self, node, value, value_perspective_player):
        """
        Backpropagation adapted for continuous value [-1, 1] from the network.
        """
        while node is not None:
            node.visits += 1
            # Convert [-1, 1] range to probability [0, 1]
            win_prob = (value + 1) / 2.0

            if value_perspective_player == self.current_player:
                score = win_prob
            else:
                score = 1.0 - win_prob

            node.wins += score
            node = node.parent

    def check_for_win(self, leaf) -> bool:
        return super().check_for_win(leaf)

    def best_move(self):
        """
        Main MCTS loop using Neural Network evaluation and strict time control.
        """
        start_time = time.time()

        # Fallback to vanilla MCTS if model is missing
        if self.model is None:
            return super().best_move()

        # 1. Initial Selection & Expansion (Root only)
        leaf = self.selection()
        if not leaf.children:
            self.expansion(leaf)

        if self.check_for_win(leaf):
            return leaf

        # 2. Main Simulation Loop
        i = 0
        while i < self.simulation_limit:
            # Time control check
            if (time.time() - start_time) >= self.timeout:
                break

            leaf = self.root

            # Selection
            while leaf.children:
                leaf = leaf.best_child()

            # Expansion
            if not self.check_for_win(leaf) and not leaf.children:
                self.expansion(leaf)
                if leaf.children:
                    leaf = random.choice(leaf.children)

            # Evaluation (Neural Network)
            val, perspective = self.evaluate_leaf(leaf)

            # Backpropagation
            self.backpropagation_neural(leaf, val, perspective)

            i += 1

        return self.root.best_child()