import torch
import numpy as np
import time
import random

# Importujemy starą klasę
from board_logic_and_mcts.MCTS import MCTS
from board_logic_and_mcts.Node import Node


class MCTS_Neural(MCTS):
    def __init__(self, initial_state, current_player, model, device, simulation_limit=800, timeout=5):
        # FIX: Nie przekazujemy 'timeout' do rodzica, bo MCTS.py go nie obsługuje
        super().__init__(initial_state, current_player, simulation_limit)
        self.timeout = timeout  # Zapisujemy timeout w nowej klasie

        self.model = model
        self.device = device

        if self.model:
            self.model.eval()

    def evaluate_leaf(self, node):
        """
        Zamiast robić losowy 'simulation', oceniamy planszę siecią neuronową.
        Zwraca wartość od -1 (przegrana) do 1 (wygrana) z perspektywy gracza, który ma ruch.
        """
        board_size = node.board.board_width
        board_np = np.array(node.board.board)

        # Ustalenie kto ma ruch (dla sieci 1=Ja, 2=Przeciwnik)
        flat_board = board_np.flatten()
        stones = np.count_nonzero(flat_board)
        # Zakładamy, że czarny (1) zaczyna. Jeśli parzysta liczba kamieni -> ruch ma 1.
        current_p = 1 if stones % 2 == 0 else 2

        plane_me = (board_np == current_p).astype(np.float32)
        plane_opp = (board_np == (3 - current_p)).astype(np.float32)

        input_tensor = np.stack([plane_me, plane_opp])
        input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            policy_logits, value_pred = self.model(input_tensor)

        value = value_pred.item()
        return value, current_p

    def backpropagation_neural(self, node, value, value_perspective_player):
        while node is not None:
            node.visits += 1
            win_prob = (value + 1) / 2.0

            if value_perspective_player == self.current_player:
                score = win_prob
            else:
                score = 1.0 - win_prob

            node.wins += score
            node = node.parent

    def check_for_win(self, leaf) -> bool:
        # Wywołujemy metodę z klasy bazowej (MCTS), jeśli jest dostępna,
        # w przeciwnym razie musimy ją zaimplementować lub użyć super().
        # W Twoim pliku MCTS.py metoda check_for_win istnieje.
        return super().check_for_win(leaf)

    def best_move(self):
        start_time = time.time()

        if self.model is None:
            return super().best_move()

        # 1. Selection & Expansion (tylko korzenia)
        leaf = self.selection()
        if not leaf.children:
            self.expansion(leaf)

        if self.check_for_win(leaf):
            return leaf

        # --- TU BYŁ BŁĄD: Usunęliśmy pętlę "Inicjalna ocena", która marnowała 8 sekund ---

        i = 0

        # Główna pętla z zabezpieczeniem czasowym
        while i < self.simulation_limit:
            # SPRAWDZENIE CZASU - TERAZ ZADZIAŁA POPRAWNIE
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

            # Evaluation
            val, perspective = self.evaluate_leaf(leaf)

            # Backpropagation
            self.backpropagation_neural(leaf, val, perspective)

            i += 1

        return self.root.best_child()