import sys
import os

# --- PATH CONFIGURATION ---
# Add project root to sys.path to allow imports from sibling directories
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
# --------------------------

import random
import time
import csv
from copy import deepcopy

from board_logic_and_mcts.Gomoku_Board import GBoard
from board_logic_and_mcts.Node import Node

# Path setup
RAW_DIR = os.path.join(current_dir, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# Output filename
FILENAME = os.path.join(RAW_DIR, "golden_data_with_value.csv")


class MCTS:
    def __init__(self, initial_state, current_player, simulation_limit=10000, timeout=60):
        self.root = initial_state
        self.simulation_limit = simulation_limit
        self.current_player = current_player
        self.gameOver = False
        self.timeout = timeout

    def selection(self):
        node = self.root
        while node.children:
            node = node.best_child()
        return node

    def expansion(self, node):
        current_player = self.current_player
        for board in node.board.getPossibleMoves(current_player):
            new_node = Node(board, parent=node)
            node.add_child(new_node)

    def simulation(self, node):
        sim_board = deepcopy(node.board)
        player = 1 if self.current_player == 2 else 2

        while not sim_board.isBoardFull():
            legal_moves = [(i, j) for i in range(sim_board.board_width) for j in range(sim_board.board_height) if
                           sim_board.isLegalMove(i, j)]
            if not legal_moves: break
            col, row = random.choice(legal_moves)
            sim_board.makeMove(col, row, player)

            if sim_board.isWon(col, row, player):
                return player

            player = 1 if player == 2 else 2
        return 0

    def backpropagation(self, node, result):
        while node is not None:
            node.visits += 1
            if result == self.current_player:
                node.wins += 1
            elif result == 0:
                node.wins += 0.5
            node = node.parent

    def check_for_win(self, leaf) -> bool:
        possible_moves = [(i, j) for i in range(leaf.board.board_width) for j in range(leaf.board.board_height) if
                          leaf.board.isLegalMove(i, j)]

        for i, j in possible_moves:
            nodeCopy = deepcopy(leaf)
            nodeCopy.board.makeMove(i, j, self.current_player)
            if nodeCopy.board.isWon(i, j, self.current_player):
                leaf.board.makeMove(i, j, self.current_player)
                self.gameOver = True
                return True
        return False

    def best_move(self):
        start_time = time.time()
        leaf = self.selection()
        if not leaf.children:
            self.expansion(leaf)
        if self.check_for_win(leaf):
            return leaf

        children_array_size = len(leaf.children)
        for child in leaf.children:
            result = self.simulation(child)
            self.backpropagation(child, result)

        children = leaf.children
        i = children_array_size

        while i < self.simulation_limit:
            if (time.time() - start_time) >= self.timeout:
                break
            leaf = random.choice(children)
            result = self.simulation(leaf)
            self.backpropagation(leaf, result)
            i += 1

        return self.root.best_child()


def run(size, starting_turn=15, ending_turn=40, timeout=60, simulation_limit_X=1000, simulation_limit_O=1000):
    game = GBoard(size)
    current_turn = 0
    x = 2
    o = 1
    playerX = True
    game_is_over = False

    # Randomize start
    for m in range(starting_turn):
        currentplayer = x if playerX else o
        current_turn += 1
        game.makeMove(random.randint(0, size - 1), random.randint(0, size - 1), currentplayer)
        playerX = not playerX

    game_history = []

    while ((not game.isBoardFull() and not game_is_over) or game.isTie()) and (current_turn < ending_turn):
        currentplayer = x if playerX else o

        if playerX:
            root = Node(game, None)
            mcts = MCTS(root, x, simulation_limit_X, timeout)
            best_node = mcts.best_move()
        else:
            root = Node(game, None)
            mcts = MCTS(root, o, simulation_limit_O, timeout)
            best_node = mcts.best_move()

        if best_node.board.last_move:
            col, row = best_node.board.last_move

            # Buffer data in RAM
            flattened_board = [cell for row_data in game.board for cell in row_data]
            game_history.append([flattened_board, currentplayer, (col, row)])

            game.makeMove(col, row, currentplayer)

            if game.isWon(col, row, currentplayer):
                game_is_over = True
                break
        else:
            break

        playerX = not playerX
        current_turn += 1

    # Determine winner and save
    winner = 0
    if game_is_over:
        winner = currentplayer  # The player who made the last move won

    with open(FILENAME, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for state, player_who_moved, move in game_history:
            # Value Calculation:
            # 1.0 if the move led to a win for the player who made it
            # -1.0 if it led to a loss
            val = 0.0
            if winner != 0:
                val = 1.0 if player_who_moved == winner else -1.0

            writer.writerow(state + [player_who_moved] + [move[0], move[1]] + [val])

    print(f"[Generator] Game finished. Winner: {winner}. Data saved.")


if __name__ == "__main__":
    # Generate games in a loop
    SIMS = 800
    GAMES_TO_PLAY = 2000

    print(f"--- STARTING DATA GENERATION ({GAMES_TO_PLAY} games) ---")
    print(f"Output file: {FILENAME}")

    for i in range(GAMES_TO_PLAY):
        print(f"Generating game {i + 1}...")
        try:
            run(size=15,
                starting_turn=8,
                ending_turn=80,
                timeout=30,
                simulation_limit_X=SIMS,
                simulation_limit_O=SIMS)
        except Exception as e:
            print(f"Error in game {i + 1}: {e}")
            continue