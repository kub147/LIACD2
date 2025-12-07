import sys
import os
import random
import time
import csv
from copy import deepcopy
import math  # Import math at module level

# --- PATH CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(current_dir, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# NEW FILENAME - ensuring fresh data
FILENAME = os.path.join(RAW_DIR, "clean_gomoku_data_FINAL_FIXED.csv")


# ==========================================
# 1. INTEGRATED AND FIXED GBOARD CLASS
# ==========================================
class GBoard:
    def __init__(self, size):
        self.counter = 0
        self.board_width = size
        self.board_height = size
        self.board = [[0 for _ in range(self.board_width)] for _ in range(self.board_height)]
        self.last_move = None

    def isBoardFull(self) -> bool:
        return self.counter == self.board_height * self.board_width

    def isEmpty(self, x, y) -> bool:
        return self.board[y][x] == 0

    def isLegalMove(self, x, y) -> bool:
        if self.isBoardFull(): return False
        if not (0 <= x < self.board_width and 0 <= y < self.board_height): return False
        return self.isEmpty(x, y)

    def makeMove(self, x, y, current_player) -> bool:
        self.board[y][x] = current_player
        self.counter += 1
        self.last_move = (x, y)
        return True

    def count_in_direction(self, current_player, dx, dy, x, y):
        count = 0
        while (0 <= x + dx < self.board_width and
               0 <= y + dy < self.board_height and
               self.board[y + dy][x + dx] == current_player):
            x += dx
            y += dy
            count += 1
        return count

    def isWon(self, x, y, current_player) -> bool:
        # --- CRITICAL FIX: Fixes "first column only" bug ---
        if x is None or y is None:
            return False
        # -------------------------------------------------------------

        directions = [((0, 1), (0, -1)),  # Horizontal
                      ((1, 0), (-1, 0)),  # Vertical
                      ((1, 1), (-1, -1)),  # Diagonal \
                      ((1, -1), (-1, 1))]  # Diagonal /

        for (dy1, dx1), (dy2, dx2) in directions:
            total = (self.count_in_direction(current_player, dx1, dy1, x, y) +
                     self.count_in_direction(current_player, dx2, dy2, x, y) + 1)
            if total >= 5:
                return True
        return False

    def isTie(self) -> bool:
        return self.isBoardFull()

    def getPossibleMoves(self, current_player):
        possible_boards = []
        # Correct iteration: y (rows), x (columns)
        for y in range(self.board_height):
            for x in range(self.board_width):
                if self.isLegalMove(x, y):
                    new_board = deepcopy(self)
                    new_board.makeMove(x, y, current_player)
                    possible_boards.append(new_board)
        return possible_boards


# ==========================================
# 2. INTEGRATED NODE CLASS (WITH ZERO DIVISION FIX)
# ==========================================
class Node:
    def __init__(self, board, parent=None):
        self.board = board
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0

    def add_child(self, child_node):
        self.children.append(child_node)

    def best_child(self):
        # --- FIX: Check children with 0 visits first ---
        # If any child has not been visited, we must select it
        # before using the formula that divides by visits.
        for child in self.children:
            if child.visits == 0:
                return child

        # If all visited, use UCT formula
        # Adding small epsilon to denominator is not needed due to loop above, but safer
        return max(self.children,
                   key=lambda c: (c.wins / c.visits) + 1.41 * math.sqrt(math.log(self.visits) / c.visits))


# ==========================================
# 3. INTEGRATED MCTS CLASS (Generator)
# ==========================================
class MCTS:
    def __init__(self, initial_state, current_player, simulation_limit=1000, timeout=60):
        self.root = initial_state
        self.simulation_limit = simulation_limit
        self.current_player = current_player
        self.timeout = timeout

    def selection(self):
        node = self.root
        while node.children:
            node = node.best_child()
        return node

    def expansion(self, node):
        moves_made = node.board.counter
        player_to_move = 1 if (moves_made % 2 == 0) else 2

        possible_states = node.board.getPossibleMoves(player_to_move)

        # --- FIX: SHUFFLE THE MOVES! ---
        random.shuffle(possible_states)
        # ---------------------------

        for state in possible_states:
            new_node = Node(state, parent=node)
            node.add_child(new_node)

    def simulation(self, node):
        sim_board = deepcopy(node.board)
        moves_made = sim_board.counter
        player = 1 if (moves_made % 2 == 0) else 2

        while not sim_board.isBoardFull():
            legal_moves = [(x, y) for y in range(sim_board.board_height) for x in range(sim_board.board_width) if
                           sim_board.isLegalMove(x, y)]
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
            elif result == 0:  # Draw
                node.wins += 0.5
            node = node.parent

    def best_move(self):
        start_time = time.time()
        leaf = self.selection()

        if not leaf.children:
            if leaf.board.last_move:
                last_x, last_y = leaf.board.last_move
                last_player = 1 if (leaf.board.counter % 2 != 0) else 2
                if leaf.board.isWon(last_x, last_y, last_player):
                    return leaf

            self.expansion(leaf)

        if leaf.children:
            child_to_sim = random.choice(leaf.children)
            result = self.simulation(child_to_sim)
            self.backpropagation(child_to_sim, result)
        else:
            result = self.simulation(leaf)
            self.backpropagation(leaf, result)

        count = 0
        while count < self.simulation_limit:
            if (time.time() - start_time) >= self.timeout:
                break

            node_to_explore = self.selection()
            if not node_to_explore.children:
                if node_to_explore.board.last_move:
                    lx, ly = node_to_explore.board.last_move
                    lp = 1 if (node_to_explore.board.counter % 2 != 0) else 2
                    if node_to_explore.board.isWon(lx, ly, lp):
                        self.backpropagation(node_to_explore, lp)
                        count += 1
                        continue

                self.expansion(node_to_explore)

            if node_to_explore.children:
                # FIX: Check if children exist before random choice
                c = random.choice(node_to_explore.children)
                res = self.simulation(c)
                self.backpropagation(c, res)
            else:
                # If no children after expansion (e.g. board full), simulate this node
                res = self.simulation(node_to_explore)
                self.backpropagation(node_to_explore, res)

            count += 1

        if not self.root.children:
            return None

        return max(self.root.children, key=lambda c: c.visits)


# ==========================================
# 4. MAIN GENERATION LOOP
# ==========================================
def run(size, starting_turn=8, ending_turn=80, timeout=30, simulation_limit=400):
    game = GBoard(size)
    current_turn = 0
    playerX = True
    game_is_over = False

    # 1. Random moves at start
    for _ in range(starting_turn):
        currentplayer = 1 if playerX else 2
        legal_moves = [(x, y) for y in range(size) for x in range(size) if game.isLegalMove(x, y)]
        if not legal_moves: break

        col, row = random.choice(legal_moves)
        game.makeMove(col, row, currentplayer)

        if game.isWon(col, row, currentplayer):
            game_is_over = True
            break

        playerX = not playerX
        current_turn += 1

    game_history = []

    # 2. MCTS vs MCTS Game
    while ((not game.isBoardFull() and not game_is_over) or game.isTie()) and (current_turn < ending_turn):
        currentplayer = 1 if playerX else 2

        root = Node(game, None)
        mcts = MCTS(root, currentplayer, simulation_limit, timeout)

        best_node = mcts.best_move()

        if best_node and best_node.board.last_move:
            col, row = best_node.board.last_move

            flattened_board = [cell for row_data in game.board for cell in row_data]
            game_history.append([flattened_board, currentplayer, (col, row)])

            game.makeMove(col, row, currentplayer)
            print(f"   Turn {current_turn}: Player {currentplayer} plays ({col}, {row})")

            if game.isWon(col, row, currentplayer):
                print(f"   -> WINNER: Player {currentplayer}")
                game_is_over = True
                winner = currentplayer
                break
        else:
            print("   -> No moves left / Error")
            break

        playerX = not playerX
        current_turn += 1

    # 3. Save Result
    winner = 0
    if game_is_over:
        if game.last_move:
            lx, ly = game.last_move
            winner = game.board[ly][lx]

    print(f"[Generator] Game finished. Winner: {winner}. Saving {len(game_history)} moves.")

    with open(FILENAME, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for state, player_who_moved, move in game_history:
            val = 0.0
            if winner != 0:
                val = 1.0 if player_who_moved == winner else -1.0

            writer.writerow(state + [player_who_moved] + [move[0], move[1]] + [val])


if __name__ == "__main__":
    SIMS = 400
    GAMES_TO_PLAY = 2000

    print(f"--- STARTING DATA GENERATION (FIXED & SAFE) ---")
    print(f"Output file: {FILENAME}")

    for i in range(GAMES_TO_PLAY):
        print(f"Generating game {i + 1}...")
        try:
            run(size=15,
                starting_turn=6,
                ending_turn=100,
                timeout=10,
                simulation_limit=SIMS)
        except Exception as e:
            print(f"Error in game {i + 1}: {e}")
            import traceback

            traceback.print_exc()
            continue