import random
import time
from copy import deepcopy
from board_logic_and_mcts.Gomoku_Board import GBoard
from board_logic_and_mcts.Node import Node
import csv

# ensure project root on sys.path
import os

# Always build paths relative to this file
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# Default filename (used ONLY when running this file directly)
FILENAME = os.path.join(RAW_DIR, "Gomoku_15x15.csv")


class MCTS:
    def __init__(self, initial_state, current_player, simulation_limit=10000, timeout=60):
        self.root = initial_state
        self.simulation_limit = simulation_limit
        self.current_player = current_player
        self.gameOver = False
        self.timeout = timeout

    def selection(self):
        # Traverse the tree starting from the root, choosing the best child
        # using UCT (Upper Confidence Bound) until a leaf node is reached.
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
        # Simulate a random playout from the given node until the game ends.
        # The simulation follows random moves until there's a winner or a tie.
        sim_board = deepcopy(node.board)
        player = 1 if self.current_player == 2 else 2

        while not sim_board.isBoardFull():
            legal_moves = [(i, j) for i in range(sim_board.board_width) for j in range(sim_board.board_height) if
                           sim_board.isLegalMove(i, j)]
            col, row = random.choice(legal_moves)
            sim_board.makeMove(col, row, player)

            if sim_board.isWon(col, row, player):
                return player  # Return the winner

            # Switch players
            player = 1 if player == 2 else 2

        return 0  # It's a tie (draw)

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
        # Start timing
        start_time = time.time()

        # Main MCTS loop: run simulations, expand, simulate, backpropagate
        leaf = self.selection()
        if not leaf.children:
            self.expansion(leaf)
        if self.check_for_win(leaf):
            return leaf

        children_array_size = len(leaf.children)
        for child in leaf.children:  # Visits all children at least once
            result = self.simulation(child)
            self.backpropagation(child, result)

        children = leaf.children  # Turn the leaf.children into a static variable
        i = children_array_size

        # Run simulations until either simulation_limit or timeout is reached
        while i < self.simulation_limit:
            # Check if timeout has been reached
            elapsed_time = time.time() - start_time
            if elapsed_time >= self.timeout:
                print(f"Timeout reached after {elapsed_time:.2f} seconds")
                break

            leaf = random.choice(children)
            result = self.simulation(leaf)
            self.backpropagation(leaf, result)
            i += 1

        # ---------------------------------------------------

        for idx, child in enumerate(children):
            print(f"NODE {idx + 1} VISITS: ", child.visits)
            print(f"NODE {idx + 1} WINS: ", child.wins)
            print(f"NODE {idx + 1} UCT: ", child.get_uct_value())

        best_node = self.root.best_child()

        print("Simulations: ", i)
        print("BEST NODE VISITS: ", best_node.visits)
        print("BEST NODE WINS", best_node.wins)
        print("BEST UCT : ", best_node.get_uct_value())
        # After all simulations, return the child with the highest win rate

        return self.root.best_child()


# Board size 15
# Starting_turn: The turn when the midgame starts, usually 10-15 moves
# Ending_turn: The turn when the midgame ends, usually 40 moves

def run(size, starting_turn=15, ending_turn=40, timeout=60, simulation_limit_X=10000, simulation_limit_O=10000):
    game = GBoard(size)
    current_turn = 0
    x = 2
    o = 1
    playerX = True  # False for player 1 first, True for player 2 first

    game_is_over = False

    # -- randomize moves in the begining
    for m in range(starting_turn):
        currentplayer = x if playerX else o
        current_turn += 1
        game.makeMove(random.randint(0, 14), random.randint(0, 14), currentplayer)
        game.printBoard()
        playerX = False if playerX else True

    while ((not game.isBoardFull() and not game_is_over) or game.isTie()) and (current_turn < ending_turn):
        currentplayer = x if playerX else o
        game.printBoard()
        print(f"It is now {currentplayer}'s turn!")
        current_turn += 1
        # N sei se a arvore do MCTS vai fzr update do root para preservar as vitórias ou cirar um árvore de zero
        if playerX:
            root = Node(game, None)
            mcts = MCTS(root, x, simulation_limit_X, timeout)
            start = time.time()
            best_node = mcts.best_move()
            col, row = best_node.board.last_move
            end = time.time()
            if mcts.gameOver:
                print(f"{currentplayer} chose column : ", col + 1, " row : ", row + 1)
                print(f"Time taken: {end - start:.2f}")
                break

        else:
            root = Node(game, None)
            mcts = MCTS(root, o, simulation_limit_O, timeout)
            start = time.time()
            best_node = mcts.best_move()
            col, row = best_node.board.last_move
            end = time.time()
            if mcts.gameOver:
                print(f"{currentplayer} chose column : ", col + 1, " row : ", row + 1)
                print(f"Time taken: {end - start:.2f}")
                break

        game.makeMove(col, row, currentplayer)
        print(f"{currentplayer} chose column : ", col + 1, " row : ", row + 1)

        flattened_board = []
        for row_data in game.board:
            for cell in row_data:
                flattened_board.append(cell)

        # -- Write the flattened row into the csv + the coordinates of the last move
        with open(FILENAME, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            row_with_move = flattened_board + [currentplayer] + [best_node.board.last_move]
            writer.writerow(row_with_move)

        playerX = False if playerX else True

    # After game is won
    game.printBoard()
    if game.isTie():
        print("It's a tie!")
    else:
        print(f"Player {currentplayer} has won!\n")

    # Use MCTS to generate the boards,
    # Write the output into the dataset.


if __name__ == "__main__":
    # Running the generator directly (not via wrapper)
    # Here you may customize defaults, but DO NOT call run() on import
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"[INFO] Direct mode. Writing to: {FILENAME}")
    # Example single run (optional, możesz zakomentować jeśli nie używasz):
    # run(size=15, starting_turn=15, ending_turn=60, timeout=60,
    #     simulation_limit_X=10000, simulation_limit_O=10000)