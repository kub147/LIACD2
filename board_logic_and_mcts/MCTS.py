import random
from copy import deepcopy
from Node import Node

class MCTS:
    def __init__(self, initial_state, current_player, simulation_limit=10000):
        self.root = initial_state
        self.simulation_limit = simulation_limit
        self.current_player = current_player
        self.gameOver = False


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
#       sim_board.print_board()


        while not sim_board.isBoardFull():            
            legal_moves = [(i, j) for i in range(sim_board.board_width) for j in range(sim_board.board_height) if sim_board.isLegalMove(i, j)]
            col, row = random.choice(legal_moves)
            sim_board.makeMove(col, row, player)
#            sim_board.print_board()

            if sim_board.isWon(col, row, player):
#                print(player, "WON")
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
        possible_moves = [(i, j) for i in range(leaf.board.board_width) for j in range(leaf.board.board_height) if leaf.board.isLegalMove(i, j)]

        for i, j in possible_moves:
            nodeCopy = deepcopy(leaf)
            nodeCopy.board.makeMove(i, j, self.current_player)
            if nodeCopy.board.isWon(i, j, self.current_player):
                leaf.board.makeMove(i, j, self.current_player)
                self.gameOver = True
                return True
        
        return False
    
    def best_move(self):  # Main MCTS loop: run simulations, expand, simulate, backpropagate
        leaf = self.selection()
        if not leaf.children:
            self.expansion(leaf)

        if self.check_for_win(leaf):
            return leaf
            
        children_array_size = len(leaf.children)    

        for child in leaf.children: # Visits all children at least once, remember children is a node
            result = self.simulation(child)
            self.backpropagation(child, result)

        children = leaf.children #Turn the leaf.children into a static variable

        for i in range(children_array_size, self.simulation_limit + 1): 
            leaf = random.choice(children)
            result = self.simulation(leaf)
            self.backpropagation(leaf, result)

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
