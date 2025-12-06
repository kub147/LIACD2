import math

class Node:
    def __init__(self, board, parent=None):
        self.board = board
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.board.getPossibleMoves())

    def best_child(self):
        return max(self.children, key=lambda child: child.get_uct_value())

    def add_child(self, child):
        self.children.append(child) 

    def get_uct_value(self, c=1.4):
        exploitation = self.wins / (self.visits + 1e-6)
        exploration = c * math.sqrt(math.log(self.parent.visits + 1) / (self.visits + 1e-6))
        return exploitation + exploration

