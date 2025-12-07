from copy import deepcopy

class GBoard:
    def __init__(self, size):
        self.counter = 0
        self.board_width = size
        self.board_height = size
        self.board = self.getBoard()
        self.last_move = None

    # returns a tuple of the board (if necessary for hashing)
    def toTuple(self) -> tuple:
        return tuple(tuple(row) for row in self.board)

    # Function that generates the board depending on the size
    def getBoard(self) -> list:
        return [[0 for _ in range(self.board_width)] for _ in range(self.board_height)]

    # Prints out the board into the terminal for visuals
    def printBoard(self):
        max_num_width = len(str(self.board_height))
        header = " " * max_num_width + "  " + " ".join(str(_ + 1).rjust(max_num_width) for _ in range(self.board_width))
        print("\n" + header)

        for row_idx, row in enumerate(self.board):
            row_num = str(row_idx + 1).rjust(max_num_width)
            row_content = " ".join(str(cell).rjust(max_num_width) for cell in row)
            print(row_num + "  " + row_content)
        print()

    # Checks if space is empty
    def isEmpty(self, x, y) -> bool:
        return self.board[y][x] == 0

    # Checks if the board is full
    def isBoardFull(self) -> bool:
        return self.counter == self.board_height * self.board_width

    # Checks if the move is legal
    def isLegalMove(self, x, y) -> bool:
        if self.isBoardFull():
            return False
        if not (0 <= x < self.board_width and 0 <= y < self.board_height):
            return False
        return self.isEmpty(x, y)

    # Makes the move
    def makeMove(self, x, y, current_player) -> bool:
        self.board[y][x] = current_player
        self.counter += 1
        self.last_move = (x, y)
        return True

    # Win condition verification helper
    def count_in_direction(self, current_player, dx, dy, x, y):
        count = 0
        while (0 <= x + dx < self.board_width and
               0 <= y + dy < self.board_height and
               self.board[y + dy][x + dx] == current_player):
            x += dx
            y += dy
            count += 1
        return count

    # Verifies if the player has won
    def isWon(self, x, y, current_player) -> bool:
        # Directions: Horizontal, Vertical, Diagonal, Anti-Diagonal
        directions = [((0, 1), (0, -1)),
                      ((1, 0), (-1, 0)),
                      ((1, 1), (-1, -1)),
                      ((1, -1), (-1, 1))]

        # CRITICAL FIX: Ensure coordinates are valid
        if x is None or y is None:
            return False

        for (dy1, dx1), (dy2, dx2) in directions:
            # Count consecutive stones in both directions + the stone itself
            total = (self.count_in_direction(current_player, dx1, dy1, x, y) +
                     self.count_in_direction(current_player, dx2, dy2, x, y) + 1)

            # Win condition: 5 or more in a row
            if total >= 5:
                return True
        return False

    def isTie(self) -> bool:
        # A tie occurs if board is full and no one has won
        # Note: Checking isWon for all cells is expensive, simplified check usually sufficient for MCTS
        if not self.isBoardFull():
            return False
        # If full, verify no winner exists (rare edge case in Gomoku, usually caught earlier)
        return True

    def getPossibleMoves(self, current_player):
        possible_boards = []
        for i in range(self.board_height):
            for j in range(self.board_width):
                if self.isLegalMove(j, i):  # Note: isLegalMove takes (x, y) -> (j, i)
                    new_board = deepcopy(self)
                    new_board.makeMove(j, i, current_player)
                    possible_boards.append(new_board)

        return possible_boards