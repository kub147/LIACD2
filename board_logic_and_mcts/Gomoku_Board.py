from copy import deepcopy

class GBoard:
    def __init__(self, size): 
        self.counter = 0; 
        self.board_width = size
        self.board_height = size
        self.board = self.getBoard()
        self.last_move = None

    # returns a tuple of the board (IF NECESSARY)
    def toTuple(self) -> tuple:
        return tuple(tuple(row) for row in self.board)


    # A function to generate custom board to test the code
    def getSimulationBoard(self) -> list:
        return []                

    # Function that generates the board depending on the size
    def getBoard(self) -> list:
        return [[0 for _ in range(self.board_width)] for _ in range(self.board_height)]


    # Prints out the board into the terminal for visuals
    def printBoard(self):
        # Calculate the width needed for row/column numbers (for alignment)
        max_num_width = len(str(self.board_height))
        
        # Print header row with column numbers
        header = " " * max_num_width + "  " + " ".join(str(_ + 1).rjust(max_num_width) for _ in range(self.board_width))
        print("\n" + header)
        
        # Print each row with row number
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
            return False    # Board is full
        if (not 0 <= x <= self.board_width or not 0 <= y <= self.board_height):
            return False    # OUT OF BOUNDS
        
        return self.isEmpty(x, y)

    # Makes the move (Places a piece at the given coordenantes depending on the player)
    def makeMove(self, x, y, current_player) -> bool:
        self.board[y][x] = current_player
        self.counter += 1
        self.last_move = (x, y)


    # Win condition verification
    # Auxiliary function that counts pieces among given direction
    def count_in_direction(self, current_player, dx, dy, x, y):
        count = 0
        while (0 <= x + dx < self.board_width and 
            0 <= y + dy < self.board_height and 
            self.board[y + dy][x + dx] == current_player):
            x += dx
            y += dy
            count += 1
        return count

    # Verifies if the player has won, given the piece location as only the last piece played would affect the baord state
    def isWon(self, x, y, current_player) -> bool:
        # Directions for auxiliary function
        directions = [((0, 1), (0, -1)),  # Horizontal
                    ((1, 0), (-1, 0)),  # Vertical
                    ((1, 1), (-1, -1)),  # Main diagonal
                    ((1, -1), (-1, 1))]  # Second-diagonal
    
        if x or y is None:
            return False

        for (dy1, dx1), (dy2, dx2) in directions:
            total = (self.count_in_direction(current_player, dx1, dy1, x, y) +
                    self.count_in_direction(current_player, dx2, dy2, x, y) + 1)  # +1 for the initial piece
            if total >= 5:
                return True
        return False

    def isTie(self) -> bool:
        return self.isBoardFull() and not any(self.isWon(p, x, y) for x in range(self.board_width) for y in range(self.board_height) for p in [1, 2])


    def getPossibleMoves(self, current_player):
        possible_boards = []

        for i in range(self.board_height):
            for j in range(self.board_width):
                if self.isLegalMove(i, j):
                    new_board = deepcopy(self)
                    new_board.makeMove(i, j, current_player)
                    possible_boards.append(new_board)
                    
        return possible_boards
