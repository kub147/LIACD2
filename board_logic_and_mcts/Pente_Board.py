from copy import deepcopy


class PBoard:
    def __init__(self, size):
        self.counter = 0
        self.board_width = size
        self.board_height = size
        self.board = self.getBoard()
        self.X_pieces_captured_by_O = 0
        self.O_pieces_captured_by_X = 0
        self.last_move = None

    def toTuple(self) -> tuple:
        return tuple(tuple(row) for row in self.board)

    def getBoard(self) -> list:
        return [[0 for _ in range(self.board_width)] for _ in range(self.board_height)]

    def printBoard(self):
        max_num_width = len(str(self.board_height))
        header = " " * max_num_width + "  " + " ".join(str(_ + 1).rjust(max_num_width) for _ in range(self.board_width))
        print("\n" + header)
        for row_idx, row in enumerate(self.board):
            row_num = str(row_idx + 1).rjust(max_num_width)
            row_content = " ".join(str(cell).rjust(max_num_width) for cell in row)
            print(row_num + "  " + row_content)
        print()

    def isEmpty(self, x, y) -> bool:
        return self.board[y][x] == 0

    def isBoardFull(self) -> bool:
        return self.counter == self.board_height * self.board_width

    def isLegalMove(self, x, y) -> bool:
        if self.isBoardFull():
            return False
        if not (0 <= x < self.board_width and 0 <= y < self.board_height):
            return False
        return self.isEmpty(x, y)

    def makeMove(self, x, y, current_player) -> None:
        self.board[y][x] = current_player
        self.counter += 1
        self.last_move = (x, y)
        self.captureLogic(x, y, current_player)

    # Capture logic: Check 8 directions for the pattern: Friendly-Opponent-Opponent-Friendly
    def scan_surround(self, x, y, current_player) -> list:
        directions = [(0, 1), (0, -1),
                      (1, 0), (-1, 0),
                      (1, 1), (-1, -1),
                      (1, -1), (-1, 1)]

        opponent = 1 if current_player == 2 else 2
        valid_directions = []

        for (dy, dx) in directions:
            adj_x = x + dx
            adj_y = y + dy
            if (0 <= adj_x < self.board_width and
                    0 <= adj_y < self.board_height and
                    self.board[adj_y][adj_x] == opponent):
                valid_directions.append((dy, dx))

        return valid_directions

    def explore_direction(self, x, y, current_player, directions) -> None:
        opponent = 1 if current_player == 2 else 2

        for (dy, dx) in directions:
            # Positions: 1 (opp), 2 (opp), 3 (friendly)
            pos1_x, pos1_y = x + dx, y + dy
            pos2_x, pos2_y = x + 2 * dx, y + 2 * dy
            pos3_x, pos3_y = x + 3 * dx, y + 3 * dy

            if (0 <= pos3_x < self.board_width and 0 <= pos3_y < self.board_height):
                if (self.board[pos1_y][pos1_x] == opponent and
                        self.board[pos2_y][pos2_x] == opponent and
                        self.board[pos3_y][pos3_x] == current_player):

                    # Capture!
                    self.board[pos1_y][pos1_x] = 0
                    self.board[pos2_y][pos2_x] = 0
                    if current_player == 2:
                        self.O_pieces_captured_by_X += 1  # Counts as 1 capture event (of 2 stones)?
                        # Rules say "5 pairs". So usually we count captures as +1 per pair.
                        # Assuming logic here increases score by 2 stones or 1 pair?
                        # Let's align with typical Pente: 5 captures = win.
                        # Your original code added 2? Let's check logic.
                        # Usually capture count tracks pairs. I will assume +1 means 1 pair.
                        # NOTE: Original code had += 2. If win condition is 10 stones, that's fine.
                        pass

                        # Update counters based on stones removed (assuming win condition checks stones)
                    if current_player == 2:
                        self.O_pieces_captured_by_X += 2  # Removed 2 stones
                    else:
                        self.X_pieces_captured_by_O += 2  # Removed 2 stones

                    self.counter -= 2

    def captureLogic(self, x, y, current_player):
        dir = self.scan_surround(x, y, current_player)
        if dir:
            self.explore_direction(x, y, current_player, dir)

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
        directions = [((0, 1), (0, -1)),
                      ((1, 0), (-1, 0)),
                      ((1, 1), (-1, -1)),
                      ((1, -1), (-1, 1))]

        # CRITICAL FIX
        if x is None or y is None:
            return False

        # 1. Check 5 in a row
        for (dy1, dx1), (dy2, dx2) in directions:
            total = (self.count_in_direction(current_player, dx1, dy1, x, y) +
                     self.count_in_direction(current_player, dx2, dy2, x, y) + 1)
            if total >= 5:
                return True

        # 2. Check Captures (5 pairs = 10 stones)
        if self.X_pieces_captured_by_O >= 10 or self.O_pieces_captured_by_X >= 10:
            return True

        return False

    def isTie(self) -> bool:
        return self.isBoardFull()

    def getPossibleMoves(self, current_player):
        possible_boards = []
        for i in range(self.board_height):
            for j in range(self.board_width):
                if self.isLegalMove(j, i):
                    new_board = deepcopy(self)
                    new_board.makeMove(j, i, current_player)
                    possible_boards.append(new_board)
        return possible_boards