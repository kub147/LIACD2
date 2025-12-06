from Gomoku_Board import GBoard
from Pente_Board import PBoard
from MCTS import MCTS
from Node import Node
import time

def human_play(game, currentplayer):
    print("Make a move by choosing your coordinates to play.")
    print("Enter coordinates as 'x y' (e.g., '3 4') or 'q' to quit:")

    userInput = input().strip()
    if userInput == 'q':
        return (-1, -1)
    else:
        # Split the input into two values and convert to integers
        x, y = userInput.split()
        if (int(x) <= 0 or int(y) <= 0): return (-1, -1)

        return (int(x) - 1, int(y) - 1)  # Return X Y (converted to 0-based indexing)



def human_vs_human(size, board_class):
    game = board_class(size)
    player = 2
    game_is_over = False

    while not game.isBoardFull() and not game_is_over:
        player = 1 if player == 2 else 2
        game.printBoard()
        print(f"It is now {player}'s turn!")
        col, row = human_play(game, player)         # Col is x, Row is y

        if row == -1 or col == -1 or row >= game.board_height or col >= game.board_width:
            break

        if not game.isLegalMove(col, row):
            print("\nWarning: Invalid Move. Try again!")
        else:
            game.makeMove(col, row, player)

        game_is_over = game.isWon(col, row, player)
        
    game.printBoard()
    if col == -1:
        print("Match abandonded")

    elif game.isTie():
        print("It's a tie!")
    
    else:
        print(f"Player {player} has won!\n")
        

def ai_vs_human(size, board_class, simulation_limit=10000):
    game = board_class(size)
    ai = 1
    human = 2
    humanplay = True # False for AI first, True for HUMAN first

    game_is_over = False 

    while not game.isBoardFull() and not game_is_over:
        player = human if humanplay else ai
        game.printBoard()
        print(f"It is now {player}'s turn!")

        if humanplay:
            col, row = human_play(game, player)
            if not game.isLegalMove(col, row):
                print("\nWarning: Invalid Move. Try again!")
            else:
                game.makeMove(col, row, human)

        else:
            root = Node(game, None) #Current game after player makes move
            mcts = MCTS(root, ai, simulation_limit=simulation_limit)
            start = time.time()
            best_node = mcts.best_move()
            end = time.time()
            col, row = best_node.board.last_move
            if mcts.gameOver:
                print("AI chose column : ", col + 1, " row : ", row + 1)
                print(f"Time taken: {end - start:.2f}")
                break
            
            game.makeMove(col, row, ai)
            print("AI chose column : ", col + 1, " row : ", row + 1)
            print(f"Time taken: {end - start:.2f}")
        
        # After game is won
        game_is_over = game.isWon(col, row, player)
        humanplay = False if humanplay else True

    game.printBoard()
    if game.isTie():
        print("It's a tie!")
    else:
        print(f"Player {player} has won!\n")

def ai_vs_ai(size, board_class, simulation_limit_X=10000, simulation_limit_O=10000):
    game = board_class(size)
    x = 2
    o = 1
    playerX = True # False for player 1 first, True for player 2 first

    game_is_over = False

    while (not game.isBoardFull() and not game_is_over) or game.isTie():
        currentplayer = x if playerX else o
        game.printBoard()
        print(f"It is now {currentplayer}'s turn!")

# N sei se a arvore do MCTS vai fzr update do root para preservar as vitórias ou cirar um árvore de zero
        if playerX:
            root = Node(game, None)
            mcts = MCTS(root, x, simulation_limit_X)
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
            mcts = MCTS(root, o, simulation_limit_O)
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
        print(f"Time taken : {end - start:.2f}")
        playerX = False if playerX else True

    # After game is won
    game.printBoard()
    if game.isTie():
        print("It's a tie!")
    else:
        print(f"Player {currentplayer} has won!\n")


def run():
    print("Choose a game type:")
    print("1. Gomoku")
    print("2. Pente")
    game_type = int(input("Enter the game type number: "))
    
    if game_type == 1:
        board_class = GBoard
        game_name = "Gomoku"
    elif game_type == 2:
        board_class = PBoard
        game_name = "Pente"
    else:
        print("Invalid game type selected!")
        return
    
    print(f"\nPlaying {game_name}!")
    print("Choose a game mode:")
    print("1. Human vs Human")
    print("2. AI vs Human")
    print("3. AI vs AI")
    game_mode = int(input("Enter the game mode number: "))
    board_size = int(input("Define board size (15x15) or (50x50): "))


    if game_mode == 1:
        human_vs_human(board_size, board_class)
    elif game_mode == 2:
        dificulty = int(input("Insert AI simulation Limit: "))
        ai_vs_human(board_size, board_class, dificulty)
    elif game_mode == 3:
        dificulty_X = int(input("Insert AI (1) simulation Limit: "))
        dificulty_O = int(input("Insert AI (2) simulation Limit: "))
        ai_vs_ai(board_size, board_class, dificulty_X, dificulty_O)
    else:
        print("Invalid game mode selected!")

run()