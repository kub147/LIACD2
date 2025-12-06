import time
from player import Player


def test():
    print("--- STARTING BOT SELF-TEST ---")

    # 1. Initialization
    try:
        bot = Player("gomoku", 15)
        print("[OK] Bot initialized.")
    except Exception as e:
        print(f"[FAIL] Could not create Player instance: {e}")
        return

    # 2. Check model loading
    if bot.model is not None:
        print("[OK] Neural Network model loaded successfully.")
    else:
        print("[WARNING] Bot running in RANDOM mode (model is None). Check .pth path!")

    # 3. Mid-game simulation
    board_size = 15
    board = [[0 for _ in range(board_size)] for _ in range(board_size)]

    # Simulate a few stones
    board[7][7] = 2
    board[7][8] = 1
    board[8][8] = 2

    turn = 3
    last_opp_move = (8, 8)

    print("\n--- MOVE TEST (Mid-Game Simulation) ---")
    start = time.time()

    try:
        move = bot.play(board, turn, last_opp_move)
        end = time.time()

        print(f"[SUCCESS] Bot returned move: {move}")
        print(f"Thinking time: {end - start:.2f} seconds")

        if isinstance(move, tuple) and len(move) == 2:
            print("[OK] Move format is correct (row, col).")
        else:
            print(f"[FAIL] Incorrect move format! Received: {type(move)}")

    except Exception as e:
        print(f"[CRITICAL FAIL] Bot crashed during play(): {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test()