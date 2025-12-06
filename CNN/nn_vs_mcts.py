"""
nn_vs_mcts.py
-------------
Play matches between a trained Neural Network and classic MCTS for Gomoku.

This script uses:
    - ResNetPolicyValueNet  (1-channel ResNet defined in resnet_model.py)
    - Weights stored in: CNN/alphazero_policy_v_best.pth
    - Gomoku logic from: board_logic_and_mcts/

USAGE
=====

1) GUI mode (Tkinter-based)
   Just run without any arguments:

       python CNN/nn_vs_mcts.py

   Default GUI setup:
       - MCTS plays as Player 2 (black, X)
       - NN plays as Player 1 (white, O)

2) CLI mode (text-only, no GUI)
   CLI is activated automatically if ANY arguments are passed:

       # NN as Player 2 (black, starts), MCTS as Player 1
       python CNN/nn_vs_mcts.py --games 10 --mcts-sims 3000

       # MCTS as Player 2 (black, starts)
       python CNN/nn_vs_mcts.py --games 10 --mcts-sims 3000 --mcts-as-p2

MODEL PATH
==========
The model is ALWAYS loaded from:

       CNN/alphazero_policy_v_best.pth

If this file does not exist, the script will exit with an error.
"""


import argparse
import sys
import random
from pathlib import Path

import numpy as np
import torch

# -------------------- Path setup --------------------

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

# Folders with game logic and CNN code
BOARD_LOGIC_DIR = PROJECT_ROOT / "board_logic_and_mcts"
CNN_DIR = PROJECT_ROOT / "CNN"

# Add needed dirs to sys.path (so we can import game logic + model)
for p in (BOARD_LOGIC_DIR, CNN_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from Gomoku_Board import GBoard
from MCTS import MCTS
from Node import Node
from resnet_model import ResNetPolicyValueNet

# Model + game defaults
DEFAULT_MODEL_PATH = CNN_DIR / "alphazero_policy_v_best.pth"
DEFAULT_BOARD_N = 15
DEFAULT_MCTS_SIMS = 5000

# Ensure the default model file exists
if not DEFAULT_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"[ERROR] Default model file not found:\n{DEFAULT_MODEL_PATH}\n"
        f"Train a model and save it as 'CNN/alphazero_policy_v_best.pth' first."
    )

# -------------------- Game Logic --------------------



def current_player_sign(board):
    """Return current player (1 or 2) based on piece counts on the board."""
    p1_count = sum(row.count(1) for row in board.board)
    p2_count = sum(row.count(2) for row in board.board)
    # In this project, Player 2 (X / black) starts on an empty board.
    return 1 if p2_count > p1_count else 2


def legal_moves(board):
    """Return list of legal move coordinates (x, y)."""
    moves = []
    for x in range(board.board_width):
        for y in range(board.board_height):
            if board.isLegalMove(x, y):
                moves.append((x, y))
    return moves


def check_terminal(board):
    """Return (is_terminal, winner). Winner: 1, 2, or 0 for draw."""
    if board.last_move is None:
        return False, 0

    x, y = board.last_move
    last_player = board.board[y][x]

    if board.isWon(x, y, last_player):
        return True, last_player
    if board.isBoardFull():
        return True, 0
    return False, 0


# -------------------- NN Policy --------------------


def to_two_planes(board, cp):
    """
    Build a single input plane (1, N, N):
        +1 for stones of current player,
        -1 for opponent stones,
        0 for empty.
    """
    N = board.board_width
    arr = np.array(board.board, dtype=np.int8)
    opponent = 1 if cp == 2 else 2

    plane = np.zeros((N, N), dtype=np.float32)
    plane[arr == cp] = 1.0
    plane[arr == opponent] = -1.0

    # Model expects (C, H, W) -> (1, N, N)
    return np.expand_dims(plane, axis=0)


def nn_choose_move(model, board, device):
    """
    Let the neural network choose a move.

    IMPORTANT: The action index mapping here must match the mapping used
    during training in dataset_policy.py:

        index = move_x * board_size + move_y

    So we:
      - encode legals as idx = x * N + y
      - decode idx back as (x = idx // N, y = idx % N)
    """
    cp = current_player_sign(board)
    N = board.board_width

    x_np = to_two_planes(board, cp)
    x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        policy_logits, _ = model(x)

    legals = legal_moves(board)
    legal_idxs = {x * N + y for (x, y) in legals}

    # Mask illegal moves with -inf so argmax only sees legal moves
    mask = torch.full((1, policy_logits.size(1)), float("-inf"), device=device)
    if legal_idxs:
        mask[0, list(legal_idxs)] = 0.0

    masked = policy_logits + mask
    move_flat = int(torch.argmax(masked, dim=1).item())

    move_x = move_flat // N
    move_y = move_flat % N
    return move_x, move_y


# -------------------- MCTS --------------------


def mcts_choose_move(board, sims=1000):
    """Let classic MCTS choose a move."""
    from copy import deepcopy

    cp = current_player_sign(board)
    board_copy = deepcopy(board)

    root = Node(board_copy)
    mcts = MCTS(initial_state=root, current_player=cp, simulation_limit=sims)
    best_node = mcts.best_move()

    if best_node and best_node.board.last_move:
        return best_node.board.last_move

    # Fallback: random legal move
    legals = legal_moves(board)
    return random.choice(legals) if legals else None


# -------------------- Game Loop (CLI) --------------------


def _winner_side_label(winner, mcts_is_player_2):
    """
    Map numeric winner (1 or 2) to 'NN' or 'MCTS', depending on who plays which side.
    Returns 'NN', 'MCTS' or 'draw'.
    """
    if winner == 0:
        return "draw"
    # If MCTS is Player 2, NN is Player 1; otherwise the opposite.
    if (winner == 2 and mcts_is_player_2) or (winner == 1 and not mcts_is_player_2):
        return "MCTS"
    return "NN"


def play_game(model, device, board_n=15, mcts_sims=1000, verbose=True, mcts_is_player_2=True):
    """Play one game between NN and MCTS. Return winner: 1, 2, or 0 for draw."""
    board = GBoard(board_n)
    move_no = 0
    PLAYER_X, PLAYER_O = 2, 1  # Consistent with existing codebase

    while True:
        if move_no > 0:
            term, winner = check_terminal(board)
            if term:
                if verbose:
                    side = _winner_side_label(winner, mcts_is_player_2)
                    if winner == 0:
                        msg = "[GAME OVER] Result: draw."
                    else:
                        msg = f"[GAME OVER] Winner: {side} (Player {winner}) after {move_no} moves."
                    print(msg)
                return winner

        cp = current_player_sign(board)

        # Decide whose turn: NN or MCTS.
        # When mcts_is_player_2 is True:
        #   - Player 2 (X) = MCTS
        #   - Player 1 (O) = NN
        if (cp == PLAYER_X and mcts_is_player_2) or (cp == PLAYER_O and not mcts_is_player_2):
            mv = mcts_choose_move(board, sims=mcts_sims)
            side = f"MCTS (Player {cp})"
        else:
            mv = nn_choose_move(model, board, device)
            side = f"NN (Player {cp})"

        if mv is None:
            if verbose:
                print("[INFO] No legal moves remaining; declaring draw.")
            return 0

        x, y = mv
        board.makeMove(x, y, cp)
        move_no += 1

        if verbose:
            print(f"Move {move_no:03d}: {side} -> ({x+1}, {y+1})")
            board.printBoard()


# -------------------- CLI --------------------


def main():
    ap = argparse.ArgumentParser(description="NN vs MCTS Gomoku Match.")
    ap.add_argument("--model-path", type=str, default="alphazero_policy_v_best.pth")
    ap.add_argument("--board-n", type=int, default=15)
    ap.add_argument("--mcts-sims", type=int, default=2000)
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--mcts-as-p2", action="store_true", help="MCTS plays as Player 2 (X, black, starts).")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[INFO] Using device: {device}")

    # Resolve model path (allow paths relative to project root or CNN dir)
    model_path = Path(args.model_path)
    if not model_path.exists() and not (CNN_DIR / model_path).exists():
        model_path = CNN_DIR / model_path.name

    model = ResNetPolicyValueNet(
        board_size=args.board_n,
        channels=64,
        num_res_blocks=5,
        dropout_p=0.3,
    ).to(device)

    if model_path.exists():
        state = torch.load(str(model_path), map_location=device)
        cleaned_state = {k: v for k, v in state.items() if "num_batches_tracked" not in k}
        try:
            model.load_state_dict(cleaned_state, strict=True)
            print(f"[OK] Loaded model from {model_path}")
        except RuntimeError:
            print("[WARN] Strict loading failed, using non-strict load.")
            model.load_state_dict(cleaned_state, strict=False)
    else:
        print(f"[WARN] Model not found at {model_path}")

    verbose = not args.quiet
    results = {1: 0, 2: 0, 0: 0}

    if args.mcts_as_p2:
        config_msg = "MCTS: Player 2 (X, black, starts) | NN: Player 1 (O, white)"
        nn_player = 1
        mcts_player = 2
    else:
        config_msg = "NN: Player 2 (X, black, starts) | MCTS: Player 1 (O, white)"
        nn_player = 2
        mcts_player = 1

    print(f"\n[CONFIG] {config_msg}")

    for g in range(1, args.games + 1):
        if verbose:
            print(f"\n===== GAME {g}/{args.games} =====")
        winner = play_game(
            model,
            device,
            board_n=args.board_n,
            mcts_sims=args.mcts_sims,
            verbose=verbose,
            mcts_is_player_2=args.mcts_as_p2,
        )
        results[winner] += 1

    total_games = args.games
    nn_wins = results[nn_player]
    mcts_wins = results[mcts_player]
    draws = results[0]

    print("\n=== SUMMARY (by player id) ===")
    print(f"Player 2 (X) wins: {results[2]}")
    print(f"Player 1 (O) wins: {results[1]}")
    print(f"Draws: {draws}")

    print("\n=== SUMMARY (NN vs MCTS) ===")
    print(f"NN wins:   {nn_wins}")
    print(f"MCTS wins: {mcts_wins}")
    print(f"Draws:     {draws}")
    if total_games > 0:
        nn_win_rate = 100.0 * nn_wins / total_games
        print(f"NN win rate: {nn_win_rate:.1f}% over {total_games} games.")


# -------------------- GUI --------------------


def draw_board_on_canvas(canvas, board, cell_size=30, margin=20):
    """Draw the Gomoku board and stones on a Tkinter canvas."""
    canvas.delete("all")
    N = board.board_width
    size = margin * 2 + cell_size * (N - 1)
    canvas.config(width=size, height=size)

    # Grid lines
    for i in range(N):
        canvas.create_line(
            margin,
            margin + i * cell_size,
            margin + cell_size * (N - 1),
            margin + i * cell_size,
        )
        canvas.create_line(
            margin + i * cell_size,
            margin,
            margin + i * cell_size,
            margin + cell_size * (N - 1),
        )

    # Stones
    for y in range(N):
        for x in range(N):
            p = board.board[y][x]
            if p == 0:
                continue
            cx, cy = margin + x * cell_size, margin + y * cell_size
            r = cell_size * 0.4
            fill_color = "black" if p == 2 else "white"
            canvas.create_oval(
                cx - r,
                cy - r,
                cx + r,
                cy + r,
                fill=fill_color,
                outline="black",
            )

    # Last move marker
    if board.last_move is not None:
        x, y = board.last_move
        cx, cy = margin + x * cell_size, margin + y * cell_size
        r = cell_size * 0.15
        canvas.create_oval(
            cx - r,
            cy - r,
            cx + r,
            cy + r,
            outline="red",
            width=2,
        )


def play_game_gui(
    model,
    device,
    canvas,
    status_var,
    board_n=DEFAULT_BOARD_N,
    mcts_sims=DEFAULT_MCTS_SIMS,
    root=None,
    mcts_is_player_2=True,
):
    """
    GUI version of play_game with board refresh after each move.

    Returns (winner, move_count).
    """
    board = GBoard(board_n)
    move_no = 0
    PLAYER_X, PLAYER_O = 2, 1

    draw_board_on_canvas(canvas, board)
    if root is not None:
        root.update_idletasks()
        root.update()

    while True:
        if move_no > 0:
            term, winner = check_terminal(board)
            if term:
                side = _winner_side_label(winner, mcts_is_player_2)
                if winner == 0:
                    text = f"Game over: draw after {move_no} moves."
                else:
                    text = f"Game over: {side} wins (Player {winner}) after {move_no} moves."
                status_var.set(text)
                if root is not None:
                    root.update()
                return winner, move_no

        cp = current_player_sign(board)

        if (cp == PLAYER_X and mcts_is_player_2) or (cp == PLAYER_O and not mcts_is_player_2):
            mv = mcts_choose_move(board, sims=mcts_sims)
            side = f"MCTS (Player {cp})"
        else:
            mv = nn_choose_move(model, board, device)
            side = f"NN (Player {cp})"

        if mv is None:
            status_var.set(f"No legal moves: draw after {move_no} moves.")
            if root is not None:
                root.update()
            return 0, move_no

        x, y = mv
        board.makeMove(x, y, cp)
        move_no += 1

        status_var.set(f"Move {move_no}: {side} -> ({x+1}, {y+1})")
        draw_board_on_canvas(canvas, board)

        if root is not None:
            root.update()


def launch_gui():
    """Simple GUI: one button, board canvas, legend, and status label."""
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.title("Gomoku NN vs MCTS")

    cell_size, margin = 30, 20
    size = margin * 2 + cell_size * (DEFAULT_BOARD_N - 1)
    canvas = tk.Canvas(root, width=size, height=size, bg="#f0d9b5")
    canvas.pack(padx=10, pady=10)

    # Legend: clarify which stones belong to NN and which to MCTS.
    legend_text = (
        "Legend: Black stones = Player 2 (X), White stones = Player 1 (O).\n"
        "In this GUI: MCTS plays as Player 2 (black), NN plays as Player 1 (white)."
    )
    tk.Label(root, text=legend_text, justify="center").pack(padx=10, pady=(0, 10))

    status_var = tk.StringVar(value="Click 'Start game' to begin.")
    tk.Label(root, textvariable=status_var).pack(padx=10, pady=(0, 10))

    def start_game():
        status_var.set("Loading model...")
        root.update_idletasks()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[GUI] Using device: {device}")

        model_path = DEFAULT_MODEL_PATH
        if not model_path.exists() and not (CNN_DIR / model_path.name).exists():
            messagebox.showerror("Error", f"Model not found:\n{model_path}")
            status_var.set("Model not found.")
            return

        if not model_path.exists():
            model_path = CNN_DIR / model_path.name

        model = ResNetPolicyValueNet(
            board_size=DEFAULT_BOARD_N,
            channels=64,
            num_res_blocks=5,
            dropout_p=0.3,
        ).to(device)

        try:
            state = torch.load(str(model_path), map_location=device)
            cleaned_state = {k: v for k, v in state.items() if "num_batches_tracked" not in k}
            try:
                model.load_state_dict(cleaned_state, strict=True)
                print(f"[GUI] Loaded model from {model_path}")
            except RuntimeError:
                model.load_state_dict(cleaned_state, strict=False)
                print("[GUI] Loaded model (non-strict).")
        except Exception as e:  # pylint: disable=broad-except
            messagebox.showerror("Error", f"Could not load model:\n{e}")
            status_var.set("Error loading model.")
            return

        status_var.set("Game started (NN vs MCTS)...")
        root.update_idletasks()

        winner, moves = play_game_gui(
            model,
            device,
            canvas,
            status_var,
            board_n=DEFAULT_BOARD_N,
            mcts_sims=DEFAULT_MCTS_SIMS,
            root=root,
            mcts_is_player_2=True,  # In GUI we fix MCTS as Player 2 (black).
        )

        # Popup with final stats
        if winner == 0:
            msg = f"Result: draw after {moves} moves."
        else:
            side = _winner_side_label(winner, mcts_is_player_2=True)
            msg = f"Result: {side} wins (Player {winner}) after {moves} moves."
        messagebox.showinfo("Game finished", msg)

    tk.Button(root, text="Start game", command=start_game).pack(pady=(0, 10))
    root.mainloop()


# -------------------- Entry Point --------------------

if __name__ == "__main__":
    if len(sys.argv) == 1:
        launch_gui()
    else:
        main()
