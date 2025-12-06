"""
nn_vs_mcts.py
-------------
Play matches between a trained Neural Network and classic MCTS for Gomoku.
Includes a GUI for visualization.
"""

import argparse
import sys
import random
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
import time

import numpy as np
import torch
from copy import deepcopy

# -------------------- Path setup (FIXED) --------------------

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # Główny folder LIACD2

# Dodajemy projekt do sys.path, żeby widzieć pakiety 'board_logic_and_mcts' i 'CNN'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Teraz importujemy używając pełnych ścieżek (jak w player.py)
from board_logic_and_mcts.Gomoku_Board import GBoard
from board_logic_and_mcts.MCTS import MCTS
from board_logic_and_mcts.Node import Node
from CNN.nn_model import PolicyValueNet

# Model + game defaults
DEFAULT_MODEL_PATH = PROJECT_ROOT / "CNN" / "alphazero_policy_v_best.pth"
DEFAULT_BOARD_N = 15

# -------------------- Game Logic Helpers --------------------

def current_player_sign(board):
    """Return current player (1 or 2) based on piece counts."""
    p1_count = sum(row.count(1) for row in board.board)
    p2_count = sum(row.count(2) for row in board.board)
    # Assuming Black (Player 1) starts.
    # If equal stones, it's P1 turn. If P1 > P2, it's P2 turn.
    return 1 if p1_count == p2_count else 2

def legal_moves(board):
    moves = []
    for x in range(board.board_width):
        for y in range(board.board_height):
            if board.isLegalMove(x, y):
                moves.append((x, y))
    return moves

def check_terminal(board):
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
    Build input planes (2, N, N) compatible with the trained model.
    Channel 0: Current Player stones
    Channel 1: Opponent stones
    """
    N = board.board_width
    arr = np.array(board.board, dtype=np.int8)
    opponent = 1 if cp == 2 else 2

    plane_me = (arr == cp).astype(np.float32)
    plane_opp = (arr == opponent).astype(np.float32)

    # Result shape: (2, N, N)
    return np.stack([plane_me, plane_opp])

def nn_choose_move(model, board, device):
    cp = current_player_sign(board)
    N = board.board_width

    # Prepare input
    x_np = to_two_planes(board, cp)
    # Add batch dim -> (1, 2, N, N)
    x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        policy_logits, _ = model(x)

    # Mask illegal moves
    legals = legal_moves(board)
    if not legals:
        return None

    legal_idxs = {y * N + x for (x, y) in legals} # Uwaga: policy_flat zazwyczaj y*N + x lub odwrotnie, zależnie od treningu
    # W Twoim treningu (dataset_policy.py): row*N + col. row=y, col=x. Czyli y*N + x.

    # Create mask
    mask = torch.full((1, policy_logits.size(1)), float("-inf"), device=device)
    valid_indices = list(legal_idxs)

    if not valid_indices:
        return None

    mask[0, valid_indices] = 0.0

    masked = policy_logits + mask
    move_flat = int(torch.argmax(masked, dim=1).item())

    move_y = move_flat // N
    move_x = move_flat % N
    return move_x, move_y

# -------------------- MCTS Logic --------------------

def mcts_choose_move(board, sims=500):
    cp = current_player_sign(board)
    board_copy = deepcopy(board)
    root = Node(board_copy)
    # Classic MCTS (Random Rollouts)
    mcts = MCTS(initial_state=root, current_player=cp, simulation_limit=sims)
    best_node = mcts.best_move()

    if best_node and best_node.board.last_move:
        return best_node.board.last_move

    # Fallback
    l = legal_moves(board)
    return random.choice(l) if l else None

# -------------------- GUI Logic --------------------

def draw_board_on_canvas(canvas, board, cell_size=30, margin=20):
    canvas.delete("all")
    N = board.board_width
    size = margin * 2 + cell_size * (N - 1)
    canvas.config(width=size, height=size)

    # Background
    canvas.create_rectangle(0, 0, size, size, fill="#eebb99", outline="")

    # Grid
    for i in range(N):
        p = margin + i * cell_size
        end = margin + cell_size * (N - 1)
        canvas.create_line(margin, p, end, p) # Horizontal
        canvas.create_line(p, margin, p, end) # Vertical

    # Stones
    for y in range(N):
        for x in range(N):
            val = board.board[y][x]
            if val == 0: continue

            cx, cy = margin + x * cell_size, margin + y * cell_size
            r = cell_size * 0.4
            color = "black" if val == 1 else "white" # Player 1 (Black), Player 2 (White)
            canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill=color, outline="black")

    # Marker for last move
    if board.last_move:
        lx, ly = board.last_move
        cx, cy = margin + lx * cell_size, margin + ly * cell_size
        r = cell_size * 0.15
        canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill="red", outline="red")

def play_step(model, device, canvas, status_var, board, root, mcts_sims, mcts_is_p2):
    """Executes one turn in the GUI loop."""
    term, winner = check_terminal(board)
    if term:
        res = "Draw" if winner == 0 else f"Player {winner} Wins!"
        status_var.set(f"Game Over: {res}")
        messagebox.showinfo("Game Over", res)
        return

    cp = current_player_sign(board)

    # Logic: Who plays who?
    # If mcts_is_p2 is True: P2(White)=MCTS, P1(Black)=NN
    is_mcts_turn = (cp == 2 and mcts_is_p2) or (cp == 1 and not mcts_is_p2)

    start_msg = f"Turn: P{cp} ({'MCTS' if is_mcts_turn else 'NeuralNet'})"
    status_var.set(start_msg)
    root.update()

    # Small delay for UI update
    # time.sleep(0.1)

    if is_mcts_turn:
        move = mcts_choose_move(board, sims=mcts_sims)
    else:
        move = nn_choose_move(model, board, device)

    if move is None:
        status_var.set("Game Over: No moves left.")
        return

    x, y = move
    board.makeMove(x, y, cp)
    draw_board_on_canvas(canvas, board)

    # Schedule next move
    root.after(50, lambda: play_step(model, device, canvas, status_var, board, root, mcts_sims, mcts_is_p2))

def launch_gui():
    root = tk.Tk()
    root.title("Gomoku: Neural Net vs Classic MCTS")

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # FIX: Use PolicyValueNet with correct channels
    model = PolicyValueNet(board_size=DEFAULT_BOARD_N, channels=128).to(device)

    # Try loading weights
    path = DEFAULT_MODEL_PATH

    loaded = False
    if path.exists():
        try:
            state = torch.load(str(path), map_location=device)
            model.load_state_dict(state)
            model.eval()
            loaded = True
            print(f"[GUI] Model loaded: {path}")
        except Exception as e:
            print(f"[GUI] Error loading model: {e}")

    # UI Elements
    frame = tk.Frame(root)
    frame.pack(side=tk.TOP, pady=5)

    lbl = tk.Label(frame, text=f"Model Loaded: {'YES' if loaded else 'NO'}", fg="green" if loaded else "red")
    lbl.pack()

    status_var = tk.StringVar(value="Press Start")
    tk.Label(root, textvariable=status_var, font=("Arial", 12)).pack(pady=5)

    canvas = tk.Canvas(root, bg="#eebb99")
    canvas.pack(padx=20, pady=20)

    def start_game():
        if not loaded:
            messagebox.showwarning("Warning", "Model not loaded! NN will play random/garbage.")

        board = GBoard(DEFAULT_BOARD_N)
        draw_board_on_canvas(canvas, board)
        # Start the loop
        # P1 (Black) = NN, P2 (White) = MCTS
        play_step(model, device, canvas, status_var, board, root, mcts_sims=300, mcts_is_p2=True)

    tk.Button(root, text="START MATCH (NN=Black, MCTS=White)", command=start_game).pack(pady=10)

    # Initial empty board draw
    dummy = GBoard(DEFAULT_BOARD_N)
    draw_board_on_canvas(canvas, dummy)

    root.mainloop()

if __name__ == "__main__":
    launch_gui()