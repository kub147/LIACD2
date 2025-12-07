"""
dataset_policy.py - FIXED VERSION
----------------------------------
CRITICAL FIX: Changed policy index from x*N+y to y*N+x (row*N+col standard)

This matches the standard 2D array flattening convention where:
- board[row][col] maps to flat_index = row * width + col
"""

import csv
import os
import random
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset


# ----------------- D4 Transform Helpers -----------------

def rot90_xy(x, y, N):  # 90° CCW
    return y, N - 1 - x


def rot180_xy(x, y, N):
    return N - 1 - x, N - 1 - y


def rot270_xy(x, y, N):  # 270° CCW
    return N - 1 - y, x


def flip_h_xy(x, y, N):  # horizontal flip (mirror across vertical axis)
    return x, N - 1 - y


def apply_transform(board_2ch, x, y, N, t):
    """
    board_2ch: numpy (2, N, N)
    (x, y): move coordinates where x=col, y=row in [0, N-1]
    t: int 0..7 (D4 symmetry transformations)
    """

    def do_rot(how, bx):
        if how == 0:
            return bx
        if how == 1:
            return np.rot90(bx, k=1, axes=(1, 2))
        if how == 2:
            return np.rot90(bx, k=2, axes=(1, 2))
        if how == 3:
            return np.rot90(bx, k=3, axes=(1, 2))
        raise ValueError(f"Invalid rotation code: {how}")

    def do_rot_xy(how, xi, yi):
        if how == 0:
            return xi, yi
        if how == 1:
            return rot90_xy(xi, yi, N)
        if how == 2:
            return rot180_xy(xi, yi, N)
        if how == 3:
            return rot270_xy(xi, yi, N)
        raise ValueError(f"Invalid rotation code: {how}")

    flip = (t >= 4)
    rot = t % 4

    b = board_2ch
    xx, yy = x, y

    if flip:
        b = b[:, :, ::-1].copy()
        xx, yy = flip_h_xy(xx, yy, N)

    b = do_rot(rot, b)
    xx, yy = do_rot_xy(rot, xx, yy)

    return np.ascontiguousarray(b), xx, yy


# ----------------- Dataset -----------------

class PolicyDataset(Dataset):
    def __init__(
            self,
            csv_path: str,
            board_size: int = 15,
            augment: bool = True,
    ):
        """
        Args:
            csv_path: cleaned CSV file path
            board_size: N (default 15)
            augment: enable random D4 symmetry transforms
        """
        super().__init__()
        self.csv_path = csv_path
        self.board_size = board_size
        self.augment = augment
        self.action_size = board_size * board_size

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        self.samples = self._load_csv(self.csv_path)
        if len(self.samples) == 0:
            raise ValueError("No valid samples found after parsing CSV.")

        print(
            f"[DATASET] Loaded {len(self.samples)} samples from {self.csv_path} "
            f"(augment={'ON' if self.augment else 'OFF'})."
        )

    def _load_csv(self, path: str) -> List[Dict[str, Any]]:
        N = self.board_size
        action_size = N * N
        out: List[Dict[str, Any]] = []

        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) not in (action_size + 3, action_size + 4, 233):
                    continue

                try:
                    board_flat = list(map(int, row[:action_size]))
                    cp_raw = int(row[action_size])

                    # CRITICAL: CSV stores [col, row] so:
                    move_col = int(row[action_size + 1])  # This is x (column)
                    move_row = int(row[action_size + 2])  # This is y (row)
                except ValueError:
                    continue

                # Basic range checks
                if not (0 <= move_col < N and 0 <= move_row < N):
                    continue

                outcome: Optional[float] = None
                if len(row) == action_size + 4:
                    try:
                        outcome = float(row[action_size + 3])
                    except ValueError:
                        outcome = None

                board_np = np.array(board_flat, dtype=np.int8).reshape(N, N)

                # Handle encoding variants
                b_min, b_max = int(board_np.min()), int(board_np.max())

                if b_min >= 0 and b_max <= 2:
                    # Absolute encoding: {0,1,2}
                    if cp_raw not in (1, 2):
                        continue
                    current_id = cp_raw
                    opp_id = 1 if current_id == 2 else 2
                    current_plane = (board_np == current_id).astype(np.float32)
                    opponent_plane = (board_np == opp_id).astype(np.float32)
                elif b_min >= -1 and b_max <= 1:
                    # Signed encoding: {-1,0,+1}
                    if cp_raw in (-1, 1):
                        cp_sign = cp_raw
                    else:
                        cp_sign = 1
                    current_plane = (board_np == cp_sign).astype(np.float32)
                    opponent_plane = (board_np == -cp_sign).astype(np.float32)
                else:
                    continue

                board_2ch = np.stack([current_plane, opponent_plane], axis=0)

                out.append(
                    {
                        "board_2ch": board_2ch,
                        "move_col": move_col,  # x (column)
                        "move_row": move_row,  # y (row)
                        "value": outcome,
                    }
                )

        return out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        board_2ch = s["board_2ch"].copy()
        x = int(s["move_col"])  # column
        y = int(s["move_row"])  # row
        N = self.board_size

        # D4 augmentation
        if self.augment:
            t = random.randint(0, 7)
            board_2ch, x, y = apply_transform(board_2ch, x, y, N, t)

        board_2ch = np.ascontiguousarray(board_2ch, dtype=np.float32)
        x_tensor = torch.from_numpy(board_2ch).float()

        # FIXED: Standard row-major flattening
        # In CSV: row[226]=col(x), row[227]=row(y)
        # Standard flattening: flat = row * N + col = y * N + x
        policy_index = y * N + x
        y_policy = torch.tensor(policy_index, dtype=torch.long)

        if s["value"] is None:
            return {"x": x_tensor, "y_policy": y_policy}
        else:
            y_value = torch.tensor(float(s["value"]), dtype=torch.float32)
            return {"x": x_tensor, "y_policy": y_policy, "y_value": y_value}