"""
train_policy.py
---------------
Policy (and optional value) training for Gomoku/Pente policy-value net.

What's inside:
- AdamW optimizer with weight_decay (anti-overfitting)
- ReduceLROnPlateau scheduler (auto LR decay on plateau)
- Early stopping with best-model restore
- Clean, consistent logging with LR display
- Indexing self-test (x,y) <-> flat
- Reproducible seeding

Assumptions:
- CSV produced by your cleaner has columns:
  b0..b224, cp, move_x, move_y[, outcome(optional in [-1, 1])]
- Dataset returns (input_tensor, policy_index, optional_value)
- Model exposes (policy_logits, value) forward()

Usage (examples from root):
python CNN/train_policy.py \
    --csv data/all_combined_clean_new.csv \
    --board-n 15 \
    --epochs 50 \
    --patience 10 \
    --batch-size 256 \
    --augment \
    --model-out CNN/alphazero_policy_v_best.pth
"""

import os
import time
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---- Local modules
from dataset_policy import PolicyDataset
from nn_model import PolicyValueNet


# ------------------------- Utilities -------------------------

def set_seed(seed: int = 1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def self_test_indexing(N: int = 15):
    """
    Verifies that flattening and unflattening agree with:
      flat = x * N + y, where x=row (0..N-1), y=col (0..N-1)
    """
    def idx(x, y): return x * N + y
    tests = [(0, 0), (0, N - 1), (N - 1, 0), (N - 1, N - 1), (7, 7), (3, 11), (11, 3)]
    for (x, y) in tests:
        flat = idx(x, y)
        rx, ry = divmod(flat, N)
        assert (rx, ry) == (x, y), f"Indexing mismatch for (x,y)=({x},{y}) -> {flat} -> ({rx},{ry})"
    print("[OK] Indexing self-test passed: (x,y) <-> flat is consistent.")


# ------------------------- Training -------------------------

def train_one_epoch(model, loader, device, optimizer, policy_criterion, value_criterion, value_weight: float):
    model.train()
    running_loss = 0.0

    for batch in loader:
        x = batch["x"].to(device)                    # (B, 2, N, N)
        y_policy = batch["y_policy"].to(device)      # (B,)
        y_value = batch.get("y_value", None)
        if y_value is not None:
            y_value = y_value.to(device).view(-1, 1)   # (B, 1)

        optimizer.zero_grad()
        policy_logits, value_pred = model(x)

        B = x.size(0)
        occupied = (x[:, 0] != 0) | (x[:, 1] != 0)
        occupied = occupied.view(B, -1)
        occupied[torch.arange(B), y_policy] = False
        policy_logits = policy_logits.masked_fill(occupied, -1e9)

        loss_policy = policy_criterion(policy_logits, y_policy)
        if y_value is not None and value_weight > 0.0:
            loss_value = value_criterion(value_pred, y_value)
            loss = loss_policy + value_weight * loss_value
        else:
            loss_value = torch.tensor(0.0, device=device)
            loss = loss_policy

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / max(1, len(loader))


@torch.no_grad()
def validate_one_epoch(model, loader, device, policy_criterion, value_criterion, value_weight: float):
    model.eval()
    running_loss = 0.0

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for batch in loader:
        x = batch["x"].to(device)
        y_policy = batch["y_policy"].to(device)
        y_value = batch.get("y_value", None)
        if y_value is not None:
            y_value = y_value.to(device).view(-1, 1)

        policy_logits, value_pred = model(x)

        B = x.size(0)
        occupied = (x[:, 0] != 0) | (x[:, 1] != 0)
        occupied = occupied.view(B, -1)
        occupied[torch.arange(B), y_policy] = False
        policy_logits = policy_logits.masked_fill(occupied, -1e9)

        loss_policy = policy_criterion(policy_logits, y_policy)
        if y_value is not None and value_weight > 0.0:
            loss_value = value_criterion(value_pred, y_value)
            loss = loss_policy + value_weight * loss_value
        else:
            loss = loss_policy

        running_loss += loss.item()

        #Top 1 accuracy
        _, pred_top1 = policy_logits.max(dim=1)
        correct_top1 += (pred_top1 == y_policy).sum().item()

        #Top 5 accuracy
        top5_indices = policy_logits.topk(5, dim=1).indices
        correct_top5 += (top5_indices == y_policy.unsqueeze(1)).any(dim=1).sum().item()

        total += y_policy.size(0)

    avg_loss = running_loss / max(1, len(loader))
    top1_accuracy = correct_top1 / max(1, total)
    top5_accuracy = correct_top5 / max(1, total)

    return avg_loss, top1_accuracy, top5_accuracy


def main():
    ap = argparse.ArgumentParser(description="Train policy/value network for Gomoku.")
    ap.add_argument("--csv", type=str, required=True, help="Path to cleaned & deduplicated CSV.")
    ap.add_argument("--board-n", type=int, default=15, help="Board size N (default: 15).")
    ap.add_argument("--epochs", type=int, default=50, help="Max epochs (default: 50).")
    ap.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10).")
    ap.add_argument("--batch-size", type=int, default=256, help="Batch size (default: 256).")
    ap.add_argument("--lr", type=float, default=1e-3, help="Base learning rate (default: 1e-3).")
    ap.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay (default: 5e-4).")
    ap.add_argument("--value-weight", type=float, default=0.0, help="Value loss weight (0 disables value loss).")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader workers (default: 4).")
    ap.add_argument("--seed", type=int, default=1234, help="Random seed (default: 1234).")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device selection.")
    ap.add_argument("--augment", action="store_true", help="Enable D4 symmetry augmentations.")
    ap.add_argument(
        "--model-out",
        type=str,
        default="CNN/alphazero_policy_v_best.pth",
        help="Where to save the best model."
    )
    args = ap.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    # Seed + indexing self-test
    set_seed(args.seed)
    self_test_indexing(args.board_n)

    # Dataset & loaders
    dataset = PolicyDataset(
        csv_path=args.csv,
        board_size=args.board_n,
        augment=args.augment
    )

    # 85/15 split (reproducible)
    n_total = len(dataset)
    n_val = max(1, int(0.15 * n_total))
    n_train = n_total - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda")
    )

    print(f"[DATA] Loaded {n_total} samples. Train: {n_train}  |  Val: {n_val}")
    print(f"[DATA] Augmentation: {'ON' if args.augment else 'OFF'}")

    # Model
    model = PolicyValueNet(board_size=args.board_n, channels=128, dropout_p=0.3).to(device)

    # Optimizer & scheduler (A + B)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Safe construction for any PyTorch version
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=3,
            min_lr=1e-6,
            verbose=True
        )
    except TypeError:
        # Older PyTorch versions (e.g. 1.12 or below) don’t have 'verbose'
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=3,
            min_lr=1e-6
        )
        print("[WARN] 'verbose' not supported in this PyTorch version — continuing without it.")

    # Losses
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    # Early stopping tracking
    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    patience = args.patience
    best_model_path = args.model_out

    print(f"[TRAIN] Starting for up to {args.epochs} epochs with patience {patience}...")
    print(f"[OPTIM] AdamW lr={args.lr:.1e} weight_decay={args.weight_decay:.1e} value_weight={args.value_weight}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, device, optimizer,
            policy_criterion, value_criterion, args.value_weight
        )
        val_loss, val_top1, val_top5 = validate_one_epoch(
            model, val_loader, device, policy_criterion, value_criterion, args.value_weight
        )

        # Scheduler step on validation metric
        scheduler.step(val_loss)

        # Early stopping bookkeeping
        improved = val_loss < (best_val - 1e-6)
        if improved:
            best_val = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            status = "new best!"
        else:
            epochs_no_improve += 1
            status = f"no improve {epochs_no_improve}/{patience}"

        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Top1: {val_top1*100:.2f}% | Val Top5: {val_top5*100:.2f}% | "
              f"{status} | LR={lr_now:.2e} | {dt:.1f}s")

        if epochs_no_improve >= patience:
            print(f"[STOP] Early stopping. Best epoch={best_epoch}, best val_loss={best_val:.4f}")
            break

    # Restore best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"[READY] Best model loaded from: {best_model_path}")
    else:
        print("[WARN] Best model file not found; leaving last epoch weights.")


if __name__ == "__main__":
    main()
