"""
test_model_inference.py
-----------------------
Test czy model produkuje sensowne predykcje.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from CNN.nn_model import PolicyValueNet
from CNN.dataset_policy import PolicyDataset


def test_model(model_path, csv_path, device='cuda'):
    """Test modelu na przykładach z datasetu"""

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = PolicyValueNet(board_size=15, channels=128).to(device)

    if Path(model_path).exists():
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        print(f"✓ Model loaded from {model_path}")
    else:
        print(f"⚠️  Model file not found: {model_path}")
        print("Testing with random weights...")

    model.eval()

    # Load dataset
    dataset = PolicyDataset(csv_path, board_size=15, augment=False)
    print(f"✓ Dataset loaded: {len(dataset)} samples\n")

    # Test on 10 samples
    print("=" * 70)
    print("TESTING MODEL PREDICTIONS")
    print("=" * 70)

    for i in range(min(10, len(dataset))):
        sample = dataset[i]

        x = sample['x'].unsqueeze(0).to(device)  # (1, 2, 15, 15)
        true_idx = sample['y_policy'].item()
        true_row = true_idx // 15
        true_col = true_idx % 15

        with torch.no_grad():
            logits, value = model(x)

        # Apply mask to occupied positions
        B = x.size(0)
        occupied = (x[:, 0] != 0) | (x[:, 1] != 0)
        occupied = occupied.view(B, -1)
        masked_logits = logits.clone()
        masked_logits[occupied] = -1e9

        # Get predictions
        probs = F.softmax(masked_logits, dim=1).squeeze(0)
        pred_idx = torch.argmax(masked_logits, dim=1).item()
        pred_row = pred_idx // 15
        pred_col = pred_idx % 15

        # Get top-5 predictions
        top5_probs, top5_indices = torch.topk(probs, k=5)

        print(f"\nSample {i}:")
        print(f"  True move: (row={true_row}, col={true_col}) [idx={true_idx}]")
        print(f"  Pred move: (row={pred_row}, col={pred_col}) [idx={pred_idx}]")
        print(f"  Correct: {'✓' if pred_idx == true_idx else '✗'}")
        print(f"  Value prediction: {value.item():.3f}")
        print(f"  True move probability: {probs[true_idx].item():.4f}")

        print(f"\n  Top-5 predictions:")
        for j, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
            r = idx.item() // 15
            c = idx.item() % 15
            marker = "← TRUE" if idx.item() == true_idx else ""
            print(f"    {j + 1}. (row={r:2d}, col={c:2d}) prob={prob.item():.4f} {marker}")

        # Check if model is biased to first row
        first_row_prob = probs[:15].sum().item()
        print(f"\n  First row total probability: {first_row_prob:.4f}")
        if first_row_prob > 0.5:
            print(f"    ⚠️  Model is biased towards first row!")

    # Statistics over more samples
    print("\n" + "=" * 70)
    print("STATISTICS OVER LARGER SAMPLE")
    print("=" * 70)

    correct_top1 = 0
    correct_top5 = 0
    first_row_predictions = 0
    total = min(1000, len(dataset))

    row_pred_dist = [0] * 15

    with torch.no_grad():
        for i in range(total):
            sample = dataset[i]
            x = sample['x'].unsqueeze(0).to(device)
            true_idx = sample['y_policy'].item()

            logits, _ = model(x)

            # Apply mask
            occupied = (x[:, 0] != 0) | (x[:, 1] != 0)
            occupied = occupied.view(1, -1)
            masked_logits = logits.clone()
            masked_logits[occupied] = -1e9

            # Predictions
            pred_idx = torch.argmax(masked_logits, dim=1).item()
            pred_row = pred_idx // 15

            row_pred_dist[pred_row] += 1

            if pred_row == 0:
                first_row_predictions += 1

            if pred_idx == true_idx:
                correct_top1 += 1

            top5_indices = torch.topk(masked_logits, k=5, dim=1).indices
            if true_idx in top5_indices:
                correct_top5 += 1

    print(f"\nAccuracy on {total} samples:")
    print(f"  Top-1: {100 * correct_top1 / total:.2f}%")
    print(f"  Top-5: {100 * correct_top5 / total:.2f}%")
    print(f"\nPrediction row distribution:")
    for row in range(15):
        pct = 100 * row_pred_dist[row] / total
        bar = '█' * int(pct / 2)
        print(f"  Row {row:2d}: {bar} {pct:5.2f}%")

    first_row_pct = 100 * first_row_predictions / total
    print(f"\n⚠️  Model predicts first row {first_row_pct:.1f}% of the time!")

    if first_row_pct > 80:
        print("\n" + "!" * 70)
        print("CRITICAL ISSUE DETECTED:")
        print("Model is severely biased towards first row!")
        print("This suggests:")
        print("  1. Training data has wrong labels (index calculation error)")
        print("  2. Model didn't learn properly (check training accuracy)")
        print("  3. Mask during inference is wrong")
        print("!" * 70)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Path to model .pth file")
    ap.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    test_model(args.model, args.csv, args.device)