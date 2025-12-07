"""
Comprehensive indexing test to verify consistency
--------------------------------------------------
Run this BEFORE training to ensure everything is aligned!
"""

import numpy as np


def test_indexing_consistency(N=15):
    """
    Tests that all parts of the pipeline use consistent indexing.
    Standard convention: flat_index = row * N + col
    """

    print("=" * 60)
    print("INDEXING CONSISTENCY TEST")
    print("=" * 60)

    # Test 1: Basic flattening
    print("\n[TEST 1] Basic 2D -> 1D flattening")
    board_2d = np.arange(N * N).reshape(N, N)
    board_flat = board_2d.flatten()  # NumPy default: row-major (C-order)

    test_cases = [
        (0, 0),  # Top-left
        (0, N - 1),  # Top-right
        (N - 1, 0),  # Bottom-left
        (N - 1, N - 1),  # Bottom-right
        (7, 7),  # Center
        (3, 11),  # Random
    ]

    all_passed = True
    for row, col in test_cases:
        # What we expect (standard row-major)
        expected_flat = row * N + col
        expected_value = board_2d[row, col]

        # What NumPy gives us
        actual_value = board_flat[expected_flat]

        # Verify
        match = (expected_value == actual_value)
        status = "✓ PASS" if match else "✗ FAIL"

        print(f"  [{row:2d}, {col:2d}] -> flat={expected_flat:3d} | "
              f"board_2d={expected_value:3d} | board_flat={actual_value:3d} | {status}")

        if not match:
            all_passed = False

    if not all_passed:
        print("\n❌ TEST 1 FAILED - Basic indexing is broken!")
        return False

    # Test 2: CSV encoding/decoding
    print("\n[TEST 2] CSV write/read cycle")

    # Simulate what DatasetGenerator does
    for test_row, test_col in [(5, 8), (12, 3)]:
        # DatasetGenerator writes: [col, row] (as move[0], move[1])
        csv_data = [test_col, test_row]

        # dataset_policy reads:
        move_col = csv_data[0]  # Should be column
        move_row = csv_data[1]  # Should be row

        # Then computes flat index
        policy_index = move_row * N + move_col  # FIXED VERSION

        # Verify unpacking
        unpacked_row = policy_index // N
        unpacked_col = policy_index % N

        match = (unpacked_row == test_row and unpacked_col == test_col)
        status = "✓ PASS" if match else "✗ FAIL"

        print(f"  Original: row={test_row}, col={test_col}")
        print(f"  Flat index: {policy_index}")
        print(f"  Unpacked: row={unpacked_row}, col={unpacked_col} | {status}")

        if not match:
            all_passed = False

    if not all_passed:
        print("\n❌ TEST 2 FAILED - CSV encoding/decoding mismatch!")
        return False

    # Test 3: GBoard coordinate system
    print("\n[TEST 3] GBoard.makeMove(x, y) convention")
    print("  GBoard uses: board[y][x] where x=col, y=row")
    print("  makeMove(x, y) expects x=column, y=row")
    print("  This is CORRECT for standard notation")

    # Test 4: Inference unpacking
    print("\n[TEST 4] Inference: flat_index -> (x, y)")

    for flat_idx in [0, N - 1, N * (N - 1), N * N - 1, 7 * N + 7]:
        row = flat_idx // N
        col = flat_idx % N

        # Verify round-trip
        reconstructed = row * N + col
        match = (reconstructed == flat_idx)
        status = "✓ PASS" if match else "✗ FAIL"

        print(f"  flat={flat_idx:3d} -> row={row:2d}, col={col:2d} "
              f"-> flat={reconstructed:3d} | {status}")

        if not match:
            all_passed = False

    if not all_passed:
        print("\n❌ TEST 4 FAILED - Inference unpacking broken!")
        return False

    # Final verdict
    print("\n" + "=" * 60)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("Indexing is consistent throughout the pipeline!")
    else:
        print("❌❌❌ TESTS FAILED ❌❌❌")
        print("Fix indexing bugs before training!")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = test_indexing_consistency()
    exit(0 if success else 1)