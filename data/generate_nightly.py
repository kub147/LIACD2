"""
generate_nightly.py
-------------------
Nightly / long-running dataset generator using your DatasetGenerator.run() method.

This script:
  - Randomizes starting turn for opening variety (min/max).
  - Repeatedly calls DatasetGenerator.run(...) N times.
  - Writes *raw* rows directly into a CSV file.
  - Never writes headers (DatasetGenerator writes raw entries only).
  - Creates data/raw/<timestamp>.csv by default.

HOW TO RUN
==========

1) Run normally:
    python data/generate_nightly.py --games 200

2) Run with caffeine (MacOS) to prevent sleep:
    caffeinate -i python data/generate_nightly.py --games 200

3) Run two processes in parallel (each writes to different file):
    caffeinate -i python data/generate_nightly.py --games 200 --output data/raw/runA.csv &
    caffeinate -i python data/generate_nightly.py --games 200 --output data/raw/runB.csv &

4) Example with custom compute limits:
    caffeinate -i python data/generate_nightly.py \
        --games 300 \
        --starting_turn_min 12 \
        --starting_turn_max 20 \
        --timeout 45 \
        --sim_limit_x 9000 \
        --sim_limit_o 9000

IMPORTANT:
 - Each process must write to a separate CSV file.
 - The CSV has no header and is immediately usable by data_cleaning.py.
"""

# ------------- Path Setup -------------
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "data"))

import argparse
import os
import time
from datetime import datetime
import random

# Import DatasetGenerator
import DatasetGenerator as DG

if not hasattr(DG, "FILENAME"):
    raise RuntimeError("DatasetGenerator.py must define global variable FILENAME")

def parse_args():
    p = argparse.ArgumentParser(description="Run DatasetGenerator overnight.")
    # Board setup
    p.add_argument("--size", type=int, default=15, help="Board size (15 for 15x15).")

    # Opening/midgame window
    p.add_argument("--starting_turn_min", type=int, default=12,
                   help="Minimum random starting turn (inclusive).")
    p.add_argument("--starting_turn_max", type=int, default=20,
                   help="Maximum random starting turn (inclusive).")
    p.add_argument("--ending_turn", type=int, default=60,
                   help="Stop the game loop once this turn is reached.")

    # MCTS compute budget
    p.add_argument("--timeout", type=int, default=60,
                   help="Seconds per move (time-based budget).")
    p.add_argument("--sim_limit_x", type=int, default=8000,
                   help="Simulation limit for player X (2).")
    p.add_argument("--sim_limit_o", type=int, default=8000,
                   help="Simulation limit for player O (1).")

    # How many games to generate in this run
    p.add_argument("--games", type=int, default=50,
                   help="Number of games to play (each contributes many rows).")

    # Output
    p.add_argument("--output", type=str, default=None,
                   help="Output CSV path. If omitted, a timestamped file in data/raw/ is used.")
    p.add_argument("--log_every", type=int, default=1,
                   help="Print a short progress line every N games.")
    p.add_argument("--sleep_between", type=float, default=0.0,
                   help="Optional sleep (seconds) between games to cool CPU (e.g., 0.5).")

    # Random seed
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    return p.parse_args()


def make_output_path(user_path: str | None) -> str:
    if user_path:
        out = user_path
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("data/raw", exist_ok=True)
        out = f"data/raw/Gomoku_15x15_{ts}.csv"
    # touch file so it's guaranteed to exist
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "a").close()
    return out


def main():
    args = parse_args()
    random.seed(args.seed)

    out_path = make_output_path(args.output)
    # Set the output file inside your original generator module
    DG.FILENAME = out_path
    print(f"[INFO] Output CSV: {out_path}")
    print(f"[INFO] Games to play: {args.games}")
    print(f"[INFO] Params: size={args.size} | starting_turn∈[{args.starting_turn_min},{args.starting_turn_max}] "
          f"| ending_turn={args.ending_turn} | timeout={args.timeout}s | simX={args.sim_limit_x} | simO={args.sim_limit_o}")

    games_done = 0
    errors = 0
    start_ts = time.time()

    for g in range(1, args.games + 1):
        # Randomize the opening length for variety
        starting_turn = random.randint(args.starting_turn_min, args.starting_turn_max)

        try:
            # Call your existing generator's run(...)
            # Signature (based on your file):
            # run(size, starting_turn=15, ending_turn=40, timeout=60,
            #     simulation_limit_X=10000, simulation_limit_O=10000)
            DG.run(
                size=args.size,
                starting_turn=starting_turn,
                ending_turn=args.ending_turn,
                timeout=args.timeout,
                simulation_limit_X=args.sim_limit_x,
                simulation_limit_O=args.sim_limit_o,
            )
            games_done += 1
        except KeyboardInterrupt:
            print("\n[WARN] Interrupted by user. Exiting gracefully.")
            break
        except Exception as e:
            errors += 1
            print(f"[ERR] Game {g} failed: {e!r}")

        if args.sleep_between > 0:
            time.sleep(args.sleep_between)

        if g % args.log_every == 0:
            elapsed = time.time() - start_ts
            print(f"[INFO] Finished game {g}/{args.games} | ok={games_done}, err={errors}, elapsed={elapsed/60:.1f} min")

    print(f"[DONE] Completed. ok={games_done}, err={errors}, output={out_path}")


if __name__ == "__main__":
    main()
