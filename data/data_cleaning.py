import csv
import re
import glob
import argparse
import os

# Default single-file mode (backwards compatible)
DEFAULT_INPUT = "raw/all_combined_new.csv"
DEFAULT_OUTPUT = "all_combined_clean_new.csv"

# Accepts '(12, 10)', "(12,10)", etc.
MOVE_TUPLE = re.compile(r"\((\d+)\s*,\s*(\d+)\)")

BOARD_SIZE = 15
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE
BOARD_COLS = [f"b{i}" for i in range(ACTION_SIZE)]


def iter_rows_from_inputs(inputs_pattern: str):
    """Yield raw rows from one file path or a glob of files."""
    paths = []
    if any(ch in inputs_pattern for ch in ["*", "?", "["]):
        paths = sorted(glob.glob(inputs_pattern))
    else:
        paths = [inputs_pattern]

    for p in paths:
        if not os.path.exists(p):
            print(f"[WARN] Skipping missing file: {p}")
            continue
        with open(p, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                yield row, p


def parse_row(row):
    """
    Expecting: b0..b224, cp, move
    where 'move' can be '(x,y)' or two trailing ints 'move_x','move_y'.
    Returns normalized: (board_flat[225], cp_int, move_x, move_y) or None if bad.
    """
    # Basic length checks
    if len(row) == ACTION_SIZE + 2:
        # b0..b224, cp, "(x,y)"
        board_flat = row[:ACTION_SIZE]
        cp = row[ACTION_SIZE]
        move_field = row[ACTION_SIZE + 1]
        m = MOVE_TUPLE.match(move_field.strip())
        if not m:
            return None, "bad_parse"
        move_x, move_y = int(m.group(1)), int(m.group(2))

    elif len(row) == ACTION_SIZE + 3:
        # b0..b224, cp, move_x, move_y
        board_flat = row[:ACTION_SIZE]
        cp = row[ACTION_SIZE]
        try:
            move_x = int(row[ACTION_SIZE + 1])
            move_y = int(row[ACTION_SIZE + 2])
        except ValueError:
            return None, "bad_parse"

    else:
        return None, "bad_len"

    # Cast board + cp
    try:
        board_flat = [int(v) for v in board_flat]
        cp = int(cp)
    except ValueError:
        return None, "bad_parse"

    # Range checks
    if not (0 <= move_x < BOARD_SIZE and 0 <= move_y < BOARD_SIZE):
        return None, "bad_vals"

    return (board_flat, cp, move_x, move_y), None


def main():
    ap = argparse.ArgumentParser(description="Clean and deduplicate Gomoku raw CSV.")
    ap.add_argument("--inputs", type=str, default=DEFAULT_INPUT,
                    help='Path or glob to input file(s). E.g. "raw/*.csv" or "raw/all_combined_new.csv"')
    ap.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                    help="Cleaned CSV output path.")
    args = ap.parse_args()

    print(f"[INFO] Reading from: {args.inputs}")
    print(f"[INFO] Writing cleaned data to: {args.output}")

    rows_in = 0
    rows_out = 0
    bad_len = bad_parse = bad_vals = 0
    dup_count = 0

    # Deduplication set (exact board + cp + move)
    seen = set()

    with open(args.output, "w", newline="") as outfile:
        writer = csv.writer(outfile)

        for row, src in iter_rows_from_inputs(args.inputs):
            rows_in += 1
            parsed, err = parse_row(row)
            if err:
                if err == "bad_len":
                    bad_len += 1
                elif err == "bad_parse":
                    bad_parse += 1
                elif err == "bad_vals":
                    bad_vals += 1
                continue

            board_flat, cp, move_x, move_y = parsed

            # Dedup key
            key = (*board_flat, cp, move_x, move_y)
            if key in seen:
                dup_count += 1
                continue
            seen.add(key)

            writer.writerow(board_flat + [cp, move_x, move_y])
            rows_out += 1

    print(f"[DONE] Clean data written to: {args.output}")
    print(f"[STATS] input_rows={rows_in} | kept={rows_out} | dropped={rows_in - rows_out}")
    print(f"[DETAILS] bad_len={bad_len} | bad_parse={bad_parse} | bad_vals={bad_vals} | duplicates={dup_count}")


if __name__ == "__main__":
    main()
