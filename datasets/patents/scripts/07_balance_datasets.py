#!/usr/bin/env python3
"""
Downsample and balance patent CSVs to a target size with 50/50 class balance.
Each input CSV becomes a `_balanced` variant with capped row count.
"""
import argparse
import csv
import gzip
from pathlib import Path
import random

csv.field_size_limit(2**31 - 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input CSV.gz with text,judgement columns")
    p.add_argument("--output", required=True)
    p.add_argument("--per-class", type=int, default=250_000,
                   help="Target rows per class (default 250K, total 500K)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    pos_rows, neg_rows = [], []
    print(f"Streaming {args.input}...")
    with gzip.open(args.input, "rt") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            j = int(row["judgement"])
            (pos_rows if j == 1 else neg_rows).append(row["text"])
            if (i + 1) % 500_000 == 0:
                print(f"  read {i+1:,} (pos={len(pos_rows):,} neg={len(neg_rows):,})")

    print(f"Total: pos={len(pos_rows):,}, neg={len(neg_rows):,}")

    # Sample (without replacement)
    target = args.per_class
    if len(pos_rows) > target:
        rng.shuffle(pos_rows)
        pos_rows = pos_rows[:target]
    if len(neg_rows) > target:
        rng.shuffle(neg_rows)
        neg_rows = neg_rows[:target]

    print(f"Sampled: pos={len(pos_rows):,}, neg={len(neg_rows):,}")

    # Combine and shuffle
    combined = [(t, 1) for t in pos_rows] + [(t, 0) for t in neg_rows]
    rng.shuffle(combined)

    print(f"Writing {len(combined):,} rows to {args.output}...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "judgement"])
        writer.writeheader()
        for text, j in combined:
            writer.writerow({"text": text, "judgement": j})
    print("Done!")


if __name__ == "__main__":
    main()
