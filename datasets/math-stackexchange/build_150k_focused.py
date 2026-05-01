#!/usr/bin/env python
"""Build a 150K focused version of the Math.SE binary dataset.

Keeps 100% of proof/intuition/soft-question tagged rows (the "interesting"
subset for elegance/quality signal), downsamples from other tags to fill
remaining quota. Maintains exact 50/50 class balance.
"""

import csv
import gzip
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent
SPLITS_DIR = DATA_DIR / "splits"
OUT_DIR = DATA_DIR / "splits_150k"

PRIORITY_TAGS = {
    "proof-writing", "proof-verification", "proof-explanation",
    "alternative-proof", "solution-verification",
    "intuition", "soft-question",
}

TARGET_TOTAL = 150000
TARGET_PER_CLASS = TARGET_TOTAL // 2
SEED = 42


def load_all_rows():
    """Load all rows from the balanced dataset."""
    rows = []
    for split in ["train", "eval", "test"]:
        path = SPLITS_DIR / f"{split}.csv.gz"
        with gzip.open(path, "rt") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["label"] = int(row["label"])
                row["answer_score"] = int(row["answer_score"])
                row["answer_len"] = int(row["answer_len"])
                rows.append(row)
    print(f"Loaded {len(rows)} total rows")
    return rows


def has_priority_tag(row):
    tags = set(row["question_tags"].split("|"))
    return bool(tags & PRIORITY_TAGS)


def main():
    rng = random.Random(SEED)
    rows = load_all_rows()

    # Separate priority vs other
    priority_pos = [r for r in rows if has_priority_tag(r) and r["label"] == 1]
    priority_neg = [r for r in rows if has_priority_tag(r) and r["label"] == 0]
    other_pos = [r for r in rows if not has_priority_tag(r) and r["label"] == 1]
    other_neg = [r for r in rows if not has_priority_tag(r) and r["label"] == 0]

    print(f"\nPriority tags ({', '.join(sorted(PRIORITY_TAGS))}):")
    print(f"  Positive: {len(priority_pos)}")
    print(f"  Negative: {len(priority_neg)}")
    print(f"Other tags:")
    print(f"  Positive: {len(other_pos)}")
    print(f"  Negative: {len(other_neg)}")

    # Keep all priority rows, but balance within priority first
    priority_min = min(len(priority_pos), len(priority_neg))
    priority_pos_keep = rng.sample(priority_pos, priority_min) if len(priority_pos) > priority_min else priority_pos
    priority_neg_keep = rng.sample(priority_neg, priority_min) if len(priority_neg) > priority_min else priority_neg

    # How many more do we need from "other"?
    need_pos = TARGET_PER_CLASS - len(priority_pos_keep)
    need_neg = TARGET_PER_CLASS - len(priority_neg_keep)

    print(f"\nAfter priority balance ({priority_min} each):")
    print(f"  Need from other: {need_pos} pos, {need_neg} neg")

    if need_pos > len(other_pos) or need_neg > len(other_neg):
        print(f"  WARNING: not enough other rows, adjusting target")
        need_pos = min(need_pos, len(other_pos))
        need_neg = min(need_neg, len(other_neg))
        need_both = min(need_pos, need_neg)
        need_pos = need_neg = need_both

    other_pos_sample = rng.sample(other_pos, need_pos)
    other_neg_sample = rng.sample(other_neg, need_neg)

    # Combine
    final_pos = priority_pos_keep + other_pos_sample
    final_neg = priority_neg_keep + other_neg_sample
    all_rows = final_pos + final_neg
    rng.shuffle(all_rows)

    print(f"\nFinal dataset:")
    print(f"  Positive: {len(final_pos)} ({len(priority_pos_keep)} priority + {len(other_pos_sample)} other)")
    print(f"  Negative: {len(final_neg)} ({len(priority_neg_keep)} priority + {len(other_neg_sample)} other)")
    print(f"  Total: {len(all_rows)}")
    print(f"  Priority fraction: {100*(len(priority_pos_keep)+len(priority_neg_keep))/len(all_rows):.1f}%")

    # Stats
    pos_qlens = [int(r.get("answer_len", 0)) for r in final_pos]
    neg_qlens = [int(r.get("answer_len", 0)) for r in final_neg]
    print(f"\n  Answer len — pos: mean={np.mean(pos_qlens):.0f} med={np.median(pos_qlens):.0f}"
          f"  neg: mean={np.mean(neg_qlens):.0f} med={np.median(neg_qlens):.0f}")

    # Tag balance check
    tag_counts = defaultdict(lambda: [0, 0])
    for r in all_rows:
        primary = r["question_tags"].split("|")[0] if r["question_tags"] else "_none_"
        tag_counts[primary][r["label"]] += 1
    print(f"\n  Top 10 tags:")
    for tag, (nc, pc) in sorted(tag_counts.items(), key=lambda x: -(x[1][0]+x[1][1]))[:10]:
        total = nc + pc
        print(f"    {tag:<35} {total:>6}  (pos: {100*pc/total:.0f}%, neg: {100*nc/total:.0f}%)")

    # Split 80/10/10
    n = len(all_rows)
    n_train = int(n * 0.8)
    n_eval = int(n * 0.1)

    splits = {
        "train": all_rows[:n_train],
        "eval": all_rows[n_train:n_train + n_eval],
        "test": all_rows[n_train + n_eval:],
    }

    OUT_DIR.mkdir(exist_ok=True)
    fieldnames = ["id", "text", "label", "question_id", "question_tags",
                  "answer_score", "answer_len"]

    for split_name, split_rows in splits.items():
        path = OUT_DIR / f"{split_name}.csv.gz"
        with gzip.open(path, "wt", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in split_rows:
                writer.writerow({k: r[k] for k in fieldnames})

        n_pos = sum(1 for r in split_rows if r["label"] == 1)
        n_neg = sum(1 for r in split_rows if r["label"] == 0)
        print(f"  {split_name}: {len(split_rows)} rows ({n_pos} pos, {n_neg} neg) → {path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
