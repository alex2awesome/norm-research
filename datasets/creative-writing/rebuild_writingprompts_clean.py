#!/usr/bin/env python3
"""
Rebuild WritingPrompts modeling dataset with leakage fixes:

  1. Drop rows where the body is the Reddit moderator removal template
     ("this submission has been removed"). All such rows are mod-bot
     boilerplate, not real stories, and concentrate in label=0.
  2. Add `prompt_id` (md5 of the prompt text) for group-aware split.
  3. Re-balance per length-bucket and per label after the drop.

Input:  writingprompts_modeling.csv.gz
Output: writingprompts_modeling_clean.csv.gz
"""
import csv
import gzip
import hashlib
import random
import re
import statistics
from collections import defaultdict

csv.field_size_limit(2**31 - 1)

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/creative-writing"
INPUT = BASE + "/writingprompts_modeling.csv.gz"
OUTPUT = BASE + "/writingprompts_modeling_clean.csv.gz"

BUCKET_EDGES = [0, 500, 1000, 2000, 3000, 5000, 10000, 999999]
SEED = 42
MOD_TEMPLATE = re.compile(r"this submission has been removed", re.IGNORECASE)

random.seed(SEED)


def bucket(length):
    for i in range(len(BUCKET_EDGES) - 1):
        if BUCKET_EDGES[i] <= length < BUCKET_EDGES[i + 1]:
            return i
    return len(BUCKET_EDGES) - 2


def prompt_id(prompt):
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()[:16]


def main():
    print("Reading", INPUT)
    rows = []
    n_dropped = 0
    with gzip.open(INPUT, "rt") as f:
        for r in csv.DictReader(f):
            parts = r["text"].split("\n\nSTORY: ", 1)
            prompt = parts[0].replace("PROMPT: ", "").strip()
            story = parts[1] if len(parts) > 1 else ""
            if MOD_TEMPLATE.search(story):
                n_dropped += 1
                continue
            rows.append({
                "text": r["text"],
                "judgement": int(r["judgement"]),
                "story_len": len(story),
                "prompt_id": prompt_id(prompt),
            })
    print(f"  Loaded {len(rows)} rows (dropped {n_dropped} mod-removal templates)")

    # Per-bucket balancing — re-do because dropping mostly-neg rows skewed things.
    buckets = defaultdict(lambda: {0: [], 1: []})
    for r in rows:
        buckets[bucket(r["story_len"])][r["judgement"]].append(r)

    print("\nPer-bucket counts after drop:")
    for b in sorted(buckets):
        lo = BUCKET_EDGES[b]
        hi = BUCKET_EDGES[b + 1] if b + 1 < len(BUCKET_EDGES) else 999999
        n_pos = len(buckets[b][1])
        n_neg = len(buckets[b][0])
        print(f"  Bucket {b} ({lo}-{hi}): pos={n_pos}, neg={n_neg}, balanced={min(n_pos, n_neg)} per class")

    balanced = []
    for b in sorted(buckets):
        pos = buckets[b][1]
        neg = buckets[b][0]
        n = min(len(pos), len(neg))
        if n == 0:
            continue
        random.shuffle(pos)
        random.shuffle(neg)
        balanced.extend(pos[:n])
        balanced.extend(neg[:n])

    random.shuffle(balanced)
    print(f"\nTotal balanced rows: {len(balanced)} ({len(balanced)//2} per class)")

    # Verify length parity
    lens = {0: [], 1: []}
    for r in balanced:
        lens[r["judgement"]].append(r["story_len"])
    if lens[0] and lens[1]:
        ratio = statistics.mean(lens[1]) / statistics.mean(lens[0])
        print(f"  Length ratio (label1/label0): {ratio:.3f}")

    # Prompt-id stats
    pid_counts = defaultdict(int)
    for r in balanced:
        pid_counts[r["prompt_id"]] += 1
    print(f"  Unique prompt_ids: {len(pid_counts)}")
    print(f"  Max rows/prompt: {max(pid_counts.values())}, median: {sorted(pid_counts.values())[len(pid_counts)//2]}")

    print("\nWriting", OUTPUT)
    with gzip.open(OUTPUT, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "judgement", "prompt_id"])
        writer.writeheader()
        for r in balanced:
            writer.writerow({"text": r["text"], "judgement": r["judgement"], "prompt_id": r["prompt_id"]})
    print("Done.")


if __name__ == "__main__":
    main()
