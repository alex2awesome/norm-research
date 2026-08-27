"""Stratified sample of N peer-review datapoints from train.csv.gz.

Stratification: by (venue × label) to keep venue distribution similar to
the full train set while balancing accept/reject within venue.

Output: runs/validity_full/full_v2/datapoints.json
  [{"datapoint_id": "d0000", "paper_id": "...", "venue": "...",
    "judgement": 0|1, "text": "...full text up to TRUNC chars..."}]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--text-trunc", type=int, default=12000,
                    help="Truncate paper text to this many chars (judge "
                         "+ code both handle up to ~3000-token chunks)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--balance-labels", action="store_true",
                    help="Force exact 50/50 accept/reject split (matches v1)")
    args = ap.parse_args()

    random.seed(args.seed)
    out_path = Path("runs/validity_full/full_v2/datapoints.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading datasets/peer-review/splits/train.csv.gz...")
    df = pd.read_csv("datasets/peer-review/splits/train.csv.gz")
    print(f"  total rows: {len(df):,}")
    print(f"  venues: {df['venue'].value_counts().to_dict()}")
    print(f"  label dist: {df['judgement'].value_counts().to_dict()}")

    # Drop rows with too-short text (less than 500 chars)
    df = df[df["text"].fillna("").str.len() >= 500].reset_index(drop=True)
    print(f"  after >=500 char filter: {len(df):,}")

    # Stratified sample by (venue, label)
    groups = df.groupby(["venue", "judgement"])
    print(f"\n  group sizes (venue, label):")
    sizes = groups.size().to_dict()
    if args.balance_labels:
        # Force 50/50 by subsampling each label class to n//2,
        # stratified by venue within each label class
        per_label = args.n // 2
        targets = {}
        for label in (0, 1):
            label_sizes = {k: v for k, v in sizes.items() if k[1] == label}
            label_total = sum(label_sizes.values())
            for k, v in label_sizes.items():
                targets[k] = max(1, round(per_label * v / label_total))
    else:
        total = sum(sizes.values())
        targets = {k: max(1, round(args.n * v / total)) for k, v in sizes.items()}
    for k, v in sorted(sizes.items()):
        print(f"    {k}: pop={v}  target={targets[k]}")

    # Sample
    sampled = []
    for (venue, label), gdf in groups:
        target = targets[(venue, label)]
        if len(gdf) <= target:
            picked = gdf
        else:
            picked = gdf.sample(n=target, random_state=args.seed)
        sampled.append(picked)
    out_df = pd.concat(sampled).reset_index(drop=True)
    # Shuffle
    out_df = out_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    if len(out_df) > args.n:
        out_df = out_df.head(args.n)

    print(f"\n  sampled: {len(out_df)}  (target {args.n})")
    print(f"  final label balance: {out_df['judgement'].value_counts().to_dict()}")
    print(f"  final venue: {out_df['venue'].value_counts().to_dict()}")

    # Format as datapoint records
    records = []
    for i, row in out_df.iterrows():
        text = (row["text"] or "")[:args.text_trunc]
        records.append({
            "datapoint_id": f"d{i:05d}",
            "paper_id": str(row.get("paper_id", "")),
            "venue": row.get("venue", ""),
            "year": row.get("year", None) if pd.notna(row.get("year")) else None,
            "domain": row.get("domain", ""),
            "judgement": int(row["judgement"]),
            "text": text,
        })

    out_path.write_text(json.dumps(records, indent=1))
    print(f"\nwrote {len(records)} datapoints to {out_path}")
    print(f"  text length: mean={int(out_df['text'].str.len().clip(upper=args.text_trunc).mean())} "
          f"max={int(out_df['text'].str.len().clip(upper=args.text_trunc).max())}")


if __name__ == "__main__":
    main()
