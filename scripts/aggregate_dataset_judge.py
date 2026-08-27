"""
Aggregate dataset/benchmark judge verdicts by venue / year / accept-reject.

Inputs:
  outputs/dataset_judge/verdicts/<run>.jsonl   (per-paper verdicts)

Outputs:
  outputs/dataset_judge/agg/<run>__by_venue_year.csv
  outputs/dataset_judge/agg/<run>__by_venue_year_decision.csv
  outputs/dataset_judge/agg/<run>__totals.csv
  prints summary tables to stdout

The judge labels each paper as one of {DATASET, BENCHMARK, BOTH, NEITHER}.
For aggregation we collapse to a binary `is_dataset_or_benchmark`:
  DATASET | BENCHMARK | BOTH  -> 1
  NEITHER                     -> 0

Run:
  python scripts/aggregate_dataset_judge.py outputs/dataset_judge/verdicts/test_v1.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
AGG = ROOT / "outputs/dataset_judge/agg"
AGG.mkdir(parents=True, exist_ok=True)


def load_verdicts(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["is_dsb"] = df["label"].isin(["DATASET", "BENCHMARK", "BOTH"]).astype(int)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verdicts", help="Path to verdicts JSONL")
    ap.add_argument("--name", default=None, help="Output name (default: stem of verdicts)")
    args = ap.parse_args()

    p = Path(args.verdicts)
    name = args.name or p.stem
    df = load_verdicts(p)

    print(f"loaded {len(df):,} verdicts from {p}")
    print(f"  parse errors (label=None): {df['label'].isna().sum():,}")
    print(f"  label distribution:")
    print(df["label"].fillna("PARSE_ERROR").value_counts().to_string())
    print()

    # Restrict to parsed verdicts for agg
    parsed = df[df["label"].notna()].copy()
    print(f"using {len(parsed):,} parsed verdicts for aggregation")
    print()

    # Pull canonical venue family from "venue" field — "ICLR 2024" -> ("ICLR", 2024)
    parsed["venue_family"] = parsed["venue"].str.extract(r"^([A-Z]+)", expand=False)
    parsed["year"] = parsed["year"].astype("Int64")

    # 1) by venue family × year
    t1 = (
        parsed.groupby(["venue_family", "year"])
              .agg(n=("paper_id","size"),
                   n_dsb=("is_dsb","sum"),
                   pct_dsb=("is_dsb", lambda s: round(100 * s.mean(), 2)))
              .reset_index()
              .sort_values(["venue_family", "year"])
    )
    t1.to_csv(AGG / f"{name}__by_venue_year.csv", index=False)
    print("=== by venue x year ===")
    print(t1.to_string(index=False))
    print()

    # 2) by venue × year × decision
    t2 = (
        parsed.groupby(["venue_family", "year", "decision_unified"])
              .agg(n=("paper_id","size"),
                   n_dsb=("is_dsb","sum"),
                   pct_dsb=("is_dsb", lambda s: round(100 * s.mean(), 2)))
              .reset_index()
              .sort_values(["venue_family", "year", "decision_unified"])
    )
    t2.to_csv(AGG / f"{name}__by_venue_year_decision.csv", index=False)
    print("=== by venue x year x decision ===")
    print(t2.to_string(index=False))
    print()

    # 3) totals + accept-rate-within-DSB
    t3 = (
        parsed.groupby("venue_family")
              .agg(n=("paper_id","size"),
                   n_dsb=("is_dsb","sum"),
                   pct_dsb=("is_dsb", lambda s: round(100 * s.mean(), 2)))
              .reset_index()
              .sort_values("n", ascending=False)
    )
    t3.to_csv(AGG / f"{name}__totals.csv", index=False)
    print("=== totals per venue family ===")
    print(t3.to_string(index=False))
    print()

    # 4) acceptance rate of dataset-papers vs non-dataset-papers
    dsb_acc = (
        parsed.groupby(["venue_family", "is_dsb", "decision_unified"])
              .size().unstack("decision_unified", fill_value=0)
    )
    if "accept" in dsb_acc.columns and "reject" in dsb_acc.columns:
        dsb_acc["pct_accepted"] = (
            100 * dsb_acc["accept"] / (dsb_acc["accept"] + dsb_acc["reject"])
        ).round(2)
        print("=== accept rate: dataset/benchmark papers vs other papers ===")
        print(dsb_acc.to_string())
        dsb_acc.to_csv(AGG / f"{name}__accept_rate_by_kind.csv")

    print()
    print(f"wrote: {AGG}/{name}__*.csv")


if __name__ == "__main__":
    main()
