"""
Consolidate open-coding output across 11 tasks.

Reads outputs/attr_open_coding/outputs/<task>.jsonl, normalises codes,
and produces:
  outputs/attr_open_coding/agg/<axis>__topcodes.csv         (frequency-ranked codes per axis)
  outputs/attr_open_coding/agg/<axis>__by_task.csv          (counts × task, top 50 codes)
  outputs/attr_open_coding/agg/code_index.parquet           (aspect × axis × code, long form)

Prints top-30 codes per axis to stdout.
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
IN_DIR = ROOT / "outputs/attr_open_coding/outputs"
AGG = ROOT / "outputs/attr_open_coding/agg"
AGG.mkdir(parents=True, exist_ok=True)

AXES = ["rule_type", "focus", "scope", "modality"]


def norm(code: str) -> str:
    s = (code or "").strip().lower()
    s = s.lstrip("?").strip()
    s = re.sub(r"\s+", " ", s)
    # collapse en/em dashes to "-"
    s = s.replace("—", "-").replace("–", "-")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=30)
    args = ap.parse_args()

    rows = []
    for fp in sorted(IN_DIR.glob("*.jsonl")):
        task = fp.stem
        with open(fp) as f:
            for line in f:
                o = json.loads(line)
                for axis in AXES:
                    for c in (o.get(axis) or []):
                        rows.append({
                            "task": task,
                            "aspect_id": o["aspect_id"],
                            "axis": axis,
                            "code": norm(c),
                            "code_raw": c,
                        })
    df = pd.DataFrame(rows)
    df.to_parquet(AGG / "code_index.parquet")
    print(f"loaded {len(df):,} (aspect × axis × code) rows from "
          f"{df['task'].nunique()} tasks")
    print()

    # Per-axis top codes
    for axis in AXES:
        sub = df[df.axis == axis]
        topall = sub["code"].value_counts().head(args.topk)
        out = topall.reset_index()
        out.columns = ["code", "n"]
        out.to_csv(AGG / f"{axis}__topcodes.csv", index=False)
        print(f"=== {axis} — top {args.topk} codes (of {sub['code'].nunique()} unique) ===")
        for code, n in topall.items():
            n_tasks = sub[sub["code"] == code]["task"].nunique()
            print(f"  {n:>4}  [in {n_tasks:>2}/11 tasks]  {code}")
        print()

        # Top-50 by task
        top50 = sub["code"].value_counts().head(50).index.tolist()
        by_task = (
            sub[sub["code"].isin(top50)]
            .groupby(["code", "task"]).size().unstack("task", fill_value=0)
            .loc[top50]
        )
        by_task.to_csv(AGG / f"{axis}__by_task.csv")

    print(f"wrote agg files to {AGG}")


if __name__ == "__main__":
    main()
