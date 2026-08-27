#!/usr/bin/env python3
"""Gather all Claude R4 editorial-similarity labels into one training table.

Sources (all Claude-labeled, the gold standard):
  1. comp_unified_claude_bank_scores_v2.parquet  -- 1,004 CC+Luogu pairs (pair_id, claude_label)
     joined to comp_unified_editorial_labels.parquet for codes + platform + canonical_pid
  2. The LC 2K Claude-labeled set (auto-discovered, see --lc-labels)
  3. comp_fourplatform_label_shards/results/*.jsonl -- new CC/CF/AC labels (in progress;
     picks up whatever has finished; resumable — rerun this script when more land)

Output: one parquet with columns
  pair_id, platform, canonical_pid, editorial_code, candidate_code, label, source
Problem-grouped splits are assigned downstream by the trainer (NOT here).
"""
import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
V2 = REPO / "outputs/v2_analysis"


def load_cc_luogu() -> pd.DataFrame:
    scores = pd.read_parquet(V2 / "comp_unified_claude_bank_scores_v2.parquet",
                             columns=["pair_id", "claude_label"])
    el = pd.read_parquet(V2 / "comp_unified_editorial_labels.parquet")
    keep = ["pair_id", "platform", "candidate_id", "editorial_id",
            "candidate_code", "editorial_code"]
    el = el[[c for c in keep if c in el.columns]]
    df = scores.merge(el, on="pair_id", how="inner")
    df["label"] = df["claude_label"].astype(int)
    df["source"] = "claude_1k"
    return df


def load_fourplatform_shards() -> pd.DataFrame:
    shard_dir = V2 / "comp_fourplatform_label_shards"
    results = sorted((shard_dir / "results").glob("*.jsonl"))
    if not results:
        return pd.DataFrame()
    labels = []
    for f in results:
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "ok" and r.get("claude_label") in (0, 1):
                labels.append({"pair_id": r["pair_id"], "label": int(r["claude_label"])})
    if not labels:
        return pd.DataFrame()
    lab = pd.DataFrame(labels).drop_duplicates("pair_id")
    # join codes from the staged shard inputs
    inputs = []
    for f in sorted((shard_dir / "shards").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            inputs.append({
                "pair_id": r["pair_id"],
                "platform": r.get("source") or r.get("platform"),
                "canonical_pid": r.get("problem_id"),
                "candidate_code": r.get("candidate_text"),
                "editorial_code": r.get("editorial_text"),
            })
    inp = pd.DataFrame(inputs).drop_duplicates("pair_id")
    df = lab.merge(inp, on="pair_id", how="inner")
    df["source"] = "claude_fourplatform"
    return df


def load_lc_2k() -> pd.DataFrame:
    labels = pd.read_parquet(V2 / "lc_python_2k_claude_labels.parquet",
                             columns=["pair_id", "label"])
    sample = pd.read_parquet(V2 / "lc_python_2k_sample.parquet",
                             columns=["pair_id", "question_slug", "code",
                                      "editorial_code"])
    df = labels.merge(sample, on="pair_id", how="inner")
    df = df.rename(columns={"code": "candidate_code",
                            "question_slug": "canonical_pid"})
    df["platform"] = "lc"
    df["source"] = "claude_lc2k"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(V2 / "distill_training_table.parquet"))
    args = ap.parse_args()

    parts = [load_cc_luogu(), load_fourplatform_shards(), load_lc_2k()]
    parts = [p for p in parts if len(p)]
    df = pd.concat(parts, ignore_index=True)

    need = ["pair_id", "platform", "canonical_pid", "editorial_code",
            "candidate_code", "label", "source"]
    for c in need:
        if c not in df.columns:
            df[c] = None
    df = df[need].dropna(subset=["editorial_code", "candidate_code", "label"])
    df = df[(df.editorial_code.str.len() >= 30) & (df.candidate_code.str.len() >= 30)]
    df = df.drop_duplicates("pair_id")

    df.to_parquet(args.out)
    print(f"wrote {len(df)} rows -> {args.out}")
    print(df.groupby(["platform", "source"]).agg(
        n=("label", "count"), pos=("label", "mean")).round(3))


if __name__ == "__main__":
    main()
