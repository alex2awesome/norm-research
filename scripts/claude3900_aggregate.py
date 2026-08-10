#!/usr/bin/env python3
"""
Stage 3 aggregation + analysis for the 3,900-pair Claude re-labeling.

Inputs:
  - 8 shard parquets in outputs/v2_analysis/comp_unified_claude3900_shards/
  - Original Qwen labels (comp_unified_editorial_labels.parquet)
  - Prior 300-pair Claude labels (qwen_validation_claude_labels.parquet)

Outputs:
  - outputs/v2_analysis/comp_unified_claude3900_labels.parquet
  - Printed summary to stdout
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/Users/spangher/Projects/stanford-research/norm-research/outputs/v2_analysis")


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    cats = sorted(set(a.tolist()) | set(b.tolist()))
    n = len(a)
    po = (a == b).mean()
    pe = 0.0
    for c in cats:
        pe += (a == c).mean() * (b == c).mean()
    return (po - pe) / (1 - pe + 1e-12)


def main():
    shard_dir = BASE / "comp_unified_claude3900_shards"
    shards = sorted(shard_dir.glob("comp_unified_claude3900_shard*.parquet"))
    parts = []
    timings = {}
    for sp in shards:
        df = pd.read_parquet(sp)
        parts.append(df)
        timings[sp.name] = {
            "n": len(df),
            "wall_total_s": float(df["wall_seconds"].sum()),
            "wall_mean_s": float(df["wall_seconds"].mean()),
            "wall_median_s": float(df["wall_seconds"].median()),
            "parse_fail": int((df["status"] == "parse_fail").sum()),
            "retry_ok": int((df["status"] == "retry_ok").sum()),
        }
    all_df = pd.concat(parts, ignore_index=True)
    print(f"Concat: {len(all_df)} rows from {len(shards)} shards")

    out_path = BASE / "comp_unified_claude3900_labels.parquet"
    all_df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")

    # Summary stats
    n_parse_fail = int((all_df["status"] == "parse_fail").sum())
    n_retry = int((all_df["status"] == "retry_ok").sum())
    n_ok = int((all_df["status"] == "ok").sum())
    print(f"\n=== status counts ===")
    print(f"ok={n_ok} retry_ok={n_retry} parse_fail={n_parse_fail}")

    valid = all_df[all_df["claude_label"].notna()].copy()
    valid["claude_label"] = valid["claude_label"].astype(int)
    valid["qwen_label_int"] = valid["qwen_label"].astype(int)

    print(f"\n=== label balance ===")
    print(f"Claude: {valid['claude_label'].value_counts().to_dict()}")
    print(f"Qwen:   {valid['qwen_label_int'].value_counts().to_dict()}")

    print(f"\n=== Claude vs Qwen agreement (N={len(valid)}) ===")
    exact = float((valid["claude_label"] == valid["qwen_label_int"]).mean())
    kappa = cohen_kappa(valid["claude_label"].values, valid["qwen_label_int"].values)
    print(f"exact agreement: {exact:.4f}")
    print(f"cohen kappa:     {kappa:.4f}")
    # Asymmetry
    c1_q0 = int(((valid["claude_label"] == 1) & (valid["qwen_label_int"] == 0)).sum())
    c0_q1 = int(((valid["claude_label"] == 0) & (valid["qwen_label_int"] == 1)).sum())
    print(f"Claude=1,Qwen=0: {c1_q0}")
    print(f"Claude=0,Qwen=1: {c0_q1}")

    # Self-consistency vs the prior 300 Claude labels
    prior = pd.read_parquet(BASE / "qwen_validation_claude_labels.parquet")
    prior = prior.rename(columns={"claude_label": "claude_label_prior", "claude_reason": "claude_reason_prior"})
    join = valid.merge(prior[["pair_id", "claude_label_prior"]], on="pair_id", how="inner")
    print(f"\n=== Claude self-consistency vs prior 300 (N={len(join)}) ===")
    if len(join) > 0:
        join["claude_label_prior"] = join["claude_label_prior"].astype(int)
        sc_exact = float((join["claude_label"] == join["claude_label_prior"]).mean())
        sc_kappa = cohen_kappa(join["claude_label"].values, join["claude_label_prior"].values)
        flip1to0 = int(((join["claude_label_prior"] == 1) & (join["claude_label"] == 0)).sum())
        flip0to1 = int(((join["claude_label_prior"] == 0) & (join["claude_label"] == 1)).sum())
        print(f"exact: {sc_exact:.4f}")
        print(f"kappa: {sc_kappa:.4f}")
        print(f"prior=1,new=0: {flip1to0}")
        print(f"prior=0,new=1: {flip0to1}")

    # Per-platform agreement
    print(f"\n=== per-platform agreement ===")
    for plat, sub in valid.groupby("platform"):
        ex = float((sub["claude_label"] == sub["qwen_label_int"]).mean())
        kp = cohen_kappa(sub["claude_label"].values, sub["qwen_label_int"].values)
        print(f"{plat}: N={len(sub)} exact={ex:.4f} kappa={kp:.4f}")

    # Per-decile (max_sim) agreement
    print(f"\n=== per-decile (max_sim) agreement ===")
    valid["decile"] = pd.cut(valid["max_sim"], bins=np.linspace(0, 1.0, 11), include_lowest=True)
    for dec, sub in valid.groupby("decile", observed=True):
        ex = float((sub["claude_label"] == sub["qwen_label_int"]).mean())
        print(f"{dec}: N={len(sub)} exact={ex:.4f} Claude=1 frac={sub['claude_label'].mean():.3f}")

    # Timing
    print(f"\n=== per-shard timing ===")
    total_serial = 0.0
    max_shard_wall = 0.0
    for name, t in timings.items():
        print(f"{name}: N={t['n']} wall_total={t['wall_total_s']:.0f}s mean={t['wall_mean_s']:.2f}s median={t['wall_median_s']:.2f}s parse_fail={t['parse_fail']}")
        total_serial += t["wall_total_s"]
        max_shard_wall = max(max_shard_wall, t["wall_total_s"])
    print(f"Serial-equivalent wall: {total_serial:.0f}s")
    print(f"Max single-shard wall:  {max_shard_wall:.0f}s")
    print(f"Throughput (effective, 8-way): {len(valid) / (max_shard_wall / 60):.1f} pairs/min")
    print(f"Throughput (per shard):        {timings[list(timings)[0]]['n'] / (timings[list(timings)[0]]['wall_total_s'] / 60):.1f} pairs/min")

    # Top 5 flip examples (Claude flipped from Qwen)
    print(f"\n=== sample Claude-flipped-from-Qwen examples ===")
    flips = valid[valid["claude_label"] != valid["qwen_label_int"]].head(5)
    for _, r in flips.iterrows():
        print(f"\n  pair_id={r['pair_id']} platform={r['platform']} max_sim={r.get('max_sim', 'NA'):.3f}")
        print(f"    Qwen={int(r['qwen_label_int'])}  ->  Claude={int(r['claude_label'])}")
        print(f"    qwen_reason:   {str(r.get('qwen_reason', ''))[:160]}")
        print(f"    claude_reason: {str(r.get('claude_reason', ''))[:160]}")


if __name__ == "__main__":
    main()
