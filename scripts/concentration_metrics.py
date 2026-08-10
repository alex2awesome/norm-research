"""Concentration metrics per (task, bucket) cell over R2 merged_groups.

For each cell:
  - merged_group leaf-count distribution
  - Zipf slope on rank-frequency (log-log linear regression)
  - Shannon entropy of size distribution (normalized to [0,1] by log(n_groups))
  - Top-k cumulative coverage: fraction of leaves in top-{1,3,10} groups
  - Per-group child-count vs leaf-count (proxy for diffuseness — high children +
    few leaves = thin spread of similar-but-distinct rubrics)

Outputs:
  - outputs/analyses/concentration_per_cell.parquet  (one row per cell)
  - outputs/analyses/per_merged_group_size.parquet   (one row per merged_group)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
HIER = ROOT / "outputs" / "hierarchy"
OUT = ROOT / "outputs" / "analyses"

TASKS = [
    "code-review", "creative-writing", "grant-funding", "humor",
    "legal-outcome-prediction", "math-stackexchange", "news-homepages",
    "notice-and-comment", "patents", "peer-review", "press-releases",
]
BUCKETS = ["general", "specific", "hyper_specific"]


def zipf_slope(sizes: list[int]) -> float | None:
    """Fit linear regression on log10(rank) vs log10(size); return slope."""
    if len(sizes) < 3:
        return None
    sizes_sorted = sorted(sizes, reverse=True)
    ranks = np.arange(1, len(sizes_sorted) + 1)
    log_ranks = np.log10(ranks)
    log_sizes = np.log10(np.array(sizes_sorted, dtype=float))
    # filter zero/null sizes
    mask = np.isfinite(log_sizes)
    if mask.sum() < 3:
        return None
    slope, _ = np.polyfit(log_ranks[mask], log_sizes[mask], 1)
    return float(slope)


def shannon_entropy(sizes: list[int]) -> float:
    """Normalized to [0, 1]: 1 = uniform, 0 = single concentrated group."""
    total = sum(sizes)
    if total == 0 or len(sizes) <= 1:
        return 0.0
    probs = np.array(sizes, dtype=float) / total
    h = -np.sum(probs * np.log(probs + 1e-12))
    h_max = math.log(len(sizes))
    return float(h / h_max) if h_max > 0 else 0.0


def top_k_coverage(sizes: list[int], k: int) -> float:
    if not sizes:
        return 0.0
    s = sorted(sizes, reverse=True)
    return sum(s[:k]) / sum(s)


def process_cell(task: str, bucket: str) -> tuple[dict | None, list[dict]]:
    p = HIER / f"{task}_{bucket}_r2_expanded.json"
    if not p.exists():
        return None, []
    d = json.loads(p.read_text())
    merged = d.get("merged_groups", [])
    grand = d.get("grandparents", [])

    if not merged and not grand:
        return None, []

    # per-group rows (one per merged_group)
    group_rows = []
    sizes = []
    for i, m in enumerate(merged):
        n_leaves = m.get("total_leaf_rubrics", 0) or len(m.get("all_leaves", []))
        n_sources = len(m.get("source_clusters", []))  # # of R1 parents folded together
        sizes.append(n_leaves)
        group_rows.append({
            "task": task,
            "bucket": bucket,
            "merged_idx": i,
            "merged_name": m.get("merged_name", ""),
            "n_leaves": n_leaves,
            "n_source_r1_parents": n_sources,
        })

    cell_summary = {
        "task": task,
        "bucket": bucket,
        "n_merged_groups": len(merged),
        "n_grandparents": len(grand),
        "n_leaves_total": sum(sizes),
        "median_size": int(np.median(sizes)) if sizes else 0,
        "mean_size": float(np.mean(sizes)) if sizes else 0.0,
        "max_size": int(max(sizes)) if sizes else 0,
        "zipf_slope": zipf_slope(sizes),
        "entropy_norm": shannon_entropy(sizes),
        "top1_cov": top_k_coverage(sizes, 1),
        "top3_cov": top_k_coverage(sizes, 3),
        "top10_cov": top_k_coverage(sizes, 10),
    }
    return cell_summary, group_rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cell_rows = []
    all_group_rows = []
    for task in TASKS:
        for bucket in BUCKETS:
            summary, group_rows = process_cell(task, bucket)
            if summary is not None:
                cell_rows.append(summary)
                all_group_rows.extend(group_rows)

    cells_df = pd.DataFrame(cell_rows)
    groups_df = pd.DataFrame(all_group_rows)

    cells_path = OUT / "concentration_per_cell.parquet"
    groups_path = OUT / "per_merged_group_size.parquet"
    cells_df.to_parquet(cells_path)
    groups_df.to_parquet(groups_path)
    print(f"wrote {cells_path}  ({len(cells_df)} rows)")
    print(f"wrote {groups_path}  ({len(groups_df)} rows)")
    print()
    print("=== summary ===")
    cols = ["task", "bucket", "n_merged_groups", "n_leaves_total", "zipf_slope", "entropy_norm", "top3_cov"]
    print(cells_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
