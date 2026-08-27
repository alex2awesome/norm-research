"""Claude-only analyses: Aspect-aspect clustering + Per-aspect Discriminativeness.

Operates on the 40 best-covered dps per task at Claude p0.

Aspect-aspect clustering:
  - Build dp × aspect matrix of Claude scores (numeric, 0/0.5/1; NaN if applicable=False)
  - For each aspect pair, compute Spearman correlation on dps where both scored
  - Cluster aspects with average-linkage on (1 - |corr|) distance, threshold 0.2
  - Output: per-task aspect cluster assignments + redundancy stats

Per-aspect Discriminativeness:
  - Load outcome labels (judgement) per dp from canonical dataset (per [[reference_v2_task_datasets]])
  - For each aspect: compute mean(score | y=1) - mean(score | y=0) and p-value (two-sample t)
  - Bonferroni-correct over # aspects
  - Output: ranked aspect list with effect sizes + significance
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_ind
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Labels live in each task's datapoints.json (per task setup), keyed by datapoint_id with col `judgement`.
def load_labels(task: str, repo: Path):
    p = repo / "runs/validity_full/v2" / task / "datapoints.json"
    if not p.exists():
        return None
    dps = json.loads(p.read_text())
    s = pd.Series(
        {d["datapoint_id"]: d["judgement"] for d in dps if "judgement" in d}
    )
    s = s[~s.isna()].astype(int)
    return s


def load_claude_matrix(task: str, repo: Path):
    """Load Claude p0 cells, return long-format DataFrame."""
    p0 = repo / "outputs/v2_analysis" / f"{task}__claude__p0.parquet"
    if not p0.exists():
        return None
    df = pd.read_parquet(p0)
    # Numeric score: 0/0.5/1 when applicable, NaN otherwise
    df["score_num"] = df["score"].where(df["applicable"], np.nan).astype(float)
    return df


def build_dp_aspect_matrix(df: pd.DataFrame):
    """Pivot to wide: index=dp_id, columns=aspect_id, values=score_num (mean if multiple)."""
    return df.groupby(["dp_id", "aspect_id"])["score_num"].mean().unstack(level="aspect_id")


def find_best_covered_dps(mat: pd.DataFrame, k: int = 40):
    """Return the k dps with most non-NaN aspect cells."""
    cov = mat.notna().sum(axis=1)
    return cov.sort_values(ascending=False).head(k).index.tolist()


def aspect_aspect_clustering(mat: pd.DataFrame, threshold: float = 0.2, min_overlap: int = 10):
    """Compute pairwise |Spearman| between aspect columns, cluster.

    Returns: DataFrame (aspect_id, cluster_id, n_in_cluster, mean_score, n_dps_seen)
    """
    aspects = mat.columns.tolist()
    n = len(aspects)
    # Pairwise distance matrix (1 - |corr|)
    dist = np.ones((n, n))
    np.fill_diagonal(dist, 0.0)
    skipped_pairs = 0
    for i in range(n):
        col_i = mat[aspects[i]]
        for j in range(i + 1, n):
            col_j = mat[aspects[j]]
            mask = col_i.notna() & col_j.notna()
            if mask.sum() < min_overlap:
                skipped_pairs += 1
                continue
            v1, v2 = col_i[mask].values, col_j[mask].values
            # If either column has no variance on this overlap, skip
            if np.std(v1) < 1e-9 or np.std(v2) < 1e-9:
                skipped_pairs += 1
                continue
            rho, _ = spearmanr(v1, v2)
            if not np.isnan(rho):
                d = 1.0 - abs(rho)
                dist[i, j] = dist[j, i] = d
    # Hierarchical clustering
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    cluster_ids = fcluster(Z, t=threshold, criterion="distance")
    out = pd.DataFrame({
        "aspect_id": aspects,
        "cluster_id": cluster_ids,
        "mean_score": [mat[a].mean() for a in aspects],
        "n_dps_seen": [mat[a].notna().sum() for a in aspects],
    })
    out["n_in_cluster"] = out.groupby("cluster_id")["aspect_id"].transform("count")
    return out, skipped_pairs


def per_aspect_discriminativeness(mat: pd.DataFrame, labels: pd.Series, min_overlap: int = 10):
    """For each aspect: mean(score|y=1) - mean(score|y=0), t-test p-value, Bonferroni-q."""
    rows = []
    aspects = mat.columns.tolist()
    # Align dps
    common = mat.index.intersection(labels.index)
    if len(common) == 0:
        return pd.DataFrame(columns=["aspect_id", "delta", "p_value", "n0", "n1"])
    mat_a = mat.loc[common]
    y = labels.loc[common]
    for a in aspects:
        col = mat_a[a]
        mask = col.notna()
        if mask.sum() < min_overlap: continue
        y_a = y[mask]
        s_a = col[mask]
        s0 = s_a[y_a == 0]
        s1 = s_a[y_a == 1]
        n0, n1 = len(s0), len(s1)
        if n0 < 3 or n1 < 3: continue
        delta = s1.mean() - s0.mean()
        try:
            _, p = ttest_ind(s1, s0, equal_var=False)
        except: p = np.nan
        rows.append({"aspect_id": a, "delta": delta, "p_value": p, "n0": n0, "n1": n1})
    res = pd.DataFrame(rows)
    if len(res) > 0:
        res["bonferroni_p"] = (res["p_value"] * len(res)).clip(upper=1.0)
        res = res.sort_values("p_value", na_position="last").reset_index(drop=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/v2_analysis")
    ap.add_argument("--top-k-dps", type=int, default=40)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    out_dir = repo / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        print(f"\n========== {task} ==========")
        df = load_claude_matrix(task, repo)
        if df is None:
            print(f"  no parquet; skip")
            continue
        wide = build_dp_aspect_matrix(df)
        print(f"  full pivot: {wide.shape[0]} dps × {wide.shape[1]} aspects")
        # Drop aspects scored on <5 dps (unreliable)
        keep_aspects = wide.notna().sum(axis=0) >= 5
        wide = wide.loc[:, keep_aspects]
        # Top-K best-covered dps
        top_dps = find_best_covered_dps(wide, k=args.top_k_dps)
        sub = wide.loc[top_dps]
        # Drop aspects that are all-NaN on this subset
        sub = sub.loc[:, sub.notna().any(axis=0)]
        print(f"  best-{args.top_k_dps}-dps subset: {sub.shape}")
        print(f"  coverage on subset: median {sub.notna().sum(axis=1).median():.0f}, "
              f"min {sub.notna().sum(axis=1).min()}, max {sub.notna().sum(axis=1).max()} aspects/dp")

        # 1. Aspect-aspect clustering
        print(f"\n  --- Aspect-aspect clustering ---")
        cluster_df, n_skipped = aspect_aspect_clustering(sub, threshold=0.2, min_overlap=10)
        cluster_df.to_parquet(out_dir / f"{task}__aspect_clusters.parquet", index=False)
        n_clusters = cluster_df["cluster_id"].nunique()
        n_singletons = (cluster_df["n_in_cluster"] == 1).sum()
        big = cluster_df[cluster_df["n_in_cluster"] >= 3].sort_values("n_in_cluster", ascending=False)
        print(f"    {len(cluster_df)} aspects → {n_clusters} clusters ({n_singletons} singletons), "
              f"{n_skipped} pairs skipped (low overlap/variance)")
        if len(big) > 0:
            top_clusters = big.groupby("cluster_id").size().sort_values(ascending=False).head(5)
            print(f"    top-5 redundancy clusters (size): {top_clusters.tolist()}")
            for cid, _ in top_clusters.items():
                members = cluster_df[cluster_df["cluster_id"] == cid]["aspect_id"].tolist()
                print(f"      cluster {cid} ({len(members)}): {members[:8]}{'...' if len(members) > 8 else ''}")

        # 2. Per-aspect Discriminativeness — need labels
        print(f"\n  --- Per-aspect Discriminativeness ---")
        labels = load_labels(task, repo)
        if labels is None or len(labels) == 0:
            print(f"    SKIP: no datapoints.json or no judgement labels for {task}")
            continue
        sub_str = sub.copy()
        sub_str.index = sub_str.index.astype(str)
        disc = per_aspect_discriminativeness(sub_str, labels)
        disc.to_parquet(out_dir / f"{task}__discriminativeness.parquet", index=False)
        n_overlap = len(set(sub_str.index) & set(labels.index))
        if len(disc) == 0:
            print(f"    NO ANALYSIS: dp overlap with labels = {n_overlap}")
            continue
        n_sig = (disc["bonferroni_p"] < 0.05).sum()
        print(f"    dp/label overlap: {n_overlap}; {len(disc)} aspects testable; "
              f"{n_sig} significant after Bonferroni (q<0.05)")
        top = disc.head(8)
        print(f"    Top-8 by p-value:")
        for _, r in top.iterrows():
            print(f"      {r['aspect_id']:<8} Δ={r['delta']:+.2f}  p={r['p_value']:.4g}  "
                  f"bonf_q={r['bonferroni_p']:.3g}  (n0={r['n0']}, n1={r['n1']})")


if __name__ == "__main__":
    main()
