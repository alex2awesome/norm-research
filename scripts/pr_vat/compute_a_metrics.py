#!/usr/bin/env python3
"""Compute A metrics (155-metric bank) on all PR diffs → feature table.
Runs on sk3 where the diffs live. CPU-bound (~2-4h for 85K diffs).

Usage:
  cd /lfs/skampere3/0/alexspan/norm-research/datasets/code-review/pr_test_execution
  python3 scripts/pr_vat/compute_a_metrics.py
"""
import sys, os, glob, time, traceback
import pandas as pd

# Add methods to path
REPO_ROOT = os.path.expanduser("~/norm-research")
sys.path.insert(0, os.path.join(REPO_ROOT, "methods"))

from existing_metrics_runner.coded.metrics import load_all

def main():
    metrics = load_all()
    _ids = ", ".join(m.ASPECT_ID for m in metrics[:5])
    print(f"Loaded {len(metrics)} metrics: {_ids}...")

    diff_files = sorted(glob.glob("batch_runs/*/diffs/pr_*.diff"))
    print(f"Found {len(diff_files)} diff files")

    rows = []
    t0 = time.time()
    for i, diff_path in enumerate(diff_files):
        parts = diff_path.split("/")
        repo = parts[1]  # batch_runs/<repo>/diffs/pr_NNN.diff
        pr_num = parts[-1].replace("pr_", "").replace(".diff", "")

        try:
            text = open(diff_path, errors="replace").read()
        except Exception:
            continue

        if not text.strip():
            continue

        row = {"repo": repo, "pr_number": pr_num}
        for m in metrics:
            mid = m.ASPECT_ID
            try:
                applied = m.applies(text)
                score = m.score(text) if applied else None
                row[f"{mid}_score"] = score
                row[f"{mid}_applied"] = int(applied)
            except Exception:
                row[f"{mid}_score"] = None
                row[f"{mid}_applied"] = 0

        rows.append(row)

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(diff_files) - i - 1) / rate
            print(f"  {i+1}/{len(diff_files)} ({rate:.1f}/s, ETA {eta/60:.0f}min)")

    df = pd.DataFrame(rows)
    outpath = os.path.join(REPO_ROOT, "datasets/code-review/pr_test_execution/outputs/pr_a_metrics.parquet")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_parquet(outpath)
    print(f"\nDone: {len(df)} rows, {df.shape[1]} columns → {outpath}")

    # summary: how many metrics applied to ≥1 PR
    applied_cols = [c for c in df.columns if c.endswith("_applied")]
    coverage = {c.replace("_applied", ""): int(df[c].sum()) for c in applied_cols}
    print(f"\nMetric coverage (top 10 by n_applied):")
    for mid, n in sorted(coverage.items(), key=lambda x: -x[1])[:10]:
        print(f"  {mid}: {n} PRs ({n/len(df)*100:.1f}%)")

if __name__ == "__main__":
    main()
