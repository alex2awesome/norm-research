"""Score every metric_implementer metric on the single-file reconstructions.

The bank's `applies(diff_text)` / `score(diff_text)` API takes a unified-diff
string. To make it work on our reconstructed single-file `file_text`, we
wrap each row's `file_text` as a synthetic all-added diff:

    diff --git a/<file_path> b/<file_path>
    --- /dev/null
    +++ b/<file_path>
    @@ -0,0 +1,<N> @@
    +<line 1>
    +<line 2>
    ...

This way the existing parsers (whatthepatch / parse_diff_added_by_file)
recognize every line as "added" inside one named file, and the metric bank
gets the file extension it needs for routing.

We process in parallel chunks. Output:
  outputs/v2_analysis/dense_4096tok_single_file_bank_scores.parquet

with columns:
  paper_id, judgement, split, file_language, n_files_in_diff,
  a*_score, a*_applied  (one pair per metric)
"""
from __future__ import annotations

import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
IN_PATH = REPO / "outputs/v2_analysis/dense_4096tok_single_file_reconstructed.parquet"
OUT_PATH = REPO / "outputs/v2_analysis/dense_4096tok_single_file_bank_scores.parquet"

os.environ["PATH"] = (f"/lfs/skampere3/0/alexspan/miniconda3/bin:"
                      + os.environ.get("PATH", ""))
sys.path.insert(0, str(REPO))


def to_synthetic_diff(file_path: str, file_text: str) -> str:
    """Wrap a single-file reconstruction as an all-added unified diff."""
    if not file_text:
        return ""
    lines = file_text.split("\n")
    body = "\n".join("+" + ln for ln in lines)
    n = len(lines)
    return (
        f"diff --git a/{file_path} b/{file_path}\n"
        f"--- /dev/null\n"
        f"+++ b/{file_path}\n"
        f"@@ -0,0 +1,{n} @@\n"
        f"{body}\n"
    )


def _process_chunk(rows):
    from methods.existing_metrics_runner.coded.metrics import load_all
    metrics = load_all()
    out = []
    for r in rows:
        diff = to_synthetic_diff(r["file_path"], r["file_text"]) \
            if r["file_path"] and r["file_text"] else ""
        rec = {
            "paper_id": r["paper_id"],
            "judgement": r["judgement"],
            "split": r["split"],
            "file_language": r["file_language"],
            "n_files_in_diff": r["n_files_in_diff"],
        }
        for m in metrics:
            applied = 0
            score = float("nan")
            if diff:
                try:
                    a = m.applies(diff)
                except Exception:
                    a = False
                applied = int(bool(a))
                if applied:
                    try:
                        s = m.score(diff)
                        if s is not None and not (isinstance(s, float)
                                                  and math.isnan(s)):
                            score = float(s)
                    except Exception:
                        pass
            rec[f"{m.ASPECT_ID}_score"] = score
            rec[f"{m.ASPECT_ID}_applied"] = applied
        out.append(rec)
    return out


def main():
    print(f"loading {IN_PATH}")
    df = pd.read_parquet(IN_PATH)
    print(f"rows: {len(df):,}; with file_text: "
          f"{df['file_text'].notna().sum():,}")
    # Drop rows we can't score at all (no file_text) — record them as
    # all-applied=0, all-NaN.
    rows = df[[
        "paper_id", "judgement", "split", "file_language", "n_files_in_diff",
        "file_path", "file_text",
    ]].to_dict(orient="records")

    # Sanity-load metrics in parent.
    from methods.existing_metrics_runner.coded.metrics import load_all
    metrics = load_all()
    print(f"loaded {len(metrics)} metrics in parent")

    n_workers = int(os.environ.get("BANK_WORKERS", "8"))
    chunk_size = max(1, len(rows) // (n_workers * 16))
    chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]
    print(f"workers={n_workers}, chunks={len(chunks)}, "
          f"chunk_size={chunk_size}")

    results: List[Dict] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_process_chunk, c) for c in chunks]
        for k, fut in enumerate(as_completed(futs)):
            results.extend(fut.result())
            elapsed = time.time() - t0
            done = len(results)
            eta_min = (elapsed / max(done, 1)) * (len(rows) - done) / 60
            if (k + 1) % 4 == 0 or (k + 1) == len(chunks):
                print(f"  chunks {k+1}/{len(chunks)} — rows {done:,}/"
                      f"{len(rows):,} elapsed={elapsed:.0f}s "
                      f"ETA={eta_min:.1f}min",
                      flush=True)

    out = pd.DataFrame(results)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH)
    print(f"\nwrote {OUT_PATH}, shape={out.shape}")

    # === Per-metric coverage report ===
    print("\n=== Per-metric coverage (top 30 by coverage) ===")
    n = len(out)
    score_cols = sorted(c for c in out.columns if c.endswith("_score"))
    cov_rows = []
    for c in score_cols:
        aid = c[:-6]
        applied = out[f"{aid}_applied"]
        scored = out[c].notna()
        cov_rows.append({
            "metric": aid,
            "applied_pct": float(applied.mean()),
            "scored_pct": float(scored.mean()),
            "scored_given_applied": (float(scored[applied == 1].mean())
                                     if applied.sum() else float("nan")),
            "mean": float(out[c].mean()),
            "std": float(out[c].std()),
        })
    cov = pd.DataFrame(cov_rows).sort_values("scored_pct", ascending=False)
    print(cov.head(30).to_string(index=False))
    print("\n=== Bottom 15 by scored_pct ===")
    print(cov.tail(15).to_string(index=False))
    cov.to_parquet(REPO / "outputs/v2_analysis/"
                          "dense_4096tok_single_file_bank_coverage.parquet")


if __name__ == "__main__":
    main()
