"""Parameterized bank scorer for four-platform competition pools.

Mirrors scripts/comp_unified_claude_bank_score.py exactly: same metric loader
(methods.existing_metrics_runner.coded.metrics.load_all), same synthetic-diff
construction, same applied/score semantics (applied=int(m.applies(diff));
score = m.score(diff) when applied else NaN; any exception -> applied=0
and/or score=NaN). Scores ONLY the candidate code.

Input parquet must have columns:
    pair_id, candidate_text, candidate_lang (or language), platform (optional)
plus any other columns to carry forward (claude_label, label_status, ...).

Usage:
    python scripts/comp_fourplatform_bank_score.py \
        --input  outputs/v2_analysis/comp_fourplatform_cells/cf_bank_input.parquet \
        --output outputs/v2_analysis/comp_fourplatform_cells/cf_bank_scores.parquet \
        [--workers 6] [--repo /path/to/norm-research]

CPU only. No GPU. Missing external binaries (clang-tidy, etc.) are recorded
as not-applied via the same try/except as the upstream scorer (so we never
crash on environment gaps).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


LANG_TO_EXT = {
    "python": "py",
    "py": "py",
    "cpp": "cpp",
    "c++": "cpp",
    "cxx": "cpp",
    "c": "c",
    "java": "java",
    "javascript": "js",
    "js": "js",
    "typescript": "ts",
    "ts": "ts",
    "go": "go",
    "ruby": "rb",
    "rb": "rb",
    "rust": "rs",
    "rs": "rs",
    "csharp": "cs",
    "c#": "cs",
    "cs": "cs",
    "php": "php",
    "kotlin": "kt",
    "kt": "kt",
    "swift": "swift",
    "scala": "scala",
    "unknown": "txt",
}


def to_synthetic_diff(file_path: str, file_text: str) -> str:
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


# Worker-side imports happen inside _process_chunk after sys.path is set.
def _process_chunk(rows, repo_path):
    sys.path.insert(0, repo_path)
    from methods.existing_metrics_runner.coded.metrics import load_all
    metrics = load_all()
    out = []
    for r in rows:
        lang = str(r.get("language") or "unknown").lower()
        ext = LANG_TO_EXT.get(lang, "txt")
        fp = f"candidate.{ext}"
        diff = to_synthetic_diff(fp, r.get("candidate_text") or "")
        rec = {k: r[k] for k in r if not k.startswith("__")}
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
                        if s is not None and not (
                            isinstance(s, float) and math.isnan(s)
                        ):
                            score = float(s)
                    except Exception:
                        pass
            rec[f"{m.ASPECT_ID}_score"] = score
            rec[f"{m.ASPECT_ID}_applied"] = applied
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=str)
    ap.add_argument("--output", required=True, type=str)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("BANK_WORKERS", "6")))
    ap.add_argument("--repo", type=str,
                    default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args()

    repo = str(Path(args.repo).resolve())
    sys.path.insert(0, repo)

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)
    print(f"Loaded {in_path}  shape={df.shape}  cols={list(df.columns)}",
          flush=True)

    # Normalize language column name
    if "language" not in df.columns and "candidate_lang" in df.columns:
        df = df.rename(columns={"candidate_lang": "language"})
    if "language" not in df.columns:
        df["language"] = "unknown"
    df["language"] = df["language"].fillna("unknown")

    if "platform" in df.columns:
        print(f"  platform dist: {df['platform'].value_counts().to_dict()}",
              flush=True)
    print(f"  language dist: {df['language'].value_counts().to_dict()}",
          flush=True)

    rows = df.to_dict(orient="records")

    # Parent-side preflight on metric loader.
    from methods.existing_metrics_runner.coded.metrics import load_all
    metrics = load_all()
    print(f"loaded {len(metrics)} metrics in parent", flush=True)

    n_workers = max(1, args.workers)
    chunk_size = max(1, len(rows) // max(1, n_workers * 8))
    chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]
    print(f"workers={n_workers}, chunks={len(chunks)}, "
          f"chunk_size={chunk_size}", flush=True)

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_process_chunk, c, repo) for c in chunks]
        for k, fut in enumerate(as_completed(futs)):
            results.extend(fut.result())
            elapsed = time.time() - t0
            done = len(results)
            eta_min = (elapsed / max(done, 1)) * (len(rows) - done) / 60
            print(f"  chunks {k+1}/{len(chunks)} — rows {done}/{len(rows)} "
                  f"elapsed={elapsed:.0f}s ETA={eta_min:.1f}min", flush=True)

    out = pd.DataFrame(results)
    # Drop heavy text payload columns to keep parquet small (kept on input).
    drop_cols = [c for c in ("candidate_text", "editorial_text") if c in out.columns]
    out = out.drop(columns=drop_cols)
    out.to_parquet(out_path)
    runtime = time.time() - t0
    print(f"wrote {out_path}, shape={out.shape}, runtime={runtime:.1f}s",
          flush=True)

    # Per-metric coverage summary.
    score_cols = sorted(c for c in out.columns if c.endswith("_score"))
    cov_rows = []
    for c in score_cols:
        aid = c[:-6]
        applied_col = f"{aid}_applied"
        if applied_col not in out.columns:
            continue
        applied = out[applied_col]
        scored = out[c].notna()
        cov_rows.append({
            "metric": aid,
            "applied_pct": float(applied.mean()),
            "scored_pct": float(scored.mean()),
            "mean": float(out[c].mean()),
            "std": float(out[c].std()),
        })
    cov = pd.DataFrame(cov_rows).sort_values("scored_pct", ascending=False)
    cov_path = out_path.with_name(out_path.stem + "_coverage.parquet")
    cov.to_parquet(cov_path)
    mean_applied = cov["applied_pct"].mean() if len(cov) else float("nan")
    n_gt20 = int((cov["applied_pct"] > 0.20).sum()) if len(cov) else 0
    print(f"coverage: n_metrics={len(cov)} mean_applied={mean_applied:.3f} "
          f"n>20%_applied={n_gt20}", flush=True)
    print("\n=== top 15 by applied_pct ===")
    print(cov.sort_values("applied_pct", ascending=False).head(15)
          .to_string(index=False))
    print("\n=== bottom 10 by applied_pct ===")
    print(cov.sort_values("applied_pct", ascending=False).tail(10)
          .to_string(index=False))


if __name__ == "__main__":
    main()
