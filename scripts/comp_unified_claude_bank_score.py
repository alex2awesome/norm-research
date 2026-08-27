"""Score the 121-metric coded bank on 1,004 Claude-labeled competition pairs.

Inputs (combined):
  1) 5K fresh shards (704 ok): outputs/v2_analysis/comp_unified_claude5k_shards/
        results/shard_*.jsonl -> pair_id, claude_label, status
        shards/shard_*.jsonl  -> pair_id, source, candidate_text, editorial_text,
                                  language, problem_id (+ cosine, NOT USED)
  2) Qwen-relabel side, ~300 Claude labels (scp'd to /tmp):
        /tmp/qwen_validation_claude_labels.parquet
        /tmp/validation_sample_300.parquet

We score ONLY the candidate code (this is the predicate the bank operates on
for the dense reward setup). The bank produces 121 (score, applied) pairs.

NO cosine is carried forward — it would leak the Claude label proxy.

Output:
  outputs/v2_analysis/comp_unified_claude_bank_scores.parquet
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

REPO = Path("/Users/spangher/Projects/stanford-research/norm-research")
sys.path.insert(0, str(REPO))

OUT_PATH = REPO / "outputs/v2_analysis/comp_unified_claude_bank_scores.parquet"

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


def load_5k() -> pd.DataFrame:
    base = REPO / "outputs/v2_analysis/comp_unified_claude5k_shards"
    res_rows = {}
    for p in sorted(glob.glob(str(base / "results" / "shard_*.jsonl"))):
        for ln in open(p):
            r = json.loads(ln)
            if r.get("claude_label") in (0, 1):
                res_rows[r["pair_id"]] = int(r["claude_label"])
    shard_rows = {}
    for p in sorted(glob.glob(str(base / "shards" / "shard_*.jsonl"))):
        for ln in open(p):
            r = json.loads(ln)
            shard_rows[r["pair_id"]] = r
    out = []
    for pid, lbl in res_rows.items():
        s = shard_rows.get(pid)
        if not s:
            continue
        out.append({
            "pair_id": pid,
            "claude_label": lbl,
            "source": s.get("source"),
            "language": s.get("language") or "unknown",
            "candidate_text": s.get("candidate_text") or "",
            "split": "5k",
        })
    return pd.DataFrame(out)


def load_300() -> pd.DataFrame:
    labels = pd.read_parquet("/tmp/qwen_validation_claude_labels.parquet")
    samp = pd.read_parquet("/tmp/validation_sample_300.parquet")
    m = samp.merge(labels[["pair_id", "claude_label"]], on="pair_id",
                   how="inner")
    m = m[m["claude_label"].isin([0, 1])].copy()
    m["claude_label"] = m["claude_label"].astype(int)
    out = pd.DataFrame({
        "pair_id": m["pair_id"],
        "claude_label": m["claude_label"],
        "source": m["platform"],
        "language": m["language"].fillna("unknown"),
        "candidate_text": m["candidate_code"].fillna(""),
        "split": "300",
    })
    return out


def _process_chunk(rows):
    sys.path.insert(0, str(REPO))
    from methods.existing_metrics_runner.coded.metrics import load_all
    metrics = load_all()
    out = []
    for r in rows:
        ext = LANG_TO_EXT.get(str(r["language"]).lower(), "txt")
        fp = f"candidate.{ext}"
        diff = to_synthetic_diff(fp, r["candidate_text"])
        rec = {
            "pair_id": r["pair_id"],
            "claude_label": r["claude_label"],
            "source": r["source"],
            "language": r["language"],
            "split": r["split"],
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
    print("Loading 5K side ...", flush=True)
    df5 = load_5k()
    print(f"  5K rows: {len(df5)}  label dist: "
          f"{df5['claude_label'].value_counts().to_dict()}")
    print("Loading 300 side ...", flush=True)
    df3 = load_300()
    print(f"  300 rows: {len(df3)}  label dist: "
          f"{df3['claude_label'].value_counts().to_dict()}")
    # Disambiguate pair_ids (5K uses 'clauded5k_*'; 300 uses 'cc_*'/'luogu_*')
    overlap = set(df5["pair_id"]).intersection(df3["pair_id"])
    if overlap:
        print(f"WARNING: pair_id overlap n={len(overlap)} — prefixing 300 side")
        df3["pair_id"] = "v300_" + df3["pair_id"]
    df = pd.concat([df5, df3], ignore_index=True)
    df = df[df["claude_label"].notna()].copy()
    print(f"Total: {len(df)} rows. label dist: "
          f"{df['claude_label'].value_counts().to_dict()}")
    print(f"language dist: {df['language'].value_counts().to_dict()}")
    print(f"source dist: {df['source'].value_counts().to_dict()}")

    rows = df.to_dict(orient="records")
    from methods.existing_metrics_runner.coded.metrics import load_all
    metrics = load_all()
    print(f"loaded {len(metrics)} metrics in parent", flush=True)

    n_workers = int(os.environ.get("BANK_WORKERS", "6"))
    chunk_size = max(1, len(rows) // (n_workers * 8))
    chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]
    print(f"workers={n_workers}, chunks={len(chunks)}, "
          f"chunk_size={chunk_size}", flush=True)

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_process_chunk, c) for c in chunks]
        for k, fut in enumerate(as_completed(futs)):
            results.extend(fut.result())
            elapsed = time.time() - t0
            done = len(results)
            eta_min = (elapsed / max(done, 1)) * (len(rows) - done) / 60
            print(f"  chunks {k+1}/{len(chunks)} — rows {done}/{len(rows)} "
                  f"elapsed={elapsed:.0f}s ETA={eta_min:.1f}min", flush=True)

    out = pd.DataFrame(results)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH)
    print(f"wrote {OUT_PATH}, shape={out.shape}", flush=True)

    # Per-metric coverage
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
            "mean": float(out[c].mean()),
            "std": float(out[c].std()),
        })
    cov = pd.DataFrame(cov_rows).sort_values("scored_pct", ascending=False)
    cov.to_parquet(REPO / "outputs/v2_analysis"
                          "/comp_unified_claude_bank_coverage.parquet")
    print("\n=== coverage (top 15 by scored_pct) ===")
    print(cov.head(15).to_string(index=False))
    print("\n=== coverage (bottom 10 by scored_pct) ===")
    print(cov.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
