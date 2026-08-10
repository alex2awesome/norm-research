"""Score ONLY the new algorithmic-fingerprint metrics (a518-a522) on the
1,004 Claude pairs, append to comp_unified_claude_bank_scores_v2.parquet
and write v3.

Mirrors the v2 scoring pattern (comp_unified_claude_bank_score_new_metrics.py)
but for a518..a522 only.
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

IN_PATH = REPO / "outputs/v2_analysis/comp_unified_claude_bank_scores_v2.parquet"
OUT_PATH = REPO / "outputs/v2_analysis/comp_unified_claude_bank_scores_v3.parquet"

NEW_METRIC_IDS = ["a518", "a519", "a520", "a521", "a522"]

LANG_TO_EXT = {
    "python": "py", "py": "py",
    "cpp": "cpp", "c++": "cpp", "cxx": "cpp",
    "c": "c",
    "java": "java", "javascript": "js", "js": "js",
    "typescript": "ts", "ts": "ts",
    "go": "go", "ruby": "rb", "rb": "rb",
    "rust": "rs", "rs": "rs", "csharp": "cs", "c#": "cs", "cs": "cs",
    "php": "php", "kotlin": "kt", "kt": "kt", "swift": "swift",
    "scala": "scala", "unknown": "txt",
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


def load_pair_texts():
    out = {}
    base = REPO / "outputs/v2_analysis/comp_unified_claude5k_shards"
    for p in sorted(glob.glob(str(base / "shards" / "shard_*.jsonl"))):
        for ln in open(p):
            r = json.loads(ln)
            out[r["pair_id"]] = {
                "candidate_text": r.get("candidate_text") or "",
                "language": r.get("language") or "unknown",
                "source": r.get("source"),
            }
    try:
        samp = pd.read_parquet("/tmp/validation_sample_300.parquet")
        for _, r in samp.iterrows():
            pid = "v300_" + str(r["pair_id"])
            out[pid] = {
                "candidate_text": r.get("candidate_code") or "",
                "language": (r.get("language") or "unknown"),
                "source": r.get("platform"),
            }
            out[str(r["pair_id"])] = {
                "candidate_text": r.get("candidate_code") or "",
                "language": (r.get("language") or "unknown"),
                "source": r.get("platform"),
            }
    except Exception as e:
        print(f"warning: {e}")
    return out


def _process_chunk(args):
    rows, new_ids = args
    sys.path.insert(0, str(REPO))
    from methods.existing_metrics_runner.coded.metrics import load_all
    all_m = load_all()
    metrics = [m for m in all_m if m.ASPECT_ID in set(new_ids)]
    out = []
    for r in rows:
        ext = LANG_TO_EXT.get(str(r["language"]).lower(), "txt")
        fp = f"candidate.{ext}"
        diff = to_synthetic_diff(fp, r["candidate_text"])
        rec = {"pair_id": r["pair_id"]}
        for m in metrics:
            applied = 0
            score_v = float("nan")
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
                            score_v = float(s)
                    except Exception:
                        pass
            rec[f"{m.ASPECT_ID}_score"] = score_v
            rec[f"{m.ASPECT_ID}_applied"] = applied
        out.append(rec)
    return out


def main():
    print(f"loading existing bank scores: {IN_PATH}", flush=True)
    bank = pd.read_parquet(IN_PATH)
    print(f"  shape: {bank.shape}")

    print("loading pair candidate_text from shards...", flush=True)
    pair_texts = load_pair_texts()
    print(f"  {len(pair_texts)} pair_id -> text entries")

    rows = []
    miss = 0
    for _, r in bank.iterrows():
        pid = r["pair_id"]
        info = pair_texts.get(pid)
        if not info:
            miss += 1
            rows.append({
                "pair_id": pid,
                "language": r.get("language") or "unknown",
                "source": r.get("source"),
                "candidate_text": "",
            })
            continue
        rows.append({
            "pair_id": pid,
            "language": info.get("language") or "unknown",
            "source": info.get("source"),
            "candidate_text": info.get("candidate_text") or "",
        })
    print(f"  missing candidate_text: {miss}/{len(rows)}")

    n_workers = int(os.environ.get("BANK_WORKERS", "6"))
    chunk_size = max(1, len(rows) // (n_workers * 4))
    chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]
    print(f"workers={n_workers}, chunks={len(chunks)}, chunk={chunk_size}",
          flush=True)

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_process_chunk, (c, NEW_METRIC_IDS)) for c in chunks]
        for k, fut in enumerate(as_completed(futs)):
            results.extend(fut.result())
            elapsed = time.time() - t0
            done = len(results)
            eta_min = (elapsed / max(done, 1)) * (len(rows) - done) / 60
            print(f"  {k+1}/{len(chunks)} rows={done}/{len(rows)} "
                  f"elapsed={elapsed:.0f}s ETA={eta_min:.1f}min", flush=True)

    new_df = pd.DataFrame(results)
    merged = bank.merge(new_df, on="pair_id", how="left")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_PATH)
    print(f"wrote {OUT_PATH}, shape={merged.shape}", flush=True)

    for aid in NEW_METRIC_IDS:
        s = merged[f"{aid}_score"]
        a = merged[f"{aid}_applied"]
        print(f"  {aid}: applied={a.mean():.3f} scored={s.notna().mean():.3f} "
              f"mean={s.mean():.3f} std={s.std():.3f}")


if __name__ == "__main__":
    main()
