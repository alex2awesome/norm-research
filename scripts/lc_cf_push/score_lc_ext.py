#!/usr/bin/env python3
"""Bank + paradigm scoring for the LC extension candidates.

EXACT replication of outputs/v2_analysis/lc_organic_relabel/scoring_tmp/score_organic.py
(same loader, same synthetic candidate.py diff — the original LC cell scored ALL
candidates as candidate.py regardless of language, so the extension must too),
but reading shards from outputs/v2_analysis/lc_cf_push/lc_ext/shards/.

Outputs: outputs/v2_analysis/lc_cf_push/lc_ext/{bank_scores,paradigm_scores}.parquet
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
OUT_DIR = ROOT / "outputs/v2_analysis/lc_cf_push/lc_ext"
SHARDS_DIR = OUT_DIR / "shards"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "outputs/v2_analysis/paradigm_metrics/src"))


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


def _bank_chunk(rows, repo_path):
    sys.path.insert(0, repo_path)
    from methods.existing_metrics_runner.coded.metrics import load_all
    metrics = load_all()
    out = []
    for r in rows:
        cid = r["candidate_id"]
        code = r.get("candidate_text") or ""
        diff = to_synthetic_diff("candidate.py", code)
        rec = {"candidate_id": cid}
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
                        if s is not None and not (isinstance(s, float) and math.isnan(s)):
                            score = float(s)
                    except Exception:
                        pass
            rec[f"{m.ASPECT_ID}_score"] = score
            rec[f"{m.ASPECT_ID}_applied"] = applied
        out.append(rec)
    return out


def _paradigm_chunk(rows, repo_path):
    sys.path.insert(0, repo_path)
    sys.path.insert(0, str(Path(repo_path) / "outputs/v2_analysis/paradigm_metrics/src"))
    from paradigm_metrics import score_all  # type: ignore
    out = []
    for r in rows:
        cid = r["candidate_id"]
        code = r.get("candidate_text") or ""
        try:
            sc = score_all(code, "python")
        except Exception:
            sc = {}
        rec = {"candidate_id": cid}
        for aid, (val, app) in sc.items():
            rec[f"{aid}_score"] = float(val) if (val is not None and not (isinstance(val, float) and math.isnan(val))) else float("nan")
            rec[f"{aid}_applied"] = int(bool(app))
        out.append(rec)
    return out


def load_unique_candidates():
    seen = set()
    rows = []
    for f in sorted(SHARDS_DIR.glob("shard_lcx_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                cid = d["candidate_id"]
                if cid in seen:
                    continue
                seen.add(cid)
                rows.append({
                    "candidate_id": cid,
                    "candidate_text": d.get("candidate_text") or "",
                })
    return rows


def run_parallel(rows, worker_fn, n_workers, label):
    chunk_size = max(1, len(rows) // max(1, n_workers * 8))
    chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]
    print(f"[{label}] workers={n_workers}, chunks={len(chunks)}", flush=True)
    results = []
    t0 = time.time()
    repo = str(ROOT)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(worker_fn, c, repo) for c in chunks]
        for k, fut in enumerate(as_completed(futs)):
            results.extend(fut.result())
            print(f"  [{label}] chunk {k+1}/{len(chunks)} rows {len(results)}/{len(rows)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return results


def main():
    n_workers = int(os.environ.get("BANK_WORKERS", "6"))
    cands = load_unique_candidates()
    print(f"unique candidates: {len(cands)}", flush=True)
    bank_df = pd.DataFrame(run_parallel(cands, _bank_chunk, n_workers, "bank"))
    bank_df.to_parquet(OUT_DIR / "bank_scores.parquet", index=False)
    print(f"wrote bank_scores {bank_df.shape}", flush=True)
    par_df = pd.DataFrame(run_parallel(cands, _paradigm_chunk, n_workers, "paradigm"))
    par_df.to_parquet(OUT_DIR / "paradigm_scores.parquet", index=False)
    print(f"wrote paradigm_scores {par_df.shape}", flush=True)


if __name__ == "__main__":
    main()
