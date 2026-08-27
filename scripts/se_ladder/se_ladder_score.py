"""SE ladder step 1: score the static metric bank (2026-06-11).

Bank = methods.metric_implementer.metrics.load_all() (172 coded metrics,
incl. python-specific a460-a477) scored on the answer's CODE blocks as a
synthetic single-file diff (per-row language extension), PLUS the six
presentation-register metrics g1-g6 computed on the FULL answer body
(markdown / de-HTMLed text). 2026-06-12: g1-g6 were promoted from inline
functions here into the bank proper (metrics/g{1..6}_*.py, ARTIFACT="body");
this script now routes any bank metric with ARTIFACT=="body" to the body
and the rest to the synthetic diff. Outputs are bit-identical to the
2026-06-11 inline version (verified on 500 bodies x 6 metrics).

Not-applicable handling: every metric emits {id}_score (NaN when not
applicable) and {id}_applied (0/1). Rows without code get applied=0 for all
bank metrics — honest non-coverage, never faked.

Resumable: input is sharded (SHARD_SIZE rows); each finished shard is an
append-only parquet under outputs/v2_analysis/se_ladder/shards/{slice}/.

Usage: python se_ladder_score.py <slice> [n_workers]
Run on sk3 from /lfs/skampere3/0/alexspan/norm-research.
"""
from __future__ import annotations

import math
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
sys.path.insert(0, str(REPO))
OUT_DIR = REPO / "outputs/v2_analysis/se_ladder"

SHARD_SIZE = 500
MAX_ARTIFACT_CHARS = 20_000
PER_CALL_TIMEOUT_S = 5


def to_synthetic_diff(file_path: str, file_text: str) -> str:
    if not file_text:
        return ""
    lines = file_text.split("\n")
    body = "\n".join("+" + ln for ln in lines)
    return (f"diff --git a/{file_path} b/{file_path}\n"
            f"--- /dev/null\n+++ b/{file_path}\n"
            f"@@ -0,0 +1,{len(lines)} @@\n{body}\n")


# ----------------------------------------------------------- bank scoring
# (g1-g6 presentation metrics now live in the bank with ARTIFACT="body";
#  they are scored on the full answer body via the routing below.)
class _CallTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _CallTimeout()


def _score_chunk(rows):
    sys.path.insert(0, str(REPO))
    from methods.metric_implementer.metrics import load_all
    all_metrics = load_all()
    diff_metrics = [m for m in all_metrics
                    if getattr(m, "ARTIFACT", "diff") != "body"]
    body_metrics = [m for m in all_metrics
                    if getattr(m, "ARTIFACT", "diff") == "body"]
    signal.signal(signal.SIGALRM, _alarm_handler)

    def _guarded(fn, arg):
        signal.alarm(PER_CALL_TIMEOUT_S)
        try:
            return fn(arg)
        finally:
            signal.alarm(0)

    def _apply(m, artifact, rec):
        applied, score = 0, float("nan")
        if artifact:
            try:
                a = _guarded(m.applies, artifact)
            except (Exception, _CallTimeout):
                a = False
            applied = int(bool(a))
            if applied:
                try:
                    s = _guarded(m.score, artifact)
                    if s is not None and not (
                            isinstance(s, float) and math.isnan(s)):
                        score = float(s)
                except (Exception, _CallTimeout):
                    pass
        rec[f"{m.ASPECT_ID}_score"] = score
        rec[f"{m.ASPECT_ID}_applied"] = applied

    out = []
    for r in rows:
        code = (r["code"] or "")[:MAX_ARTIFACT_CHARS]
        diff = to_synthetic_diff(f"candidate.{r['ext']}", code)
        body = r["body"] if isinstance(r["body"], str) else ""
        rec = {"row_id": r["row_id"]}
        for m in diff_metrics:
            _apply(m, diff, rec)
        for m in body_metrics:
            _apply(m, body, rec)
        out.append(rec)
    return out


def main():
    slice_name = sys.argv[1]
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    inp = OUT_DIR / f"{slice_name}_input.parquet"
    shard_dir = OUT_DIR / "shards" / slice_name
    shard_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(inp,
                         columns=["row_id", "ext", "code", "body"])
    rows = df.to_dict(orient="records")
    shards = [(i, rows[i * SHARD_SIZE:(i + 1) * SHARD_SIZE])
              for i in range((len(rows) + SHARD_SIZE - 1) // SHARD_SIZE)]
    todo = [(i, c) for i, c in shards
            if not (shard_dir / f"shard_{i:05d}.parquet").exists()]
    print(f"[{slice_name}] {len(rows)} rows, {len(shards)} shards, "
          f"{len(todo)} todo, {n_workers} workers", flush=True)

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_score_chunk, c): i for i, c in todo}
        for f in as_completed(futs):
            i = futs[f]
            recs = f.result()
            pd.DataFrame(recs).to_parquet(
                shard_dir / f"shard_{i:05d}.parquet", index=False)
            done += 1
            if done % 5 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"[{slice_name}] shard {done}/{len(todo)} "
                      f"elapsed={el:.0f}s rate={done/max(el,1)*3600:.0f} "
                      f"shards/h", flush=True)
    print(f"[{slice_name}] DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
