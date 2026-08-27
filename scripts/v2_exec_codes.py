"""In-process execution of v2 code variants against all 5K datapoints.

Imports each <aspect>_v<N>_<style>.py from codegen_claude_r2/, runs
score(text) under a signal.alarm timeout, writes results to JSONL.

Output: runs/validity_full/full_v2/codegen_exec_results.jsonl
  {"aspect_id": "a0", "variant": "v0_keyword",
   "datapoint_id": "d00042", "score": 0.43, "error": null}
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import signal
import sys
import time
from pathlib import Path


class _Timeout(Exception): pass

def _handler(signum, frame):
    raise _Timeout()


def _load_score(path: Path):
    """Import a .py file and return its top-level score(text) -> float."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "score", None)
    if fn is None or not callable(fn):
        raise ValueError("no callable score() found")
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout-sec", type=int, default=2)
    ap.add_argument("--limit-aspects", type=int, default=0,
                    help="0 = all aspects")
    ap.add_argument("--task", default=None,
                    help="task name under runs/validity_full/v2/<task>/. "
                         "If unset, uses legacy runs/validity_full/full_v2/.")
    args = ap.parse_args()

    if args.task:
        v2 = Path(f"runs/validity_full/v2/{args.task}")
        code_dir = v2 / "codegen_claude"
    else:
        v2 = Path("runs/validity_full/full_v2")
        code_dir = v2 / "codegen_claude_r2"
    datapoints = json.loads((v2 / "datapoints.json").read_text())

    # Group code files by aspect_id
    files = sorted(code_dir.glob("a*.py"))
    pat = re.compile(r"^(a\d+)_(v\d+_[a-z]+)\.py$")
    by_aspect = {}
    for fp in files:
        m = pat.match(fp.name)
        if not m: continue
        aid, variant = m.group(1), m.group(2)
        by_aspect.setdefault(aid, []).append((variant, fp))

    aspect_ids = sorted(by_aspect.keys(), key=lambda s: int(s[1:]))
    if args.limit_aspects > 0:
        aspect_ids = aspect_ids[:args.limit_aspects]
    print(f"aspects: {len(aspect_ids)}  datapoints: {len(datapoints)}")
    print(f"total executions: ~{len(aspect_ids) * 3 * len(datapoints):,}")

    out_path = v2 / "codegen_exec_results.jsonl"
    out_path.unlink(missing_ok=True)
    signal.signal(signal.SIGALRM, _handler)

    n_ok = n_err = n_timeout = 0
    t0 = time.time()
    with out_path.open("a") as fh:
        for ai, aid in enumerate(aspect_ids):
            for variant, fp in by_aspect[aid]:
                try:
                    score_fn = _load_score(fp)
                except Exception as e:
                    print(f"  IMPORT FAIL {fp.name}: {e}", flush=True)
                    n_err += len(datapoints)
                    continue
                for dp in datapoints:
                    try:
                        signal.alarm(args.timeout_sec)
                        s = score_fn(dp["text"])
                        signal.alarm(0)
                        # Clamp to [0,1]
                        try:
                            s = float(s)
                            if s != s:  # NaN
                                raise ValueError("NaN")
                            s = max(0.0, min(1.0, s))
                        except Exception:
                            raise ValueError(f"non-numeric: {s!r}")
                        fh.write(json.dumps({
                            "aspect_id": aid, "variant": variant,
                            "datapoint_id": dp["datapoint_id"],
                            "score": s, "error": None,
                        }) + "\n")
                        n_ok += 1
                    except _Timeout:
                        signal.alarm(0)
                        fh.write(json.dumps({
                            "aspect_id": aid, "variant": variant,
                            "datapoint_id": dp["datapoint_id"],
                            "score": None, "error": "timeout",
                        }) + "\n")
                        n_timeout += 1
                    except Exception as e:
                        signal.alarm(0)
                        fh.write(json.dumps({
                            "aspect_id": aid, "variant": variant,
                            "datapoint_id": dp["datapoint_id"],
                            "score": None, "error": str(e)[:200],
                        }) + "\n")
                        n_err += 1
            if (ai + 1) % 20 == 0:
                elapsed = time.time() - t0
                rate = (ai + 1) / elapsed
                eta = (len(aspect_ids) - ai - 1) / rate
                print(f"  [{ai+1}/{len(aspect_ids)}] ok={n_ok} err={n_err} "
                      f"timeout={n_timeout}  rate={rate:.1f} aspects/s  "
                      f"eta={eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\n=== DONE ===")
    print(f"  ok={n_ok:,}  err={n_err:,}  timeout={n_timeout:,}")
    print(f"  elapsed: {elapsed:.1f}s ({n_ok/elapsed:.0f}/s)")
    print(f"  output: {out_path}")


if __name__ == "__main__":
    main()
