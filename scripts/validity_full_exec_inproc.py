"""Fast in-process code executor (~100x faster than subprocess per-call).

Each worker loads ALL codes once into its own globals, then runs each on each
datapoint via direct function call. Signal-based timeout for runaway code.

Output: same format as subprocess version (codegen_exec_results.jsonl)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import signal
import sys
import time
from pathlib import Path


def _timeout_handler(signum, frame):
    raise TimeoutError("code timeout")


def compile_code(code):
    g = {"__name__": "__sandbox__"}
    try:
        exec(code, g)
        if "score" not in g or not callable(g["score"]):
            return None
        return g["score"]
    except Exception:
        return None


def worker(args):
    """Each worker processes a CHUNK of codes."""
    code_chunk, datapoints = args
    results = []
    signal.signal(signal.SIGALRM, _timeout_handler)
    for entry, code_text in code_chunk:
        fn = compile_code(code_text)
        if fn is None:
            for dp in datapoints:
                results.append({**entry, "datapoint_id": dp["datapoint_id"],
                                 "score": None, "error": "compile_or_no_score"})
            continue
        for dp in datapoints:
            signal.alarm(3)
            try:
                s = fn(dp["text"])
                signal.alarm(0)
                if isinstance(s, (int, float)):
                    s = float(s)
                    if not (0.0 <= s <= 1.0):
                        s = max(0.0, min(1.0, s))
                else:
                    s = None
                results.append({**entry, "datapoint_id": dp["datapoint_id"],
                                 "score": s, "error": None if s is not None else "non-numeric"})
            except Exception as e:
                signal.alarm(0)
                results.append({**entry, "datapoint_id": dp["datapoint_id"],
                                 "score": None, "error": str(e)[:120]})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="full_v1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=20,
                    help="Codes per worker chunk (each chunk runs all datapoints)")
    args = ap.parse_args()
    base = Path(f"runs/validity_full/{args.run_name}")
    manifest = json.loads((base / "codegen_manifest.json").read_text())
    datapoints = json.loads((base / "datapoints.json").read_text())
    code_dir = base / "codegen_responses_llama"
    out_path = base / "codegen_exec_results.jsonl"

    # Build (entry, code_text) pairs
    pairs = []
    n_missing = 0
    for entry in manifest:
        cp = code_dir / f"{entry['key']}.py"
        if not cp.exists():
            n_missing += 1; continue
        pairs.append((entry, cp.read_text()))
    print(f"loaded {len(pairs)} codes ({n_missing} missing), "
          f"{len(datapoints)} datapoints")

    # Chunk
    chunks = [pairs[i:i + args.chunk_size]
              for i in range(0, len(pairs), args.chunk_size)]
    print(f"split into {len(chunks)} chunks of ≤{args.chunk_size}")

    n_done = 0
    t0 = time.time()
    with out_path.open("w") as f:
        with mp.Pool(args.workers) as pool:
            for results in pool.imap_unordered(
                    worker, [(c, datapoints) for c in chunks], chunksize=1):
                for r in results:
                    f.write(json.dumps(r) + "\n")
                n_done += len(results) // len(datapoints)
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1e-6)
                eta = (len(pairs) - n_done) / max(rate, 1e-6)
                print(f"  {n_done}/{len(pairs)} codes done "
                      f"({rate:.1f}/s, eta {eta/60:.1f} min)", flush=True)
    print(f"DONE in {(time.time()-t0)/60:.1f} min — "
          f"{len(pairs) * len(datapoints)} total executions")


if __name__ == "__main__":
    main()
