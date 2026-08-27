"""Execute all Llama-generated codes against all datapoints.

For each (metric_id, paraphrase_idx) code variant, run on each datapoint.
Saves scores to a parquet (or jsonl) for downstream analysis.

Output:
  runs/validity_full/<run>/codegen_exec_results.jsonl
    each row: {metric_id, paraphrase_idx, parent_aspect_id, datapoint_id, score, error}
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable


def run_one(code, text, timeout=4.0):
    runner = (
        f"{code}\n\n"
        "import json, sys\n"
        "text = json.loads(sys.stdin.read())\n"
        "try:\n"
        "    s = float(score(text))\n"
        "    if not (0.0 <= s <= 1.0): s = max(0.0, min(1.0, s))\n"
        "    print(json.dumps({'score': s}))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'error': str(e)[:120]}))\n"
    )
    try:
        proc = subprocess.run([PY, "-c", runner],
                              input=json.dumps(text), capture_output=True,
                              text=True, timeout=timeout)
        if proc.returncode != 0: return None, "nonzero exit"
        try:
            out = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:
            return None, "no JSON"
        if "error" in out: return None, out["error"]
        return out.get("score"), None
    except subprocess.TimeoutExpired:
        return None, "timeout"


def worker(args):
    entry, code, datapoints = args
    results = []
    for dp in datapoints:
        s, err = run_one(code, dp["text"])
        results.append({**entry, "datapoint_id": dp["datapoint_id"],
                        "score": s, "error": err})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="full_v1")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    base = Path(f"runs/validity_full/{args.run_name}")
    manifest = json.loads((base / "codegen_manifest.json").read_text())
    datapoints = json.loads((base / "datapoints.json").read_text())
    code_dir = base / "codegen_responses_llama"
    out_path = base / "codegen_exec_results.jsonl"

    # Gather (entry, code) for codes that exist + compile
    jobs = []
    n_missing = n_syntax = 0
    for entry in manifest:
        cp = code_dir / f"{entry['key']}.py"
        if not cp.exists():
            n_missing += 1; continue
        code = cp.read_text()
        try: compile(code, cp.name, "exec")
        except SyntaxError:
            n_syntax += 1; continue
        jobs.append((entry, code, datapoints))
    print(f"jobs: {len(jobs)} codes × {len(datapoints)} dp = "
          f"{len(jobs) * len(datapoints)} executions "
          f"(missing={n_missing}, syntax_err={n_syntax})")

    n_done = 0
    t0 = time.time()
    with out_path.open("w") as f:
        with mp.Pool(args.workers) as pool:
            for results in pool.imap_unordered(worker, jobs, chunksize=4):
                for r in results:
                    f.write(json.dumps(r) + "\n")
                n_done += 1
                if n_done % 100 == 0:
                    elapsed = time.time() - t0
                    rate = n_done / elapsed
                    eta = (len(jobs) - n_done) / max(rate, 1e-6)
                    print(f"  {n_done}/{len(jobs)} codes done "
                          f"({rate:.1f}/s, eta {eta/60:.1f} min)", flush=True)
    print(f"Done. {len(jobs)} codes × {len(datapoints)} dp = "
          f"{len(jobs) * len(datapoints)} rows in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
