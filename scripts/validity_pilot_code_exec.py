"""Execute each generated score function against each datapoint.

For every (metric, level, model, trial) code response, runs it via subprocess
with timeout against every datapoint in the run. Records:
  - score (float in [0,1]) or None on failure
  - error message (if any)
  - execution time

Outputs:
  runs/validity_pilot/<run>/codegen/exec_results.jsonl
  runs/validity_pilot/<run>/codegen/exec_summary.json (per-code success rate)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run_one(code: str, text: str, timeout: float = 5.0):
    runner = (
        f"{code}\n\n"
        "import json, sys\n"
        "text = json.loads(sys.stdin.read())\n"
        "try:\n"
        "    s = float(score(text))\n"
        "    if not (0.0 <= s <= 1.0):\n"
        "        s = max(0.0, min(1.0, s))\n"
        "    print(json.dumps({'score': s}))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'error': str(e)[:200]}))\n"
    )
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner],
            input=json.dumps(text),
            capture_output=True, text=True, timeout=timeout,
        )
        dt = time.time() - t0
        if proc.returncode != 0:
            return None, f"nonzero exit: {proc.stderr[:200]}", dt
        try:
            out = json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            return None, f"no JSON on stdout: {proc.stdout[:200]}", dt
        if "error" in out:
            return None, out["error"], dt
        return out.get("score"), None, dt
    except subprocess.TimeoutExpired:
        return None, "timeout", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="smoke")
    args = ap.parse_args()

    base = Path(f"runs/validity_pilot/{args.run_name}")
    manifest = json.loads((base / "codegen" / "manifest.json").read_text())
    datapoints = json.loads((base / "datapoints.json").read_text())

    out_path = base / "codegen" / "exec_results.jsonl"
    summary = {}
    n_runs = len(manifest) * len(datapoints)
    print(f"Running {len(manifest)} code variants × {len(datapoints)} datapoints "
          f"= {n_runs} executions")
    n_done = 0
    with out_path.open("w") as f:
        for entry in manifest:
            code_path = base / "codegen" / "responses" / f"{entry['key']}.py"
            if not code_path.exists():
                print(f"  MISSING {entry['key']}")
                continue
            code = code_path.read_text()
            # quick syntax check
            try:
                compile(code, code_path.name, "exec")
            except SyntaxError as e:
                print(f"  SYNTAX ERROR {entry['key']}: {e}")
                summary[entry["key"]] = {"status": "syntax_error", "n_ok": 0,
                                          "n_total": len(datapoints)}
                continue
            n_ok = 0
            for dp in datapoints:
                s, err, dt = run_one(code, dp["text"])
                rec = {**entry, "datapoint_id": dp["datapoint_id"],
                       "score": s, "error": err, "exec_ms": int(dt * 1000)}
                f.write(json.dumps(rec) + "\n")
                if s is not None: n_ok += 1
                n_done += 1
                if n_done % 50 == 0:
                    print(f"  {n_done}/{n_runs} done", flush=True)
            summary[entry["key"]] = {"status": "ok", "n_ok": n_ok,
                                      "n_total": len(datapoints),
                                      "success_rate": n_ok / len(datapoints)}

    (base / "codegen" / "exec_summary.json").write_text(
        json.dumps(summary, indent=1))
    bad = [k for k, v in summary.items() if v.get("success_rate", 0) < 0.5]
    print(f"\nDone. {len(summary)} code variants executed.")
    print(f"  {len([v for v in summary.values() if v.get('success_rate',0) >= 0.9])} "
          f"with success >=90%")
    print(f"  {len(bad)} with success < 50% (flagged): {bad[:10]}")


if __name__ == "__main__":
    main()
