"""Execute Llama-generated code on datapoints (mirror of validity_pilot_code_exec.py
but reads from codegen/responses_llama/ instead of responses/).

Outputs results to codegen/exec_results_llama.jsonl. Then the analysis script
can read BOTH exec_results.jsonl (Claude) and exec_results_llama.jsonl (Llama)
to compute inter-model consistency.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run_one(code, text, timeout=5.0):
    runner = (
        f"{code}\n\n"
        "import json, sys\n"
        "text = json.loads(sys.stdin.read())\n"
        "try:\n"
        "    s = float(score(text))\n"
        "    if not (0.0 <= s <= 1.0): s = max(0.0, min(1.0, s))\n"
        "    print(json.dumps({'score': s}))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'error': str(e)[:200]}))\n"
    )
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner], input=json.dumps(text),
            capture_output=True, text=True, timeout=timeout)
        dt = time.time() - t0
        if proc.returncode != 0:
            return None, f"nonzero exit: {proc.stderr[:200]}", dt
        try:
            out = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:
            return None, f"no JSON on stdout", dt
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

    # The manifest entries' `model` field says "claude", but the Llama runner
    # used the same prompts. So we re-use the manifest as a job list, but
    # read from responses_llama/ instead.
    manifest = json.loads((base / "codegen" / "manifest.json").read_text())
    datapoints = json.loads((base / "datapoints.json").read_text())

    out_path = base / "codegen" / "exec_results_llama.jsonl"
    summary = {}
    n_total = sum(1 for e in manifest
                  if (base / "codegen" / "responses_llama" / f"{e['key']}.py").exists())
    print(f"Found {n_total} Llama code responses (out of {len(manifest)} manifest entries)")
    n_done = 0
    with out_path.open("w") as f:
        for entry in manifest:
            code_path = base / "codegen" / "responses_llama" / f"{entry['key']}.py"
            if not code_path.exists():
                continue
            code = code_path.read_text()
            try:
                compile(code, code_path.name, "exec")
            except SyntaxError as e:
                summary[entry["key"]] = {"status": "syntax_error"}
                continue
            # Overwrite model field to mark as llama
            entry_l = {**entry, "model": "llama"}
            n_ok = 0
            for dp in datapoints:
                s, err, dt = run_one(code, dp["text"])
                rec = {**entry_l, "datapoint_id": dp["datapoint_id"],
                       "score": s, "error": err, "exec_ms": int(dt * 1000)}
                f.write(json.dumps(rec) + "\n")
                if s is not None: n_ok += 1
                n_done += 1
                if n_done % 50 == 0:
                    print(f"  {n_done}/{n_total * len(datapoints)} runs done", flush=True)
            summary[entry["key"]] = {"status": "ok", "n_ok": n_ok,
                                      "n_total": len(datapoints),
                                      "success_rate": n_ok / len(datapoints)}
    (base / "codegen" / "exec_summary_llama.json").write_text(json.dumps(summary, indent=1))
    n_high = sum(1 for v in summary.values() if v.get("success_rate", 0) >= 0.9)
    n_low = sum(1 for v in summary.values() if v.get("success_rate", 1) < 0.5)
    print(f"Done. {n_high} with >=90% success, {n_low} with <50%.")


if __name__ == "__main__":
    main()
