#!/usr/bin/env python3
"""Execute the frozen 443-call Math-a12 full-context Sonnet arm."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from methods.metric_seam.verifiers.math_a12_context_contract import validate_response_envelope

ROOT = Path(__file__).resolve().parents[2]
REQ = ROOT / "outputs/metric_seam_pilot/hierarchy_r123/requests/math_a12_context_train_v1"
OUT = ROOT / "outputs/metric_seam_pilot/hierarchy_r123/results/math_a12_context_train_v1"


def invoke(request: dict, timeout: int) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    try:
        cp = subprocess.run([
            "claude", "--model", request["model"], "--output-format", "text",
            "--system-prompt", request["system_prompt"], "-p", request["user_prompt"]],
            capture_output=True, text=True, timeout=timeout, check=False)
        raw, error, returncode = cp.stdout, cp.stderr, cp.returncode
    except subprocess.TimeoutExpired as exc:
        raw, error, returncode = exc.stdout or "", exc.stderr or "timeout", None
    envelope = {
        "request_sha256": request["request_sha256"], "raw_response": raw,
        "stderr": error, "returncode": returncode, "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        envelope["validated"] = validate_response_envelope(envelope, request)
        envelope["valid"] = True
    except Exception as exc:
        envelope["valid"] = False
        envelope["validation_error"] = f"{type(exc).__name__}: {exc}"
    return envelope


def run_batch(requests, timeout, concurrency):
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(invoke, r, timeout): r for r in requests}
        return [future.result() for future in as_completed(futures)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--request-dir", type=Path, default=REQ)
    ap.add_argument("--output-dir", type=Path, default=OUT)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    if not 1 <= args.concurrency <= 4:
        raise ValueError("concurrency must be 1..4")
    requests = [json.loads(line) for line in (args.request_dir / "requests.jsonl").read_text().splitlines()]
    if len(requests) != 443:
        raise AssertionError("expected exactly 443 frozen requests")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    smoke = run_batch(requests[:10], args.timeout, args.concurrency)
    smoke_valid = sum(r["valid"] for r in smoke)
    rows = smoke
    stopped = smoke_valid != 10
    if not stopped:
        rows += run_batch(requests[10:], args.timeout, args.concurrency)
    (args.output_dir / "responses.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows))
    manifest = {
        "schema": "metric-seam.math-a12-context-execution.v1",
        "status": "stopped_after_smoke" if stopped else "complete",
        "smoke": {"n": 10, "valid": smoke_valid, "required": 10},
        "request_count": 443, "executed_count": len(rows),
        "valid_count": sum(r["valid"] for r in rows),
        "model": requests[0]["model"], "concurrency": args.concurrency,
        "fresh_process_per_request": True, "retries": 0, "gpu_used": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(args.output_dir / "manifest.json")
    return 2 if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
