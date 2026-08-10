#!/usr/bin/env python3
"""Launch one predeclared variant from a frozen cross-encoder queue.

The three variants in a queue are scientifically independent, but GPU
availability is not always simultaneous.  This launcher preserves the queue
verbatim, validates every bound artifact with the all-variant launcher, and
starts exactly one named variant on an explicitly idle GPU.  Each variant gets
an append-only launch directory, so later variants can be launched serially
without rewriting the frozen queue.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .common import sha256_file
from .gpu_host_policy import validate_gpu_indices_for_host, validate_launch_gpus
from .launch_frozen_cross_encoder_queues import (
    _gpu_snapshot,
    _verify_runtime_inventory,
    validate_queue,
)


def _write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--launch-root", required=True)
    parser.add_argument("--runtime-inventory", required=True)
    parser.add_argument("--max-used-memory-mib", type=int, default=2048)
    parser.add_argument("--startup-probe-seconds", type=int, default=30)
    args = parser.parse_args()
    validate_gpu_indices_for_host([args.gpu], hostname=socket.gethostname())

    queue_path = Path(args.queue).resolve()
    # A serial launcher must require freshness only for the requested variant.
    # Other predeclared variants may already have completed reports.
    queue, artifacts = validate_queue(queue_path, {args.variant})
    commands = [
        entry
        for entry in queue["commands"]
        if str(entry["variant"]["name"]) == args.variant
    ]
    if len(commands) != 1:
        raise ValueError(f"variant is not uniquely predeclared: {args.variant}")
    entry = commands[0]
    runtime = _verify_runtime_inventory(Path(args.runtime_inventory).resolve())
    snapshot = _gpu_snapshot()
    target = next((row for row in snapshot if row["index"] == args.gpu), None)
    if target is None or target["memory_used_mib"] > args.max_used_memory_mib:
        raise RuntimeError(f"target GPU is not idle: {args.gpu}/{target}")
    gpu_launch_guard = validate_launch_gpus(
        [args.gpu],
        hostname=socket.gethostname(),
        maximum_idle_memory_mib=args.max_used_memory_mib,
    )

    launch_root = Path(args.launch_root).resolve()
    if launch_root.exists():
        raise FileExistsError(launch_root)
    launch_root.mkdir(parents=True, exist_ok=False)
    log_path = launch_root / "training.log"
    pid_path = launch_root / "training.pid"
    plan = {
        "schema_version": "silver-match-v3-cross-encoder-variant-launch-v1",
        "status": "LAUNCHING",
        "launched_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "task": queue["task"],
        "variant": entry["variant"],
        "gpu": target,
        "gpu_launch_guard": gpu_launch_guard,
        "queue": {"path": str(queue_path), "sha256": sha256_file(queue_path)},
        "runtime_inventory": runtime,
        "verified_artifacts": artifacts,
        "command": entry["command"],
        "expected_report": entry["expected_report"],
        "permanent_blind_consumed": False,
    }
    _write_new(launch_root / "launch_plan.json", plan)

    environment = dict(os.environ)
    repo_root = Path(queue["repo_root"]).resolve()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONPATH": os.pathsep.join((str(Path(runtime["root"])), str(repo_root))),
            "HF_HOME": str(repo_root / "cache" / "huggingface"),
            "XDG_CACHE_HOME": str(repo_root / "cache"),
            "TORCHINDUCTOR_CACHE_DIR": str(repo_root / "cache" / "torchinductor"),
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "4",
        }
    )
    with log_path.open("x", encoding="utf-8") as handle:
        process = subprocess.Popen(
            entry["command"],
            cwd=repo_root,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    pid_path.write_text(f"{process.pid}\n")
    if not 1 <= args.startup_probe_seconds <= 120:
        raise ValueError("startup probe must be between 1 and 120 seconds")
    try:
        process.wait(timeout=args.startup_probe_seconds)
    except subprocess.TimeoutExpired:
        pass
    returncode = process.poll()
    record = {
        **plan,
        "status": "LAUNCHED" if returncode in (None, 0) else "STARTUP_FAILURE",
        "pid": process.pid,
        "log": str(log_path),
        "returncode_after_startup_probe": returncode,
        "startup_probe_seconds": args.startup_probe_seconds,
    }
    _write_new(launch_root / "launch_record.json", record)
    print(json.dumps(record, sort_keys=True))
    if returncode not in (None, 0):
        raise RuntimeError(f"variant failed during startup: {returncode}")


if __name__ == "__main__":
    main()
