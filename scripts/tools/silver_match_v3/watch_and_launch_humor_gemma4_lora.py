#!/usr/bin/env python3
"""Wait for a genuinely idle allowed GPU, then launch the Humor Gemma LoRA once."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SK3_ALLOWED_GPU_INDICES = frozenset({0, 5, 6, 7})
SK3_PROHIBITED_GPU_INDICES = frozenset({1, 2, 3, 4})


def is_sk3_host() -> bool:
    host = socket.gethostname().split(".", 1)[0].lower()
    return host in {"sk3", "skampere3"} or host.startswith("skampere3-")


def selectable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not is_sk3_host():
        return rows
    return [row for row in rows if row["index"] in SK3_ALLOWED_GPU_INDICES]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle, fcntl.LOCK_UN)


def gpu_rows() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [value.strip() for value in line.split(",")]
        if len(parts) != 4:
            raise ValueError(f"unexpected nvidia-smi row: {line!r}")
        rows.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "memory_used_mib": int(parts[2]),
                "utilization_percent": int(parts[3]),
            }
        )
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPUs")
    return rows


def active_gpu_uuids() -> set[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        line.split(",", 1)[0].strip()
        for line in result.stdout.splitlines()
        if line.strip() and "," in line
    }


def genuinely_idle(
    rows: list[dict[str, Any]],
    idle_memory_mib: int,
    process_gpu_uuids: set[str],
) -> list[dict[str, Any]]:
    return sorted(
        (
            row
            for row in rows
            if row["uuid"] not in process_gpu_uuids
            and row["memory_used_mib"] <= idle_memory_mib
            and row["utilization_percent"] == 0
        ),
        key=lambda row: row["index"],
    )


def verify_bindings(queue: dict[str, Any]) -> dict[str, str]:
    hashes = {}
    for name, binding in sorted((queue.get("bindings") or {}).items()):
        path = Path(str(binding.get("path") or "")).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"queue binding missing: {name}/{path}")
        actual = sha256_file(path)
        if actual != binding.get("sha256"):
            raise ValueError(f"queue binding drift: {name}: {actual} != {binding.get('sha256')}")
        hashes[name] = actual
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-polls", type=int, default=1440)
    args = parser.parse_args()
    if not 5 <= args.poll_seconds <= 60 or args.max_polls <= 0:
        parser.error("poll-seconds must be 5..60 and max-polls must be positive")

    queue_path = Path(args.queue).resolve()
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    if (
        queue.get("schema_version")
        != "silver-match-v3-humor-gemma4-typed-lora-queue-v1"
        or queue.get("status") != "FROZEN_AWAITING_SERVER_CAPACITY"
        or queue.get("task") != "humor"
    ):
        raise ValueError("unexpected frozen queue")
    queue_sha = sha256_file(queue_path)
    outputs = queue.get("outputs") or {}
    launch_record = Path(outputs["launch_record"]).resolve()
    training_log = Path(outputs["log"]).resolve()
    adapter_output = Path(outputs["adapter"]).resolve()
    training_report = Path(outputs["training_report"]).resolve()
    events = queue_path.with_suffix(queue_path.suffix + ".watcher.events.jsonl")
    lock_path = queue_path.with_suffix(queue_path.suffix + ".watcher.lock")
    for path in (launch_record, training_log, adapter_output, training_report):
        if path.exists():
            raise FileExistsError(f"non-overwriting launch target exists: {path}")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another watcher already owns this queue") from exc
        initial_hashes = verify_bindings(queue)
        policy = queue["gpu_policy"]
        idle_memory_mib = int(policy["target_max_memory_used_mib"])
        prior_candidate: tuple[int, str] | None = None
        stable_polls = 0
        append_event(
            events,
            {
                "at": datetime.now(timezone.utc).isoformat(),
                "event": "WATCHER_STARTED",
                "queue_sha256": queue_sha,
                "bindings": initial_hashes,
                "max_polls": args.max_polls,
                "poll_seconds": args.poll_seconds,
            },
        )
        for poll in range(1, args.max_polls + 1):
            rows = gpu_rows()
            process_gpu_uuids = active_gpu_uuids()
            idle = genuinely_idle(
                selectable_rows(rows), idle_memory_mib, process_gpu_uuids
            )
            candidate = (idle[0]["index"], idle[0]["uuid"]) if idle else None
            capacity = candidate is not None
            if capacity and candidate == prior_candidate:
                stable_polls += 1
            elif capacity:
                stable_polls = 1
            else:
                stable_polls = 0
            prior_candidate = candidate if capacity else None
            if poll == 1 or poll % 10 == 0 or stable_polls:
                append_event(
                    events,
                    {
                        "at": datetime.now(timezone.utc).isoformat(),
                        "event": "CAPACITY_POLL",
                        "poll": poll,
                        "candidate": candidate,
                        "gpu_count_gate_applied": False,
                        "stable_candidate_polls": stable_polls,
                        "process_gpu_uuids": sorted(process_gpu_uuids),
                        "gpus": rows,
                    },
                )
            if stable_polls >= int(policy["stable_idle_polls_required"]):
                # Rehash and repoll immediately before launch.  The selected
                # target must remain genuinely idle and host-allowed.
                final_hashes = verify_bindings(queue)
                if final_hashes != initial_hashes:
                    raise ValueError("binding hash set changed during capacity wait")
                final_rows = gpu_rows()
                final_process_gpu_uuids = active_gpu_uuids()
                selected = next(
                    (row for row in final_rows if (row["index"], row["uuid"]) == candidate),
                    None,
                )
                if (
                    selected is None
                    or (
                        is_sk3_host()
                        and selected["index"] not in SK3_ALLOWED_GPU_INDICES
                    )
                    or selected["uuid"] in final_process_gpu_uuids
                    or selected["memory_used_mib"] > idle_memory_mib
                    or selected["utilization_percent"] != 0
                ):
                    stable_polls = 0
                    prior_candidate = None
                    time.sleep(args.poll_seconds)
                    continue
                for path in (launch_record, training_log, adapter_output, training_report):
                    if path.exists():
                        raise FileExistsError(f"launch target appeared while waiting: {path}")
                training_log.parent.mkdir(parents=True, exist_ok=True)
                environment = os.environ.copy()
                environment.update({str(k): str(v) for k, v in queue["environment"].items()})
                environment["CUDA_VISIBLE_DEVICES"] = str(selected["index"])
                command = [str(value) for value in queue["command"]]
                log_handle = training_log.open("x", encoding="utf-8")
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    cwd=str(queue_path.parent),
                    start_new_session=True,
                    text=True,
                )
                log_handle.close()
                record = {
                    "schema_version": "silver-match-v3-humor-gemma4-lora-launch-v1",
                    "status": "LAUNCHED",
                    "launched_at": datetime.now(timezone.utc).isoformat(),
                    "queue": {"path": str(queue_path), "sha256": queue_sha},
                    "pid": process.pid,
                    "gpu": selected,
                    "gpu_count_gate_applied": False,
                    "projected_owner_count_check_applied": False,
                    "sk3_allowed_gpu_indices": sorted(SK3_ALLOWED_GPU_INDICES),
                    "sk3_prohibited_gpu_indices": sorted(
                        SK3_PROHIBITED_GPU_INDICES
                    ),
                    "command": command,
                    "environment_overrides": {
                        **queue["environment"],
                        "CUDA_VISIBLE_DEVICES": str(selected["index"]),
                    },
                    "bindings": final_hashes,
                    "log": str(training_log),
                }
                launch_record.parent.mkdir(parents=True, exist_ok=True)
                with launch_record.open("x", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                append_event(
                    events,
                    {
                        "at": datetime.now(timezone.utc).isoformat(),
                        "event": "TRAINING_LAUNCHED",
                        "pid": process.pid,
                        "gpu": selected,
                        "launch_record": str(launch_record),
                        "launch_record_sha256": sha256_file(launch_record),
                    },
                )
                print(json.dumps(record, sort_keys=True), flush=True)
                return
            time.sleep(args.poll_seconds)
        append_event(
            events,
            {
                "at": datetime.now(timezone.utc).isoformat(),
                "event": "WATCHER_EXHAUSTED_WITHOUT_LAUNCH",
                "polls": args.max_polls,
            },
        )
        raise TimeoutError("no genuinely idle allowed GPU became stable")


if __name__ == "__main__":
    main()
