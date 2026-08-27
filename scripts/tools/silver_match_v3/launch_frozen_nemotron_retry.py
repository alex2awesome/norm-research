#!/usr/bin/env python3
"""Launch one hash-frozen task-local Nemotron LoRA on an explicit idle GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .gpu_host_policy import validate_gpu_indices_for_host, validate_launch_gpus


SCHEMA = "silver-match-v3-frozen-nemotron-retry-queue-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_bindings(queue: dict[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for binding in queue["bindings"]:
        name = str(binding["name"])
        path = Path(binding["path"])
        expected = str(binding["sha256"])
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen binding {name}: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen binding mismatch {name}: {actual} != {expected}")
        observed[name] = actual
    return observed


def verify_model_inventory(queue: dict[str, Any]) -> dict[str, Any]:
    inventory_path = Path(queue["model_inventory"]["path"])
    inventory = json.loads(inventory_path.read_text())
    expected_content = queue["model_inventory"]["content_inventory_sha256"]
    if inventory.get("content_inventory_sha256") != expected_content:
        raise ValueError("model inventory content hash differs from frozen queue")
    root = Path(queue["model"])
    if Path(inventory["root"]).resolve() != root.resolve():
        raise ValueError("model inventory root differs from frozen queue")
    files = inventory.get("files") or []
    observed_paths = {
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    }
    expected_paths = {str(row["relative_path"]) for row in files}
    if observed_paths != expected_paths:
        raise ValueError("model file set differs from frozen inventory")
    total = 0
    for row in files:
        path = root / row["relative_path"]
        size = path.stat().st_size
        if size != int(row["size_bytes"]) or sha256_file(path) != row["sha256"]:
            raise ValueError(f"model artifact differs from inventory: {path}")
        total += size
    if total != int(inventory["total_size_bytes"]):
        raise ValueError("model total size differs from frozen inventory")
    return {
        "content_inventory_sha256": expected_content,
        "file_count": len(files),
        "total_size_bytes": total,
    }


def gpu_sample(index: int) -> dict[str, Any]:
    output = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    values = [value.strip() for value in output.split(",")]
    if len(values) != 5:
        raise ValueError(f"unexpected nvidia-smi output: {output!r}")
    return {
        "index": int(values[0]),
        "uuid": values[1],
        "memory_used_mib": int(values[2]),
        "memory_free_mib": int(values[3]),
        "utilization_percent": int(values[4]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    args = parser.parse_args()
    validate_gpu_indices_for_host([args.gpu], hostname=socket.gethostname())

    queue_path = Path(args.queue).resolve()
    queue = json.loads(queue_path.read_text())
    if queue.get("schema_version") != SCHEMA or queue.get("status") != "FROZEN_READY":
        raise ValueError("queue is not a frozen Nemotron retry queue")
    if args.gpu != int(queue["gpu"]["index"]):
        raise ValueError("requested GPU differs from frozen queue")
    command = [str(value) for value in queue["command"]]
    forbidden = {
        str(binding["path"])
        for binding in queue["bindings"]
        if binding.get("training_access") == "FORBIDDEN"
    }
    if forbidden.intersection(command):
        raise ValueError("a training-forbidden input appears in the command")
    if command[1:4] != ["-u", "-m", "scripts.tools.silver_match_v3.train_nemotron_lora"]:
        raise ValueError("unexpected frozen training entry point")

    paths = queue["outputs"]
    launch_path = Path(paths["launch_record"])
    pid_path = Path(paths["pid"])
    log_path = Path(paths["log"])
    output_root = Path(paths["training_output_root"]) / queue["task"]
    for path in (launch_path, pid_path, log_path):
        if path.exists():
            raise FileExistsError(path)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"nonempty training output: {output_root}")
    launch_path.parent.mkdir(parents=True, exist_ok=True)

    bindings = verify_bindings(queue)
    model = verify_model_inventory(queue)
    sample = gpu_sample(args.gpu)
    if sample["uuid"] != queue["gpu"]["uuid"]:
        raise ValueError("physical GPU UUID differs from frozen queue")
    if sample["memory_used_mib"] > 2048 or sample["utilization_percent"] > 5:
        raise RuntimeError(f"frozen GPU is not idle: {sample}")
    gpu_launch_guard = validate_launch_gpus(
        [args.gpu], hostname=socket.gethostname(), maximum_idle_memory_mib=2048
    )

    environment = dict(os.environ)
    environment.update({str(k): str(v) for k, v in queue["environment"].items()})
    with log_path.open("x", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=queue["repo"],
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    pid_path.write_text(f"{process.pid}\n")
    record = {
        "schema_version": "silver-match-v3-frozen-nemotron-retry-launch-v1",
        "status": "LAUNCHED",
        "launched_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "pid": process.pid,
        "gpu": sample,
        "gpu_launch_guard": gpu_launch_guard,
        "queue": str(queue_path),
        "queue_sha256": sha256_file(queue_path),
        "verified_bindings": bindings,
        "verified_model": model,
        "command": command,
        "external_dev_test_consumed": False,
    }
    atomic_json(launch_path, record)
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
