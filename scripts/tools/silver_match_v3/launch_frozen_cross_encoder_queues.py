#!/usr/bin/env python3
"""Validate and launch frozen cross-encoder queues on explicit idle GPUs.

The launcher is intentionally mechanical: it does not alter commands, labels,
roles, or scientific settings.  It verifies every hash-bound queue artifact,
requires a one-to-one explicit GPU mapping, writes an append-only launch plan,
and starts each frozen command as an independent direct process.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import sha256_file
from .gpu_host_policy import validate_gpu_indices_for_host, validate_launch_gpus


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _verify_artifact(value: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(value["path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != value["sha256"]:
        raise ValueError(f"frozen artifact hash mismatch: {path}")
    return {"path": str(path), "sha256": observed, "size_bytes": path.stat().st_size}


def _verify_runtime_inventory(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version")
        != "silver-match-v3-directory-content-inventory-v1"
        or payload.get("status") != "FROZEN_CONTENT_HASH_INVENTORY"
    ):
        raise ValueError("unsupported runtime inventory")
    root = Path(str(payload["root"])).resolve()
    for row in payload.get("files") or []:
        artifact = root / str(row["relative_path"])
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(row["size_bytes"])
            or sha256_file(artifact) != row["sha256"]
        ):
            raise ValueError(f"runtime inventory mismatch: {artifact}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "root": str(root),
        "file_count": payload["file_count"],
        "total_size_bytes": payload["total_size_bytes"],
        "content_inventory_sha256": payload["content_inventory_sha256"],
    }


def validate_queue(
    path: Path, selected_variants: set[str] | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    queue = json.loads(path.read_text(encoding="utf-8"))
    if (
        queue.get("schema_version")
        != "silver-match-v3-cross-encoder-training-queue-v1"
        or queue.get("status") != "FROZEN_NOT_LAUNCHED"
    ):
        raise ValueError(f"unsupported or unfrozen queue: {path}")
    role = queue.get("role_audit") or {}
    if (
        role.get("complete") is not True
        or int(role.get("cross_role_uid_count", -1)) != 0
        or int(role.get("cross_role_source_group_count", -1)) != 0
    ):
        raise ValueError(f"queue role audit is not source-disjoint: {path}")
    artifacts = []
    for key in ("policy", "manifest", "bank", "implementation"):
        artifacts.append(_verify_artifact(queue[key]))
    if queue.get("policy_eligibility"):
        artifacts.append(_verify_artifact(queue["policy_eligibility"]))
    for role_name in ("train", "dev"):
        for value in queue["teacher_inputs"][role_name]:
            artifacts.append(_verify_artifact(value))
    for value in queue["candidate_inputs"]:
        artifacts.append(_verify_artifact(value))
    for value in queue.get("extra_bindings") or []:
        verified = _verify_artifact(value)
        verified["name"] = value.get("name")
        artifacts.append(verified)
    commands = queue.get("commands") or []
    if len(commands) != 3:
        raise ValueError(f"expected exactly three predeclared variants: {path}")
    repo = Path(str(queue["repo_root"])).resolve()
    if not repo.is_dir():
        raise FileNotFoundError(repo)
    implementation_path = Path(str(queue["implementation"]["path"])).resolve()
    try:
        implementation_relative = implementation_path.relative_to(repo)
    except ValueError as exc:
        raise ValueError("queue implementation is outside frozen repo root") from exc
    expected_module = ".".join(implementation_relative.with_suffix("").parts)
    for entry in commands:
        command = entry.get("command") or []
        if len(command) < 5 or command[2:4] != ["-m", expected_module]:
            raise ValueError(f"unexpected frozen command: {entry.get('variant')}")
        if not Path(command[0]).is_file():
            raise FileNotFoundError(command[0])
        expected = Path(str(entry["expected_report"]))
        output_root = Path(str(entry["output_root"])) / str(queue["task"])
        name = str((entry.get("variant") or {}).get("name") or "")
        check_fresh_output = selected_variants is None or name in selected_variants
        if check_fresh_output and (
            expected.exists() or (output_root.exists() and any(output_root.iterdir()))
        ):
            raise FileExistsError(f"variant output is already nonempty: {output_root}")
    return queue, artifacts


def _gpu_snapshot() -> list[dict[str, int]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in output.splitlines():
        values = [int(item.strip()) for item in line.split(",")]
        if len(values) != 4:
            raise ValueError(f"unexpected nvidia-smi row: {line}")
        rows.append(
            {
                "index": values[0],
                "memory_total_mib": values[1],
                "memory_used_mib": values[2],
                "utilization_percent": values[3],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", action="append", required=True)
    parser.add_argument("--gpu", action="append", type=int, required=True)
    parser.add_argument(
        "--variant-name",
        action="append",
        help="Launch only named frozen variants; the full three-variant queue is still verified.",
    )
    parser.add_argument("--launch-root", required=True)
    parser.add_argument("--runtime-inventory", required=True)
    parser.add_argument("--max-used-memory-mib", type=int, default=128)
    parser.add_argument("--startup-probe-seconds", type=int, default=30)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    validate_gpu_indices_for_host(args.gpu, hostname=socket.gethostname())

    queue_paths = [Path(value).resolve() for value in args.queue]
    runtime_inventory = _verify_runtime_inventory(
        Path(args.runtime_inventory).resolve()
    )
    selected_variants = set(args.variant_name or []) or None
    validated = [validate_queue(path, selected_variants) for path in queue_paths]
    entries = [
        (queue_path, queue_and_artifacts, command)
        for queue_path, queue_and_artifacts in zip(
            queue_paths, validated, strict=True
        )
        for command in queue_and_artifacts[0]["commands"]
    ]
    if selected_variants is not None:
        entries = [
            entry
            for entry in entries
            if str((entry[2].get("variant") or {}).get("name")) in selected_variants
        ]
        observed_variants = {
            str((entry[2].get("variant") or {}).get("name")) for entry in entries
        }
        if observed_variants != selected_variants:
            raise ValueError(
                f"requested variants absent or ambiguous: requested={sorted(selected_variants)}, "
                f"observed={sorted(observed_variants)}"
            )
    if len(args.gpu) != len(entries) or len(set(args.gpu)) != len(args.gpu):
        raise ValueError("provide one unique --gpu for every frozen command")
    snapshot = _gpu_snapshot()
    by_gpu = {row["index"]: row for row in snapshot}
    for gpu in args.gpu:
        row = by_gpu.get(gpu)
        if row is None or row["memory_used_mib"] > args.max_used_memory_mib:
            raise ValueError(f"target GPU is not idle: {gpu}/{row}")

    launch_root = Path(args.launch_root).resolve()
    if launch_root.exists():
        raise FileExistsError(launch_root)
    tool_path = Path(__file__).resolve()
    plan = {
        "schema_version": "silver-match-v3-cross-encoder-launch-plan-v1",
        "status": "VALIDATED_NOT_LAUNCHED" if not args.run else "LAUNCHING",
        "at": _utc_now(),
        "host": socket.gethostname(),
        "launcher": {"path": str(tool_path), "sha256": sha256_file(tool_path)},
        "runtime_inventory": runtime_inventory,
        "gpu_snapshot": snapshot,
        "jobs": [],
    }
    for gpu, (queue_path, (queue, artifacts), command_entry) in zip(
        args.gpu, entries, strict=True
    ):
        plan["jobs"].append(
            {
                "task": queue["task"],
                "variant": command_entry["variant"],
                "gpu": gpu,
                "queue": {"path": str(queue_path), "sha256": sha256_file(queue_path)},
                "verified_artifacts": artifacts,
                "command": command_entry["command"],
                "expected_report": command_entry["expected_report"],
            }
        )
    if not args.run:
        print(json.dumps(plan, sort_keys=True))
        return

    plan["gpu_launch_guard"] = validate_launch_gpus(
        args.gpu,
        hostname=socket.gethostname(),
        maximum_idle_memory_mib=args.max_used_memory_mib,
    )
    launch_root.mkdir(parents=True, exist_ok=False)
    _write_new(launch_root / "launch_plan.json", plan)
    processes = []
    for job in plan["jobs"]:
        slug = f"{job['task']}.{job['variant']['name']}.gpu{job['gpu']}"
        log_path = launch_root / f"{slug}.log"
        pid_path = launch_root / f"{slug}.pid"
        environment = dict(os.environ)
        repo = Path(job["queue"]["path"]).resolve()
        queue = json.loads(repo.read_text(encoding="utf-8"))
        repo_root = Path(queue["repo_root"]).resolve()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(job["gpu"]),
                "PYTHONPATH": os.pathsep.join(
                    (str(Path(runtime_inventory["root"])), str(repo_root))
                ),
                "HF_HOME": str(repo_root / "cache" / "huggingface"),
                "XDG_CACHE_HOME": str(repo_root / "cache"),
                "TORCHINDUCTOR_CACHE_DIR": str(repo_root / "cache" / "torchinductor"),
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "4",
            }
        )
        with log_path.open("x", encoding="utf-8") as handle:
            process = subprocess.Popen(
                job["command"],
                cwd=repo_root,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
        job["pid"] = process.pid
        job["log"] = str(log_path)
        processes.append((job, process))

    if args.startup_probe_seconds < 1 or args.startup_probe_seconds > 120:
        raise ValueError("startup probe must be between 1 and 120 seconds")
    time.sleep(args.startup_probe_seconds)
    failed = []
    for job, process in processes:
        returncode = process.poll()
        job["returncode_after_startup_probe"] = returncode
        if returncode not in (None, 0):
            failed.append(job)
    plan["status"] = "LAUNCHED" if not failed else "STARTUP_FAILURE"
    plan["startup_probe_seconds"] = args.startup_probe_seconds
    _write_new(launch_root / "launch_record.json", plan)
    print(json.dumps(plan, sort_keys=True), flush=True)
    if failed:
        raise RuntimeError(f"{len(failed)} CE jobs failed during startup probe")


if __name__ == "__main__":
    main()
