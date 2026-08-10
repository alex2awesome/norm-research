#!/usr/bin/env python3
"""Nonpreemptingly launch a sealed task LoRA after stable GPU-free evidence."""

from __future__ import annotations

import argparse
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def verify_queue_hashes(queue: dict[str, Any]) -> dict[str, str]:
    checks = {
        "manifest": (
            Path(queue["inputs"]["manifest"]["path"]),
            queue["inputs"]["manifest"]["sha256"],
        ),
        "combined_teachers": (
            Path(queue["inputs"]["combined_teachers"]["path"]),
            queue["inputs"]["combined_teachers"]["sha256"],
        ),
        "external_dev": (
            Path(queue["inputs"]["external_dev"]["path"]),
            queue["inputs"]["external_dev"]["sha256"],
        ),
        "external_dev_test": (
            Path(queue["inputs"]["external_dev_test"]["path"]),
            queue["inputs"]["external_dev_test"]["sha256"],
        ),
        "promotion_policy": (
            Path(queue["inputs"]["promotion_policy"]["path"]),
            queue["inputs"]["promotion_policy"]["sha256"],
        ),
        "trainer": (
            Path(queue["trainer"]["path"]),
            queue["trainer"]["expected_sha256_before_launch"],
        ),
    }
    observed = {}
    for name, (path, expected) in checks.items():
        if not path.is_file():
            raise FileNotFoundError(f"sealed queue input missing: {name}/{path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"sealed queue hash mismatch: {name}/{path}: {actual} != {expected}"
            )
        observed[name] = actual
    return observed


def verify_training_python(python: Path) -> dict[str, str]:
    probe = (
        "import json, peft, sentence_transformers, torch, transformers; "
        "print(json.dumps({'peft': peft.__version__, "
        "'sentence_transformers': sentence_transformers.__version__, "
        "'torch': torch.__version__, 'transformers': transformers.__version__}, "
        "sort_keys=True))"
    )
    completed = subprocess.run(
        [str(python), "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip())


def query_gpus() -> list[dict[str, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        values = [int(value.strip()) for value in line.split(",")]
        if len(values) != 4:
            raise ValueError(f"unexpected nvidia-smi row: {line!r}")
        rows.append(
            {
                "index": values[0],
                "memory_free_mib": values[1],
                "memory_used_mib": values[2],
                "utilization_percent": values[3],
            }
        )
    if not rows:
        raise ValueError("nvidia-smi returned no GPUs")
    return rows


def active_gpu_indices() -> set[int]:
    gpu_output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
    )
    uuid_to_index = {
        uuid.strip(): int(index.strip())
        for line in gpu_output.splitlines()
        if line.strip()
        for index, uuid in [line.split(",", 1)]
    }
    apps_output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    pid_to_gpu = {
        int(pid.strip()): uuid_to_index[uuid.strip()]
        for line in apps_output.splitlines()
        if line.strip() and "," in line
        for uuid, pid in [line.split(",", 1)]
        if uuid.strip() in uuid_to_index and pid.strip().isdigit()
    }
    return set(pid_to_gpu.values())


def is_gpu_free(
    row: dict[str, int],
    *,
    excluded: set[int],
    minimum_free_memory_mib: int,
    maximum_used_memory_mib: int,
    maximum_utilization_percent: int,
    active_gpus: set[int] | None = None,
) -> bool:
    return (
        row["index"] not in excluded
        and row["index"] not in (active_gpus or set())
        and row["memory_free_mib"] >= minimum_free_memory_mib
        and row["memory_used_mib"] <= maximum_used_memory_mib
        and row["utilization_percent"] <= maximum_utilization_percent
    )


def training_command(queue: dict[str, Any], python: Path) -> list[str]:
    config = queue["trainer"]["hyperparameters"]
    command = [
        str(python),
        "-u",
        "-m",
        "scripts.tools.silver_match_v3.train_nemotron_lora",
        "--task",
        str(queue["task"]),
        "--manifest",
        str(queue["inputs"]["manifest"]["path"]),
        "--teachers",
        str(queue["inputs"]["combined_teachers"]["path"]),
        "--output-root",
        str(queue["trainer"]["output_root"]),
        "--device",
        "cuda",
        "--epochs",
        str(config["epochs"]),
        "--batch-size",
        str(config["batch_size"]),
        "--gradient-accumulation-steps",
        str(config["gradient_accumulation_steps"]),
        "--max-seq-length",
        str(config["max_seq_length"]),
        "--learning-rate",
        str(config["learning_rate"]),
        "--margin",
        str(config["margin"]),
        "--hard-negative-pool",
        str(config["hard_negative_pool"]),
        "--negatives-per-positive",
        str(config["negatives_per_positive"]),
        "--lora-rank",
        str(config["lora_rank"]),
        "--lora-alpha",
        str(config["lora_alpha"]),
        "--lora-dropout",
        str(config["lora_dropout"]),
        "--train-percent",
        str(config["train_percent"]),
        "--dev-percent",
        str(config["dev_percent"]),
        "--selection-k",
        str(config["selection_k"]),
        "--epoch-selection-policy",
        str(config["epoch_selection_policy"]),
        "--seed",
        str(config["seed"]),
        "--split-seed",
        str(config["split_seed"]),
        "--no-enforce-promotion-gate",
    ]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--exclude-gpu", action="append", type=int, default=[])
    parser.add_argument("--stable-samples", type=int, default=6)
    parser.add_argument("--sample-interval-seconds", type=int, default=60)
    parser.add_argument("--minimum-free-memory-mib", type=int, default=120000)
    parser.add_argument("--maximum-used-memory-mib", type=int, default=2048)
    parser.add_argument("--maximum-utilization-percent", type=int, default=5)
    args = parser.parse_args()
    if args.stable_samples < 1 or args.sample_interval_seconds < 1:
        parser.error("stable samples and sample interval must be positive")
    queue_path = Path(args.queue).resolve()
    repo = Path(args.repo).resolve()
    python = Path(args.python).resolve()
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    if queue.get("status") != "QUEUED_AWAITING_LIVE_GPU_CAPACITY":
        raise ValueError("queue is not launchable")
    if (
        queue.get("inputs", {}).get("external_dev", {}).get("training_access")
        != "FORBIDDEN"
    ):
        raise ValueError("external dev training boundary is not sealed")
    if (
        queue.get("inputs", {}).get("external_dev_test", {}).get("training_access")
        != "FORBIDDEN"
    ):
        raise ValueError("external dev/test training boundary is not sealed")
    if not repo.is_dir() or not python.is_file():
        raise FileNotFoundError("repo or Python executable missing")

    queue_dir = queue_path.parent
    lock_path = queue_dir / "lora.watcher.lock"
    events_path = queue_dir / "lora.watcher.events.jsonl"
    launch_path = queue_dir / "lora.launch.record.json"
    training_log = queue_dir / "lora.training.log"
    training_pid_path = queue_dir / "lora.training.pid"
    output_root = Path(queue["trainer"]["output_root"]).resolve() / str(queue["task"])
    if launch_path.exists() or training_pid_path.exists():
        raise FileExistsError("sealed queue already has a launch record")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"adapter output is already nonempty: {output_root}")
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(lock_fd, f"{os.getpid()}\n".encode())
    os.close(lock_fd)

    excluded = set(args.exclude_gpu)
    if is_sk3_host():
        # This is a hard runtime overlay.  It also protects historical queues
        # that predate the operator's permanent sk3 device prohibition.
        excluded.update(SK3_PROHIBITED_GPU_INDICES)
    stable = {}
    try:
        initial_hashes = verify_queue_hashes(queue)
        initial_python_libraries = verify_training_python(python)
        append_event(
            events_path,
            {
                "event": "watcher_started",
                "at": utc_now(),
                "host": socket.gethostname(),
                "watcher_pid": os.getpid(),
                "queue": str(queue_path),
                "queue_sha256": sha256_file(queue_path),
                "verified_hashes": initial_hashes,
                "training_python": str(python),
                "training_python_libraries": initial_python_libraries,
                "excluded_gpus": sorted(excluded),
                "sk3_allowed_gpu_indices": sorted(SK3_ALLOWED_GPU_INDICES),
                "sk3_prohibited_gpu_indices": sorted(SK3_PROHIBITED_GPU_INDICES),
                "stable_samples_required": args.stable_samples,
                "gpu_count_gate_applied": False,
                "projected_owner_count_check_applied": False,
            },
        )
        while True:
            rows = query_gpus()
            all_active = active_gpu_indices()
            current_free = set()
            for row in rows:
                index = row["index"]
                free = is_gpu_free(
                    row,
                    excluded=excluded,
                    minimum_free_memory_mib=args.minimum_free_memory_mib,
                    maximum_used_memory_mib=args.maximum_used_memory_mib,
                    maximum_utilization_percent=args.maximum_utilization_percent,
                    active_gpus=all_active,
                )
                stable[index] = stable.get(index, 0) + 1 if free else 0
                if free:
                    current_free.add(index)
            append_event(
                events_path,
                {
                    "event": "gpu_sample",
                    "at": utc_now(),
                    "gpus": rows,
                    "stable_free_counts": dict(sorted(stable.items())),
                    "all_active_gpu_indices": sorted(all_active),
                    "gpu_count_gate_applied": False,
                },
            )
            ready = sorted(
                index
                for index in current_free
                if stable.get(index, 0) >= args.stable_samples
            )
            if not ready:
                time.sleep(args.sample_interval_seconds)
                continue
            gpu = ready[0]
            if is_sk3_host() and gpu not in SK3_ALLOWED_GPU_INDICES:
                raise RuntimeError(f"sk3 GPU policy rejected selected device {gpu}")

            # Recheck all sealed hashes and live capacity immediately before launch.
            final_hashes = verify_queue_hashes(queue)
            final_python_libraries = verify_training_python(python)
            fresh = {row["index"]: row for row in query_gpus()}
            fresh_all_active = active_gpu_indices()
            if gpu not in fresh or not is_gpu_free(
                fresh[gpu],
                excluded=excluded,
                minimum_free_memory_mib=args.minimum_free_memory_mib,
                maximum_used_memory_mib=args.maximum_used_memory_mib,
                maximum_utilization_percent=args.maximum_utilization_percent,
                active_gpus=fresh_all_active,
            ):
                stable[gpu] = 0
                append_event(
                    events_path,
                    {"event": "launch_recheck_failed", "at": utc_now(), "gpu": gpu},
                )
                time.sleep(args.sample_interval_seconds)
                continue
            if output_root.exists() and any(output_root.iterdir()):
                raise FileExistsError(f"adapter output became nonempty: {output_root}")

            command = training_command(queue, python)
            environment = dict(os.environ)
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "HOME": "/lfs/skampere3/0/alexspan",
                    "XDG_CACHE_HOME": "/lfs/skampere3/0/alexspan/.cache",
                    "HF_MODULES_CACHE": "/lfs/skampere3/0/alexspan/.cache/huggingface/modules",
                    "TORCHINDUCTOR_CACHE_DIR": "/lfs/skampere3/0/alexspan/.cache/torchinductor",
                    "PYTHONPATH": str(repo),
                }
            )
            with training_log.open("x", encoding="utf-8") as handle:
                process = subprocess.Popen(
                    command,
                    cwd=repo,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            atomic_text(training_pid_path, f"{process.pid}\n")
            launch = {
                "schema_version": "silver-match-v3-task-lora-launch-v1",
                "event": "launched",
                "at": utc_now(),
                "host": socket.gethostname(),
                "watcher_pid": os.getpid(),
                "training_pid": process.pid,
                "gpu_index": gpu,
                "final_gpu_sample": fresh[gpu],
                "stable_free_samples": stable[gpu],
                "gpu_count_gate_applied": False,
                "projected_owner_count_check_applied": False,
                "excluded_gpus": sorted(excluded),
                "queue": str(queue_path),
                "queue_sha256": sha256_file(queue_path),
                "verified_hashes_immediately_before_launch": final_hashes,
                "training_python_libraries_immediately_before_launch": final_python_libraries,
                "command": command,
                "training_log": str(training_log),
                "output_root": str(output_root),
                "external_dev_test_consumed": False,
            }
            atomic_json(launch_path, launch)
            append_event(events_path, launch)
            return
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
