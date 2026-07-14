#!/usr/bin/env python3
"""Run the frozen similarity LoRA DAG, fail-closed on sk2 GPU availability."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def assert_sk2() -> None:
    host = socket.gethostname().split(".", 1)[0].lower()
    if host not in {"sk2", "skampere2"} and not host.startswith("skampere2-"):
        raise RuntimeError(f"job DAG is sk2-only; refusing {socket.gethostname()}")


def gpu_idle(index: int) -> bool:
    query = subprocess.check_output(
        [
            "nvidia-smi", f"--id={index}",
            "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits",
        ], text=True,
    ).strip()
    memory, utilization = [int(value.strip()) for value in query.split(",", 1)]
    processes = subprocess.check_output(
        ["nvidia-smi", f"--id={index}", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    return memory <= 2048 and utilization == 0 and not processes


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_frozen_inputs(plan: dict[str, Any]) -> None:
    for key in ("dataset_inventory", "dataset_manifest"):
        reference = plan[key]
        path = Path(reference["remote_path"])
        if not path.is_file() or sha256_file(path) != reference["sha256"]:
            raise RuntimeError(f"frozen {key} missing or hash-drifted: {path}")
    model_reference = plan.get("model_inventory")
    if not isinstance(model_reference, dict):
        raise RuntimeError("job plan does not bind the model inventory")
    model_inventory = Path(model_reference["remote_path"])
    if not model_inventory.is_file() or sha256_file(model_inventory) != model_reference["sha256"]:
        raise RuntimeError(f"missing or hash-drifted sk2 model inventory: {model_inventory}")
    frozen_model = json.loads(model_inventory.read_text(encoding="utf-8"))
    model_root = Path(frozen_model["model"])
    if str(model_root.resolve()) != str(Path(plan["model"]).resolve()):
        raise RuntimeError("model path differs from its frozen inventory")
    for name, reference in frozen_model["files"].items():
        path = model_root / name
        if (
            not path.is_file()
            or path.stat().st_size != int(reference["bytes"])
            or sha256_file(path) != reference["sha256"]
        ):
            raise RuntimeError(f"model snapshot file missing or hash-drifted: {path}")
    runtime = frozen_model["runtime"]
    import torch
    installed = {
        "torch": str(torch.__version__),
        "transformers": importlib.metadata.version("transformers"),
        "peft": importlib.metadata.version("peft"),
        "accelerate": importlib.metadata.version("accelerate"),
    }
    if installed != runtime:
        raise RuntimeError(f"runtime drift: installed={installed} frozen={runtime}")
    implementation_files = plan.get("implementation_files")
    if not isinstance(implementation_files, dict) or not implementation_files:
        raise RuntimeError("job plan does not bind implementation files")
    for relative, reference in implementation_files.items():
        path = Path(reference["remote_path"])
        if not path.is_file() or sha256_file(path) != reference["sha256"]:
            raise RuntimeError(f"implementation file missing or hash-drifted: {relative} ({path})")
    manifest = json.loads(Path(plan["dataset_manifest"]["remote_path"]).read_text(encoding="utf-8"))
    root = Path(plan["dataset_manifest"]["remote_path"]).parent
    required = {
        "protocols.json", "inventory.json",
        *(f"{level}_train.jsonl" for level in ("R1", "R2", "R3")),
        *(f"{level}_eval.jsonl" for level in ("R1", "R2", "R3")),
    }
    for name in required:
        path = root / name
        expected = manifest["artifacts"].get(name, {}).get("sha256")
        if not expected or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"required dataset artifact missing or hash-drifted: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-parallel-gpu-jobs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    assert_sk2()
    args = parse_args()
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not plan.get("sk3_forbidden") or "skampere2" not in plan.get("host_allowlist", []):
        raise ValueError("plan lacks the sk2-only safety declaration")
    validate_frozen_inputs(plan)
    repo = Path(plan["repo"]).resolve()
    run_root = repo / "outputs/lexicon/similarity_lora_v1"
    logs = run_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (run_root / "reports").mkdir(parents=True, exist_ok=True)
    (run_root / "predictions").mkdir(parents=True, exist_ok=True)
    (run_root / "adapters").mkdir(parents=True, exist_ok=True)
    status_path = run_root / "status.json"
    jobs = {job["job_id"]: job for job in plan["jobs"]}
    status: dict[str, dict[str, Any]] = {
        job_id: {"state": "pending", "updated_at": now()} for job_id in jobs
    }
    for job_id, job in jobs.items():
        if all(Path(path).exists() for path in job["outputs"]):
            status[job_id] = {"state": "complete", "updated_at": now(), "reused_outputs": True}
    running: dict[str, tuple[subprocess.Popen[bytes], Any]] = {}
    atomic_json(status_path, {"schema_version": "gemma4-similarity-sk2-run-status-v1", "jobs": status})
    while True:
        for job_id, (process, handle) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            state = "complete" if code == 0 and all(Path(path).exists() for path in jobs[job_id]["outputs"]) else "failed"
            status[job_id] = {**status[job_id], "state": state, "returncode": code, "completed_at": now(), "updated_at": now()}
            del running[job_id]
        for job_id, job in jobs.items():
            if status[job_id]["state"] != "pending":
                continue
            dependency_states = [status[dependency]["state"] for dependency in job["depends_on"]]
            if any(state in {"failed", "blocked"} for state in dependency_states):
                status[job_id] = {"state": "blocked", "updated_at": now(), "reason": "failed_dependency"}
                continue
            if not all(state == "complete" for state in dependency_states):
                continue
            gpu = job.get("gpu")
            active_gpu_jobs = sum(jobs[running_id].get("gpu") is not None for running_id in running)
            if gpu is not None:
                if active_gpu_jobs >= args.max_parallel_gpu_jobs:
                    continue
                if any(jobs[running_id].get("gpu") == gpu for running_id in running):
                    continue
                if not gpu_idle(int(gpu)):
                    continue
            log_path = logs / f"{job_id}.log"
            handle = log_path.open("ab")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(repo)
            environment["TOKENIZERS_PARALLELISM"] = "false"
            environment["HF_HOME"] = "/lfs/skampere2/0/alexspan/.cache/huggingface"
            environment["XDG_CACHE_HOME"] = "/lfs/skampere2/0/alexspan/.cache"
            environment["TMPDIR"] = "/lfs/skampere2/0/alexspan/tmp"
            if gpu is not None:
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                job["argv"], cwd=repo, env=environment, stdout=handle, stderr=subprocess.STDOUT
            )
            running[job_id] = (process, handle)
            status[job_id] = {
                "state": "running", "pid": process.pid, "gpu": gpu,
                "started_at": now(), "updated_at": now(), "log": str(log_path),
            }
        atomic_json(status_path, {"schema_version": "gemma4-similarity-sk2-run-status-v1", "updated_at": now(), "jobs": status})
        terminal = {"complete", "failed", "blocked"}
        if not running and all(row["state"] in terminal for row in status.values()):
            break
        time.sleep(max(5, args.poll_seconds))
    failures = [job_id for job_id, row in status.items() if row["state"] != "complete"]
    if failures:
        raise SystemExit(f"DAG incomplete: {failures}")


if __name__ == "__main__":
    main()
