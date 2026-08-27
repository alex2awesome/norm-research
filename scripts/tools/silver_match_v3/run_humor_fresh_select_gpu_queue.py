#!/usr/bin/env python3
"""Validate and execute a frozen Humor fresh-select direct-batch queue.

Validation is the default.  ``--run`` executes one frozen stage at a time,
records append-only launch/completion evidence, and skips only outputs that pass
their module-specific completeness audit.  Partial Gemma JSONL outputs are
resumed by the frozen ``--resume`` command; inconsistent sealed outputs fail
closed.
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


SCHEMA = "silver-match-v3-humor-fresh-select-gpu-queue-v1"
GPU_MODULES = {
    "scripts.tools.silver_match_v3.adjudicate_gemma",
    "scripts.tools.silver_match_v3.verify_gemma",
}
CPU_MODULES = {"scripts.tools.silver_match_v3.build_two_order_consensus_proposals"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _verify_artifact(value: dict[str, Any]) -> None:
    path = Path(str(value["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if "bytes" in value and path.stat().st_size != int(value["bytes"]):
        raise ValueError(f"frozen artifact size mismatch: {path}")
    if sha256_file(path) != str(value["sha256"]):
        raise ValueError(f"frozen artifact hash mismatch: {path}")


def _verify_nested_artifacts(value: Any) -> None:
    if isinstance(value, dict):
        if {"path", "sha256"} <= set(value):
            _verify_artifact(value)
            return
        for child in value.values():
            _verify_nested_artifacts(child)
    elif isinstance(value, list):
        for child in value:
            _verify_nested_artifacts(child)


def _arg(argv: list[str], name: str) -> str:
    positions = [index for index, value in enumerate(argv) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"command must contain exactly one {name}")
    return argv[positions[0] + 1]


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _inference_complete(cell: dict[str, Any]) -> bool:
    argv = [str(value) for value in cell["argv"]]
    output = Path(_arg(argv, "--output"))
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    if not output.exists() and not meta_path.exists():
        return False
    if meta_path.exists() and not output.exists():
        raise ValueError(f"inference metadata exists without output: {meta_path}")
    if not meta_path.exists():
        # A direct-batch process may have appended a valid prefix before an
        # interruption.  The frozen command is explicitly resumable.
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("output_sha256") != sha256_file(output):
        raise ValueError(f"sealed inference output hash mismatch: {output}")
    expected = int(meta.get("eligible_count", -1))
    if expected < 0 or _line_count(output) != expected:
        raise ValueError(f"sealed inference output lacks exact coverage: {output}")
    if int(meta.get("invalid_count", 0)) != 0:
        raise ValueError(f"sealed inference contains invalid outputs: {output}")
    if meta.get("model") != _arg(argv, "--model"):
        raise ValueError(f"sealed inference model mismatch: {output}")
    if meta.get("order_mode") != _arg(argv, "--order-mode"):
        raise ValueError(f"sealed inference order mismatch: {output}")
    frozen_prompts = [
        Path(_arg(argv, "--prompt")),
        *[
            Path(argv[index + 1])
            for index, value in enumerate(argv)
            if value == "--prompt-addon"
        ],
    ]
    observed = meta.get("prompt_component_sha256") or {}
    if {
        str(path): sha256_file(path) for path in frozen_prompts
    } != {str(key): str(value) for key, value in observed.items()}:
        raise ValueError(f"sealed inference prompt identity mismatch: {output}")
    return True


def _consensus_complete(cell: dict[str, Any]) -> bool:
    argv = [str(value) for value in cell["argv"]]
    output = Path(_arg(argv, "--output"))
    report_path = output.with_suffix(output.suffix + ".report.json")
    if not output.exists() and not report_path.exists():
        return False
    if not output.exists() or not report_path.exists():
        raise ValueError(f"partial consensus output fails closed: {output}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    identity = report.get("output") or {}
    if identity.get("sha256") != sha256_file(output):
        raise ValueError(f"consensus output hash mismatch: {output}")
    if int(report.get("consensus_match_count", -1)) != _line_count(output):
        raise ValueError(f"consensus output count mismatch: {output}")
    for name in ("original", "hashed"):
        path = Path(_arg(argv, f"--{name}"))
        observed = (report.get("inputs") or {}).get(name) or {}
        if observed.get("sha256") != sha256_file(path):
            raise ValueError(f"consensus input hash mismatch: {path}")
    return True


def _cell_complete(cell: dict[str, Any]) -> bool:
    module = str(cell["module"])
    if module in GPU_MODULES:
        return _inference_complete(cell)
    if module in CPU_MODULES:
        return _consensus_complete(cell)
    raise ValueError(f"unsupported frozen module: {module}")


def validate_queue(queue: dict[str, Any]) -> None:
    if (
        queue.get("schema_version") != SCHEMA
        or queue.get("status") != "FROZEN_BEFORE_FRESH_SELECT_MODEL_PREDICTIONS"
        or queue.get("task") != "humor"
        or queue.get("backend") != "direct_vllm_batch"
        or queue.get("fresh_select_truth_read") is not False
        or queue.get("permanent_blind_consumed") is not False
    ):
        raise ValueError("unsupported, unfrozen, or contaminated Humor queue")
    purity = queue.get("backend_purity") or {}
    if purity.get("openai_server_forbidden") is not True or purity.get(
        "never_mix_backends_within_cell"
    ) is not True:
        raise ValueError("queue lacks backend-purity guarantees")
    repo = Path(str(queue["repo"]))
    if not repo.is_dir():
        raise FileNotFoundError(repo)
    _verify_artifact(queue["python"])
    _verify_nested_artifacts(queue["inputs"])
    model = queue["model_snapshot"]
    model_root = Path(str(model["path"]))
    if model_root.name != str(model["revision"]):
        raise ValueError("model snapshot revision/path mismatch")
    for value in model["identity_files"].values():
        _verify_artifact(value)
    for name, size in model["weight_shard_bytes"].items():
        path = model_root / str(name)
        if not path.is_file() or path.stat().st_size != int(size):
            raise ValueError(f"model weight shard identity mismatch: {path}")

    policy = queue.get("gpu_policy") or {}
    gpu_ids = [int(value) for value in policy.get("physical_gpu_ids") or []]
    validate_gpu_indices_for_host(gpu_ids, hostname=socket.gethostname())
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("queue has invalid physical GPU IDs")
    if int(policy.get("maximum_concurrent_gpus", 0)) != len(gpu_ids):
        raise ValueError("queue GPU concurrency does not match frozen IDs")
    if policy.get("global_gpu_count_gate_applied", False) is not False:
        raise ValueError("queue must not enable global GPU-count gating")

    stages = queue.get("stages") or []
    if not stages:
        raise ValueError("queue has no stages")
    seen: set[str] = set()
    for stage in stages:
        name = str(stage.get("stage") or "")
        if not name or name in seen:
            raise ValueError(f"missing/duplicate stage: {name!r}")
        dependencies = [str(value) for value in stage.get("depends_on") or []]
        if any(value not in seen for value in dependencies):
            raise ValueError(f"stage dependency is absent or not earlier: {name}")
        cells = stage.get("cells") or []
        if not cells or (not stage.get("parallel") and len(cells) != 1):
            raise ValueError(f"invalid cell/parallel declaration: {name}")
        stage_gpus: set[int] = set()
        for cell in cells:
            module = str(cell.get("module") or "")
            argv = [str(value) for value in cell.get("argv") or []]
            if module not in GPU_MODULES | CPU_MODULES or not argv:
                raise ValueError(f"unsupported or empty cell in {name}")
            if any(value in argv for value in ("--api-base-url", "--api-key-file")):
                raise ValueError(f"API/server argument contaminates direct-batch cell: {name}")
            gpu = cell.get("cuda_visible_devices")
            if module in GPU_MODULES:
                if gpu is None or int(gpu) not in gpu_ids:
                    raise ValueError(f"GPU cell is outside frozen physical IDs: {name}")
                if int(gpu) in stage_gpus:
                    raise ValueError(f"two parallel cells share a physical GPU: {name}")
                stage_gpus.add(int(gpu))
                if "--resume" not in argv:
                    raise ValueError(f"inference cell lacks exact-resume flag: {name}")
            elif gpu is not None:
                raise ValueError(f"CPU consensus cell declares a GPU: {name}")
            _arg(argv, "--output")
        if len(stage_gpus) > int(policy["maximum_concurrent_gpus"]):
            raise ValueError(f"stage exceeds frozen GPU concurrency: {name}")
        seen.add(name)


def _gpu_is_free(gpu: int) -> bool:
    gpu_rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
    )
    uuid_by_index = {
        int(index.strip()): uuid.strip()
        for line in gpu_rows.splitlines()
        if line.strip()
        for index, uuid in [line.split(",", 1)]
    }
    if gpu not in uuid_by_index:
        raise ValueError(f"physical GPU does not exist: {gpu}")
    apps = subprocess.check_output(
        [
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    return not any(
        line.split(",", 1)[0].strip() == uuid_by_index[gpu]
        for line in apps.splitlines()
        if line.strip() and "," in line
    )


def _wait_for_gpus(gpus: set[int], poll_seconds: int) -> None:
    while not all(_gpu_is_free(gpu) for gpu in gpus):
        print(
            json.dumps(
                {"status": "WAITING_FOR_GPU", "gpus": sorted(gpus), "at": _utc_now()},
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(poll_seconds)


def _run_cell(
    *,
    queue: dict[str, Any],
    queue_sha256: str,
    stage_name: str,
    cell_index: int,
    cell: dict[str, Any],
    launch_root: Path,
) -> subprocess.Popen[str]:
    python = str(queue["python"]["path"])
    module = str(cell["module"])
    argv = [str(value) for value in cell["argv"]]
    command = [python, "-u", "-m", module, *argv]
    cell_name = f"{stage_name}.cell-{cell_index:02d}"
    log_path = launch_root / "logs" / f"{cell_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    # Preserve an explicitly frozen runtime compatibility overlay while keeping
    # the queue repo first for the audited executor modules.  The previous
    # assignment silently discarded PYTHONPATH inherited by the launch wrapper,
    # which made an append-only Transformers compatibility projection impossible.
    inherited_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(queue["repo"]), inherited_pythonpath) if value
    )
    gpu = cell.get("cuda_visible_devices")
    if gpu is None:
        environment.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_handle = log_path.open("x", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(queue["repo"]),
        env=environment,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    process._silver_log_handle = log_handle  # type: ignore[attr-defined]
    _write_new_json(
        launch_root / "cells" / f"{cell_name}.launched.json",
        {
            "schema_version": "silver-match-v3-humor-queue-cell-launch-v1",
            "queue_sha256": queue_sha256,
            "stage": stage_name,
            "cell_index": cell_index,
            "module": module,
            "argv": argv,
            "command": command,
            "cuda_visible_devices": gpu,
            "pid": process.pid,
            "host": socket.gethostname(),
            "started_at": _utc_now(),
            "log": str(log_path),
        },
    )
    return process


def run_queue(queue: dict[str, Any], queue_path: Path, *, poll_seconds: int) -> Path:
    queue_sha = sha256_file(queue_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    launch_root = queue_path.parent / "launches" / f"{stamp}.{os.getpid()}"
    launch_root.mkdir(parents=True, exist_ok=False)
    _write_new_json(
        launch_root / "RUN.json",
        {
            "schema_version": "silver-match-v3-humor-queue-run-v1",
            "queue": str(queue_path),
            "queue_sha256": queue_sha,
            "host": socket.gethostname(),
            "runner_pid": os.getpid(),
            "started_at": _utc_now(),
            "resume_policy": "skip_only_module_specific_exact_complete_outputs",
        },
    )
    for stage in queue["stages"]:
        stage_name = str(stage["stage"])
        incomplete = [
            (index, cell)
            for index, cell in enumerate(stage["cells"])
            if not _cell_complete(cell)
        ]
        if not incomplete:
            _write_new_json(
                launch_root / "stages" / f"{stage_name}.skipped.json",
                {"stage": stage_name, "status": "SKIPPED_EXACT_COMPLETE", "at": _utc_now()},
            )
            continue
        gpus = {
            int(cell["cuda_visible_devices"])
            for _, cell in incomplete
            if cell.get("cuda_visible_devices") is not None
        }
        if gpus:
            _wait_for_gpus(gpus, poll_seconds)
            validate_launch_gpus(sorted(gpus), hostname=socket.gethostname())
        processes = [
            (
                index,
                cell,
                _run_cell(
                    queue=queue,
                    queue_sha256=queue_sha,
                    stage_name=stage_name,
                    cell_index=index,
                    cell=cell,
                    launch_root=launch_root,
                ),
            )
            for index, cell in incomplete
        ]
        failures = []
        for index, cell, process in processes:
            returncode = process.wait()
            process._silver_log_handle.close()  # type: ignore[attr-defined]
            complete = returncode == 0 and _cell_complete(cell)
            _write_new_json(
                launch_root / "cells" / f"{stage_name}.cell-{index:02d}.completed.json",
                {
                    "stage": stage_name,
                    "cell_index": index,
                    "pid": process.pid,
                    "returncode": returncode,
                    "exact_complete": complete,
                    "completed_at": _utc_now(),
                },
            )
            if not complete:
                failures.append((index, returncode))
        if failures:
            raise RuntimeError(f"stage failed closed: {stage_name}: {failures}")
        _write_new_json(
            launch_root / "stages" / f"{stage_name}.completed.json",
            {"stage": stage_name, "status": "EXACT_COMPLETE", "at": _utc_now()},
        )
    _write_new_json(
        launch_root / "COMPLETE.json",
        {
            "schema_version": "silver-match-v3-humor-queue-run-complete-v1",
            "queue_sha256": queue_sha,
            "completed_at": _utc_now(),
            "all_cells_exact_complete": all(
                _cell_complete(cell)
                for stage in queue["stages"]
                for cell in stage["cells"]
            ),
        },
    )
    return launch_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args()
    if not 5 <= args.poll_seconds <= 60:
        parser.error("--poll-seconds must be in [5, 60]")
    queue_path = Path(args.queue).resolve()
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    validate_queue(queue)
    result: dict[str, Any] = {
        "status": "VALIDATED",
        "queue": str(queue_path),
        "queue_sha256": sha256_file(queue_path),
        "stage_count": len(queue["stages"]),
    }
    if args.run:
        result.update(
            {
                "status": "COMPLETE",
                "launch_root": str(
                    run_queue(queue, queue_path, poll_seconds=args.poll_seconds)
                ),
            }
        )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
