"""Hard host-specific GPU safety policy for silver-match-v3 launchers.

This module is deliberately small and dependency-free so queue runners can
fail closed before setting ``CUDA_VISIBLE_DEVICES``.  Historical frozen queue
files remain immutable; a runtime launcher must apply this policy even when an
older queue names a now-prohibited device.
"""

from __future__ import annotations

import socket
import subprocess
from collections.abc import Iterable, Mapping
from typing import Any


SK3_ALLOWED_GPU_INDICES = frozenset({0, 5, 6, 7})
SK3_PROHIBITED_GPU_INDICES = frozenset({1, 2, 3, 4})


def is_sk3_host(hostname: str | None = None) -> bool:
    """Return whether *hostname* denotes the sk3 / skampere3 server."""

    value = (hostname or socket.gethostname()).split(".", 1)[0].lower()
    return value in {"sk3", "skampere3"} or value.startswith("skampere3-")


def validate_gpu_indices_for_host(
    indices: Iterable[int],
    *,
    hostname: str | None = None,
) -> tuple[int, ...]:
    """Validate an explicit GPU selection and return it as an integer tuple.

    On sk3, devices 1--4 are a permanent fail-closed prohibition.  No queue,
    fallback, or explicit command-line selection may override it.
    """

    selected = tuple(int(value) for value in indices)
    if not selected:
        raise ValueError("GPU selection must not be empty")
    if len(selected) != len(set(selected)):
        raise ValueError(f"GPU selection contains duplicates: {selected}")
    if is_sk3_host(hostname):
        prohibited = sorted(set(selected) & SK3_PROHIBITED_GPU_INDICES)
        outside_allowlist = sorted(set(selected) - SK3_ALLOWED_GPU_INDICES)
        if prohibited or outside_allowlist:
            raise ValueError(
                "sk3 GPU policy violation: "
                f"selected={list(selected)} prohibited={prohibited} "
                f"allowed={sorted(SK3_ALLOWED_GPU_INDICES)}"
            )
    return selected


def filter_gpu_rows_for_host(
    rows: Iterable[Mapping[str, Any]], *, hostname: str | None = None
) -> list[dict[str, Any]]:
    """Remove devices that this pipeline may not select on the current host."""

    materialized = [dict(row) for row in rows]
    if not is_sk3_host(hostname):
        return materialized
    return [
        row
        for row in materialized
        if int(row["index"]) in SK3_ALLOWED_GPU_INDICES
    ]


def validate_launch_gpus(
    indices: Iterable[int],
    *,
    hostname: str | None = None,
    maximum_idle_memory_mib: int = 2048,
) -> dict[str, Any]:
    """Fail unless every target is process-free, idle, and host-allowed.

    This is an immediate pre-launch guard.  Long-lived watchers should still
    require stable repeated samples and re-run this check just before spawn.
    GPU-count and projected-owner-count gates are intentionally not part of
    scheduling policy on sk1, sk2, or sk3.
    """

    selected = validate_gpu_indices_for_host(indices, hostname=hostname)
    gpu_output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    gpus: list[dict[str, Any]] = []
    for line in gpu_output.splitlines():
        if not line.strip():
            continue
        index, uuid, memory, utilization = [
            value.strip() for value in line.split(",", 3)
        ]
        gpus.append(
            {
                "index": int(index),
                "uuid": uuid,
                "memory_used_mib": int(memory),
                "utilization_percent": int(utilization),
            }
        )
    by_index = {row["index"]: row for row in gpus}
    missing = sorted(set(selected) - set(by_index))
    if missing:
        raise ValueError(f"selected GPUs do not exist: {missing}")

    app_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if app_result.returncode:
        raise RuntimeError(f"unable to query GPU processes: {app_result.stderr.strip()}")
    uuid_to_index = {row["uuid"]: row["index"] for row in gpus}
    pid_to_gpu = {
        int(pid.strip()): uuid_to_index[uuid.strip()]
        for line in app_result.stdout.splitlines()
        if line.strip() and "," in line
        for uuid, pid in [line.split(",", 1)]
        if uuid.strip() in uuid_to_index and pid.strip().isdigit()
    }
    target_processes = {
        pid: gpu for pid, gpu in pid_to_gpu.items() if gpu in set(selected)
    }
    nonidle = [
        by_index[gpu]
        for gpu in selected
        if by_index[gpu]["memory_used_mib"] > maximum_idle_memory_mib
        or by_index[gpu]["utilization_percent"] != 0
    ]
    if target_processes or nonidle:
        raise RuntimeError(
            "selected GPUs are not genuinely idle: "
            f"processes={target_processes} nonidle={nonidle}"
        )

    return {
        "host": hostname or socket.gethostname(),
        "selected_gpu_indices": list(selected),
        "selected_gpu_rows": [by_index[gpu] for gpu in selected],
        "target_processes": target_processes,
        "gpu_count_gate_applied": False,
        "projected_owner_count_check_applied": False,
    }
