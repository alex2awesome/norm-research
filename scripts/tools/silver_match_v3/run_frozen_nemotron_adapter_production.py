#!/usr/bin/env python3
"""Validate or run a frozen direct Nemotron-LoRA production queue."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audit_candidate_outputs import audit_candidates
from .common import sha256_file
from .gpu_host_policy import validate_gpu_indices_for_host, validate_launch_gpus


def verify_artifact(value: dict[str, Any]) -> None:
    path = Path(value["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(value["size_bytes"])
        or sha256_file(path) != value["sha256"]
    ):
        raise ValueError(f"frozen artifact changed: {path}")


def validate(plan: dict[str, Any]) -> None:
    if (
        plan.get("schema_version")
        != "silver-match-v3-frozen-nemotron-production-queue-v1"
        or plan.get("status") != "FROZEN_READY_NOT_LAUNCHED"
        or (plan.get("safety") or {}).get("external_test_consumed") is not False
        or (plan.get("safety") or {}).get("external_labels_opened") is not False
        or int(plan.get("expected_k", -1)) != 50
    ):
        raise ValueError("queue schema, status, depth, or safety policy is invalid")
    if not re.fullmatch(
        str((plan.get("execution") or {}).get("host_pattern") or ""), platform.node()
    ):
        raise ValueError(f"queue cannot run on this host: {platform.node()}")
    for value in (plan.get("bindings") or {}).values():
        verify_artifact(value)
    for package, expected in (plan.get("runtime") or {}).get("packages", {}).items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise ValueError(f"runtime package changed: {package} {actual} != {expected}")
    inventory_path = Path(plan["bindings"]["model_inventory"]["path"])
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    model = Path(plan["model"]["path"])
    for value in inventory.get("files") or []:
        path = model / str(value["relative_path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(value["size_bytes"])
            or sha256_file(path) != value["sha256"]
        ):
            raise ValueError(f"base model changed: {path}")
    selection = json.loads(
        Path(plan["bindings"]["selection"]["path"]).read_text(encoding="utf-8")
    )
    if (
        selection.get("status") != "SELECTED_FOR_PRODUCTION_RETRIEVAL"
        or selection.get("frozen_external_test_consumed") is not False
        or ((selection.get("chosen") or {}).get("external_dev_metrics") or {})
        .get("promotion_gate", {})
        .get("passed")
        is not True
    ):
        raise ValueError("promoted selection no longer validates")


def candidate_valid(plan: dict[str, Any]) -> bool:
    candidate = Path(plan["outputs"]["candidate"])
    if not candidate.is_file():
        return False
    try:
        audit_candidates(
            manifest_path=Path(plan["bindings"]["manifest"]["path"]),
            corpus=plan["corpus"],
            candidate_paths=[candidate],
            expected_k=plan["expected_k"],
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return False
    return True


def run(plan: dict[str, Any], queue_path: Path) -> None:
    outputs = {name: Path(path) for name, path in plan["outputs"].items()}
    if outputs["run_record"].exists() or outputs["audit"].exists():
        raise FileExistsError("sealed run record or audit already exists")
    execution = plan["execution"]
    gpu = int(execution["gpu_index"])
    validate_gpu_indices_for_host([gpu])
    gpu_launch_guard = validate_launch_gpus([gpu])

    outputs["candidate"].parent.mkdir(parents=True, exist_ok=True)
    outputs["log"].parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({str(k): str(v) for k, v in plan["environment"].items()})
    started = datetime.now(timezone.utc)
    already_valid = candidate_valid(plan)
    if not already_valid:
        with outputs["log"].open("ab") as log_handle:
            log_handle.write(
                (f"\n=== frozen queue attempt {started.isoformat()} ===\n").encode()
            )
            log_handle.flush()
            result = subprocess.run(
                plan["command"],
                cwd=execution["repo_root"],
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"retrieval failed closed with return code {result.returncode}; "
                f"partial output remains resume-eligible and unsealed"
            )
    elif not outputs["log"].exists():
        outputs["log"].write_text(
            "retrieval candidate was already exact-valid; resumed at audit stage\n",
            encoding="utf-8",
        )
    if not candidate_valid(plan):
        raise ValueError("retrieval command returned success without exact audited coverage")
    subprocess.run(
        plan["audit_command"],
        cwd=execution["repo_root"],
        env=environment,
        check=True,
    )
    completed = datetime.now(timezone.utc)
    candidate = outputs["candidate"]
    run_record = {
        "schema_version": "silver-match-v3-nemotron-production-run-v1",
        "status": "COMPLETED_EXACT_K50",
        "task": plan["task"],
        "corpus": plan["corpus"],
        "expected_rows": plan["expected_rows"],
        "expected_k": plan["expected_k"],
        "host": platform.node(),
        "physical_gpu": gpu,
        "gpu_launch_guard": gpu_launch_guard,
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "elapsed_seconds": (completed - started).total_seconds(),
        "queue": str(queue_path),
        "queue_sha256": sha256_file(queue_path),
        "candidate": str(candidate),
        "candidate_sha256": sha256_file(candidate),
        "candidate_meta_sha256": sha256_file(outputs["candidate_meta"]),
        "audit_sha256": sha256_file(outputs["audit"]),
        "log_sha256": sha256_file(outputs["log"]),
        "external_labels_opened": False,
        "external_test_consumed": False,
    }
    outputs["run_record"].write_text(
        json.dumps(run_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_record, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    queue_path = Path(args.queue).resolve()
    plan = json.loads(queue_path.read_text(encoding="utf-8"))
    validate(plan)
    if not args.run:
        print(
            json.dumps(
                {
                    "status": "VALIDATED_NOT_LAUNCHED",
                    "queue": str(queue_path),
                    "queue_sha256": sha256_file(queue_path),
                    "expected_rows": plan["expected_rows"],
                    "expected_k": plan["expected_k"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    run(plan, queue_path)


if __name__ == "__main__":
    main()
