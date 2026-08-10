#!/usr/bin/env python3
"""Launch exact frozen Codex passes after specified competing jobs have exited."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _running_blockers(needles: list[str]) -> list[str]:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=True
    )
    blockers = []
    for line in completed.stdout.splitlines():
        if all(needle in line for needle in needles) and (
            "codex exec" in line or "run_codex_pack_labels" in line
        ):
            blockers.append(line.strip())
    return blockers


def _validate_refs(value: Any) -> None:
    if isinstance(value, dict):
        if set(value) >= {"path", "sha256"}:
            path = Path(str(value["path"])).resolve()
            if not path.is_file() or sha256_file(path) != value["sha256"]:
                raise ValueError(f"frozen artifact missing or drifted: {path}")
        for child in value.values():
            _validate_refs(child)
    elif isinstance(value, list):
        for child in value:
            _validate_refs(child)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--pass-name", action="append", required=True)
    parser.add_argument("--wait-needle", action="append", default=[])
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--receipt-root", required=True)
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        parser.error("--poll-seconds must be in [1, 60]")
    plan_path = Path(args.plan).resolve()
    receipt_root = Path(args.receipt_root).resolve()
    launch_receipt = receipt_root / "LAUNCH_RECEIPT.json"
    completion_receipt = receipt_root / "COMPLETION_RECEIPT.json"
    if launch_receipt.exists() or completion_receipt.exists():
        raise FileExistsError("refusing to reuse a frozen-pass launch receipt")
    if sha256_file(plan_path) != args.expected_plan_sha256:
        raise ValueError("execution plan hash drift")
    plan = json.loads(plan_path.read_text())
    if (
        plan.get("schema_version")
        != "silver-match-v3-independent-codex-label-execution-plan-v1"
        or plan.get("status") != "FROZEN_BEFORE_EITHER_INDEPENDENT_LABEL_PASS"
        or set(args.pass_name) != set(plan.get("commands") or {})
        or len(args.pass_name) != len(set(args.pass_name))
    ):
        raise ValueError("invalid frozen pass universe")
    _validate_refs(plan.get("implementation") or {})
    _validate_refs(plan.get("inputs") or {})

    while _running_blockers(args.wait_needle):
        time.sleep(args.poll_seconds)
    receipt_root.mkdir(parents=True, exist_ok=True)
    launch = {
        "schema_version": "silver-match-v3-frozen-codex-launch-receipt-v1",
        "status": "LAUNCHED_AFTER_COMPETING_QUOTA_CLEARED",
        "launched_at": datetime.now(timezone.utc).isoformat(),
        "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "passes": args.pass_name,
        "wait_needles": args.wait_needle,
        "blockers_at_launch": _running_blockers(args.wait_needle),
    }
    launch_receipt.write_text(json.dumps(launch, indent=2, sort_keys=True) + "\n")
    processes: dict[str, subprocess.Popen[bytes]] = {}
    logs = {}
    for name in args.pass_name:
        command = plan["commands"][name]
        environment = os.environ.copy()
        environment.update(command.get("environment") or {})
        log_path = receipt_root / f"pass_{name}.runtime.log"
        log = log_path.open("wb")
        logs[name] = log
        processes[name] = subprocess.Popen(
            command["argv"],
            cwd=command["cwd"],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    returncodes = {name: process.wait() for name, process in processes.items()}
    for log in logs.values():
        log.close()
    completion = {
        "schema_version": "silver-match-v3-frozen-codex-completion-receipt-v1",
        "status": "COMPLETE" if all(code == 0 for code in returncodes.values()) else "FAILED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "returncodes": returncodes,
        "logs": {
            name: {
                "path": str(receipt_root / f"pass_{name}.runtime.log"),
                "sha256": sha256_file(receipt_root / f"pass_{name}.runtime.log"),
            }
            for name in args.pass_name
        },
    }
    completion_receipt.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )
    if any(code != 0 for code in returncodes.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
