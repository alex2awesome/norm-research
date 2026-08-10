#!/usr/bin/env python3
"""Seal a single selected Nemotron adapter's external-dev decision."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import sha256_file


def _load(value: str) -> tuple[Path, dict[str, Any]]:
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--predeclaration", required=True)
    parser.add_argument("--internal-selection", required=True)
    parser.add_argument("--dev-report", required=True)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--run-record", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)

    lock_path, lock = _load(args.predeclaration)
    selection_path, selection = _load(args.internal_selection)
    report_path, report = _load(args.dev_report)
    queue_path, queue = _load(args.queue)
    run_path, run = _load(args.run_record)
    if any(value.get("task") != args.task for value in (lock, selection, report, queue)):
        raise ValueError("task mismatch across external-dev decision evidence")
    if (
        selection.get("status") != "FROZEN_SELECTED_BEFORE_EXTERNAL_DEV"
        or selection.get("external_dev_consumed") is not False
        or selection.get("external_test_consumed") is not False
    ):
        raise ValueError("adapter selection was not frozen before external dev")
    if report.get("split") != "dev" or report.get("selection_role") != "promotion_dev":
        raise ValueError("report is not external promotion dev")
    if (
        queue.get("external_dev_audit", {}).get("external_test_consumed") is not False
        or queue.get("external_dev_audit", {}).get("foreign_task_or_split_rows") != 0
        or queue["command"][queue["command"].index("--split") + 1] != "dev"
    ):
        raise ValueError("evaluation queue was not isolated to dev")
    if (
        run.get("status") != "COMPLETED"
        or run.get("external_test_consumed") is not False
        or run.get("queue_sha256") != sha256_file(queue_path)
        or run.get("output_sha256") != sha256_file(report_path)
    ):
        raise ValueError("external-dev run record is incomplete or inconsistent")
    policy = lock.get("external_promotion") or {}
    gate = report.get("promotion_gate") or {}
    minimum_gain = float(policy["minimum_dev_exact_recall_at_50_gain"])
    if float(gate.get("minimum_gain")) != minimum_gain:
        raise ValueError("report gate differs from predeclaration")
    expected_pass = (
        float(gate["actual_gain"]) >= minimum_gain
        and gate.get("secondary_passed") is True
    )
    if gate.get("passed") is not expected_pass:
        raise ValueError("reported gate decision is internally inconsistent")

    passed = bool(gate["passed"])
    payload = {
        "schema_version": "silver-match-v3-nemotron-external-dev-decision-v1",
        "status": "FROZEN_PROMOTE_PENDING_TEST" if passed else "FROZEN_REJECT_RETAIN_BASE",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "selected_variant": selection["selected_variant"],
        "decision": (
            "PROMOTE_SELECTED_ADAPTER_PENDING_FROZEN_TEST_ONCE"
            if passed
            else "REJECT_SELECTED_ADAPTER_RETAIN_FROZEN_BASE"
        ),
        "external_dev_gate": {
            "passed": passed,
            "minimum_exact_recall_at_50_gain": minimum_gain,
            "actual_exact_recall_at_50_gain": gate["actual_gain"],
            "recall_at_80_non_decrease_passed": gate["secondary_passed"],
            "before": report["before"]["exact"],
            "after": report["after"]["exact"],
        },
        "bindings": {
            "predeclaration": _artifact(lock_path),
            "internal_selection": _artifact(selection_path),
            "dev_report": _artifact(report_path),
            "queue": _artifact(queue_path),
            "run_record": _artifact(run_path),
        },
        "external_test": {
            "status": "SEALED_UNCONSUMED",
            "consumed_during_training": False,
            "consumed_during_internal_selection": False,
            "consumed_during_external_dev": False,
            "next_action": (
                "one hash-frozen evaluation exactly once"
                if passed
                else "DO_NOT_OPEN_FOR_REJECTED_ADAPTER"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "sha256": sha256_file(output), **payload}, sort_keys=True))


if __name__ == "__main__":
    main()
