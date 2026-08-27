#!/usr/bin/env python3
"""Freeze a two-run Nemotron dev confirmation before any test is opened."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import sha256_file


def _load(path: str) -> tuple[Path, dict[str, Any]]:
    value = Path(path).resolve()
    if not value.is_file():
        raise FileNotFoundError(value)
    return value, json.loads(value.read_text(encoding="utf-8"))


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _validate_dev(
    report: dict[str, Any], task: str, minimum_gain: float
) -> dict[str, Any]:
    if (
        report.get("task") != task
        or report.get("split") != "dev"
        or report.get("selection_role") != "promotion_dev"
    ):
        raise ValueError("report is not the requested task's promotion dev split")
    gate = report.get("promotion_gate") or {}
    gain = float(gate.get("actual_gain", float("-inf")))
    before80 = float(report["before"]["exact"]["recall_at_80"])
    after80 = float(report["after"]["exact"]["recall_at_80"])
    if (
        gate.get("passed") is not True
        or gain < minimum_gain
        or gate.get("secondary_passed") is not True
        or after80 < before80
    ):
        raise ValueError("one run does not satisfy the predeclared joint dev gate")
    return {
        "exact_recall_at_50_before": report["before"]["exact"]["recall_at_50"],
        "exact_recall_at_50_after": report["after"]["exact"]["recall_at_50"],
        "exact_recall_at_50_gain": gain,
        "exact_recall_at_80_before": before80,
        "exact_recall_at_80_after": after80,
        "mrr_before": report["before"]["exact"]["mrr"],
        "mrr_after": report["after"]["exact"]["mrr"],
        "external_dev_labels_sha256": report["input_hashes"]["labels"],
        "adapter_hashes": report["input_hashes"]["adapter"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--rule", required=True)
    parser.add_argument("--primary-dev-report", required=True)
    parser.add_argument("--confirmation-dev-report", required=True)
    parser.add_argument("--confirmation-training-report", required=True)
    parser.add_argument("--confirmation-adapter", required=True)
    parser.add_argument("--confirmation-queue", required=True)
    parser.add_argument("--confirmation-run-record", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    rule_path, rule = _load(args.rule)
    primary_path, primary = _load(args.primary_dev_report)
    confirmation_path, confirmation = _load(args.confirmation_dev_report)
    training_path, training = _load(args.confirmation_training_report)
    queue_path, queue = _load(args.confirmation_queue)
    run_path, run = _load(args.confirmation_run_record)
    adapter = Path(args.confirmation_adapter).resolve()

    if (
        rule.get("task") != args.task
        or rule.get("schema_version")
        != "silver-match-v3-math-retriever-reproducibility-confirmation-v1"
        or rule.get("joint_promotion_gate", {}).get("both_runs_must_independently_pass")
        is not True
    ):
        raise ValueError("unsupported joint confirmation rule")
    minimum_gain = float(
        rule["joint_promotion_gate"][
            "minimum_external_dev_exact_recall_at_50_gain_each"
        ]
    )
    primary_metrics = _validate_dev(primary, args.task, minimum_gain)
    confirmation_metrics = _validate_dev(confirmation, args.task, minimum_gain)
    if (
        primary_metrics["external_dev_labels_sha256"]
        != confirmation_metrics["external_dev_labels_sha256"]
    ):
        raise ValueError("the two runs were not scored on identical external dev labels")
    if training.get("task") != args.task or training.get("status") != "PROMOTABLE":
        raise ValueError("confirmation training report is not promotable")
    expected_adapter = training.get("generated_hashes", {}).get("adapter") or {}
    observed_adapter = {
        path.name: sha256_file(path)
        for path in sorted(adapter.iterdir())
        if path.is_file()
    }
    if not expected_adapter or observed_adapter != expected_adapter:
        raise ValueError("confirmation adapter differs from its training report")
    if observed_adapter != confirmation_metrics["adapter_hashes"]:
        raise ValueError("confirmation dev report scored different adapter bytes")
    if (
        queue.get("task") != args.task
        or queue.get("external_dev_audit", {}).get("external_test_consumed") is not False
        or "--split" not in queue.get("command", [])
        or queue["command"][queue["command"].index("--split") + 1] != "dev"
    ):
        raise ValueError("confirmation queue was not dev-only with test sealed")
    if (
        run.get("status") != "COMPLETED"
        or run.get("external_test_consumed") is not False
        or run.get("output_sha256") != sha256_file(confirmation_path)
        or run.get("queue_sha256") != sha256_file(queue_path)
    ):
        raise ValueError("confirmation run record is incomplete or inconsistent")

    payload = {
        "schema_version": "silver-match-v3-nemotron-joint-dev-selection-v1",
        "status": "SELECTION_FROZEN_TEST_UNCONSUMED",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "decision": "PROMOTE_DETERMINISTIC_CONFIRMATION_ADAPTER_PENDING_TEST_ONCE",
        "selection_basis": "predeclared two-run joint external-dev gate; confirmation role selected, not post-hoc metric ranking",
        "joint_gate": {
            "passed": True,
            "minimum_exact_recall_at_50_gain_each": minimum_gain,
            "recall_at_80_may_decrease": False,
            "primary": {**_artifact(primary_path), "metrics": primary_metrics},
            "confirmation": {
                **_artifact(confirmation_path),
                "metrics": confirmation_metrics,
            },
        },
        "selected_adapter": {
            "path": str(adapter),
            "hashes": observed_adapter,
            "training_report": _artifact(training_path),
        },
        "bindings": {
            "rule": _artifact(rule_path),
            "confirmation_queue": _artifact(queue_path),
            "confirmation_run_record": _artifact(run_path),
        },
        "external_test": {
            "status": "SEALED_UNCONSUMED",
            "consumed_during_training": False,
            "consumed_during_dev_selection": False,
            "permitted_next_action": "one hash-frozen evaluation exactly once",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "sha256": sha256_file(output), **payload}, sort_keys=True))


if __name__ == "__main__":
    main()
