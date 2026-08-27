#!/usr/bin/env python3
"""Freeze truth-blind full-bank candidates for one predeclared panel role."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


FORBIDDEN_ASSIGNMENT_FIELDS = {
    "acceptable_metric_ids",
    "candidate_ids",
    "decision",
    "label",
    "metric_id",
    "outcome",
    "prediction",
    "raw_response",
    "reason",
}


def _ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    pack_root = Path(args.pack_root).resolve()
    partition_path = Path(args.partition).resolve()
    partition_freeze_path = Path(args.partition_freeze).resolve()
    output = Path(args.output).resolve()
    report = Path(args.report).resolve()
    if output.exists() or report.exists():
        raise FileExistsError("refusing to overwrite partition-role candidates")

    validation_path = pack_root / "validation.json"
    items_path = pack_root / "items.jsonl"
    bank_path = pack_root / "bank.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    partition_freeze = json.loads(partition_freeze_path.read_text(encoding="utf-8"))
    if (
        validation.get("schema_version")
        != "silver-match-v3-frozen-identity-full-bank-source-pack-v1"
        or validation.get("status")
        != "FROZEN_CANDIDATE_AND_TRUTH_HIDDEN_BEFORE_LABELING"
        or validation.get("task") != args.task
        or validation.get("truth_hidden") is not True
        or validation.get("candidate_proposals_hidden") is not True
        or validation.get("prior_labels_predictions_mi_and_outcomes_not_read") is not True
        or sha256_file(items_path)
        != ((validation.get("outputs") or {}).get("items") or {}).get("sha256")
        or sha256_file(bank_path)
        != ((validation.get("outputs") or {}).get("bank") or {}).get("sha256")
    ):
        raise ValueError("source pack is not an exact truth/candidate-hidden pack")
    if (
        partition_freeze.get("schema_version")
        != "silver-match-v3-frozen-identity-partition-freeze-v1"
        or partition_freeze.get("status")
        != "FROZEN_BEFORE_ANY_DISTILLATION_LABELS_OR_PREDICTIONS"
        or partition_freeze.get("task") != args.task
        or (partition_freeze.get("output") or {}).get("sha256")
        != sha256_file(partition_path)
        or (partition_freeze.get("content_contract") or {}).get(
            "identity_and_source_group_fields_only"
        )
        is not True
        or (partition_freeze.get("content_contract") or {}).get(
            "labels_predictions_metric_ids_reasons_mi_or_outcomes_used"
        )
        is not False
    ):
        raise ValueError("partition is not a truth-blind pre-label freeze")

    items = list(read_jsonl(items_path))
    item_by_uid = {str(row.get("norm_uid") or ""): row for row in items}
    if not items or "" in item_by_uid or len(item_by_uid) != len(items):
        raise ValueError("source pack has missing or duplicate UIDs")
    assignments = list(read_jsonl(partition_path))
    assignment_by_uid = {
        str(row.get("norm_uid") or ""): row for row in assignments
    }
    if (
        not assignments
        or "" in assignment_by_uid
        or len(assignment_by_uid) != len(assignments)
        or set(assignment_by_uid) != set(item_by_uid)
        or len(assignments) != int((partition_freeze.get("output") or {}).get("count", -1))
    ):
        raise ValueError("partition does not exactly cover the frozen source pack")
    selected: list[dict[str, Any]] = []
    for uid, assignment in assignment_by_uid.items():
        item = item_by_uid[uid]
        if (
            assignment.get("schema_version")
            != "silver-match-v3-frozen-identity-partition-v1"
            or assignment.get("task") != args.task
            or assignment.get("corpus") != item.get("corpus")
            or assignment.get("source_group") != item.get("source_group")
            or assignment.get("labels_predictions_metric_ids_reasons_mi_or_outcomes_used")
            is not False
            or FORBIDDEN_ASSIGNMENT_FIELDS & set(assignment)
        ):
            raise ValueError(f"invalid or label-bearing partition assignment: {uid}")
        if assignment.get("remediation_role") == args.role:
            selected.append(item)
    selected.sort(key=lambda row: str(row["norm_uid"]))
    expected_count = int(
        (partition_freeze.get("role_counts") or {}).get(args.role, -1)
    )
    if len(selected) != expected_count or len(selected) != args.expected_count:
        raise ValueError("selected partition-role count differs from the freeze")

    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    metrics = list(bank.get("metrics") or [])
    metric_ids = [str(row.get("metric_id") or "") for row in metrics]
    if (
        bank.get("task") != args.task
        or bank.get("source_sha256") != validation.get("bank_source_sha256")
        or not metric_ids
        or "" in metric_ids
        or len(metric_ids) != len(set(metric_ids))
    ):
        raise ValueError("source bank is invalid or identity-drifted")
    candidates = [
        {
            "metric_id": metric_id,
            "rank": rank,
            "score": None,
            "candidate_source": "truth_blind_complete_frozen_bank",
        }
        for rank, metric_id in enumerate(metric_ids, 1)
    ]
    rows = [
        {
            "schema_version": item.get("schema_version") or "silver-match-v3.0",
            "norm_uid": item["norm_uid"],
            "corpus": item["corpus"],
            "task": args.task,
            "row": item["row"],
            "bank_source_sha256": validation["bank_source_sha256"],
            "candidates": candidates,
            "candidate_depth": len(candidates),
            "truth_hidden": True,
            "prior_predictions_hidden": True,
            "partition_role": args.role,
        }
        for item in selected
    ]
    write_jsonl(output, rows)
    result = {
        "schema_version": "silver-match-v3-partition-role-full-bank-candidates-freeze-v1",
        "status": "FROZEN_BEFORE_INFERENCE",
        "task": args.task,
        "partition_role": args.role,
        "count": len(rows),
        "unique_uids": len(rows),
        "unique_source_groups": len({str(row["source_group"]) for row in selected}),
        "candidate_depth": len(metric_ids),
        "bank_source_sha256": validation["bank_source_sha256"],
        "truth_hidden": True,
        "select_rows_read": False,
        "prior_decisions_metric_ids_predictions_proposals_mi_and_outcomes_read": False,
        "inputs": {
            "pack_validation": _ref(validation_path),
            "items": _ref(items_path),
            "bank": _ref(bank_path),
            "partition": _ref(partition_path),
            "partition_freeze": _ref(partition_freeze_path),
        },
        "output": _ref(output),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**result, "report": _ref(report)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--partition-freeze", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if args.expected_count < 1:
        parser.error("--expected-count must be positive")
    print(json.dumps(materialize(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
