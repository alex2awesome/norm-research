#!/usr/bin/env python3
"""Fail closed unless two truth-hidden full-bank packs are independent views.

The packs must contain exactly the same frozen identities and bank metrics, but
must expose both in different orders.  No manual truth or model prediction may
already be present in either view.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file


MANUAL_FIELDS = (
    "manual_decision",
    "manual_metric_id",
    "manual_confidence",
    "manual_reason",
    "auditor",
)
PREDICTION_FIELDS = (
    "decision",
    "metric_id",
    "predicted_metric_id",
    "prediction",
    "adjudicator_decision",
    "adjudicator_metric_id",
)


def _load_pack(root: Path) -> dict[str, Any]:
    validation_path = root / "validation.json"
    items_path = root / "items.jsonl"
    bank_path = root / "bank.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not validation.get("truth_hidden"):
        raise ValueError(f"pack is not truth hidden: {root}")
    if validation.get("adjudicator_outputs_read") or validation.get(
        "label_pass_outputs_read"
    ):
        raise ValueError(f"pack reports prior outputs read: {root}")
    if sha256_file(items_path) != validation["outputs"]["items"]["sha256"]:
        raise ValueError(f"item hash mismatch: {root}")
    if sha256_file(bank_path) != validation["outputs"]["bank"]["sha256"]:
        raise ValueError(f"bank hash mismatch: {root}")
    rows = list(read_jsonl(items_path))
    uids = [str(row.get("norm_uid") or "") for row in rows]
    if not all(uids) or len(uids) != len(set(uids)):
        raise ValueError(f"missing or duplicate item UIDs: {root}")
    groups = [str(row.get("source_group") or "") for row in rows]
    if not all(groups) or len(groups) != len(set(groups)):
        raise ValueError(f"missing or duplicate source groups: {root}")
    for row in rows:
        if any(row.get(field) is not None for field in MANUAL_FIELDS):
            raise ValueError(f"manual truth exposed for {row['norm_uid']}: {root}")
        if any(field in row for field in PREDICTION_FIELDS):
            raise ValueError(f"prediction exposed for {row['norm_uid']}: {root}")
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    metric_ids = [str(row.get("metric_id") or "") for row in bank["metrics"]]
    if not all(metric_ids) or len(metric_ids) != len(set(metric_ids)):
        raise ValueError(f"missing or duplicate bank metric IDs: {root}")
    if len(rows) != int(validation["count"]):
        raise ValueError(f"item count mismatch: {root}")
    if len(metric_ids) != int(validation["bank_metric_count"]):
        raise ValueError(f"bank count mismatch: {root}")
    return {
        "root": str(root),
        "validation": validation,
        "validation_sha256": sha256_file(validation_path),
        "items_sha256": sha256_file(items_path),
        "bank_sha256": sha256_file(bank_path),
        "uids": uids,
        "groups_by_uid": {
            str(row["norm_uid"]): str(row["source_group"]) for row in rows
        },
        "metric_ids": metric_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-root", required=True)
    parser.add_argument("--right-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    left = _load_pack(Path(args.left_root).resolve())
    right = _load_pack(Path(args.right_root).resolve())
    left_validation = left["validation"]
    right_validation = right["validation"]
    for field in ("task", "count", "bank_metric_count", "bank_source_sha256"):
        if left_validation[field] != right_validation[field]:
            raise ValueError(f"pack validation differs for {field}")
    if set(left["uids"]) != set(right["uids"]):
        raise ValueError("pack UID sets differ")
    if left["groups_by_uid"] != right["groups_by_uid"]:
        raise ValueError("pack source-group identities differ")
    if set(left["metric_ids"]) != set(right["metric_ids"]):
        raise ValueError("pack bank metric sets differ")
    if len(left["uids"]) > 1 and left["uids"] == right["uids"]:
        raise ValueError("item order was not independently permuted")
    if len(left["metric_ids"]) > 1 and left["metric_ids"] == right["metric_ids"]:
        raise ValueError("bank order was not independently permuted")
    report = {
        "schema_version": "silver-match-v3-independent-semantic-pack-audit-v1",
        "complete": True,
        "task": left_validation["task"],
        "count": left_validation["count"],
        "bank_metric_count": left_validation["bank_metric_count"],
        "bank_source_sha256": left_validation["bank_source_sha256"],
        "same_uid_set": True,
        "same_source_group_identity": True,
        "different_item_order": True,
        "same_bank_metric_set": True,
        "different_bank_order": True,
        "truth_hidden": True,
        "outputs_read": False,
        "left": {
            key: left[key]
            for key in ("root", "validation_sha256", "items_sha256", "bank_sha256")
        },
        "right": {
            key: right[key]
            for key in ("root", "validation_sha256", "items_sha256", "bank_sha256")
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "output_sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
