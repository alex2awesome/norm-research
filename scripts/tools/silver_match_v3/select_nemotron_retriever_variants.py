#!/usr/bin/env python3
"""Freeze a task-local Nemotron variant choice using internal dev only."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import sha256_file


METRIC_ORDER = (
    "recall_at_50",
    "mrr",
    "recall_at_30",
    "recall_at_16",
    "recall_at_10",
    "recall_at_5",
    "recall_at_3",
    "recall_at_1",
)


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def _parse_report(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--report must be VARIANT=PATH")
    variant, raw_path = value.split("=", 1)
    variant = variant.strip()
    if not variant:
        raise ValueError("empty variant name")
    return variant, Path(raw_path).resolve()


def _selection_key(report: dict[str, Any]) -> tuple[float, ...]:
    exact = report["after"]["dev"]["all"]["exact"]
    return tuple(float(exact[name]) for name in METRIC_ORDER)


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    lock_path = Path(args.predeclaration).resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    task = str(lock["task"])
    rule = lock["internal_variant_selection"]
    if rule.get("data") != "internal_dev_only":
        raise ValueError("predeclaration does not require internal-dev-only selection")
    expected_order = [f"higher_{name}" for name in METRIC_ORDER]
    expected_order.append("lexicographically_smaller_variant_name")
    if rule.get("order") != expected_order:
        raise ValueError("unsupported internal variant selection order")
    if rule.get("external_dev_may_select_between_variants") is not False:
        raise ValueError("external dev is allowed to select variants")

    declared = {
        str(row["name"]): row for row in lock.get("predeclared_variants", [])
    }
    supplied = dict(_parse_report(value) for value in args.report)
    if set(supplied) != set(declared):
        raise ValueError("report variants differ from the predeclared variants")

    expected_teacher = str(lock["inputs"]["exact_teacher_union_sha256"])
    expected_manifest = str(lock["inputs"]["relocated_manifest_sha256"])
    rows: list[dict[str, Any]] = []
    for variant in sorted(supplied):
        path = supplied[variant]
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("task") != task or report.get("status") != "PROMOTABLE":
            raise ValueError(f"ineligible report: {variant}")
        if report.get("input_hashes", {}).get("manifest") != expected_manifest:
            raise ValueError(f"manifest mismatch: {variant}")
        teachers = report.get("input_hashes", {}).get("teachers", {})
        if list(teachers.values()) != [expected_teacher]:
            raise ValueError(f"teacher mismatch: {variant}")
        audit = report.get("split_audit", {})
        if audit.get("source_group_overlap"):
            raise ValueError(f"source-group leakage: {variant}")
        if int(audit.get("rows", {}).get("dev", 0)) < 1:
            raise ValueError(f"empty internal dev: {variant}")
        key = _selection_key(report)
        adapter = Path(str(report["adapter_path"])).resolve()
        adapter_artifacts = [
            _artifact(adapter / name)
            for name in sorted(report.get("adapter_files", []))
        ]
        rows.append(
            {
                "variant": variant,
                "declared": declared[variant],
                "report": _artifact(path),
                "best_epoch": int(report["best_epoch"]),
                "selection_key_names": list(METRIC_ORDER),
                "selection_key": list(key),
                "adapter_path": str(adapter),
                "adapter_artifacts": adapter_artifacts,
            }
        )

    # Sort descending by every metric, then ascending by variant name.
    ranked = sorted(rows, key=lambda row: tuple(-x for x in row["selection_key"]) + (row["variant"],))
    selected = ranked[0]
    return {
        "schema_version": "silver-match-v3-nemotron-internal-dev-selection-v1",
        "status": "FROZEN_SELECTED_BEFORE_EXTERNAL_DEV",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "predeclaration": _artifact(lock_path),
        "selector": _artifact(Path(__file__)),
        "selection_data": "internal_dev_only",
        "selection_order": expected_order,
        "external_dev_consumed": False,
        "external_test_consumed": False,
        "ranked_variants": ranked,
        "selected_variant": selected["variant"],
        "selected_report": selected["report"],
        "selected_adapter_path": selected["adapter_path"],
        "selected_adapter_artifacts": selected["adapter_artifacts"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predeclaration", required=True)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    result = freeze(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": sha256_file(output), **result}, sort_keys=True))


if __name__ == "__main__":
    main()
