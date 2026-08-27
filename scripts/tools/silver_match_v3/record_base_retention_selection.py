#!/usr/bin/env python3
"""Rebind a rejected-adapter decision as a production base selection.

External-dev retry jobs can end in an explicit rejection of the trained
adapter.  The original decision is immutable and may have been produced on a
different host, so this recorder verifies copied decision/report bytes and
emits a portable, append-only retrieval-selection record for the canonical
production host.  It never opens the frozen test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import sha256_file


SCHEMA = "silver-match-v3-retrieval-selection-v2"
DECISION_SCHEMA = "silver-match-v3-nemotron-external-dev-decision-v1"
REJECT_DECISION = "REJECT_SELECTED_ADAPTER_RETAIN_FROZEN_BASE"
REJECT_STATUS = "FROZEN_REJECT_RETAIN_BASE"


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def build_selection(
    *,
    task: str,
    decision_path: Path,
    report_path: Path,
    fusion_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    decision_path = decision_path.resolve()
    report_path = report_path.resolve()
    fusion_path = fusion_path.resolve()
    manifest_path = manifest_path.resolve()
    decision = _load(decision_path)
    report = _load(report_path)
    fusion = _load(fusion_path)
    manifest = _load(manifest_path)

    if (
        decision.get("schema_version") != DECISION_SCHEMA
        or decision.get("task") != task
        or decision.get("decision") != REJECT_DECISION
        or decision.get("status") != REJECT_STATUS
    ):
        raise ValueError("not a frozen rejected-adapter/base-retention decision")
    external_test = decision.get("external_test") or {}
    consumed_fields = (
        "consumed_during_training",
        "consumed_during_internal_selection",
        "consumed_during_external_dev",
    )
    if (
        external_test.get("status") != "SEALED_UNCONSUMED"
        or any(external_test.get(key) is not False for key in consumed_fields)
    ):
        raise ValueError("frozen external test was consumed or is not explicitly sealed")

    gate = decision.get("external_dev_gate") or {}
    minimum_gain = float(gate.get("minimum_exact_recall_at_50_gain"))
    actual_gain = float(gate.get("actual_exact_recall_at_50_gain"))
    if (
        gate.get("passed") is not False
        or actual_gain >= minimum_gain
        or gate.get("recall_at_80_non_decrease_passed") is not True
    ):
        raise ValueError("base-retention decision has an inconsistent external-dev gate")

    expected_report_sha = str(
        ((decision.get("bindings") or {}).get("dev_report") or {}).get("sha256") or ""
    )
    if expected_report_sha != sha256_file(report_path):
        raise ValueError("copied dev report does not match the frozen decision")
    report_gate = report.get("promotion_gate") or {}
    if (
        report.get("task") != task
        or report.get("split") != "dev"
        or report.get("selection_role") != "promotion_dev"
        or report_gate.get("passed") is not False
        or float(report_gate.get("minimum_gain")) != minimum_gain
        or float(report_gate.get("actual_gain")) != actual_gain
        or report_gate.get("secondary_passed") is not True
        or (report.get("before") or {}).get("exact") != gate.get("before")
        or (report.get("after") or {}).get("exact") != gate.get("after")
    ):
        raise ValueError("dev report differs from the frozen base-retention decision")

    banks = manifest.get("banks") or {}
    if task not in banks:
        raise ValueError("task is absent from the canonical manifest")
    bank_meta = banks[task]
    bank_path = Path(bank_meta["path"]).resolve()
    if not bank_path.is_file():
        raise FileNotFoundError(bank_path)
    if (
        fusion.get("task") != task
        or fusion.get("selection_split") != "dev"
        or int(fusion.get("bank_size", -1)) != int(bank_meta["count"])
        or any(
            split != "dev" and int(count) != 0
            for split, count in (fusion.get("split_counts") or {}).items()
        )
    ):
        raise ValueError("base fusion is not a dev-only full-bank report for this task")
    components = ((fusion.get("selected") or {}).get("component_weights") or {})
    if not components or not any(float(value) > 0 for value in components.values()):
        raise ValueError("base fusion has no selected retrieval geometry")

    chosen = {
        "name": "nemotron-base",
        "kind": "nemotron_base",
        "fusion_report": str(fusion_path),
        "fusion_report_sha256": sha256_file(fusion_path),
        "candidate_inputs": fusion.get("candidate_inputs") or {},
        "dev_metrics": (fusion.get("metrics") or {}).get("dev") or {},
        "items": fusion.get("items") or [],
    }
    return {
        "schema_version": SCHEMA,
        "task": task,
        "selection_split": "external_dev_only",
        "frozen_test_consumed": False,
        "selection": {
            "chosen_name": chosen["name"],
            "chosen_kind": chosen["kind"],
            "adapter_reference": "nemotron-base",
            "decision": REJECT_DECISION,
            "minimum_adapter_recall_at_50_gain": minimum_gain,
            "actual_adapter_recall_at_50_gain": actual_gain,
        },
        "chosen": chosen,
        "variants": [chosen],
        "label_inputs": fusion.get("label_inputs") or {},
        "base_retention_evidence": {
            "external_dev_decision": _artifact(decision_path),
            "external_dev_report": _artifact(report_path),
            "selected_variant_rejected": decision.get("selected_variant"),
            "external_test": external_test,
        },
        "canonical_release_bindings": {
            "manifest": _artifact(manifest_path),
            "bank": {
                **_artifact(bank_path),
                "count": int(bank_meta["count"]),
                "source_sha256": bank_meta["source_sha256"],
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--dev-report", required=True)
    parser.add_argument("--base-fusion", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    payload = build_selection(
        task=args.task,
        decision_path=Path(args.decision),
        report_path=Path(args.dev_report),
        fusion_path=Path(args.base_fusion),
        manifest_path=Path(args.manifest),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": _artifact(output), "task": args.task}, sort_keys=True))


if __name__ == "__main__":
    main()
