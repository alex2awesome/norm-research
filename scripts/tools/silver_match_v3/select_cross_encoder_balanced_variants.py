#!/usr/bin/env python3
"""Select metric-balanced CE variants from unchanged frozen dev reports.

The frozen balanced trainer deliberately wraps the original policy validator.
Its validator returns ``(policy, binding)``; the underlying v3 trainer records
that tuple in ``frozen_policy``, which JSON serializes as ``[policy, binding]``.
The originally pinned selector accepts only the legacy scalar binding.  This
append-only compatibility selector validates both tuple members, verifies the
original selector still has its policy-pinned hash, and otherwise applies the
same frozen-dev-only selection contract without rewriting training reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import sha256_file
from . import select_cross_encoder_variants as legacy


SCHEMA_VERSION = "silver-match-v4-metric-balanced-cross-encoder-selection-v1"
ENCODING = "balanced-validator-tuple-json-array-v1"


def _decode_binding(
    report: dict[str, Any], policy: dict[str, Any], policy_sha256: str
) -> tuple[dict[str, Any], str]:
    raw = report.get("frozen_policy")
    if isinstance(raw, dict):
        binding = raw
        encoding = "legacy-scalar-binding-v1"
    elif isinstance(raw, list) and len(raw) == 2:
        embedded_policy, binding = raw
        if not isinstance(embedded_policy, dict) or not isinstance(binding, dict):
            raise ValueError("balanced frozen_policy tuple members must be objects")
        if embedded_policy != policy:
            raise ValueError("embedded balanced policy differs from frozen policy")
        encoding = ENCODING
    else:
        raise ValueError("unsupported frozen_policy report encoding")

    if binding.get("sha256") != policy_sha256:
        raise ValueError("report binding policy hash differs from frozen policy")
    if binding.get("balanced_training") != policy.get("balanced_training"):
        raise ValueError("report binding balanced-training contract differs")
    expected_trainer = (policy.get("implementation") or {}).get(
        "balanced_train_cross_encoder_sha256"
    )
    observed_trainer = (binding.get("balanced_trainer") or {}).get("sha256")
    if expected_trainer and observed_trainer != expected_trainer:
        raise ValueError("report binding balanced-trainer hash differs")
    return binding, encoding


def select(policy_path: Path, task: str, report_paths: list[Path]) -> dict[str, Any]:
    policy_path = policy_path.resolve()
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if not legacy._supported_policy_task(policy, task):
        raise ValueError("unsupported frozen CE policy/task")
    if policy.get("balanced_objective_revision") != (
        "cross-encoder-metric-balanced-v4"
    ):
        raise ValueError("policy is not the metric-balanced v4 objective")

    policy_sha256 = sha256_file(policy_path)
    eligibility_path = policy_path.with_suffix(".ELIGIBILITY.json")
    if not eligibility_path.exists():
        raise ValueError("balanced policy requires an eligibility registry")
    eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
    if (
        eligibility.get("policy_sha256") != policy_sha256
        or task not in eligibility.get("eligible_primary_tasks", [])
    ):
        raise ValueError("frozen policy eligibility registry restricts this task")

    implementation = policy.get("implementation") or {}
    expected_legacy_sha = implementation.get(
        "select_cross_encoder_variants_sha256"
    )
    legacy_path = Path(legacy.__file__).resolve()
    if not expected_legacy_sha or sha256_file(legacy_path) != expected_legacy_sha:
        raise ValueError("original policy-pinned CE selector differs")

    expected_variants = {
        str(value["name"]) for value in policy.get("predeclared_variants") or []
    }
    reports: list[dict[str, Any]] = []
    seen: set[str] = set()
    manifest_hashes: set[str] = set()
    bank_hashes: set[str] = set()
    dev_inputs: set[tuple[tuple[str, str], ...]] = set()
    encodings: set[str] = set()

    for path in report_paths:
        path = path.resolve()
        value = json.loads(path.read_text(encoding="utf-8"))
        legacy._verify_model_files(value)
        binding, encoding = _decode_binding(value, policy, policy_sha256)
        encodings.add(encoding)
        name = str(binding.get("variant_name") or "")
        if (
            value.get("task") != task
            or name not in expected_variants
            or name in seen
            or value.get("teacher_split_mode") != "explicit_role"
            or (value.get("source_group_split_audit") or {}).get(
                "cross_role_source_group_count"
            )
            != 0
            or value.get("frozen_test_consumed") is not False
            or (value.get("grouped_listwise_evaluation_contract") or {}).get(
                "blind_status"
            )
            != "SEALED_UNCONSUMED"
        ):
            raise ValueError(f"ineligible or inconsistent balanced CE report: {path}")
        seen.add(name)
        manifest_hashes.add(str(value["manifest_sha256"]))
        bank_hashes.add(str(value["bank_source_sha256"]))
        role_inputs = value.get("explicit_role_inputs") or {}
        dev_inputs.add(
            tuple(
                sorted(
                    (source, str(meta["sha256"]))
                    for source, meta in role_inputs.items()
                    if meta.get("role") == "dev"
                )
            )
        )
        reports.append(
            {
                "name": name,
                "path": str(path),
                "sha256": sha256_file(path),
                "frozen_policy_encoding": encoding,
                "status": value.get("status"),
                "eligible": value.get("status")
                == "DEV_PROMOTABLE_PENDING_BLIND"
                and value.get("dev_promotable") is True,
                "selected_dev": value["selected_dev"],
                "model_dir": value["model_dir"],
                "model_hashes": value["model_hashes"],
            }
        )

    if seen != expected_variants:
        raise ValueError(
            "reports must cover every predeclared variant; "
            f"missing={sorted(expected_variants - seen)}"
        )
    if len(manifest_hashes) != 1 or len(bank_hashes) != 1 or len(dev_inputs) != 1:
        raise ValueError("CE variants differ in manifest, bank, or frozen dev inputs")

    eligible_reports = [row for row in reports if row["eligible"]]
    eligible_reports.sort(key=lambda row: row["name"])
    eligible_reports.sort(
        key=lambda row: legacy._score({"selected_dev": row["selected_dev"]}),
        reverse=True,
    )
    chosen = eligible_reports[:2]
    status = (
        "TWO_VARIANT_CE_PROPOSAL_PATH_SELECTED"
        if len(chosen) == 2
        else "NO_AUTOMATIC_CE_MATCH_PATH"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "task": task,
        "policy": {"path": str(policy_path), "sha256": policy_sha256},
        "policy_eligibility": {
            "path": str(eligibility_path),
            "sha256": sha256_file(eligibility_path),
        },
        "selection_data": "frozen_dev_only",
        "frozen_test_consumed": False,
        "blind_status": "SEALED_UNCONSUMED",
        "manifest_sha256": next(iter(manifest_hashes)),
        "bank_source_sha256": next(iter(bank_hashes)),
        "compatibility": {
            "reason": "balanced validator tuple was JSON-serialized as an array",
            "accepted_report_encodings": sorted(encodings),
            "original_selector": {
                "path": str(legacy_path),
                "sha256": sha256_file(legacy_path),
                "policy_pinned_sha256": expected_legacy_sha,
            },
            "training_reports_rewritten": False,
        },
        "all_variants": sorted(reports, key=lambda row: row["name"]),
        "chosen": chosen,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--training-report", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    result = select(
        Path(args.policy), args.task, [Path(value) for value in args.training_report]
    )
    result["compatibility"]["selector"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256_file(Path(__file__).resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"output": str(output), "sha256": sha256_file(output), **result},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
