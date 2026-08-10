#!/usr/bin/env python3
"""Select two independently trained CE variants using frozen dev evidence only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import sha256_file


ALLTASK_POLICY_V1 = "silver-match-v3-cross-encoder-alltask-policy-v1"
PRESS_RELEASES_POLICY_V2 = (
    "silver-match-v3-cross-encoder-press-releases-policy-v2"
)


def _supported_policy_task(policy: dict[str, Any], task: str) -> bool:
    schema = policy.get("schema_version")
    scope = policy.get("scope") or []
    if schema == ALLTASK_POLICY_V1:
        return task in scope
    if schema == PRESS_RELEASES_POLICY_V2:
        return task == "press-releases" and scope == ["press-releases"]
    return False


def _score(report: dict[str, Any]) -> tuple[Any, ...]:
    dev = report["selected_dev"]
    interval = dev.get("exact_match_precision_wilson_95") or [-1.0, -1.0]
    # ``max`` uses this tuple; invert the lexical name separately in the caller.
    return (
        float(interval[0]),
        float(dev["exact_f_beta_0_5"]),
        float(dev["exact_match_precision"]),
        float(dev["exact_match_recall"]),
    )


def _verify_model_files(report: dict[str, Any]) -> None:
    model_dir = Path(str(report["model_dir"]))
    expected = report.get("model_hashes") or {}
    if not model_dir.is_dir() or not expected:
        raise ValueError(f"missing frozen CE model: {model_dir}")
    observed = {
        str(path.relative_to(model_dir)): sha256_file(path)
        for path in sorted(model_dir.rglob("*"))
        if path.is_file()
    }
    if observed != expected:
        raise ValueError(f"CE model files differ from training report: {model_dir}")


def select(policy_path: Path, task: str, report_paths: list[Path]) -> dict[str, Any]:
    policy_path = policy_path.resolve()
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if not _supported_policy_task(policy, task):
        raise ValueError("unsupported frozen CE policy/task")
    eligibility_path = policy_path.with_suffix(".ELIGIBILITY.json")
    eligibility_artifact = None
    if eligibility_path.exists():
        eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
        if (
            eligibility.get("policy_sha256") != sha256_file(policy_path)
            or task not in eligibility.get("eligible_primary_tasks", [])
        ):
            raise ValueError("frozen policy eligibility registry restricts this task")
        eligibility_artifact = {
            "path": str(eligibility_path),
            "sha256": sha256_file(eligibility_path),
        }
    elif policy.get("schema_version") == PRESS_RELEASES_POLICY_V2:
        raise ValueError("press-releases v2 policy requires an eligibility registry")
    implementation = policy.get("implementation") or {}
    expected_selector_sha = implementation.get("select_cross_encoder_variants_sha256")
    if expected_selector_sha and sha256_file(Path(__file__).resolve()) != expected_selector_sha:
        raise ValueError("CE selector implementation differs from frozen policy")
    expected_variants = {
        str(value["name"]) for value in policy.get("predeclared_variants") or []
    }
    reports: list[dict[str, Any]] = []
    seen: set[str] = set()
    manifest_hashes: set[str] = set()
    bank_hashes: set[str] = set()
    dev_inputs: set[tuple[tuple[str, str], ...]] = set()
    for path in report_paths:
        path = path.resolve()
        value = json.loads(path.read_text(encoding="utf-8"))
        _verify_model_files(value)
        binding = value.get("frozen_policy") or {}
        name = str(binding.get("variant_name") or "")
        if (
            value.get("task") != task
            or binding.get("sha256") != sha256_file(policy_path)
            or name not in expected_variants
            or name in seen
            or value.get("teacher_split_mode") != "explicit_role"
            or (value.get("source_group_split_audit") or {}).get(
                "cross_role_source_group_count"
            )
            != 0
            or value.get("frozen_test_consumed") is not False
        ):
            raise ValueError(f"ineligible or inconsistent CE report: {path}")
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
                "status": value.get("status"),
                "eligible": value.get("status") == "DEV_PROMOTABLE_PENDING_BLIND"
                and value.get("dev_promotable") is True,
                "selected_dev": value["selected_dev"],
                "model_dir": value["model_dir"],
                "model_hashes": value["model_hashes"],
            }
        )
    if seen != expected_variants:
        raise ValueError(
            f"reports must cover every predeclared variant; missing={sorted(expected_variants-seen)}"
        )
    if len(manifest_hashes) != 1 or len(bank_hashes) != 1 or len(dev_inputs) != 1:
        raise ValueError("CE variants differ in manifest, bank, or frozen dev inputs")
    eligible = [row for row in reports if row["eligible"]]
    eligible.sort(key=lambda row: row["name"])
    eligible.sort(key=lambda row: _score({"selected_dev": row["selected_dev"]}), reverse=True)
    chosen = eligible[:2]
    status = (
        "TWO_VARIANT_CE_PROPOSAL_PATH_SELECTED"
        if len(chosen) == 2
        else "NO_AUTOMATIC_CE_MATCH_PATH"
    )
    return {
        "schema_version": "silver-match-v3-cross-encoder-selection-v1",
        "status": status,
        "task": task,
        "policy": {"path": str(policy_path), "sha256": sha256_file(policy_path)},
        "policy_eligibility": eligibility_artifact,
        "selection_data": "frozen_dev_only",
        "frozen_test_consumed": False,
        "manifest_sha256": next(iter(manifest_hashes)),
        "bank_source_sha256": next(iter(bank_hashes)),
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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": sha256_file(output), **result}, sort_keys=True))


if __name__ == "__main__":
    main()
