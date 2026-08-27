#!/usr/bin/env python3
"""Reclassify a consumed selection panel without leaking upstream holdouts.

The old selection truth remains immutable.  This utility emits an append-only
view in which authoritative upstream-train rows may be used for task-local
prompt optimization and CE training, while upstream dev/test rows are retained
as audit evidence with every gradient flag disabled.  A fresh selection freeze
must already exist and be source-group disjoint from the consumed panel.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


def _index(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    indexed = {str(row.get("norm_uid") or ""): row for row in rows}
    if not rows or "" in indexed or len(indexed) != len(rows):
        raise ValueError(f"empty, missing, or duplicate norm_uid values: {path}")
    return indexed


def reclassify(
    *,
    task: str,
    identities_path: Path,
    selection_audit_path: Path,
    truth_path: Path,
    truth_report_path: Path,
    policy_path: Path,
    upstream_roles_path: Path,
    fresh_identities_path: Path,
    fresh_freeze_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    identities = _index(identities_path)
    truth = _index(truth_path)
    roles = _index(upstream_roles_path)
    fresh = _index(fresh_identities_path)
    audit = json.loads(selection_audit_path.read_text(encoding="utf-8"))
    truth_report = json.loads(truth_report_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    fresh_freeze = json.loads(fresh_freeze_path.read_text(encoding="utf-8"))

    selected = audit.get("selected") or {}
    if (
        audit.get("task") != task
        or selected.get("identity_sha256") != sha256_file(identities_path)
        or int(selected.get("count", -1)) != len(identities)
    ):
        raise ValueError("consumed selection identity audit mismatch")
    if set(truth) != set(identities):
        raise ValueError("resolved truth does not exactly cover consumed selection")
    if (
        truth_report.get("task") != task
        or truth_report.get("complete") is not True
        or int(truth_report.get("resolved_count", -1)) != len(truth)
        or ((truth_report.get("outputs") or {}).get("resolved") or {}).get("sha256")
        != sha256_file(truth_path)
    ):
        raise ValueError("resolved truth report is incomplete or hash-mismatched")
    if (
        policy.get("task") != task
        or policy.get("status") != "failed_closed_no_eligible_policy"
        or policy.get("permanent_blind_consumed") is not False
    ):
        raise ValueError("selection policy is not failed-closed with blind unconsumed")
    if not set(identities) <= set(roles):
        raise ValueError("upstream role map does not cover consumed selection")
    if (
        fresh_freeze.get("task") != task
        or fresh_freeze.get("role") != "select"
        or fresh_freeze.get("status")
        != "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES"
        or (((fresh_freeze.get("outputs") or {}).get("identities") or {}).get("sha256"))
        != sha256_file(fresh_identities_path)
    ):
        raise ValueError("fresh selection identity freeze mismatch")

    consumed_groups = {str(row.get("source_group") or "") for row in identities.values()}
    fresh_groups = {str(row.get("source_group") or "") for row in fresh.values()}
    if "" in consumed_groups or "" in fresh_groups:
        raise ValueError("selection identity lacks canonical source_group")
    if set(identities) & set(fresh) or consumed_groups & fresh_groups:
        raise ValueError("fresh selection overlaps consumed selection")

    optimize: list[dict[str, Any]] = []
    audit_only: list[dict[str, Any]] = []
    role_counts: Counter[str] = Counter()
    for uid in sorted(truth):
        row = truth[uid]
        if row.get("task") != task:
            raise ValueError(f"truth task mismatch: {uid}")
        upstream = str(roles[uid].get("split") or "")
        if upstream not in {"train", "dev", "test"}:
            raise ValueError(f"invalid upstream role: {uid}/{upstream!r}")
        role_counts[upstream] += 1
        common = {
            **row,
            "selection_consumed": True,
            "original_gepa_role": row.get("gepa_role"),
            "original_split": row.get("split"),
            "predeclared_split": upstream,
            "fresh_select_uid_overlap": 0,
            "fresh_select_source_group_overlap": 0,
        }
        if upstream == "train":
            optimize.append(
                {
                    **common,
                    "split": "train",
                    "gepa_role": "optimize",
                    "evaluation_only": False,
                    "prompt_gradient_eligible": True,
                    "prompt_selection_eligible": False,
                    "task_local_ce_training_eligible": True,
                    "retriever_training_eligible": False,
                    "audit_only": False,
                }
            )
        else:
            audit_only.append(
                {
                    **common,
                    "split": upstream,
                    "gepa_role": "consumed_audit_only",
                    "evaluation_only": True,
                    "prompt_gradient_eligible": False,
                    "prompt_selection_eligible": False,
                    "task_local_ce_training_eligible": False,
                    "retriever_training_eligible": False,
                    "audit_only": True,
                }
            )

    report = {
        "schema_version": "silver-match-v3-consumed-select-reclassification-v1",
        "status": "CONSUMED_SELECT_RECLASSIFIED_WITH_UPSTREAM_HOLDOUTS_PROTECTED",
        "task": task,
        "consumed_count": len(truth),
        "authoritative_upstream_roles": dict(sorted(role_counts.items())),
        "prompt_gradient_and_task_local_ce_count": len(optimize),
        "audit_only_nontrain_count": len(audit_only),
        "fresh_select_count": len(fresh),
        "fresh_select_uid_overlap": 0,
        "fresh_select_source_group_overlap": 0,
        "permanent_blind_consumed": False,
        "usage_contract": {
            "optimize_rows_may_mutate_prompts": True,
            "optimize_rows_may_train_task_local_ce": True,
            "optimize_rows_may_train_retriever": False,
            "audit_only_rows_may_mutate_prompts_or_weights": False,
            "consumed_rows_may_be_reused_as_selection_or_blind": False,
        },
        "inputs": {
            "consumed_identities": {
                "path": str(identities_path),
                "sha256": sha256_file(identities_path),
            },
            "selection_audit": {
                "path": str(selection_audit_path),
                "sha256": sha256_file(selection_audit_path),
            },
            "resolved_truth": {"path": str(truth_path), "sha256": sha256_file(truth_path)},
            "truth_report": {
                "path": str(truth_report_path),
                "sha256": sha256_file(truth_report_path),
            },
            "failed_policy": {"path": str(policy_path), "sha256": sha256_file(policy_path)},
            "upstream_roles": {
                "path": str(upstream_roles_path),
                "sha256": sha256_file(upstream_roles_path),
            },
            "fresh_select_identities": {
                "path": str(fresh_identities_path),
                "sha256": sha256_file(fresh_identities_path),
            },
            "fresh_select_freeze": {
                "path": str(fresh_freeze_path),
                "sha256": sha256_file(fresh_freeze_path),
            },
        },
    }
    return optimize, audit_only, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--identities", required=True)
    parser.add_argument("--selection-audit", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--truth-report", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--upstream-roles", required=True)
    parser.add_argument("--fresh-identities", required=True)
    parser.add_argument("--fresh-freeze", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    output = Path(args.output_root).resolve()
    if output.exists():
        raise FileExistsError(output)
    optimize, audit_only, report = reclassify(
        task=args.task,
        identities_path=Path(args.identities).resolve(),
        selection_audit_path=Path(args.selection_audit).resolve(),
        truth_path=Path(args.truth).resolve(),
        truth_report_path=Path(args.truth_report).resolve(),
        policy_path=Path(args.policy).resolve(),
        upstream_roles_path=Path(args.upstream_roles).resolve(),
        fresh_identities_path=Path(args.fresh_identities).resolve(),
        fresh_freeze_path=Path(args.fresh_freeze).resolve(),
    )
    output.mkdir(parents=True, exist_ok=False)
    optimize_path = output / "upstream_train.optimize_truth.jsonl"
    audit_path = output / "upstream_nontrain.audit_only_truth.jsonl"
    write_jsonl(optimize_path, optimize)
    write_jsonl(audit_path, audit_only)
    report["outputs"] = {
        "upstream_train_optimize_truth": {
            "path": str(optimize_path),
            "sha256": sha256_file(optimize_path),
        },
        "upstream_nontrain_audit_only_truth": {
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
        },
    }
    report_path = output / "RECLASSIFICATION.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "report_sha256": sha256_file(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
