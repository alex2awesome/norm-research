#!/usr/bin/env python3
"""Freeze one fully audited task for leakage-safe downstream analysis.

Staggered task release is allowed only after that task's matcher and every
canonical decision are immutable.  This preserves the outcome/MI firewall
while allowing completed tasks to be analyzed before unrelated large tasks
finish production.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import sha256_file
from .common import read_jsonl


def _artifact(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def _resolve(raw: str, anchor: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else anchor.parent / path


def freeze_release(
    *,
    manifest_path: Path,
    task: str,
    plan_path: Path,
    final_audit_path: Path,
    final_paths: list[Path],
    blind_risk_audit_path: Path,
    analysis_exclusion_paths: list[Path],
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if task not in manifest.get("banks", {}):
        raise KeyError(task)
    expected_corpora = sorted(
        corpus
        for corpus, meta in manifest["corpora"].items()
        if meta.get("task") == task
    )
    expected_rows = sum(int(manifest["corpora"][c]["count"]) for c in expected_corpora)
    if not analysis_exclusion_paths:
        raise ValueError("at least one calibration/test exclusion input is required")
    task_uids = {
        str(row["norm_uid"])
        for corpus in expected_corpora
        for row in read_jsonl(
            _resolve(str(manifest["corpora"][corpus]["path"]), manifest_path)
        )
    }
    exclusion_uids: set[str] = set()
    exclusion_inputs = {}
    for path in analysis_exclusion_paths:
        exclusion_inputs[str(path)] = sha256_file(path)
        for row in read_jsonl(path):
            uid = str(row.get("norm_uid") or "")
            if not uid:
                raise ValueError(f"analysis exclusion lacks norm_uid: {path}")
            exclusion_uids.add(uid)
    unknown_exclusions = exclusion_uids - task_uids
    if unknown_exclusions:
        raise ValueError(
            f"analysis exclusions contain non-task UIDs: {sorted(unknown_exclusions)[:3]}"
        )

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan.get("status") != "FROZEN_READY_FOR_UNLABELED_PRODUCTION"
        or plan.get("task") != task
        or plan.get("manifest", {}).get("sha256") != sha256_file(manifest_path)
        or plan.get("bank_source_sha256")
        != manifest["banks"][task]["source_sha256"]
    ):
        raise ValueError("production plan is not the frozen task/manifest plan")

    final_audit = json.loads(final_audit_path.read_text(encoding="utf-8"))
    scope = final_audit.get("scope") or {}
    if (
        final_audit.get("schema_version") != "silver-match-v3-final-audit-v1"
        or final_audit.get("complete") is not True
        or final_audit.get("manifest_sha256") != sha256_file(manifest_path)
        or int(final_audit.get("audited_rows", -1)) != expected_rows
        or scope.get("tasks") != [task]
        or sorted((final_audit.get("by_corpus") or {}).keys()) != expected_corpora
    ):
        raise ValueError("final audit is not a complete task-scoped exact audit")
    final_artifacts = [_artifact(path) for path in final_paths]
    supplied_hashes = {value["path"]: value["sha256"] for value in final_artifacts}
    if supplied_hashes != (final_audit.get("input_hashes") or {}):
        raise ValueError("final files differ from exact-audit inputs")

    risk = json.loads(blind_risk_audit_path.read_text(encoding="utf-8"))
    task_risk = (risk.get("by_task") or {}).get(task)
    prediction_hashes = risk.get("prediction_inputs") or {}
    risk_exclusions = risk.get("analysis_exclusions") or {}
    if (
        risk.get("schema_version") != "silver-match-v3-false-abstention-audit-v1"
        or not task_risk
        or supplied_hashes != prediction_hashes
        or (risk_exclusions.get("inputs") or {})
        != dict(sorted(exclusion_inputs.items()))
        or int(risk_exclusions.get("count", -1)) != len(exclusion_uids)
        or int(task_risk.get("audited_rows", 0)) < 1
    ):
        raise ValueError("blind risk audit is missing, task-mismatched, or not final-linked")

    return {
        "schema_version": "silver-match-v3-task-analysis-release-v1",
        "status": "TASK_FROZEN_ANALYSIS_READY",
        "task": task,
        "corpora": expected_corpora,
        "expected_rows": expected_rows,
        "manifest": _artifact(manifest_path),
        "bank_source_sha256": manifest["banks"][task]["source_sha256"],
        "production_plan": _artifact(plan_path),
        "final_audit": _artifact(final_audit_path),
        "final_outputs": final_artifacts,
        "blind_risk_audit": _artifact(blind_risk_audit_path),
        "blind_risk": task_risk,
        "analysis_exclusions": {
            "policy": "exclude every labeled retriever/GEPA/verifier train/dev/test norm from MI and outcome estimation",
            "inputs": dict(sorted(exclusion_inputs.items())),
            "count": len(exclusion_uids),
            "norm_uids": sorted(exclusion_uids),
        },
        "precision_claim_supported": bool(
            task_risk.get("predicted_match_precision_claim_supported")
        ),
        "false_abstention_claim_supported": bool(task_risk.get("claim_supported")),
        "analysis_firewall": {
            "task_matcher_is_immutable": True,
            "may_join_mi_and_outcomes": True,
            "may_tune_this_or_other_task_matchers_from_results": False,
            "cross_task_prompt_or_threshold_transfer_after_release": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--final-audit", required=True)
    parser.add_argument("--final", action="append", required=True)
    parser.add_argument("--blind-risk-audit", required=True)
    parser.add_argument("--analysis-exclusion", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    release = freeze_release(
        manifest_path=Path(args.manifest).resolve(),
        task=args.task,
        plan_path=Path(args.plan).resolve(),
        final_audit_path=Path(args.final_audit).resolve(),
        final_paths=[Path(path).resolve() for path in args.final],
        blind_risk_audit_path=Path(args.blind_risk_audit).resolve(),
        analysis_exclusion_paths=[
            Path(path).resolve() for path in args.analysis_exclusion
        ],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**release, "output": _artifact(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
