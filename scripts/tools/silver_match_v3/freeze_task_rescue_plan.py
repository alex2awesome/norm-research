#!/usr/bin/env python3
"""Freeze the repeated full-bank rescue contract for one production task."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import sha256_file


IMPLEMENTATIONS = (
    "freeze_task_rescue_plan.py",
    "run_task_rescue.py",
    "common.py",
    "build_abstention_rescue.py",
    "aggregate_abstention_rescue.py",
    "adjudicate_gemma.py",
    "verify_gemma.py",
    "verify_abstention_gemma.py",
    "combine_two_order_verifications.py",
    "combine_ordered_verifications.py",
    "combine_two_order_abstention_verifications.py",
    "merge_rescue_decisions.py",
    "prepare_unresolved_decision_pack.py",
    "filter_labels.py",
    "audit_final_outputs.py",
    "prepare_final_decision_audit.py",
    "prepare_false_abstention_audit.py",
    "prepare_final_decision_label_pack.py",
    "permute_independent_teacher_pack.py",
    "audit_independent_pack_views.py",
    "audit_isolated_labeler_transcripts.py",
    "validate_independent_teacher_labels.py",
    "finalize_exact_multi_pass_truth.py",
    "audit_false_abstentions.py",
    "freeze_final_risk_gold_consensus.py",
    "freeze_task_final_risk_release.py",
)


def _artifact(path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def _parse_system(values: list[str]) -> dict[str, list[Path]]:
    systems: dict[str, list[Path]] = defaultdict(list)
    for value in values:
        name, separator, raw_path = value.partition("=")
        name = name.strip()
        if not separator or not name or not raw_path.strip():
            raise ValueError("--candidate-system must be NAME=PATH")
        systems[name].append(Path(raw_path).resolve())
    if len(systems) < 2:
        raise ValueError("repeated rescue requires at least two retrieval systems")
    return dict(systems)


def freeze_plan(
    *,
    production_plan_path: Path,
    production_report_path: Path,
    candidate_system_values: list[str],
    candidate_audit_paths: list[Path],
    repo_root: Path,
    abstention_prompt_path: Path,
    blind_audit_exclusion_paths: list[Path],
    block_size: int = 50,
    coverage_repeats: int = 2,
    max_finalists: int = 16,
    blind_audit_n: int = 300,
) -> dict[str, Any]:
    production_plan_path = production_plan_path.resolve()
    production_report_path = production_report_path.resolve()
    repo_root = repo_root.resolve()
    plan = json.loads(production_plan_path.read_text(encoding="utf-8"))
    report = json.loads(production_report_path.read_text(encoding="utf-8"))
    if plan.get("status") != "FROZEN_READY_FOR_UNLABELED_PRODUCTION":
        raise ValueError("primary production plan is not frozen")
    task = str(plan.get("task") or "")
    expected_corpora = set(plan.get("corpora") or [])
    if (
        report.get("schema_version") not in {
            "silver-match-v3-task-production-run-v1",
            "silver-match-v3-task-production-run-v2",
        }
        or report.get("status") != "COMPLETE_PRE_RESCUE_ONLY"
        or report.get("task") != task
        or (report.get("plan") or {}).get("sha256") != sha256_file(production_plan_path)
        or int(report.get("candidate_count", -1)) != int(plan.get("expected_count", -2))
    ):
        raise ValueError("pre-rescue production report is not linked and complete")
    verifier_orders = list((plan.get("verifier") or {}).get("orders") or ["original", "hashed"])
    if verifier_orders not in (
        ["original", "hashed"],
        ["original", "hashed", "reverse"],
    ):
        raise ValueError("primary production plan has unsupported verifier topology")
    if report.get("schema_version") == "silver-match-v3-task-production-run-v2" and (
        (report.get("strict_verification") or {}).get("orders") != verifier_orders
    ):
        raise ValueError("production report and plan verifier topology differ")
    manifest_path = Path(plan["manifest"]["path"]).resolve()
    if sha256_file(manifest_path) != plan["manifest"]["sha256"]:
        raise ValueError("canonical manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bank = manifest["banks"][task]
    bank_count = int(bank["count"])
    bank_sha = str(bank["source_sha256"])

    primary: dict[str, dict[str, Any]] = {}
    reported_finals = report.get("final_pre_rescue") or {}
    if set(reported_finals) != expected_corpora:
        raise ValueError("pre-rescue finals do not cover the task's exact corpora")
    for corpus, artifacts in sorted(reported_finals.items()):
        output = Path(artifacts["output"]["path"]).resolve()
        final_report = Path(artifacts["report"]["path"]).resolve()
        if (
            sha256_file(output) != artifacts["output"]["sha256"]
            or sha256_file(final_report) != artifacts["report"]["sha256"]
        ):
            raise ValueError(f"pre-rescue final changed: {corpus}")
        final_meta = json.loads(final_report.read_text(encoding="utf-8"))
        if (
            final_meta.get("complete") is not True
            or final_meta.get("strict_production") is not True
            or final_meta.get("task") != task
            or final_meta.get("corpus") != corpus
            or final_meta.get("output_sha256") != artifacts["output"]["sha256"]
            or ((final_meta.get("production_plan") or {}).get("sha256"))
            != sha256_file(production_plan_path)
        ):
            raise ValueError(f"invalid strict pre-rescue report: {corpus}")
        primary[corpus] = artifacts

    systems = _parse_system(candidate_system_values)
    audit_by_candidate: dict[Path, tuple[Path, dict[str, Any]]] = {}
    for audit_path in candidate_audit_paths:
        audit_path = audit_path.resolve()
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            audit.get("schema_version")
            != "silver-match-v3-production-candidate-audit-v1"
            or audit.get("complete") is not True
            or audit.get("task") != task
            or audit.get("corpus") not in expected_corpora
            or audit.get("manifest_sha256") != sha256_file(manifest_path)
            or audit.get("bank_source_sha256") != bank_sha
            or int(audit.get("bank_count", -1)) != bank_count
            or int(audit.get("materialized_k", -1)) != bank_count
            or int(audit.get("expected_k", -1)) != bank_count
        ):
            raise ValueError(
                f"candidate audit is not exact full-bank evidence: {audit_path}"
            )
        for raw_path, value in (audit.get("candidate_inputs") or {}).items():
            candidate = Path(raw_path).resolve()
            if candidate in audit_by_candidate:
                raise ValueError(f"candidate appears in multiple audits: {candidate}")
            if sha256_file(candidate) != value["sha256"]:
                raise ValueError(f"full-bank candidate changed: {candidate}")
            meta = Path(value["meta"]).resolve()
            if sha256_file(meta) != value["meta_sha256"]:
                raise ValueError(f"full-bank candidate metadata changed: {candidate}")
            audit_by_candidate[candidate] = (audit_path, audit)

    supplied_candidates = {path for paths in systems.values() for path in paths}
    if supplied_candidates != set(audit_by_candidate):
        raise ValueError("candidate systems and exact full-bank audits differ")
    system_records: dict[str, Any] = {}
    artifact_hashes = set()
    for name, paths in sorted(systems.items()):
        corpora = []
        inputs = []
        for path in paths:
            audit_path, audit = audit_by_candidate[path]
            corpora.append(str(audit["corpus"]))
            artifact = _artifact(path)
            artifact_hashes.add(artifact["sha256"])
            inputs.append(
                {
                    "candidate": artifact,
                    "audit": _artifact(audit_path),
                    "corpus": audit["corpus"],
                    "count": int(audit["observed_count"]),
                }
            )
        if len(corpora) != len(set(corpora)) or set(corpora) != expected_corpora:
            raise ValueError(f"candidate system {name!r} lacks exact corpus coverage")
        system_records[name] = {"inputs": inputs}
    if len(artifact_hashes) != len(supplied_candidates):
        raise ValueError("candidate systems contain byte-identical duplicate captures")

    abstention_prompt_path = abstention_prompt_path.resolve()
    if not blind_audit_exclusion_paths:
        raise ValueError("blind final audits require frozen analysis exclusions")
    blind_exclusions = [_artifact(path) for path in blind_audit_exclusion_paths]
    implementations = {
        name: _artifact(repo_root / "scripts/tools/silver_match_v3" / name)
        for name in IMPLEMENTATIONS
    }
    if (
        coverage_repeats < 2
        or block_size < 1
        or max_finalists < 1
        or max_finalists > block_size
        or blind_audit_n < 60
    ):
        raise ValueError("rescue parameters are weaker than the production contract")
    return {
        "schema_version": "silver-match-v3-task-rescue-plan-v3",
        "status": "FROZEN_READY_FOR_REPEATED_FULL_BANK_RESCUE",
        "task": task,
        "manifest": _artifact(manifest_path),
        "bank_source_sha256": bank_sha,
        "bank_count": bank_count,
        "corpora": sorted(expected_corpora),
        "expected_count": int(plan["expected_count"]),
        "production_plan": _artifact(production_plan_path),
        "production_report": _artifact(production_report_path),
        "primary_final_pre_rescue": primary,
        "candidate_systems": system_records,
        "rescue_policy": {
            "block_size": block_size,
            "primary_k": int(plan["adjudicator"]["candidate_depth"]),
            "coverage_repeats": coverage_repeats,
            "reinclude_primary": True,
            "include_all_abstentions": True,
            "include_low_confidence": True,
            "max_finalists": max_finalists,
            "finalist_adjudication_orders": ["original", "hashed"],
            "contrastive_verification_orders": verifier_orders,
            "typed_abstention_verification_orders": ["original", "hashed"],
            "strict_two_order_finalist_adjudication": True,
            "strict_all_selected_order_contrastive_verification": True,
            "strict_two_order_contrastive_verification": verifier_orders
            == ["original", "hashed"],
            "strict_two_order_typed_abstention_verification": True,
            "blind_match_and_abstention_audits_required": True,
        },
        "adjudicator": plan["adjudicator"],
        "verifier": plan["verifier"],
        "abstention_verifier": {
            "prompt": _artifact(abstention_prompt_path),
            "model": plan["adjudicator"]["model"],
            "orders": ["original", "hashed"],
            "max_model_len": 8192,
            "max_tokens": 180,
            "seed": 43,
        },
        "blind_audit_exclusions": blind_exclusions,
        "final_risk_policy": {
            "sample_schema": "silver-match-v3-final-decision-sample-v2",
            "uniform_match_sample_n": blind_audit_n,
            "uniform_abstention_sample_n": blind_audit_n,
            "independent_full_bank_passes_minimum": 2,
            "unique_exact_two_vote_consensus_required": True,
            "disagreement_only_resolvers_required": True,
            "unresolved_gold_rows_may_not_be_dropped_from_sample": True,
            "strict_transcript_isolation_required_for_every_pass": True,
            "alpha_one_sided": 0.05,
            "minimum_support_per_sample": 60,
            "false_abstention_upper_target": 0.05,
            "match_exact_precision_point_target": 0.90,
            "match_exact_precision_lower_target": 0.90,
            "typed_abstention_exact_point_target": 0.90,
            "typed_abstention_exact_lower_target": 0.80,
        },
        "implementations": implementations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-plan", required=True)
    parser.add_argument("--production-report", required=True)
    parser.add_argument("--candidate-system", action="append", required=True)
    parser.add_argument("--candidate-audit", action="append", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--abstention-prompt",
        default="scripts/tools/silver_match_v3/prompts/verify_abstention_v1.txt",
    )
    parser.add_argument("--blind-audit-exclusion", action="append", required=True)
    parser.add_argument("--block-size", type=int, default=50)
    parser.add_argument("--coverage-repeats", type=int, default=2)
    parser.add_argument("--max-finalists", type=int, default=16)
    parser.add_argument("--blind-audit-n", type=int, default=300)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    report = freeze_plan(
        production_plan_path=Path(args.production_plan),
        production_report_path=Path(args.production_report),
        candidate_system_values=args.candidate_system,
        candidate_audit_paths=[Path(path) for path in args.candidate_audit],
        repo_root=Path(args.repo_root),
        abstention_prompt_path=Path(args.abstention_prompt),
        blind_audit_exclusion_paths=[Path(path) for path in args.blind_audit_exclusion],
        block_size=args.block_size,
        coverage_repeats=args.coverage_repeats,
        max_finalists=args.max_finalists,
        blind_audit_n=args.blind_audit_n,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"output": str(output), "sha256": sha256_file(output)}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
