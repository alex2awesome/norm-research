#!/usr/bin/env python3
"""Freeze/watch the CPU-only Humor consensus-to-training-handoff seam.

The queue is frozen before exact-consensus outputs exist.  It content-locks
every already-available input, waits for the canonical consensus manifest and
CE partition, invokes the existing final-stack freezer unchanged, materializes
the complete unlabeled production CE pair universe, and seals a
content-addressed receipt.  It never launches a trainer, scorer, or GPU job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import read_jsonl, sha256_file
from .freeze_humor_final_stack_handoff import (
    EXPECTED_PROMPT_COMPONENT_SHA256,
    QUEUE_SCHEMA as FINAL_QUEUE_SCHEMA,
    SCHEMA as HANDOFF_SCHEMA,
    TASK,
    _bank_source_hash,
    _bound_output,
    _parse_named_paths,
    _validate_train_only_prompt_audit,
    add_arguments as add_handoff_arguments,
    load_full_candidate_bundle,
    validate_args as validate_handoff_args,
    validate_pilot_recipe,
)
from .freeze_nemotron_ce_production_queue import validate_production_pair_report
from .materialize_nemotron_ce_production_pairs import (
    _load_scope,
    _parse_candidates,
    _validate_candidate_meta,
)


SCHEMA = "silver-match-v3-humor-consensus-completion-queue-v1"
STATUS = "FROZEN_WAITING_FOR_EXACT_CONSENSUS_CPU_ONLY"
RECEIPT_SCHEMA = "silver-match-v3-humor-content-addressed-training-handoff-v1"
STATIC_AUDIT_SCHEMA = "silver-match-v3-humor-remote-static-handoff-audit-v1"
IMPLEMENTATIONS = (
    "build_gemma4_typed_dataset.py",
    "build_nemotron_ce_pairs.py",
    "common.py",
    "freeze_humor_consensus_completion_queue.py",
    "freeze_humor_final_stack_handoff.py",
    "freeze_nemotron_ce_production_queue.py",
    "materialize_nemotron_ce_production_pairs.py",
    "gate_humor_k200_consensus_dev.py",
    "prepare_ce_eligible_truth.py",
    "relocate_consensus_truth_handoff.py",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _canonical_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(Path(args.python).resolve()),
        "-u",
        "-m",
        "scripts.tools.silver_match_v3.freeze_humor_final_stack_handoff",
    ]
    for name in (
        "manifest",
        "bank",
        "hierarchy",
        "existing_truth",
        "existing_truth_report",
        "consensus_truth",
        "consensus_truth_manifest",
        "candidate_capture_freeze",
        "pilot_selection",
        "ce_model",
        "gemma_model",
        "independent_labeling_guide",
        "python",
        "ce_trainer",
        "ce_scorer",
        "gemma_trainer",
        "runtime_root",
        "output_root",
    ):
        command.extend((f"--{name.replace('_', '-')}", str(Path(getattr(args, name)).resolve())))
    for seed in args.ce_seed:
        command.extend(("--ce-seed", str(seed)))
    for value in args.gepa_rule:
        name, _, raw_path = value.partition("=")
        command.extend(("--gepa-rule", f"{name}={Path(raw_path).resolve()}"))
    for value in args.gepa_train_only_audit:
        name, _, raw_path = value.partition("=")
        command.extend(
            ("--gepa-train-only-audit", f"{name}={Path(raw_path).resolve()}")
        )
    for name in (
        "gemma_seed",
        "pair_seed",
        "maximum_pairs",
        "global_negatives_per_norm",
        "ce_context_chars",
        "gemma_max_candidates",
        "gemma_order_seed",
        "gemma_context_chars",
        "gemma_description_chars",
        "gemma_example_chars",
        "gemma_max_examples",
    ):
        command.extend((f"--{name.replace('_', '-')}", str(getattr(args, name))))
    return command


def _production_command(
    args: argparse.Namespace, candidates: Mapping[str, Path]
) -> list[str]:
    command = [
        str(Path(args.python).resolve()),
        "-u",
        "-m",
        "scripts.tools.silver_match_v3.materialize_nemotron_ce_production_pairs",
        "--manifest",
        str(Path(args.manifest).resolve()),
        "--task",
        TASK,
    ]
    for corpus, path in candidates.items():
        command.extend(("--candidate", f"{corpus}={path}"))
    command.extend(
        (
            "--output",
            str(Path(args.production_pairs).resolve()),
            "--norm-universe",
            str(Path(args.production_norm_universe).resolve()),
            "--expected-k",
            str(args.production_k),
            "--context-chars",
            str(args.production_context_chars),
        )
    )
    return command


def _validate_production_candidate_audit(
    *,
    audit_path: Path,
    candidate_path: Path,
    candidate_meta: Mapping[str, Any],
    manifest_path: Path,
    corpus: str,
    bank_hash: str,
    bank_count: int,
    expected_count: int,
    expected_k: int,
) -> dict[str, Any]:
    """Bind the complete multi-lane candidate audit to its exact candidate bytes."""

    audit_path = audit_path.resolve()
    candidate_path = candidate_path.resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_inputs = audit.get("candidate_inputs") or {}
    candidate_ref = audit_inputs.get(str(candidate_path)) or {}
    distribution = audit.get("candidate_count_distribution") or {}
    expected_meta_sha = sha256_file(Path(str(candidate_meta["meta"])))
    if (
        audit.get("schema_version")
        != "silver-match-v3-production-candidate-audit-v1"
        or audit.get("complete") is not True
        or audit.get("task") != TASK
        or audit.get("corpus") != corpus
        or int(audit.get("expected_count", -1)) != expected_count
        or int(audit.get("observed_count", -1)) != expected_count
        or int(audit.get("expected_k", -1)) != expected_k
        or int(audit.get("materialized_k", -1)) != expected_k
        or int(audit.get("bank_count", -1)) != bank_count
        or audit.get("bank_source_sha256") != bank_hash
        or audit.get("manifest_sha256") != sha256_file(manifest_path)
        or distribution != {str(expected_k): expected_count}
        or set(audit_inputs) != {str(candidate_path)}
        or candidate_ref.get("sha256") != sha256_file(candidate_path)
        or candidate_ref.get("meta_sha256") != expected_meta_sha
        or int(candidate_ref.get("count", -1)) != expected_count
    ):
        raise ValueError(f"production candidate audit failed closed: {corpus}")
    return {
        "path": str(audit_path),
        "sha256": sha256_file(audit_path),
        "candidate_sha256": candidate_ref["sha256"],
        "candidate_meta_sha256": candidate_ref["meta_sha256"],
    }


def _validate_capture_report(
    path: Path,
    *,
    role: str,
    candidate_path: Path,
    bank_hash: str,
    require_gate: bool,
) -> dict[str, Any]:
    """Validate train diagnostic or untouched-dev K200 capture evidence."""

    path = path.resolve()
    candidate_path = candidate_path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    group = (payload.get("groups") or {}).get(f"task_split:{TASK}:{role}") or {}
    candidate_inputs = payload.get("candidate_inputs") or {}
    label_inputs = payload.get("label_inputs") or {}
    if (
        role not in {"train", "dev"}
        or payload.get("schema_version") != "silver-match-v3-candidate-capture-v1"
        or int(payload.get("k", -1)) != 200
        or candidate_inputs != {str(candidate_path): sha256_file(candidate_path)}
        or not isinstance(label_inputs, Mapping)
        or not label_inputs
        or not isinstance(group, Mapping)
        or int(group.get("gold_matches", -1)) < 1
        or float(group.get("confidence_level_one_sided", -1)) != 0.95
        or float(group.get("target_upper_bound", -1)) != 0.05
        or int((group.get("unique_candidate_union_size") or {}).get("max", -1))
        > 200
    ):
        raise ValueError(f"Humor K200 {role} capture report contract differs")
    if require_gate and (
        group.get("under_target_supported") is not True
        or float(group.get("union_miss_upper_bound", 1.0)) > 0.05
    ):
        raise ValueError("untouched Humor dev capture does not support <5% miss")
    label_refs = []
    for raw_path, expected in label_inputs.items():
        label_path = Path(str(raw_path)).resolve()
        if sha256_file(label_path) != expected:
            raise ValueError(f"Humor {role} capture label hash differs")
        rows = list(read_jsonl(label_path))
        if not rows or any(
            row.get("task") != TASK
            or row.get("split") != role
            or row.get("decision") != "MATCH"
            or row.get("current_bank_source_sha256") != bank_hash
            or not row.get("norm_uid")
            or not row.get("metric_id")
            for row in rows
        ):
            raise ValueError(f"Humor {role} capture label role/content differs")
        label_refs.append(_artifact(label_path))
    return {
        "report": _artifact(path),
        "role": role,
        "gold_matches": int(group["gold_matches"]),
        "capture_rate": float(group["union_capture_rate"]),
        "miss_upper_bound_one_sided_95": float(group["union_miss_upper_bound"]),
        "under_five_percent_supported": group.get("under_target_supported") is True,
        "label_inputs": label_refs,
        "promotion_gate": require_gate,
    }


def _validate_progressive_policy_gate(
    path: Path,
    *,
    bank_path: Path,
    primary_candidate: Path,
    rescue_candidate: Path,
    capture_report: Path,
) -> dict[str, Any]:
    """Bind the dev-selected progressive schedule without authorizing CE stopping."""

    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    inputs = payload.get("inputs") or {}
    outputs = payload.get("outputs") or {}
    selection = payload.get("progressive_policy_selection") or {}
    selected = selection.get("selected_policy") or {}
    scoring = payload.get("production_scoring_contract") or {}
    refs = {
        "bank": bank_path.resolve(),
        "k200_candidates": primary_candidate.resolve(),
        "fullbank285_candidates": rescue_candidate.resolve(),
    }
    for name, expected_path in refs.items():
        ref = inputs.get(name) or {}
        if (
            Path(str(ref.get("path") or "")).resolve() != expected_path
            or ref.get("sha256") != sha256_file(expected_path)
        ):
            raise ValueError(f"progressive Humor dev gate input differs: {name}")
    capture_ref = outputs.get("capture_report") or {}
    if (
        payload.get("schema_version")
        != "silver-match-v3-humor-k200-untouched-dev-gate-v1"
        or payload.get("status") != "K200_UNTOUCHED_DEV_GATE_PASSED"
        or payload.get("task") != TASK
        or payload.get("selection_role") != "untouched_development_only"
        or payload.get("test_or_blind_labels_used_for_policy_selection") is not False
        or payload.get("training_labels_used_for_promotion") is not False
        or (payload.get("gate") or {}).get("passed") is not True
        or selection.get("multiple_policy_correction")
        != "bonferroni_simultaneous_one_sided_95"
        or selection.get("passed") is not True
        or selected.get("passed_deployment_gate") is not True
        or (
            selected.get("kind") == "component_union"
            and (
                selected.get("passed_simultaneous_gate") is not True
                or float(
                    selected.get(
                        "miss_upper_bound_simultaneous_one_sided_95", 1.0
                    )
                )
                >= 0.05
            )
        )
        or (
            selected.get("kind") == "fused_rank_prefix"
            and (
                selected.get("passed_fixed_fallback_gate") is not True
                or float(
                    selected.get("miss_upper_bound_pointwise_one_sided_95", 1.0)
                )
                >= 0.05
            )
        )
        or selected.get("kind") not in {"component_union", "fused_rank_prefix"}
        or Path(str(capture_ref.get("path") or "")).resolve()
        != capture_report.resolve()
        or capture_ref.get("sha256") != sha256_file(capture_report)
        or scoring.get("score_all_fused_k200_pairs_unconditionally") is not False
        or scoring.get("pre_ce_early_stopping_authorized") is not False
        or scoring.get("ce_confidence_early_stopping_requires_separate_untouched_dev_audit")
        is not True
        or scoring.get("fullbank285_rescue_required_for_primary_abstentions")
        is not True
        or int(payload.get("gpu_processes_launched", -1)) != 0
        or payload.get("training_or_model_scoring_executed") is not False
        or payload.get("release_ready") is not False
    ):
        raise ValueError("progressive Humor untouched-dev policy gate differs")
    return {
        "report": _artifact(path),
        "selected_policy": selected,
        "scoring_contract": scoring,
    }


def _validate_consensus_relocation(
    path: Path,
    *,
    manifest_path: Path,
    ce_report_path: Path,
    source_validation_path: Path,
) -> dict[str, Any]:
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    relocated = payload.get("relocated") or {}
    expected = {
        "manifest": manifest_path.resolve(),
        "ce_report": ce_report_path.resolve(),
        "source_validation": source_validation_path.resolve(),
    }
    for name, expected_path in expected.items():
        ref = relocated.get(name) or {}
        if (
            Path(str(ref.get("path") or "")).resolve() != expected_path
            or ref.get("sha256") != sha256_file(expected_path)
            or int(ref.get("size_bytes", -1)) != expected_path.stat().st_size
        ):
            raise ValueError(f"consensus relocation binding differs: {name}")
    if (
        payload.get("schema_version")
        != "silver-match-v3-consensus-truth-path-relocation-v1"
        or payload.get("status")
        != "BYTE_EXACT_TRUTH_AND_CE_OUTPUTS_PATHS_ONLY_RELOCATED"
        or payload.get("task") != TASK
        or payload.get("truth_or_ce_output_bytes_changed") is not False
        or int(payload.get("truth_rows_parsed_by_relocator", -1)) != 0
        or int(payload.get("test_or_blind_truth_rows_parsed", -1)) != 0
        or payload.get("test_or_blind_outcomes_used") is not False
        or int(payload.get("gpu_processes_launched", -1)) != 0
    ):
        raise ValueError("consensus relocation safety contract differs")
    return _artifact(path)


def _audit_static_inputs(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.ce_seed) != 2 or len(set(args.ce_seed)) != 2:
        raise ValueError("completion queue requires exactly two distinct CE seeds")
    if args.production_k != 200 or args.production_context_chars < 1:
        raise ValueError("Humor primary production pair materialization is frozen to K200")
    bank = Path(args.bank).resolve()
    bank_hash = _bank_source_hash(bank)
    existing = Path(args.existing_truth).resolve()
    existing_report = Path(args.existing_truth_report).resolve()
    _bound_output(
        existing_report,
        existing,
        expected_schema="silver-match-v3-humor-ce-existing-truth-report-v1",
        expected_status="CANONICAL_EXISTING_TRUTH_READY",
    )
    candidate_specs, candidate_audit = load_full_candidate_bundle(
        Path(args.candidate_capture_freeze).resolve(), bank_hash=bank_hash
    )
    recipe, pilot_audit = validate_pilot_recipe(
        Path(args.pilot_selection).resolve(), ce_model=Path(args.ce_model).resolve()
    )
    consensus_relocation = _validate_consensus_relocation(
        Path(args.consensus_relocation_report),
        manifest_path=Path(args.consensus_truth_manifest),
        ce_report_path=Path(args.ce_truth_report),
        source_validation_path=Path(args.consensus_source_validation),
    )

    rules = _parse_named_paths(
        args.gepa_rule,
        expected={f"R{value}" for value in range(1, 10)},
        label="GEPA rule",
    )
    guide = Path(args.independent_labeling_guide).resolve()
    components = {"GUIDE": guide, **rules}
    for name, expected in EXPECTED_PROMPT_COMPONENT_SHA256.items():
        if sha256_file(components[name]) != expected:
            raise ValueError(f"frozen Humor prompt component hash drift: {name}")
    prompt_audits = _parse_named_paths(
        args.gepa_train_only_audit,
        expected={"R7", "R8", "R9"},
        label="train-only GEPA audit",
    )
    for name, path in prompt_audits.items():
        _validate_train_only_prompt_audit(path, round_name=name, prompt_path=rules[name])

    manifest_path = Path(args.manifest).resolve()
    _, _, production_bank_hash, metric_by_id, ordered_corpora = _load_scope(
        manifest_path, TASK
    )
    if production_bank_hash != bank_hash:
        raise ValueError("handoff and production materializer bank identities differ")
    production_candidates = _parse_candidates(args.production_candidate)
    if set(production_candidates) != set(ordered_corpora):
        raise ValueError("production candidates do not cover every Humor corpus exactly")
    production_candidate_audits = _parse_named_paths(
        args.production_candidate_audit,
        expected=set(ordered_corpora),
        label="production candidate audit",
    )
    production_rescue_candidates = _parse_candidates(
        args.production_rescue_candidate
    )
    if set(production_rescue_candidates) != set(ordered_corpora):
        raise ValueError("full-bank rescue candidates do not cover every Humor corpus")
    production_rescue_audits = _parse_named_paths(
        args.production_rescue_candidate_audit,
        expected=set(ordered_corpora),
        label="production full-bank rescue audit",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    production_refs = {}
    production_audits = {}
    rescue_refs = {}
    rescue_audits = {}
    for corpus in ordered_corpora:
        meta = _validate_candidate_meta(
            path=production_candidates[corpus],
            manifest_path=manifest_path,
            corpus=corpus,
            task=TASK,
            bank_hash=bank_hash,
            expected_count=int(manifest["corpora"][corpus]["count"]),
            expected_k=args.production_k,
        )
        production_refs[corpus] = meta
        expected_count = int(manifest["corpora"][corpus]["count"])
        production_audits[corpus] = _validate_production_candidate_audit(
            audit_path=production_candidate_audits[corpus],
            candidate_path=production_candidates[corpus],
            candidate_meta=meta,
            manifest_path=manifest_path,
            corpus=corpus,
            bank_hash=bank_hash,
            bank_count=len(metric_by_id),
            expected_count=expected_count,
            expected_k=args.production_k,
        )
        rescue_meta = _validate_candidate_meta(
            path=production_rescue_candidates[corpus],
            manifest_path=manifest_path,
            corpus=corpus,
            task=TASK,
            bank_hash=bank_hash,
            expected_count=expected_count,
            expected_k=len(metric_by_id),
        )
        if int(rescue_meta["output_k"]) != len(metric_by_id):
            raise ValueError(f"Humor rescue is not exactly full-bank: {corpus}")
        rescue_refs[corpus] = rescue_meta
        rescue_audits[corpus] = _validate_production_candidate_audit(
            audit_path=production_rescue_audits[corpus],
            candidate_path=production_rescue_candidates[corpus],
            candidate_meta=rescue_meta,
            manifest_path=manifest_path,
            corpus=corpus,
            bank_hash=bank_hash,
            bank_count=len(metric_by_id),
            expected_count=expected_count,
            expected_k=len(metric_by_id),
        )

    if len(ordered_corpora) != 1:
        raise ValueError("Humor completion expects one canonical production corpus")
    primary_candidate = production_candidates[ordered_corpora[0]]
    train_capture = _validate_capture_report(
        Path(args.production_train_capture_diagnostic),
        role="train",
        candidate_path=primary_candidate,
        bank_hash=bank_hash,
        require_gate=False,
    )
    dev_capture = _validate_capture_report(
        Path(args.production_dev_capture_gate),
        role="dev",
        candidate_path=primary_candidate,
        bank_hash=bank_hash,
        require_gate=True,
    )
    progressive_gate = _validate_progressive_policy_gate(
        Path(args.production_dev_policy_gate),
        bank_path=bank,
        primary_candidate=primary_candidate,
        rescue_candidate=production_rescue_candidates[ordered_corpora[0]],
        capture_report=Path(args.production_dev_capture_gate),
    )

    repo_root = Path(args.repo_root).resolve()
    bindings = {
        "manifest": _artifact(manifest_path),
        "bank": _artifact(bank),
        "hierarchy": _artifact(Path(args.hierarchy)),
        "existing_truth": _artifact(existing),
        "existing_truth_report": _artifact(existing_report),
        "candidate_capture_freeze": _artifact(Path(args.candidate_capture_freeze)),
        "pilot_selection": _artifact(Path(args.pilot_selection)),
        "independent_labeling_guide": _artifact(guide),
        "consensus_source_validation": _artifact(Path(args.consensus_source_validation)),
        "consensus_relocation_report": consensus_relocation,
        "production_train_capture_diagnostic": train_capture["report"],
        "production_dev_capture_gate": dev_capture["report"],
        "production_dev_policy_gate": progressive_gate["report"],
        "python": _artifact(Path(args.python)),
        "ce_trainer": _artifact(Path(args.ce_trainer)),
        "ce_scorer": _artifact(Path(args.ce_scorer)),
        "gemma_trainer": _artifact(Path(args.gemma_trainer)),
        **{f"prompt_{name}": _artifact(path) for name, path in rules.items()},
        **{f"prompt_audit_{name}": _artifact(path) for name, path in prompt_audits.items()},
        **{
            f"implementation_{name}": _artifact(
                repo_root / "scripts" / "tools" / "silver_match_v3" / name
            )
            for name in IMPLEMENTATIONS
        },
    }
    for lane, value in (candidate_audit.get("inputs") or {}).items():
        bindings[f"handoff_candidate_{lane}"] = {
            "path": value["path"],
            "sha256": value["sha256"],
            "size_bytes": Path(value["path"]).stat().st_size,
        }
    for corpus, value in production_refs.items():
        bindings[f"production_candidate_{corpus}"] = _artifact(Path(value["path"]))
        bindings[f"production_candidate_meta_{corpus}"] = _artifact(Path(value["meta"]))
        bindings[f"production_candidate_audit_{corpus}"] = _artifact(
            production_candidate_audits[corpus]
        )
        bindings[f"production_rescue_candidate_{corpus}"] = _artifact(
            Path(rescue_refs[corpus]["path"])
        )
        bindings[f"production_rescue_candidate_meta_{corpus}"] = _artifact(
            Path(rescue_refs[corpus]["meta"])
        )
        bindings[f"production_rescue_candidate_audit_{corpus}"] = _artifact(
            production_rescue_audits[corpus]
        )
    for role, capture in (("train", train_capture), ("dev", dev_capture)):
        for index, ref in enumerate(capture["label_inputs"]):
            bindings[f"production_{role}_capture_labels_{index}"] = ref
    return {
        "bindings": bindings,
        "candidate_audit": candidate_audit,
        "pilot_audit": pilot_audit,
        "pilot_recipe": recipe,
        "production_candidates": {
            corpus: {
                "path": str(path),
                "sha256": sha256_file(path),
                "meta": production_refs[corpus],
            }
            for corpus, path in production_candidates.items()
        },
        "corpus_order": ordered_corpora,
        "production_candidate_audits": production_audits,
        "production_rescue_candidates": {
            corpus: {
                "path": str(path),
                "sha256": sha256_file(path),
                "meta": rescue_refs[corpus],
            }
            for corpus, path in production_rescue_candidates.items()
        },
        "production_rescue_candidate_audits": rescue_audits,
        "production_capture_evidence": {
            "train_diagnostic_only": train_capture,
            "untouched_dev_promotion_gate": dev_capture,
            "progressive_scoring_policy_gate": progressive_gate,
        },
        "bank_source_sha256": bank_hash,
    }


def _command_contract(args: argparse.Namespace) -> dict[str, Any]:
    candidates = _parse_candidates(args.production_candidate)
    return {
        "handoff": _canonical_command(args),
        "production_pairs": _production_command(args, candidates),
        "production_fullbank_rescue": {
            "candidates": [
                f"{corpus}={path}"
                for corpus, path in _parse_candidates(
                    args.production_rescue_candidate
                ).items()
            ],
            "audits": list(args.production_rescue_candidate_audit),
        },
        "production_capture_evidence": {
            "train_diagnostic_only": str(
                Path(args.production_train_capture_diagnostic).resolve()
            ),
            "untouched_dev_promotion_gate": str(
                Path(args.production_dev_capture_gate).resolve()
            ),
            "progressive_scoring_policy_gate": str(
                Path(args.production_dev_policy_gate).resolve()
            ),
        },
        "watch": {
            "consensus_truth": str(Path(args.consensus_truth).resolve()),
            "consensus_truth_manifest": str(
                Path(args.consensus_truth_manifest).resolve()
            ),
            "ce_truth_report": str(Path(args.ce_truth_report).resolve()),
            "consensus_source_validation": str(
                Path(args.consensus_source_validation).resolve()
            ),
            "consensus_relocation_report": str(
                Path(args.consensus_relocation_report).resolve()
            ),
        },
    }


def build_static_audit_receipt(args: argparse.Namespace) -> dict[str, Any]:
    """Validate remote bytes once and emit a small copyable exact receipt."""

    static = _audit_static_inputs(args)
    return {
        "schema_version": STATIC_AUDIT_SCHEMA,
        "status": "VERIFIED_EXACT_REMOTE_STATIC_INPUTS_CPU_ONLY",
        "created_at": _now(),
        "host": platform.node(),
        "task": TASK,
        "command_contract": _command_contract(args),
        "static_audit": static,
        "consensus_outcomes_opened": False,
        "gpu_processes_launched": 0,
    }


def _static_from_receipt(args: argparse.Namespace, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    expected_receipt_sha = getattr(args, "static_audit_receipt_sha256", None)
    if not expected_receipt_sha or sha256_file(path) != expected_receipt_sha:
        raise ValueError("remote static-audit receipt SHA-256 differs")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != STATIC_AUDIT_SCHEMA
        or payload.get("status") != "VERIFIED_EXACT_REMOTE_STATIC_INPUTS_CPU_ONLY"
        or payload.get("task") != TASK
        or payload.get("consensus_outcomes_opened") is not False
        or int(payload.get("gpu_processes_launched", -1)) != 0
        or payload.get("command_contract") != _command_contract(args)
        or not isinstance(payload.get("static_audit"), Mapping)
    ):
        raise ValueError("remote static-audit receipt contract differs")
    static = dict(payload["static_audit"])
    if not isinstance(static.get("bindings"), Mapping) or not static["bindings"]:
        raise ValueError("remote static-audit receipt lacks exact artifact bindings")
    expected_candidates = _parse_candidates(args.production_candidate)
    expected_rescue = _parse_candidates(args.production_rescue_candidate)
    receipt_candidates = static.get("production_candidates") or {}
    receipt_audits = static.get("production_candidate_audits") or {}
    receipt_rescue = static.get("production_rescue_candidates") or {}
    receipt_rescue_audits = (
        static.get("production_rescue_candidate_audits") or {}
    )
    if (
        set(receipt_candidates) != set(expected_candidates)
        or set(receipt_audits) != set(expected_candidates)
        or list(static.get("corpus_order") or []) != list(expected_candidates)
        or set(receipt_rescue) != set(expected_rescue)
        or set(receipt_rescue_audits) != set(expected_rescue)
    ):
        raise ValueError("remote static-audit receipt corpus identities differ")
    for corpus, expected_path in expected_candidates.items():
        candidate = receipt_candidates[corpus]
        audit = receipt_audits[corpus]
        candidate_binding = static["bindings"].get(f"production_candidate_{corpus}") or {}
        meta_binding = static["bindings"].get(
            f"production_candidate_meta_{corpus}"
        ) or {}
        audit_binding = static["bindings"].get(
            f"production_candidate_audit_{corpus}"
        ) or {}
        if (
            Path(str(candidate.get("path") or "")).resolve()
            != expected_path.resolve()
            or candidate.get("sha256") != candidate_binding.get("sha256")
            or (candidate.get("meta") or {}).get("meta_sha256")
            != meta_binding.get("sha256")
            or audit.get("sha256") != audit_binding.get("sha256")
            or audit.get("candidate_sha256") != candidate_binding.get("sha256")
            or audit.get("candidate_meta_sha256") != meta_binding.get("sha256")
        ):
            raise ValueError(
                f"remote static-audit receipt artifact identities differ: {corpus}"
            )
        rescue = receipt_rescue[corpus]
        rescue_audit = receipt_rescue_audits[corpus]
        rescue_binding = static["bindings"].get(
            f"production_rescue_candidate_{corpus}"
        ) or {}
        rescue_meta_binding = static["bindings"].get(
            f"production_rescue_candidate_meta_{corpus}"
        ) or {}
        rescue_audit_binding = static["bindings"].get(
            f"production_rescue_candidate_audit_{corpus}"
        ) or {}
        if (
            Path(str(rescue.get("path") or "")).resolve()
            != expected_rescue[corpus].resolve()
            or rescue.get("sha256") != rescue_binding.get("sha256")
            or (rescue.get("meta") or {}).get("meta_sha256")
            != rescue_meta_binding.get("sha256")
            or rescue_audit.get("sha256")
            != rescue_audit_binding.get("sha256")
            or rescue_audit.get("candidate_sha256")
            != rescue_binding.get("sha256")
            or rescue_audit.get("candidate_meta_sha256")
            != rescue_meta_binding.get("sha256")
        ):
            raise ValueError(
                f"remote static-audit full-bank rescue differs: {corpus}"
            )
    return static, {
        "mode": "COPIED_EXACT_REMOTE_STATIC_AUDIT_RECEIPT",
        "receipt_source_path": str(path),
        "receipt_sha256": expected_receipt_sha,
        "verified_host": payload.get("host"),
    }


def freeze_queue(args: argparse.Namespace) -> dict[str, Any]:
    receipt_path = getattr(args, "static_audit_receipt", None)
    receipt_sha = getattr(args, "static_audit_receipt_sha256", None)
    if bool(receipt_path) != bool(receipt_sha):
        raise ValueError(
            "--static-audit-receipt and --static-audit-receipt-sha256 are paired"
        )
    if receipt_path:
        static, static_provenance = _static_from_receipt(
            args, Path(receipt_path)
        )
    else:
        static = _audit_static_inputs(args)
        static_provenance = {
            "mode": "STATIC_INPUTS_VERIFIED_ON_FREEZER_HOST",
            "verified_host": platform.node(),
        }
    final_root = Path(args.output_root).resolve()
    pair_path = Path(args.production_pairs).resolve()
    universe_path = Path(args.production_norm_universe).resolve()
    pair_report = pair_path.with_suffix(pair_path.suffix + ".meta.json")
    receipt_dir = Path(args.receipt_directory).resolve()
    queue_output = Path(args.queue_output).resolve()
    pair_state = [pair_path.exists(), universe_path.exists(), pair_report.exists()]
    if any(pair_state) and not all(pair_state):
        raise ValueError("partial precomputed production pair universe fails closed")
    precomputed_production = None
    if all(pair_state):
        precomputed_production = validate_production_pair_report(
            pair_report, expected_task=TASK, num_shards=2
        )
        if (
            Path(precomputed_production["pairs"]["path"]).resolve()
            != pair_path
            or Path(precomputed_production["norm_universe"]["path"]).resolve()
            != universe_path
            or int(precomputed_production.get("candidate_depth", -1)) != 200
        ):
            raise ValueError("precomputed Humor K200 production universe differs")
        static["bindings"] = dict(static["bindings"])
        static["bindings"].update(
            {
                "precomputed_production_pairs": _artifact(pair_path),
                "precomputed_production_norm_universe": _artifact(universe_path),
                "precomputed_production_pair_report": _artifact(pair_report),
            }
        )
    prohibited = [final_root]
    existing = [str(path) for path in prohibited if path.exists()]
    if receipt_dir.exists():
        existing.extend(str(path) for path in receipt_dir.glob("*.json"))
    if queue_output.exists() or existing:
        raise FileExistsError(
            f"completion queue output/runtime target already exists: {existing[:5]}"
        )
    handoff_command = _canonical_command(args)
    production_candidates = {
        corpus: Path(value["path"])
        for corpus, value in static["production_candidates"].items()
    }
    production_command = _production_command(args, production_candidates)
    return {
        "schema_version": SCHEMA,
        "status": STATUS,
        "created_at": _now(),
        "task": TASK,
        "bindings": static["bindings"],
        "watch": {
            "consensus_truth": str(Path(args.consensus_truth).resolve()),
            "consensus_truth_manifest": str(
                Path(args.consensus_truth_manifest).resolve()
            ),
            "ce_truth_report": str(Path(args.ce_truth_report).resolve()),
            "poll_seconds": args.poll_seconds,
            "source_validation_sha256": static["bindings"][
                "consensus_source_validation"
            ]["sha256"],
        },
        "static_audit": {
            "candidate_bundle": static["candidate_audit"],
            "pilot_recipe": static["pilot_recipe"],
            "pilot_audit": static["pilot_audit"],
            "production_candidates": static["production_candidates"],
            "production_candidate_audits": static[
                "production_candidate_audits"
            ],
            "production_rescue_candidates": static[
                "production_rescue_candidates"
            ],
            "production_rescue_candidate_audits": static[
                "production_rescue_candidate_audits"
            ],
            "production_capture_evidence": static[
                "production_capture_evidence"
            ],
            "production_corpus_order": static["corpus_order"],
            "bank_source_sha256": static["bank_source_sha256"],
            "provenance": static_provenance,
        },
        "commands": {
            "freeze_final_training_handoff": handoff_command,
            "materialize_unlabeled_production_pairs": production_command,
        },
        "outputs": {
            "final_handoff_root": str(final_root),
            "handoff_manifest": str(final_root / "HANDOFF_MANIFEST.json"),
            "final_stack_queue": str(final_root / "FINAL_STACK_QUEUE.json"),
            "production_pairs": str(pair_path),
            "production_norm_universe": str(universe_path),
            "production_pair_report": str(pair_report),
            "receipt_directory": str(receipt_dir),
            "receipt_filename_rule": "<sha256-of-exact-json-bytes>.json",
            "cpu_log": str(queue_output.with_suffix(queue_output.suffix + ".run.log")),
        },
        "execution": {
            "repo_root": str(Path(args.repo_root).resolve()),
            "python": str(Path(args.python).resolve()),
            "only_permitted_modules": [
                "scripts.tools.silver_match_v3.freeze_humor_final_stack_handoff",
                "scripts.tools.silver_match_v3.materialize_nemotron_ce_production_pairs",
            ],
        },
        "safety": {
            "gpu_launches_permitted": False,
            "trainers_or_scorers_executed": False,
            "final_stack_freezer_reused_unchanged": True,
            "consensus_outcomes_available_before_static_freeze": False,
            "remote_artifacts_require_exact_copied_static_audit_receipt": True,
            "test_or_blind_selection_permitted": False,
            "production_pair_labels_materialized": False,
            "precomputed_k200_pair_universe_reused": precomputed_production
            is not None,
            "primary_candidate_depth": 200,
            "fullbank_rescue_bound": True,
            "train_capture_is_diagnostic_only": True,
            "untouched_dev_capture_gate_passed": True,
            "progressive_scoring_policy_bound": True,
            "pre_ce_confidence_early_stopping_authorized": False,
            "training_handoff_is_not_release": True,
            "release_ready": False,
        },
    }


def _write_queue(path: Path, plan: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _verify_artifact(ref: Mapping[str, Any]) -> None:
    path = Path(str(ref.get("path") or ""))
    if (
        not path.is_file()
        or path.stat().st_size != int(ref.get("size_bytes", -1))
        or sha256_file(path) != ref.get("sha256")
    ):
        raise ValueError(f"frozen completion-queue artifact changed: {path}")


def validate_queue(plan: Mapping[str, Any]) -> None:
    safety = plan.get("safety") or {}
    if (
        plan.get("schema_version") != SCHEMA
        or plan.get("status") != STATUS
        or plan.get("task") != TASK
        or safety.get("gpu_launches_permitted") is not False
        or safety.get("trainers_or_scorers_executed") is not False
        or safety.get("final_stack_freezer_reused_unchanged") is not True
        or safety.get("production_pair_labels_materialized") is not False
        or int(safety.get("primary_candidate_depth", -1)) != 200
        or safety.get("fullbank_rescue_bound") is not True
        or safety.get("train_capture_is_diagnostic_only") is not True
        or safety.get("untouched_dev_capture_gate_passed") is not True
        or safety.get("progressive_scoring_policy_bound") is not True
        or safety.get("pre_ce_confidence_early_stopping_authorized") is not False
        or safety.get("release_ready") is not False
    ):
        raise ValueError("completion queue schema/status/safety contract failed")
    for ref in (plan.get("bindings") or {}).values():
        _verify_artifact(ref)
    commands = plan.get("commands") or {}
    allowed = set((plan.get("execution") or {}).get("only_permitted_modules") or [])
    if allowed != {
        "scripts.tools.silver_match_v3.freeze_humor_final_stack_handoff",
        "scripts.tools.silver_match_v3.materialize_nemotron_ce_production_pairs",
    }:
        raise ValueError("CPU-only module allowlist differs")
    for name, expected_module in (
        (
            "freeze_final_training_handoff",
            "scripts.tools.silver_match_v3.freeze_humor_final_stack_handoff",
        ),
        (
            "materialize_unlabeled_production_pairs",
            "scripts.tools.silver_match_v3.materialize_nemotron_ce_production_pairs",
        ),
    ):
        command = commands.get(name) or []
        if (
            len(command) < 4
            or command[:3] != [plan["execution"]["python"], "-u", "-m"]
            or command[3] != expected_module
            or any(value in command for value in ("torchrun", "accelerate", "vllm"))
        ):
            raise ValueError(f"completion command escaped CPU-only allowlist: {name}")
    outputs = plan.get("outputs") or {}
    if outputs.get("receipt_filename_rule") != "<sha256-of-exact-json-bytes>.json":
        raise ValueError("content-addressed receipt rule differs")


def _verify_output_ref(ref: Mapping[str, Any], anchor: Path) -> Path:
    path = Path(str(ref.get("path") or ""))
    if not path.is_absolute():
        path = (anchor.parent / path).resolve()
    else:
        path = path.resolve()
    if not path.is_file() or sha256_file(path) != ref.get("sha256"):
        raise ValueError(f"completion output reference changed: {path}")
    return path


def consensus_completion(plan: Mapping[str, Any]) -> dict[str, Any] | None:
    watch = plan["watch"]
    manifest_path = Path(watch["consensus_truth_manifest"])
    truth_path = Path(watch["consensus_truth"])
    ce_report_path = Path(watch["ce_truth_report"])
    if not manifest_path.exists() or not ce_report_path.exists():
        return None
    if not truth_path.is_file():
        raise ValueError("consensus manifest exists without its exact truth output")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_ref = (manifest.get("outputs") or {}).get("all") or {}
    source_ref = (manifest.get("inputs") or {}).get("pack_validation") or {}
    if (
        manifest.get("schema_version")
        != "silver-match-v3-consensus-training-truth-manifest-v1"
        or manifest.get("status") != "COMPLETE_EXACT_CONSENSUS_WITH_FROZEN_SPLITS"
        or manifest.get("task") != TASK
        or int(manifest.get("source_group_cross_split_count", -1)) != 0
        or int(manifest.get("blind_rows_training_eligible", -1)) != 0
        or source_ref.get("sha256") != watch["source_validation_sha256"]
        or _verify_output_ref(all_ref, manifest_path) != truth_path.resolve()
        or int(all_ref.get("count", -1)) != sum(1 for _ in read_jsonl(truth_path))
    ):
        raise ValueError("exact consensus completion manifest failed closed")
    ce_report = json.loads(ce_report_path.read_text(encoding="utf-8"))
    ce_input = ce_report.get("input") or {}
    ce_outputs = ce_report.get("outputs") or {}
    if (
        ce_report.get("schema_version") != "silver-match-v3-ce-eligible-truth-report-v1"
        or ce_report.get("status") != "PARTITIONED_WITHOUT_INFERRED_FAMILY_ANCHORS"
        or ce_report.get("task") != TASK
        or int(ce_report.get("source_groups_crossing_splits", -1)) != 0
        or Path(str(ce_input.get("path") or "")).resolve() != truth_path.resolve()
        or ce_input.get("sha256") != sha256_file(truth_path)
        or set(ce_outputs) != {"eligible", "typed_only"}
    ):
        raise ValueError("CE truth partition is not bound to completed consensus")
    for ref in ce_outputs.values():
        output = _verify_output_ref(ref, ce_report_path)
        if int(ref.get("count", -1)) != sum(1 for _ in read_jsonl(output)):
            raise ValueError("CE truth partition output count differs")
    return {
        "manifest": _artifact(manifest_path),
        "truth": _artifact(truth_path),
        "ce_truth_report": _artifact(ce_report_path),
        "truth_count": int(all_ref["count"]),
    }


def _run_cpu(command: Sequence[str], plan: Mapping[str, Any]) -> None:
    log_path = Path(plan["outputs"]["cpu_log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        handle.write((f"\n=== CPU-only completion { _now() } ===\n").encode())
        result = subprocess.run(
            list(command),
            cwd=plan["execution"]["repo_root"],
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"CPU-only completion stage failed closed: {result.returncode}")


def _validate_handoff(plan: Mapping[str, Any]) -> dict[str, Any] | None:
    manifest_path = Path(plan["outputs"]["handoff_manifest"])
    root = Path(plan["outputs"]["final_handoff_root"])
    if not root.exists() and not manifest_path.exists():
        return None
    if not root.is_dir() or not manifest_path.is_file():
        raise ValueError("partial final-stack handoff is not resume-eligible")
    handoff = json.loads(manifest_path.read_text(encoding="utf-8"))
    queue_path = Path(plan["outputs"]["final_stack_queue"])
    if (
        handoff.get("schema_version") != HANDOFF_SCHEMA
        or handoff.get("status") != "FROZEN_HANDOFF_NOT_PRODUCTION_OR_RELEASE_READY"
        or handoff.get("task") != TASK
        or Path(str(handoff.get("output_root") or "")).resolve() != root.resolve()
        or (handoff.get("queue") or {}).get("sha256") != sha256_file(queue_path)
    ):
        raise ValueError("final-stack handoff manifest differs")
    for name in (
        "truth_manifest",
        "ce_partition_report",
        "ce_builder_report",
        "ce_split_report",
        "gemma_dataset_report",
        "gemma_composite_prompt_manifest",
        "queue",
    ):
        _verify_output_ref(handoff.get(name) or {}, manifest_path)
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    readiness = queue.get("readiness") or {}
    if (
        queue.get("schema_version") != FINAL_QUEUE_SCHEMA
        or queue.get("status")
        != "FROZEN_TRAINING_AND_HELDOUT_SCORING_QUEUE_NOT_RELEASE_READY"
        or queue.get("task") != TASK
        or len(((queue.get("ce") or {}).get("runs") or [])) != 2
        or (queue.get("gemma") or {}).get("status") != "FROZEN_AWAITING_EXECUTION"
        or readiness.get("training_queue_frozen") is not True
        or readiness.get("production_ready") is not False
        or readiness.get("release_ready") is not False
    ):
        raise ValueError("final CE/Gemma training queue contract differs")
    for ref in (queue.get("bindings") or {}).values():
        _verify_output_ref(ref, queue_path)
    return {
        "manifest": _artifact(manifest_path),
        "queue": _artifact(queue_path),
        "ce_train": queue["bindings"]["ce_train"],
        "ce_dev": queue["bindings"]["ce_dev"],
        "gemma_train": queue["bindings"]["gemma_train"],
        "gemma_dev": queue["bindings"]["gemma_dev"],
        "ce_seed_count": 2,
        "gemma_queue_count": 1,
    }


def _validate_production(plan: Mapping[str, Any]) -> dict[str, Any] | None:
    report_path = Path(plan["outputs"]["production_pair_report"])
    pairs = Path(plan["outputs"]["production_pairs"])
    universe = Path(plan["outputs"]["production_norm_universe"])
    if not report_path.exists():
        if pairs.exists() or universe.exists():
            raise ValueError("partial production pair materialization is not resume-eligible")
        return None
    audit = validate_production_pair_report(
        report_path, expected_task=TASK, num_shards=2
    )
    if Path(audit["pairs"]["path"]).resolve() != pairs.resolve() or Path(
        audit["norm_universe"]["path"]
    ).resolve() != universe.resolve():
        raise ValueError("production pair materialization paths differ")
    return audit


def _receipt_bytes(
    queue_path: Path,
    consensus: Mapping[str, Any],
    handoff: Mapping[str, Any],
    production: Mapping[str, Any],
) -> bytes:
    payload = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "CONTENT_ADDRESSED_TRAINING_HANDOFFS_AND_PRODUCTION_PAIRS_COMPLETE",
        "task": TASK,
        "queue": _artifact(queue_path),
        "consensus": dict(consensus),
        "training_handoff": dict(handoff),
        "production_pairs": dict(production),
        "gpu_processes_launched": 0,
        "trainers_or_scorers_executed": False,
        "release_ready": False,
    }
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _seal_receipt(
    plan: Mapping[str, Any],
    queue_path: Path,
    consensus: Mapping[str, Any],
    handoff: Mapping[str, Any],
    production: Mapping[str, Any],
) -> tuple[Path, str]:
    raw = _receipt_bytes(queue_path, consensus, handoff, production)
    digest = hashlib.sha256(raw).hexdigest()
    root = Path(plan["outputs"]["receipt_directory"])
    path = root / f"{digest}.json"
    root.mkdir(parents=True, exist_ok=True)
    extras = [candidate for candidate in root.glob("*.json") if candidate != path]
    if extras:
        raise ValueError(f"competing completion receipts exist: {extras[:3]}")
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("content-addressed completion receipt bytes differ")
    else:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    return path, digest


def run_once(plan: Mapping[str, Any], queue_path: Path) -> dict[str, Any]:
    validate_queue(plan)
    consensus = consensus_completion(plan)
    if consensus is None:
        return {"status": "WAITING_FOR_EXACT_CONSENSUS", "mutations_performed": 0}
    handoff = _validate_handoff(plan)
    if handoff is None:
        _run_cpu(plan["commands"]["freeze_final_training_handoff"], plan)
        handoff = _validate_handoff(plan)
    assert handoff is not None
    production = _validate_production(plan)
    if production is None:
        _run_cpu(plan["commands"]["materialize_unlabeled_production_pairs"], plan)
        production = _validate_production(plan)
    assert production is not None
    receipt, digest = _seal_receipt(
        plan, queue_path.resolve(), consensus, handoff, production
    )
    return {
        "status": "COMPLETE_CPU_ONLY",
        "receipt": str(receipt),
        "receipt_sha256": digest,
        "truth_count": consensus["truth_count"],
        "production_norm_count": production["norm_count"],
        "production_pair_count": production["pair_count"],
        "gpu_processes_launched": 0,
    }


def _add_completion_inputs(parser: argparse.ArgumentParser) -> None:
    add_handoff_arguments(parser)
    parser.add_argument("--consensus-source-validation", required=True)
    parser.add_argument("--consensus-relocation-report", required=True)
    parser.add_argument("--ce-truth-report", required=True)
    parser.add_argument("--production-candidate", action="append", required=True)
    parser.add_argument(
        "--production-candidate-audit", action="append", required=True
    )
    parser.add_argument(
        "--production-rescue-candidate", action="append", required=True
    )
    parser.add_argument(
        "--production-rescue-candidate-audit", action="append", required=True
    )
    parser.add_argument("--production-train-capture-diagnostic", required=True)
    parser.add_argument("--production-dev-capture-gate", required=True)
    parser.add_argument("--production-dev-policy-gate", required=True)
    parser.add_argument("--production-pairs", required=True)
    parser.add_argument("--production-norm-universe", required=True)
    parser.add_argument("--production-k", type=int, default=200)
    parser.add_argument("--production-context-chars", type=int, default=1400)
    parser.add_argument("--receipt-directory", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--queue-output", required=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    auditor = sub.add_parser(
        "audit-static",
        help="validate exact static bytes on the execution host and write a copyable receipt",
    )
    _add_completion_inputs(auditor)
    auditor.add_argument("--static-audit-output", required=True)
    freezer = sub.add_parser("freeze", help="freeze all static inputs before consensus")
    _add_completion_inputs(freezer)
    freezer.add_argument("--static-audit-receipt")
    freezer.add_argument("--static-audit-receipt-sha256")
    runner = sub.add_parser("run", help="validate once or watch; never launch GPUs")
    runner.add_argument("--queue", required=True)
    runner.add_argument("--watch", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command in {"audit-static", "freeze"}:
        validate_handoff_args(parser, args)
        if args.poll_seconds < 1:
            parser.error("--poll-seconds must be positive")
    if args.command == "audit-static":
        receipt = build_static_audit_receipt(args)
        output = Path(args.static_audit_output).resolve()
        _write_queue(output, receipt)
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "host": receipt["host"],
                    "static_audit_receipt": str(output),
                    "static_audit_receipt_sha256": sha256_file(output),
                },
                sort_keys=True,
            )
        )
        return
    if args.command == "freeze":
        plan = freeze_queue(args)
        output = Path(args.queue_output).resolve()
        _write_queue(output, plan)
        print(
            json.dumps(
                {
                    "status": STATUS,
                    "queue": str(output),
                    "queue_sha256": sha256_file(output),
                },
                sort_keys=True,
            )
        )
        return
    queue_path = Path(args.queue).resolve()
    plan = json.loads(queue_path.read_text(encoding="utf-8"))
    while True:
        result = run_once(plan, queue_path)
        print(json.dumps(result, sort_keys=True), flush=True)
        if result["status"] != "WAITING_FOR_EXACT_CONSENSUS" or not args.watch:
            return
        time.sleep(int(plan["watch"]["poll_seconds"]))


if __name__ == "__main__":
    main()
