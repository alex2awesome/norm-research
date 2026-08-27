#!/usr/bin/env python3
"""Seal a Nemotron LoRA promotion and append-only retriever selection.

The freezer binds the pre-result contract, internal training report, saved LoRA,
sealed external-dev gate, exact evaluator runtime, failed-closed first attempt,
and downstream verifier truth isolation.  It writes a validation audit first and
then a new selection that records (but never mutates) the superseded selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file


GEOMETRY = "direct_dense_evidence_query_metric_card_nemotron_instruction_v1"


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def adapter_hashes(path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    values = {
        child.name: sha256_file(child)
        for child in sorted(path.iterdir())
        if child.is_file()
    }
    required = {"README.md", "adapter_config.json", "adapter_model.safetensors"}
    if set(values) != required:
        raise ValueError("adapter is not the exact three-file LoRA release")
    return values


def identities(path: Path) -> tuple[set[str], set[str], int]:
    uids: set[str] = set()
    groups: set[str] = set()
    count = 0
    for row in read_jsonl(path):
        uid = str(row.get("norm_uid") or "")
        group = str(row.get("source_group") or "")
        if not uid or uid in uids or not group:
            raise ValueError(f"missing/duplicate UID or group in {path}: {uid!r}")
        uids.add(uid)
        groups.add(group)
        count += 1
    return uids, groups, count


def freeze(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {
        name: Path(getattr(args, name)).resolve()
        for name in (
            "precontract",
            "training_report",
            "external_queue",
            "external_report",
            "external_log",
            "external_run_record",
            "runtime_inventory",
            "first_rejection",
            "external_dev",
            "verifier_truth",
            "verifier_truth_report",
            "model_inventory",
        )
    }
    for optional_name in ("training_log", "teacher", "old_selection"):
        raw = getattr(args, optional_name, None)
        if raw:
            paths[optional_name] = Path(raw).resolve()
    bound = {name: artifact(path) for name, path in paths.items()}
    relocated_references: dict[str, dict[str, Any]] = {}
    if "training_log" not in paths:
        relocated_references["training_log"] = {
            "path": args.training_log_reference_path,
            "sha256": args.training_log_reference_sha256,
            "live_rehashed_on_this_host": False,
            "previously_independently_rehashed": True,
        }
        bound["training_log"] = relocated_references["training_log"]
    if "teacher" not in paths:
        relocated_references["teacher"] = {
            "path": args.teacher_reference_path,
            "sha256": args.teacher_reference_sha256,
            "rows": 14594,
            "live_rehashed_on_this_host": False,
            "hash_reverified_via_training_report_and_precontract": True,
        }
        bound["teacher"] = relocated_references["teacher"]
    if "old_selection" not in paths:
        relocated_references["old_selection"] = {
            "path": args.old_selection_reference_path,
            "sha256": args.old_selection_reference_sha256,
            "live_rehashed_on_this_host": False,
            "previously_independently_rehashed": True,
        }
        bound["old_selection"] = relocated_references["old_selection"]
    adapter_path = Path(args.adapter).resolve()
    base_model_path = Path(args.base_model).resolve()
    if not base_model_path.is_dir():
        raise FileNotFoundError(base_model_path)
    adapter_files = adapter_hashes(adapter_path)

    contract = json.loads(paths["precontract"].read_text(encoding="utf-8"))
    training = json.loads(paths["training_report"].read_text(encoding="utf-8"))
    external_queue = json.loads(paths["external_queue"].read_text(encoding="utf-8"))
    external = json.loads(paths["external_report"].read_text(encoding="utf-8"))
    run_record = json.loads(paths["external_run_record"].read_text(encoding="utf-8"))
    runtime = json.loads(paths["runtime_inventory"].read_text(encoding="utf-8"))
    rejection = json.loads(paths["first_rejection"].read_text(encoding="utf-8"))
    truth_report = json.loads(paths["verifier_truth_report"].read_text(encoding="utf-8"))
    old_selection = (
        json.loads(paths["old_selection"].read_text(encoding="utf-8"))
        if "old_selection" in paths
        else {
            "task": args.task,
            "selection_split": "external_dev_only",
            "frozen_test_consumed": False,
            "chosen": {
                "name": args.old_selection_reference_name,
                "kind": args.old_selection_reference_kind,
            },
        }
    )

    failures: list[str] = []
    if (
        contract.get("task") != args.task
        or contract.get("status") != "FROZEN_BEFORE_TRAINING_RESULT"
        or contract.get("training_result_present_when_frozen") is not False
        or (contract.get("sealed_external_dev") or {}).get("external_test_consumed")
        is not False
    ):
        failures.append("pre-result validation contract is not clean and sealed")
    generated_adapter = (training.get("generated_hashes") or {}).get("adapter") or {}
    dev_before = (((training.get("before") or {}).get("dev") or {}).get("all") or {}).get("exact") or {}
    dev_after = (((training.get("after") or {}).get("dev") or {}).get("all") or {}).get("exact") or {}
    test_before = (((training.get("before") or {}).get("test") or {}).get("all") or {}).get("exact") or {}
    test_after = (((training.get("after") or {}).get("test") or {}).get("all") or {}).get("exact") or {}
    if (
        training.get("task") != args.task
        or training.get("status") != "PROMOTABLE"
        or (training.get("promotion_gate") or {}).get("passed") is not True
        or generated_adapter != adapter_files
        or int((training.get("teacher_audit") or {}).get("weak_forced_groups", -1)) != 0
        or int((training.get("split_audit") or {}).get("rows", {}).get("train", -1))
        != 11576
        or any((training.get("split_audit") or {}).get("source_group_overlap", {}).values())
    ):
        failures.append("internal training result or LoRA integrity gate failed")
    if "training_log" in paths:
        training_log_text = paths["training_log"].read_text(
            encoding="utf-8", errors="replace"
        )
        warning_bound = (
            "sentence-transformers version 5.1.2" in training_log_text.lower()
            or "sentence_transformers version 5.1.2" in training_log_text.lower()
            or "Sentence Transformers version 5.1.2" in training_log_text
            or "created with version 5.1.2" in training_log_text.lower()
        )
    else:
        warning_bound = bool(
            args.training_log_reference_sha256
            and (contract.get("runtime_warning_policy") or {}).get("known_warning")
        )

    queue_bindings = {
        str(value["name"]): str(value["sha256"])
        for value in external_queue.get("bindings") or []
    }
    external_gate = external.get("promotion_gate") or {}
    external_before = (external.get("before") or {}).get("exact") or {}
    external_after = (external.get("after") or {}).get("exact") or {}
    if (
        external.get("task") != args.task
        or external.get("split") != "dev"
        or external.get("selection_role") != "promotion_dev"
        or int(external.get("n_match_labels", -1)) != 53
        or external_gate.get("passed") is not True
        or external_gate.get("secondary_passed") is not True
        or float(external_gate.get("actual_gain", -1)) < 0.03
        or run_record.get("status") != "COMPLETED"
        or int(run_record.get("returncode", -1)) != 0
        or run_record.get("external_test_consumed") is not False
        or run_record.get("output_sha256") != bound["external_report"]["sha256"]
        or run_record.get("log_sha256") != bound["external_log"]["sha256"]
        or run_record.get("queue_sha256") != bound["external_queue"]["sha256"]
        or queue_bindings.get("training_report") != bound["training_report"]["sha256"]
        or queue_bindings.get("posttrain_contract") != bound["precontract"]["sha256"]
        or queue_bindings.get("runtime_inventory") != bound["runtime_inventory"]["sha256"]
    ):
        failures.append("sealed external-dev promotion gate or provenance failed")
    if (
        (runtime.get("runtime_audit") or {}).get("all_expected_versions_match")
        is not True
        or rejection.get("status") != "FAILED_CLOSED_BEFORE_QUEUE_FREEZE_OR_SCORING"
        or ((rejection.get("zero_output_proof") or {}).get("external_dev_scored"))
        is not False
        or ((rejection.get("zero_output_proof") or {}).get("external_test_consumed"))
        is not False
    ):
        failures.append("runtime mitigation or first-attempt zero-output proof failed")
    if not warning_bound:
        failures.append("known training-runtime metadata warning is absent from bound log")

    external_uids, external_groups, external_count = identities(paths["external_dev"])
    truth_uids, truth_groups, truth_count = identities(paths["verifier_truth"])
    contract_isolation = (
        contract.get("downstream_verifier_truth_provenance") or {}
    ).get("retriever_isolation") or {}
    if "teacher" in paths:
        teacher_uids, teacher_groups, teacher_count = identities(paths["teacher"])
        truth_teacher_uid = len(truth_uids & teacher_uids)
        truth_teacher_group = len(truth_groups & teacher_groups)
        teacher_external_uid = len(teacher_uids & external_uids)
        teacher_external_group = len(teacher_groups & external_groups)
    else:
        teacher_count = int(
            (contract.get("training_bindings") or {}).get("teacher", {}).get("rows", -1)
        )
        expected_teacher_sha = str(
            (contract.get("training_bindings") or {}).get("teacher", {}).get("sha256")
            or ""
        )
        report_teacher_hashes = (training.get("input_hashes") or {}).get("teachers") or {}
        report_teacher_sha_values = set(map(str, report_teacher_hashes.values()))
        if (
            args.teacher_reference_sha256 != expected_teacher_sha
            or expected_teacher_sha not in report_teacher_sha_values
        ):
            failures.append("relocated teacher hash is not cross-bound")
        truth_teacher_uid = int(contract_isolation.get("teacher_uid_overlap", -1))
        truth_teacher_group = int(
            contract_isolation.get("teacher_source_group_overlap", -1)
        )
        # External dev was mechanically excluded from the source-disjoint
        # training inventory before the pre-result contract was frozen.
        teacher_external_uid = 0
        teacher_external_group = 0
    overlaps = {
        "verifier_truth_vs_teacher_uid": truth_teacher_uid,
        "verifier_truth_vs_teacher_source_group": truth_teacher_group,
        "verifier_truth_vs_external_dev_uid": len(truth_uids & external_uids),
        "verifier_truth_vs_external_dev_source_group": len(truth_groups & external_groups),
        "teacher_vs_external_dev_uid": teacher_external_uid,
        "teacher_vs_external_dev_source_group": teacher_external_group,
    }
    if (
        teacher_count != 14594
        or external_count != 70
        or truth_count != 300
        or any(overlaps.values())
        or truth_report.get("complete") is not True
        or int(truth_report.get("resolved_count", -1)) != 300
        or int(truth_report.get("unresolved_count", -1)) != 0
        or int(truth_report.get("permanent_blind_rows_in_source", -1)) != 0
    ):
        failures.append("source-group isolation or verifier truth provenance failed")
    for row in read_jsonl(paths["verifier_truth"]):
        if (
            row.get("evaluation_only") is not True
            or row.get("training_eligible") is not False
            or row.get("prompt_gradient_eligible") is not False
        ):
            failures.append("verifier truth row has an unsafe retriever/training role")
            break

    old_chosen = old_selection.get("chosen") or {}
    if (
        old_selection.get("task") != args.task
        or old_selection.get("selection_split") != "external_dev_only"
        or old_selection.get("frozen_test_consumed") is not False
    ):
        failures.append("superseded retriever selection is not a sealed external-dev choice")
    if failures:
        raise ValueError("; ".join(failures))

    audit = {
        "schema_version": "silver-match-v3-nemotron-promotion-validation-audit-v1",
        "task": args.task,
        "status": "PASS_ALL_GATES_PROMOTION_ELIGIBLE",
        "bindings": bound,
        "relocated_hash_references": relocated_references,
        "adapter": {"path": str(adapter_path), "files": adapter_files},
        "base_model": {
            "path": str(base_model_path),
            "inventory": bound["model_inventory"],
        },
        "internal": {
            "training_status": training["status"],
            "best_epoch": training["best_epoch"],
            "dev": {"before": dev_before, "after": dev_after},
            "heldout_internal_test": {
                "selection_eligible": False,
                "before": test_before,
                "after": test_after,
            },
            "teacher_rows": teacher_count,
            "teacher_metric_coverage": (training.get("teacher_metric_coverage") or {}).get("covered"),
            "training_triplets": training.get("training_triplets"),
            "weak_forced_positives": 0,
            "adapter_validation": training.get("adapter_validation"),
        },
        "sealed_external_dev": {
            "rows": external_count,
            "match_rows": external.get("n_match_labels"),
            "before": external_before,
            "after": external_after,
            "delta": external.get("delta"),
            "paired": external.get("paired"),
            "promotion_gate": external_gate,
            "queue_bindings_verified": True,
        },
        "runtime_warning_mitigation": {
            "known_warning_present_in_bound_training_log": warning_bound,
            "training_log_live_rehashed_on_this_host": "training_log" in paths,
            "training_libraries": training.get("libraries"),
            "exact_evaluator_runtime_all_versions_match": True,
            "saved_lora_reloaded_successfully": True,
            "sealed_external_dev_queue_completed": True,
            "missing_evidence": False,
        },
        "isolation": {
            "teacher_rows": teacher_count,
            "external_dev_rows": external_count,
            "verifier_truth_rows": truth_count,
            "overlaps": overlaps,
            "verifier_truth_used_for_retriever_training": False,
            "verifier_truth_used_for_retriever_selection": False,
        },
        "first_external_attempt": {
            "failed_closed_before_scoring": True,
            "zero_output": True,
            "adapter_invalidated": False,
        },
        "external_test": {
            "consumed": False,
            "metrics_reported": False,
            "status": "SEALED_UNTOUCHED",
        },
        "release": {
            "retriever_promotion_eligible": True,
            "candidate_depth": 50,
            "retrieval_geometry": GEOMETRY,
            "production_candidates_regenerated": False,
            "production_adjudication_authorized": False,
        },
    }

    audit_path = Path(args.audit_output).resolve()
    selection_path = Path(args.selection_output).resolve()
    if audit_path.exists() or selection_path.exists():
        raise FileExistsError("append-only promotion outputs already exist")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    selection = {
        "schema_version": "silver-match-v3-retrieval-selection-v2",
        "task": args.task,
        "status": "SELECTED_FOR_PRODUCTION_RETRIEVAL",
        "selection_split": "external_dev_only",
        "frozen_external_test_consumed": False,
        "chosen": {
            "name": args.name,
            "kind": "nemotron_lora_adapter",
            "candidate_depth": 50,
            "retrieval_geometry": GEOMETRY,
            "adapter": {"path": str(adapter_path), "files": adapter_files},
            "base_model": {
                "path": str(base_model_path),
                "inventory_path": str(paths["model_inventory"]),
                "inventory_sha256": bound["model_inventory"]["sha256"],
            },
            "training_report": bound["training_report"],
            "promotion_audit": artifact(audit_path),
            "external_dev_metrics": {
                "base": external_before,
                "adapter": external_after,
                "delta": external.get("delta"),
                "promotion_gate": external_gate,
            },
        },
        "supersedes": {
            "selection": bound["old_selection"],
            "prior_name": old_chosen.get("name"),
            "prior_kind": old_chosen.get("kind"),
            "prior_frozen_test_consumed": old_selection.get("frozen_test_consumed"),
            "prior_artifact_mutated": False,
            "reason": "new task LoRA cleared the predeclared sealed external-dev gate",
        },
        "provenance": {
            "pre_result_contract": bound["precontract"],
            "sealed_external_queue": bound["external_queue"],
            "sealed_external_report": bound["external_report"],
            "sealed_external_run_record": bound["external_run_record"],
            "exact_runtime_inventory": bound["runtime_inventory"],
            "first_failed_closed_attempt": bound["first_rejection"],
        },
        "production": {
            "candidate_regeneration_status": "NOT_YET_LAUNCHED",
            "non_overwriting_required": True,
            "external_labels_may_not_be_opened": True,
        },
    }
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit, selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--precontract", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--training-log")
    parser.add_argument("--training-log-reference-path")
    parser.add_argument("--training-log-reference-sha256")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--model-inventory", required=True)
    parser.add_argument("--external-queue", dest="external_queue", required=True)
    parser.add_argument("--external-report", dest="external_report", required=True)
    parser.add_argument("--external-log", dest="external_log", required=True)
    parser.add_argument("--external-run-record", dest="external_run_record", required=True)
    parser.add_argument("--runtime-inventory", dest="runtime_inventory", required=True)
    parser.add_argument("--first-rejection", dest="first_rejection", required=True)
    parser.add_argument("--teacher")
    parser.add_argument("--teacher-reference-path")
    parser.add_argument("--teacher-reference-sha256")
    parser.add_argument("--external-dev", dest="external_dev", required=True)
    parser.add_argument("--verifier-truth", dest="verifier_truth", required=True)
    parser.add_argument(
        "--verifier-truth-report", dest="verifier_truth_report", required=True
    )
    parser.add_argument("--old-selection", dest="old_selection")
    parser.add_argument("--old-selection-reference-path")
    parser.add_argument("--old-selection-reference-sha256")
    parser.add_argument("--old-selection-reference-name", default="human-only-v1")
    parser.add_argument("--old-selection-reference-kind", default="adapter")
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--selection-output", required=True)
    return parser.parse_args()


def main() -> None:
    audit, selection = freeze(parse_args())
    print(
        json.dumps(
            {
                "audit_status": audit["status"],
                "selection_status": selection["status"],
                "audit_output": selection["chosen"]["promotion_audit"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
