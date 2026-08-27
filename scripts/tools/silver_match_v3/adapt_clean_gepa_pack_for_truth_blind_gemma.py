#!/usr/bin/env python3
"""Adapt one frozen clean-GEPA optimize pack for the generic Gemma queue.

The clean-GEPA pack predates the generic partition-role Gemma launcher and has
an intentionally different validation schema.  This adapter does not create
or inspect labels.  It revalidates the original clean pack, its pre-label
identity/exclusion lineage, and a mutually prediction-hidden pre-label audit;
then it emits:

* a compatibility source pack containing the same truth-hidden items and the
  exact canonical current bank;
* an all-item ``optimize`` partition (there is no data-dependent choice); and
* a compact inference manifest plus a source-pack audit understood by the
  existing generic queue freezer/auditor.

Every output is append-only.  Any non-empty prior-model-output root causes a
fail-closed rejection before the output directory is created.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


FORBIDDEN_LABEL_FIELDS = {
    "acceptable_metric_ids",
    "candidate_ids",
    "candidates",
    "decision",
    "label",
    "metric_id",
    "outcome",
    "prediction",
    "raw_response",
    "reason",
}


def _ref(path: Path, **extra: Any) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        **extra,
    }


def _bound_hash(entry: dict[str, Any], path: Path, name: str) -> None:
    if str(entry.get("sha256") or "") != sha256_file(path):
        raise ValueError(f"{name} hash differs from the frozen clean-GEPA lineage")


def _read_unique(path: Path, *, name: str) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    uids = [str(row.get("norm_uid") or "") for row in rows]
    if not rows or "" in uids or len(uids) != len(set(uids)):
        raise ValueError(f"{name} is empty or has missing/duplicate norm_uid values")
    return rows


def _copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as read_handle, destination.open("xb") as write_handle:
        shutil.copyfileobj(read_handle, write_handle)
    if sha256_file(source) != sha256_file(destination):
        raise AssertionError(f"byte-exact copy failed: {source}")


def adapt(args: argparse.Namespace) -> dict[str, Any]:
    task = str(args.task)
    role = str(args.role)
    if role != "optimize":
        raise ValueError("the truth-blind baseline adapter is optimize-only")
    expected_count = int(args.expected_count)
    if expected_count < 1:
        raise ValueError("expected count must be positive")

    manifest_path = Path(args.manifest).resolve()
    clean_root = Path(args.clean_pack_root).resolve()
    clean_validation_path = clean_root / "validation.json"
    clean_items_path = clean_root / "items.jsonl"
    clean_bank_path = clean_root / "bank.json"
    identities_path = Path(args.identities).resolve()
    identity_freeze_path = Path(args.identity_freeze).resolve()
    exclusion_inventory_path = Path(args.exclusion_inventory).resolve()
    prelabel_audit_path = Path(args.prelabel_independence_audit).resolve()
    canonical_bank_path = Path(args.canonical_bank).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)

    prior_output_roots: list[Path] = []
    prior_output_files: list[Path] = []
    for raw in args.prior_model_output_root:
        root = Path(raw).resolve()
        prior_output_roots.append(root)
        if root.exists() and not root.is_dir():
            raise ValueError(f"prior-model-output root is not a directory: {root}")
        if root.is_dir():
            prior_output_files.extend(sorted(path for path in root.rglob("*") if path.is_file()))
    if prior_output_files:
        raise ValueError(
            "successful or partial prior model output exists; refusing a post-prediction "
            f"partition: {prior_output_files[0]}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clean_validation = json.loads(clean_validation_path.read_text(encoding="utf-8"))
    identity_freeze = json.loads(identity_freeze_path.read_text(encoding="utf-8"))
    exclusion_inventory = json.loads(
        exclusion_inventory_path.read_text(encoding="utf-8")
    )
    prelabel_audit = json.loads(prelabel_audit_path.read_text(encoding="utf-8"))

    clean_inputs = clean_validation.get("inputs") or {}
    clean_outputs = clean_validation.get("outputs") or {}
    usage = clean_validation.get("usage_contract") or {}
    if (
        clean_validation.get("schema_version")
        != "silver-match-v3-clean-gepa-label-pack-v1"
        or clean_validation.get("status") != "FROZEN_TRUTH_HIDDEN_BEFORE_LABELING"
        or clean_validation.get("task") != task
        or clean_validation.get("gepa_role") != role
        or int(clean_validation.get("count", -1)) != expected_count
        or int(clean_validation.get("source_groups", -1)) != expected_count
        or clean_validation.get("truth_hidden") is not True
        or clean_validation.get(
            "prior_decisions_proposals_predictions_and_outcomes_hidden"
        )
        is not True
        or usage.get("optimize_may_mutate_prompts") is not True
        or usage.get("may_train_or_select_retriever") is not False
        or usage.get("may_use_for_mi_or_outcome_estimation") is not False
        or usage.get("may_use_as_test_or_blind_audit") is not False
    ):
        raise ValueError("source is not a frozen truth-hidden clean-GEPA optimize pack")
    for name, path in (
        ("manifest", manifest_path),
        ("identities", identities_path),
        ("identity_freeze", identity_freeze_path),
    ):
        _bound_hash(clean_inputs.get(name) or {}, path, f"clean-pack {name}")
    for name, path in (("items", clean_items_path), ("bank", clean_bank_path)):
        _bound_hash(clean_outputs.get(name) or {}, path, f"clean-pack {name}")

    freeze_contract = identity_freeze.get("content_contract") or {}
    frozen_identities = (identity_freeze.get("outputs") or {}).get("identities") or {}
    frozen_exclusions = identity_freeze.get("exclusion_union") or {}
    if (
        identity_freeze.get("schema_version")
        != "silver-match-v3-clean-gepa-panel-freeze-v1"
        or identity_freeze.get("status")
        != "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES"
        or identity_freeze.get("task") != task
        or identity_freeze.get("role") != role
        or identity_freeze.get("required_upstream_split") != "train"
        or int(identity_freeze.get("selected_count", -1)) != expected_count
        or int(identity_freeze.get("selected_source_groups", -1)) != expected_count
        or str(frozen_identities.get("sha256") or "") != sha256_file(identities_path)
        or freeze_contract.get("selection_uses_identity_and_source_group_only") is not True
        or any(
            freeze_contract.get(field) is not False
            for field in (
                "downstream_outcomes_read",
                "metric_ids_read",
                "model_prediction_fields_read",
                "truth_fields_read",
            )
        )
        or int(frozen_exclusions.get("selected_uid_overlap", -1)) != 0
        or int(frozen_exclusions.get("selected_source_group_overlap", -1)) != 0
    ):
        raise ValueError("identity panel is not a valid pre-label optimize freeze")

    exclusion_contract = exclusion_inventory.get("content_contract") or {}
    union_entry = exclusion_inventory.get("identity_union") or {}
    union_path = (
        Path(args.exclusion_union).resolve()
        if args.exclusion_union
        else Path(str(union_entry.get("path") or "")).resolve()
    )
    if (
        exclusion_inventory.get("schema_version")
        != "silver-match-v3-gepa-exclusion-union-v1"
        or exclusion_inventory.get("status")
        != "FROZEN_BEFORE_NEW_PANEL_SELECTION_PREDICTIONS_OR_LABELS"
        or exclusion_inventory.get("task") != task
        or exclusion_inventory.get("all_required_categories_present") is not True
        or exclusion_contract.get(
            "model_predictions_metric_ids_reasons_and_outcomes_used"
        )
        is not False
        or exclusion_contract.get("parsed_sources_used_only_identity_fields") is not True
        or exclusion_contract.get("sealed_test_or_outcome_structured_content_parsed")
        is not False
        or not union_path.is_file()
        or sha256_file(union_path) != str(union_entry.get("sha256") or "")
    ):
        raise ValueError("exclusion inventory is absent, label-bearing, or hash-drifted")

    if (
        prelabel_audit.get("schema_version")
        != "silver-match-v3-independent-pack-view-audit-v1"
        or prelabel_audit.get("status")
        != "FROZEN_MUTUALLY_PREDICTION_HIDDEN_BEFORE_LABELING"
        or prelabel_audit.get("task") != task
        or int(prelabel_audit.get("count", -1)) != expected_count
        or int(prelabel_audit.get("bank_metric_count", -1))
        != int(clean_validation.get("bank_metric_count", -2))
        or prelabel_audit.get("same_uid_set") is not True
        or prelabel_audit.get("same_bank_leaf_set") is not True
        or prelabel_audit.get("same_canonical_item_content_by_uid") is not True
        or prelabel_audit.get("same_frozen_source_pack") is not True
        or prelabel_audit.get("prior_truth_or_predictions_exposed_to_either_pass")
        is not False
        or prelabel_audit.get("candidate_proposals_exposed_to_either_pass") is not False
        or prelabel_audit.get("pass_predictions_mutually_visible") is not False
        or prelabel_audit.get("post_label_artifacts_present") is not False
    ):
        raise ValueError("independent pack views lack a clean pre-label attestation")

    identities = _read_unique(identities_path, name="optimize identities")
    items = _read_unique(clean_items_path, name="clean-GEPA items")
    exclusions = _read_unique(union_path, name="exclusion union")
    if len(identities) != expected_count or len(items) != expected_count:
        raise ValueError("identity or item count differs from the frozen optimize count")
    identity_by_uid = {str(row["norm_uid"]): row for row in identities}
    item_by_uid = {str(row["norm_uid"]): row for row in items}
    if set(identity_by_uid) != set(item_by_uid):
        raise ValueError("clean pack and identity freeze have different UID sets")
    identity_groups = [str(row.get("source_group") or "") for row in identities]
    item_groups = [str(row.get("source_group") or "") for row in items]
    if (
        "" in identity_groups
        or "" in item_groups
        or len(identity_groups) != len(set(identity_groups))
        or len(item_groups) != len(set(item_groups))
    ):
        raise ValueError("optimize source groups are missing or duplicated")
    for uid, identity in identity_by_uid.items():
        item = item_by_uid[uid]
        if (
            identity.get("task") != task
            or identity.get("gepa_role") != role
            or identity.get("upstream_split") != "train"
            or identity.get("corpus") != item.get("corpus")
            or identity.get("source_group") != item.get("source_group")
            or item.get("task") != task
            or item.get("gepa_role") != role
            or item.get("truth_hidden") is not True
            or item.get("predeclared_split") != "train"
            or FORBIDDEN_LABEL_FIELDS & set(identity)
            or FORBIDDEN_LABEL_FIELDS & set(item)
        ):
            raise ValueError(f"truth-hidden identity/item contract violation: {uid}")
    excluded_uids = {str(row["norm_uid"]) for row in exclusions}
    excluded_groups = {str(row.get("source_group") or "") for row in exclusions}
    if set(identity_by_uid) & excluded_uids or set(identity_groups) & excluded_groups:
        raise ValueError("optimize panel overlaps the frozen exclusion union")
    if (
        len(exclusions) != int(union_entry.get("uids", -1))
        or len(excluded_groups) != int(union_entry.get("source_groups", -1))
    ):
        raise ValueError("exclusion union counts drifted")

    canonical_bank = json.loads(canonical_bank_path.read_text(encoding="utf-8"))
    clean_bank = json.loads(clean_bank_path.read_text(encoding="utf-8"))
    canonical_ids = [str(row.get("metric_id") or "") for row in canonical_bank.get("metrics") or []]
    clean_ids = [str(row.get("metric_id") or "") for row in clean_bank.get("metrics") or []]
    bank_meta = (manifest.get("banks") or {}).get(task) or {}
    if (
        canonical_bank.get("task") != task
        or clean_bank.get("task") != task
        or not canonical_ids
        or "" in canonical_ids
        or len(canonical_ids) != len(set(canonical_ids))
        or len(clean_ids) != len(set(clean_ids))
        or set(clean_ids) != set(canonical_ids)
        or len(canonical_ids) != int(clean_validation.get("bank_metric_count", -1))
        or canonical_bank.get("source_sha256")
        != clean_validation.get("bank_source_sha256")
        or clean_bank.get("source_sha256") != canonical_bank.get("source_sha256")
        or bank_meta.get("source_sha256") != canonical_bank.get("source_sha256")
    ):
        raise ValueError("clean pack does not contain the exact complete current bank")
    _bound_hash(clean_inputs.get("bank_source") or {}, canonical_bank_path, "canonical bank")

    # All validation above happens before the first output path is created.
    source_pack = output_root / "source_pack"
    partition_root = output_root / "all_optimize_partition"
    source_pack.mkdir(parents=True, exist_ok=False)
    partition_root.mkdir(parents=True, exist_ok=False)
    adapted_items_path = source_pack / "items.jsonl"
    adapted_bank_path = source_pack / "bank.json"
    _copy_exact(clean_items_path, adapted_items_path)
    _copy_exact(canonical_bank_path, adapted_bank_path)
    chunk_paths: list[Path] = []
    for start in range(0, len(items), 25):
        path = source_pack / "chunks" / f"part-{start // 25:03d}.jsonl"
        write_jsonl(path, items[start : start + 25])
        chunk_paths.append(path)

    source_validation = {
        "schema_version": "silver-match-v3-frozen-identity-full-bank-source-pack-v1",
        "status": "FROZEN_CANDIDATE_AND_TRUTH_HIDDEN_BEFORE_LABELING",
        "task": task,
        "role": role,
        "binding_panel_key": "all_clean_gepa_optimize",
        "required_split": "train",
        "count": expected_count,
        "source_groups": expected_count,
        "corpora": dict(sorted(Counter(str(row["corpus"]) for row in items).items())),
        "bank_metric_count": len(canonical_ids),
        "bank_source_sha256": canonical_bank["source_sha256"],
        "truth_hidden": True,
        "candidate_proposals_hidden": True,
        "prior_labels_predictions_mi_and_outcomes_not_read": True,
        "inputs": {
            "manifest": _ref(manifest_path),
            "identities": _ref(identities_path),
            "identity_freeze": _ref(identity_freeze_path),
            "clean_gepa_validation": _ref(clean_validation_path),
            "exclusion_inventory": _ref(exclusion_inventory_path),
            "exclusion_union": _ref(union_path),
            "prelabel_independence_audit": _ref(prelabel_audit_path),
            "canonical_bank": _ref(canonical_bank_path),
        },
        "outputs": {
            "items": _ref(adapted_items_path),
            "bank": _ref(adapted_bank_path),
            "chunks": {str(path.resolve()): sha256_file(path) for path in chunk_paths},
        },
        "usage_contract": {
            "may_train_or_select_retriever": False,
            "may_use_for_mi_or_outcome_estimation": False,
            "may_use_as_final_blind_audit": False,
            "may_use_for_prompt_optimization_after_independent_truth_release": True,
            "baseline_outputs_may_enter_truth_consensus": False,
        },
    }
    source_validation_path = source_pack / "validation.json"
    source_validation_path.write_text(
        json.dumps(source_validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    assignments = [
        {
            "schema_version": "silver-match-v3-frozen-identity-partition-v1",
            "task": task,
            "norm_uid": row["norm_uid"],
            "corpus": row["corpus"],
            "source_group": row["source_group"],
            "upstream_split": row["upstream_split"],
            "remediation_role": role,
            "labels_predictions_metric_ids_reasons_mi_or_outcomes_used": False,
        }
        for row in identities
    ]
    assignments.sort(key=lambda row: str(row["norm_uid"]))
    assignments_path = partition_root / "assignments.jsonl"
    write_jsonl(assignments_path, assignments)
    partition_freeze = {
        "schema_version": "silver-match-v3-frozen-identity-partition-freeze-v1",
        "status": "FROZEN_BEFORE_ANY_DISTILLATION_LABELS_OR_PREDICTIONS",
        "task": task,
        "seed": None,
        "partition_method": "all_frozen_clean_gepa_optimize_identities_assigned_optimize",
        "role_counts": {role: expected_count},
        "role_by_corpus": {
            role: dict(sorted(Counter(str(row["corpus"]) for row in identities).items()))
        },
        "cross_role_uid_or_source_group_overlap": 0,
        "inputs": {
            "identity_freeze": _ref(identity_freeze_path),
            "identities": _ref(identities_path),
            "clean_gepa_validation": _ref(clean_validation_path),
            "exclusion_inventory": _ref(exclusion_inventory_path),
            "prelabel_independence_audit": _ref(prelabel_audit_path),
        },
        "output": _ref(assignments_path, count=expected_count),
        "content_contract": {
            "identity_and_source_group_fields_only": True,
            "all_frozen_rows_assigned_one_preexisting_role_without_selection": True,
            "labels_predictions_metric_ids_reasons_mi_or_outcomes_used": False,
            "successful_prior_model_output_files_observed": 0,
        },
    }
    partition_freeze_path = partition_root / "FREEZE.json"
    partition_freeze_path.write_text(
        json.dumps(partition_freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    inference_manifest = {
        "schema_version": "silver-match-v3-truth-blind-task-local-inference-manifest-v1",
        "source_manifest": _ref(manifest_path),
        "corpora": {
            corpus: {
                "task": task,
                "path": str(adapted_items_path),
                "sha256": sha256_file(adapted_items_path),
            }
            for corpus in sorted({str(row["corpus"]) for row in items})
        },
        "banks": {
            task: {
                "path": str(adapted_bank_path),
                "sha256": sha256_file(adapted_bank_path),
                "source_sha256": canonical_bank["source_sha256"],
                "metric_count": len(canonical_ids),
            }
        },
    }
    inference_manifest_path = output_root / "inference_manifest.json"
    inference_manifest_path.write_text(
        json.dumps(inference_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    output_absence = {
        "schema_version": "silver-match-v3-prior-model-output-absence-v1",
        "status": "ZERO_SUCCESSFUL_OR_PARTIAL_PRIOR_MODEL_OUTPUT_FILES",
        "task": task,
        "checked_roots": [str(path) for path in prior_output_roots],
        "observed_files": 0,
        "checked_before_adapter_outputs_created": True,
    }
    output_absence_path = output_root / "PRIOR_MODEL_OUTPUT_ABSENCE.json"
    output_absence_path.write_text(
        json.dumps(output_absence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    source_audit = {
        "schema_version": "silver-match-v3-clean-gepa-gemma-source-adapter-audit-v1",
        "status": "EXACT_TRUTH_AND_CANDIDATE_HIDDEN_PACK_PASS",
        "task": task,
        "role": role,
        "count": expected_count,
        "source_groups": expected_count,
        "bank_metric_count": len(canonical_ids),
        "bank_source_sha256": canonical_bank["source_sha256"],
        "uid_overlap_with_all_exclusions": 0,
        "source_group_overlap_with_all_exclusions": 0,
        "truth_hidden": True,
        "candidate_proposals_hidden": True,
        "labels_predictions_mi_and_outcomes_read": False,
        "successful_prior_model_output_files_observed": 0,
        "artifacts": {
            "manifest": _ref(manifest_path),
            "clean_gepa_validation": _ref(clean_validation_path),
            "identities": _ref(identities_path),
            "identity_freeze": _ref(identity_freeze_path),
            "exclusion_inventory": _ref(exclusion_inventory_path),
            "exclusion_union": _ref(union_path),
            "prelabel_independence_audit": _ref(prelabel_audit_path),
            "canonical_bank": _ref(canonical_bank_path),
            "source_pack_validation": _ref(source_validation_path),
            "source_pack_items": _ref(adapted_items_path),
            "source_pack_bank": _ref(adapted_bank_path),
            "partition": _ref(assignments_path),
            "partition_freeze": _ref(partition_freeze_path),
            "inference_manifest": _ref(inference_manifest_path),
            "prior_model_output_absence": _ref(output_absence_path),
        },
    }
    source_audit_path = output_root / "SOURCE_PACK_AUDIT.json"
    source_audit_path.write_text(
        json.dumps(source_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **source_audit,
        "source_pack_validation": _ref(source_validation_path),
        "source_pack_audit": _ref(source_audit_path),
        "partition": _ref(assignments_path),
        "partition_freeze": _ref(partition_freeze_path),
        "inference_manifest": _ref(inference_manifest_path),
        "prior_model_output_absence": _ref(output_absence_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--role", default="optimize")
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--clean-pack-root", required=True)
    parser.add_argument("--identities", required=True)
    parser.add_argument("--identity-freeze", required=True)
    parser.add_argument("--exclusion-inventory", required=True)
    parser.add_argument(
        "--exclusion-union",
        help="relocated exclusion-union JSONL; its hash must match the inventory",
    )
    parser.add_argument("--prelabel-independence-audit", required=True)
    parser.add_argument("--canonical-bank", required=True)
    parser.add_argument("--prior-model-output-root", action="append", default=[])
    parser.add_argument("--output-root", required=True)
    result = adapt(parser.parse_args())
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
