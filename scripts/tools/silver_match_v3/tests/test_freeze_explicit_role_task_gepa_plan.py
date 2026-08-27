import copy
import hashlib
import json
from argparse import Namespace
from collections import Counter
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.common import sha256_file, write_jsonl
from scripts.tools.silver_match_v3.freeze_explicit_role_task_gepa_plan import freeze
from scripts.tools.silver_match_v3.make_calibration import split_for, split_group_for
from scripts.tools.silver_match_v3.train_nemotron_lora import (
    source_group_key,
    split_source_group,
)


def _ref_for_test(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _write_role(root: Path, task: str, role: str, norms: list[dict]) -> dict[str, str]:
    role_root = root / role
    role_root.mkdir(parents=True)
    truth_path = role_root / "truth.jsonl"
    candidates_path = role_root / "candidates.jsonl"
    identities_path = role_root / "identities.jsonl"
    freeze_path = role_root / "FREEZE.json"
    truth = []
    identities = []
    candidates = []
    for index, row in enumerate(norms):
        group = split_group_for(row)
        truth.append(
            {
                "norm_uid": row["norm_uid"],
                "task": task,
                "source_group": group,
                "gepa_role": role,
                "split": "train",
                "decision": "MATCH" if index == 0 else "NO_CANDIDATE_FITS",
                "metric_id": "a0" if index == 0 else None,
                "confidence": "high",
            }
        )
        identities.append(
            {
                "schema_version": "silver-match-v3-clean-gepa-panel-identity-v1",
                "norm_uid": row["norm_uid"],
                "task": task,
                "corpus": row["corpus"],
                "source_group": group,
                "gepa_role": role,
                "upstream_split": "train",
                "permanently_excluded_from_mi_and_outcome_estimation": True,
                "permanently_excluded_from_retriever_gradients": True,
            }
        )
        candidates.append(
            {
                "norm_uid": row["norm_uid"],
                "task": task,
                "bank_source_sha256": "bank-hash",
                "candidates": [
                    {"metric_id": "a0", "rank": 1},
                    {"metric_id": "a1", "rank": 2},
                ],
            }
        )
    write_jsonl(truth_path, truth)
    write_jsonl(candidates_path, candidates)
    write_jsonl(identities_path, identities)
    freeze_path.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-clean-gepa-panel-freeze-v1",
                "status": "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES",
                "task": task,
                "role": role,
                "required_upstream_split": "train",
                "selected_count": len(identities),
                "selected_source_groups": len(identities),
                "outputs": {
                    "identities": {
                        "path": str(identities_path),
                        "sha256": sha256_file(identities_path),
                    }
                },
                "content_contract": {
                    "selection_uses_identity_and_source_group_only": True,
                    "downstream_outcomes_read": False,
                    "metric_ids_read": False,
                    "model_prediction_fields_read": False,
                    "truth_fields_read": False,
                },
            }
        )
    )
    return {
        "truth": str(truth_path),
        "candidates": str(candidates_path),
        "identities": str(identities_path),
        "freeze": str(freeze_path),
    }


def _refresh_role_freeze(paths: dict[str, str]) -> None:
    freeze_path = Path(paths["freeze"])
    payload = json.loads(freeze_path.read_text())
    payload["outputs"]["identities"]["sha256"] = sha256_file(Path(paths["identities"]))
    freeze_path.write_text(json.dumps(payload))


def _write_role_bindings(
    paths: dict[str, str], task: str, role: str, manifest_path: Path, bank_path: Path
) -> None:
    role_root = Path(paths["truth"]).parent
    candidates_path = Path(paths["candidates"])
    identities_path = Path(paths["identities"])
    freeze_path = Path(paths["freeze"])
    truth_path = Path(paths["truth"])
    source_path = role_root / "candidate_source.jsonl"
    write_jsonl(
        source_path,
        [json.loads(line) for line in candidates_path.read_text().splitlines()],
    )
    fusion_path = role_root / "fusion.json"
    fusion_path.write_text("{}\n")
    Path(str(source_path) + ".meta.json").write_text(
        json.dumps(
            {
                "output_sha256": sha256_file(source_path),
                "manifest_sha256": sha256_file(manifest_path),
                "output_k": 2,
                "query_format": "nemotron",
                "query_views": "evidence+statement",
                "dense_query_instruction": True,
                "encoder": "local-encoder",
                "fusion_weights": str(fusion_path),
                "fusion_weights_sha256": sha256_file(fusion_path),
            }
        )
    )
    run_config_path = role_root / "run_config.json"
    run_config_path.write_text(
        json.dumps(
            {
                "task": task,
                "selection_k": 2,
                "query_instruction": "retrieve exact metric",
            }
        )
    )
    all_norms = {}
    manifest = json.loads(manifest_path.read_text())
    for corpus in manifest["corpora"].values():
        for norm in (
            json.loads(line) for line in Path(corpus["path"]).read_text().splitlines()
        ):
            all_norms[norm["norm_uid"]] = norm
    identity_rows = [
        json.loads(line) for line in identities_path.read_text().splitlines()
    ]
    role_rows = []
    for identity in identity_rows:
        norm = all_norms[identity["norm_uid"]]
        retriever_group = source_group_key(norm)
        split = split_source_group(retriever_group)
        assert split == "train"
        role_rows.append(
            {
                "schema_version": "silver-match-v3-upstream-role-reference-v1",
                "norm_uid": identity["norm_uid"],
                "task": task,
                "corpus": norm["corpus"],
                "source_group": split_group_for(norm),
                "retriever_source_group": retriever_group,
                "split": split,
                "split_seed": 73129,
                "train_percent": 80,
                "dev_percent": 10,
                "test_percent": 10,
            }
        )
    roles_path = role_root / "roles.jsonl"
    audited_roles_path = role_root / "audited_roles.jsonl"
    write_jsonl(roles_path, role_rows)
    write_jsonl(
        audited_roles_path,
        ({"norm_uid": row["norm_uid"], "split": row["split"]} for row in role_rows),
    )
    role_counts = dict(Counter(row["split"] for row in role_rows))
    group_counts = {
        split: len(
            {
                row["retriever_source_group"]
                for row in role_rows
                if row["split"] == split
            }
        )
        for split in ("train", "dev", "test")
    }
    upstream_path = role_root / "UPSTREAM_FREEZE.json"
    upstream_path.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-upstream-role-reference-freeze-v1",
                "status": "FROZEN_AND_AUDIT_VERIFIED",
                "task": task,
                "bank_source_sha256": "bank-hash",
                "minimum_k": 2,
                "candidate_rows": len(role_rows),
                "roles": role_counts,
                "role_source_groups": group_counts,
                "split_policy": {
                    "function": "train_nemotron_lora.split_source_group",
                    "group_function": "train_nemotron_lora.source_group_key",
                    "split_seed": 73129,
                    "train_percent": 80,
                    "dev_percent": 10,
                    "test_percent": 10,
                },
                "output": {
                    "path": str(roles_path),
                    "sha256": sha256_file(roles_path),
                },
                "audit_verification": {
                    "path": str(audited_roles_path),
                    "sha256": sha256_file(audited_roles_path),
                    "overlap": len(role_rows),
                    "mismatches": 0,
                    "exact_role_matches": len(role_rows),
                },
                "inputs": {
                    "manifest": {
                        "path": str(manifest_path),
                        "sha256": sha256_file(manifest_path),
                    },
                    "candidates": {
                        "path": str(source_path),
                        "sha256": sha256_file(source_path),
                    },
                    "run_config": {
                        "path": str(run_config_path),
                        "sha256": sha256_file(run_config_path),
                    },
                },
            }
        )
    )
    role_freeze = json.loads(freeze_path.read_text())
    role_freeze["inputs"] = {
        "upstream_role_reference": {
            "path": str(roles_path),
            "sha256": sha256_file(roles_path),
            "field": "split",
            "role_counts": role_counts,
            "authoritative": True,
        }
    }
    freeze_path.write_text(json.dumps(role_freeze))
    candidate_audit_path = role_root / "CANDIDATE_AUDIT.json"
    candidate_audit_path.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-clean-gepa-label-pack-v1",
                "status": "FROZEN_TRUTH_HIDDEN_BEFORE_LABELING",
                "truth_hidden": True,
                "task": task,
                "gepa_role": role,
                "count": 2,
                "candidate_k": 2,
                "bank_source_sha256": "bank-hash",
                "inputs": {
                    "manifest": {
                        "path": str(manifest_path),
                        "sha256": sha256_file(manifest_path),
                    },
                    "bank_source": {
                        "path": str(bank_path),
                        "sha256": sha256_file(bank_path),
                    },
                    "candidate_source": {
                        "path": str(source_path),
                        "sha256": sha256_file(source_path),
                    },
                    "identities": {
                        "path": str(identities_path),
                        "sha256": sha256_file(identities_path),
                    },
                    "identity_freeze": {
                        "path": str(freeze_path),
                        "sha256": sha256_file(freeze_path),
                    },
                    "upstream_role_freeze": {
                        "path": str(upstream_path),
                        "sha256": sha256_file(upstream_path),
                    },
                },
                "outputs": {
                    "candidates": {
                        "path": str(candidates_path),
                        "sha256": sha256_file(candidates_path),
                    }
                },
            }
        )
    )
    consensus_path = role_root / "EXACT_CONSENSUS_REPORT.json"
    consensus_path.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-exact-multi-pass-truth-report-v1",
                "complete": True,
                "task": task,
                "gepa_role": role,
                "resolved_count": 2,
                "unresolved_count": 0,
                "outputs": {"resolved": {"sha256": sha256_file(truth_path)}},
            }
        )
    )
    independence_path = role_root / "INDEPENDENCE_AUDIT.json"
    independence_path.write_text("{}\n")
    released_passes = []
    for pass_name in ("pass_a", "pass_b"):
        pass_root = role_root / pass_name
        (pass_root / "raw_labels").mkdir(parents=True)
        (pass_root / "logs").mkdir()
        labels_path = pass_root / "labels.validated.jsonl"
        validation_path = pass_root / "labels.validation.json"
        pack_path = pass_root / "validation.json"
        raw_path = pass_root / "raw_labels" / "part-000.json"
        log_path = pass_root / "logs" / "part-000.log"
        transcript_path = pass_root / "STRICT_TRANSCRIPT_ISOLATION_AUDIT.json"
        write_jsonl(
            labels_path,
            [json.loads(line) for line in truth_path.read_text().splitlines()],
        )
        validation_path.write_text("{}\n")
        pack_path.write_text("{}\n")
        raw_path.write_text("{}\n")
        log_path.write_text("sandbox: read-only\napproval: never\n")
        transcript_path.write_text('{"status":"PASS"}\n')
        released_passes.append(
            {
                "name": pass_name,
                "labels": {
                    "path": str(labels_path),
                    "sha256": sha256_file(labels_path),
                },
                "label_validation": {
                    "path": str(validation_path),
                    "sha256": sha256_file(validation_path),
                },
                "pack_validation": {
                    "path": str(pack_path),
                    "sha256": sha256_file(pack_path),
                },
                "transcript_audit": {
                    "mode": "strict_isolation_audit",
                    "path": str(transcript_path),
                    "sha256": sha256_file(transcript_path),
                    "full_pack_artifact_binding": True,
                    "artifact_equivalence_verified": True,
                    "guide_sha256": "1" * 64,
                },
            }
        )
    truth_release_path = role_root / "TRUTH_RELEASE.json"
    truth_release_path.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-clean-gepa-exact-truth-release-v2",
                "status": "FROZEN_EXACT_TRUTH_RELEASE_AUDITED",
                "task": task,
                "role": role,
                "count": 2,
                "truth": {"path": str(truth_path), "sha256": sha256_file(truth_path)},
                "consensus_report": {
                    "path": str(consensus_path),
                    "sha256": sha256_file(consensus_path),
                },
                "role_freeze": {
                    "path": str(freeze_path),
                    "sha256": sha256_file(freeze_path),
                },
                "identities": {
                    "path": str(identities_path),
                    "sha256": sha256_file(identities_path),
                },
                "independence_audit": {
                    "path": str(independence_path),
                    "sha256": sha256_file(independence_path),
                },
                "candidate_release": {
                    "path": str(candidate_audit_path),
                    "sha256": sha256_file(candidate_audit_path),
                    "candidate_sha256": sha256_file(candidates_path),
                },
                "consensus_replay": {
                    "round_count": 2,
                    "resolved_count": 2,
                    "unresolved_count": 0,
                    "round_metadata_verified": True,
                    "released_decision_metric_confidence_supporters_exact": True,
                },
                "passes": released_passes,
                "scientific_contract": {
                    "exact_decision_and_leaf_consensus_complete": True,
                    "consensus_recomputed_from_bound_pass_labels": True,
                    "all_pass_labels_and_validations_hash_bound": True,
                    "transcripts_hash_bound_and_leakage_audited": True,
                    "strict_transcript_pass_required_for_every_consensus_pass": True,
                    "cross_workspace_artifacts_hash_equivalent": True,
                    "legacy_transcripts_allowed": False,
                    "truth_may_be_used_only_for_declared_gepa_role": True,
                },
            }
        )
    )
    paths["candidate_audit"] = str(candidate_audit_path)
    paths["truth_release"] = str(truth_release_path)
    paths["upstream_roles"] = str(roles_path)
    paths["upstream_freeze"] = str(upstream_path)


def _refresh_role_bindings(paths: dict[str, str]) -> None:
    freeze_path = Path(paths["freeze"])
    freeze = json.loads(freeze_path.read_text())
    upstream_ref = freeze["inputs"]["upstream_role_reference"]
    roles_path = Path(upstream_ref["path"])
    role_rows = [json.loads(line) for line in roles_path.read_text().splitlines()]
    candidate_audit_path = Path(paths["candidate_audit"])
    candidate_audit = json.loads(candidate_audit_path.read_text())
    upstream_path = Path(candidate_audit["inputs"]["upstream_role_freeze"]["path"])
    upstream = json.loads(upstream_path.read_text())
    audit_path = Path(upstream["audit_verification"]["path"])
    write_jsonl(
        audit_path,
        ({"norm_uid": row["norm_uid"], "split": row["split"]} for row in role_rows),
    )
    role_counts = dict(Counter(row["split"] for row in role_rows))
    upstream["roles"] = role_counts
    upstream["role_source_groups"] = {
        split: len(
            {
                row["retriever_source_group"]
                for row in role_rows
                if row["split"] == split
            }
        )
        for split in ("train", "dev", "test")
    }
    upstream["output"]["sha256"] = sha256_file(roles_path)
    upstream["audit_verification"].update(
        {
            "sha256": sha256_file(audit_path),
            "overlap": len(role_rows),
            "mismatches": 0,
            "exact_role_matches": len(role_rows),
        }
    )
    upstream_path.write_text(json.dumps(upstream))
    freeze["outputs"]["identities"]["sha256"] = sha256_file(Path(paths["identities"]))
    freeze["inputs"]["upstream_role_reference"].update(
        {"sha256": sha256_file(roles_path), "role_counts": role_counts}
    )
    freeze_path.write_text(json.dumps(freeze))
    candidate_audit["inputs"]["upstream_role_freeze"]["sha256"] = sha256_file(
        upstream_path
    )
    candidate_audit["inputs"]["identities"]["sha256"] = sha256_file(
        Path(paths["identities"])
    )
    candidate_audit["inputs"]["identity_freeze"]["sha256"] = sha256_file(freeze_path)
    candidate_audit_path.write_text(json.dumps(candidate_audit))
    consensus_path = Path(paths["truth"]).parent / "EXACT_CONSENSUS_REPORT.json"
    consensus = json.loads(consensus_path.read_text())
    consensus["outputs"]["resolved"]["sha256"] = sha256_file(Path(paths["truth"]))
    consensus_path.write_text(json.dumps(consensus))
    release_path = Path(paths["truth_release"])
    release = json.loads(release_path.read_text())
    release["truth"]["sha256"] = sha256_file(Path(paths["truth"]))
    release["role_freeze"]["sha256"] = sha256_file(freeze_path)
    release["identities"]["sha256"] = sha256_file(Path(paths["identities"]))
    release["candidate_release"]["sha256"] = sha256_file(candidate_audit_path)
    release["consensus_report"]["sha256"] = sha256_file(consensus_path)
    release_path.write_text(json.dumps(release))


def _fixture(tmp_path: Path) -> Namespace:
    task = "code-review"
    upstream_train_norms = []
    upstream_train_calibration_nontrain = []
    ordinal = 0
    while len(upstream_train_norms) < 6 or not upstream_train_calibration_nontrain:
        ordinal += 1
        row = {
            "norm_uid": f"u-{ordinal}",
            "task": task,
            "corpus": "c",
            "source_id": f"source-{ordinal}",
            "row": ordinal,
            "norm": f"criterion {ordinal}",
        }
        if split_source_group(source_group_key(row)) == "train":
            if split_for(split_group_for(row)) != "train":
                upstream_train_calibration_nontrain.append(row)
            else:
                upstream_train_norms.append(row)
    train_norms = [upstream_train_calibration_nontrain[0], *upstream_train_norms[:5]]
    norms_path = tmp_path / "norms.jsonl"
    write_jsonl(norms_path, train_norms)
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(
        json.dumps(
            {
                "source_sha256": "bank-hash",
                "metrics": [
                    {"metric_id": "a0", "name": "zero", "description": "zero"},
                    {"metric_id": "a1", "name": "one", "description": "one"},
                ],
            }
        )
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "corpora": {"c": {"task": task, "path": str(norms_path)}},
                "banks": {task: {"path": str(bank_path), "source_sha256": "bank-hash"}},
            }
        )
    )
    optimize = _write_role(tmp_path, task, "optimize", train_norms[:2])
    select = _write_role(tmp_path, task, "select", train_norms[2:4])
    _write_role_bindings(optimize, task, "optimize", manifest_path, bank_path)
    _write_role_bindings(select, task, "select", manifest_path, bank_path)
    exclusion_path = tmp_path / "exclude.jsonl"
    excluded_norm = train_norms[4]
    exclusion_row = {
        "schema_version": "silver-match-v3-gepa-exclusion-identity-v1",
        "norm_uid": excluded_norm["norm_uid"],
        "task": task,
        "corpus": excluded_norm["corpus"],
        "source_group": split_group_for(excluded_norm),
        "upstream_split": "train",
    }
    write_jsonl(exclusion_path, [exclusion_row])
    source_path = tmp_path / "prior_inspected.jsonl"
    write_jsonl(source_path, [exclusion_row])
    inventory_path = tmp_path / "EXCLUSION_INVENTORY.json"
    inventory_path.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-gepa-exclusion-union-v1",
                "status": "FROZEN_BEFORE_NEW_PANEL_SELECTION_PREDICTIONS_OR_LABELS",
                "task": task,
                "all_required_categories_present": True,
                "required_categories": ["legacy_anchor_pack"],
                "observed_categories": ["legacy_anchor_pack"],
                "content_contract": {
                    "parsed_sources_used_only_identity_fields": True,
                    "model_predictions_metric_ids_reasons_and_outcomes_used": False,
                    "sealed_test_or_outcome_structured_content_parsed": False,
                },
                "inputs": {"manifest": _ref_for_test(manifest_path)},
                "identity_union": {
                    **_ref_for_test(exclusion_path),
                    "uids": 1,
                    "source_groups": 1,
                    "by_corpus": {excluded_norm["corpus"]: 1},
                    "by_upstream_split": {"train": 1},
                },
                "sources": {
                    str(source_path): {
                        "sha256": sha256_file(source_path),
                        "format": "jsonl",
                        "category": "legacy_anchor_pack",
                        "fields_used": ["norm_uid", "source_group"],
                        "canonical_source_group_recomputed": True,
                        "structured_content_parsed": True,
                        "supplied_source_group_mismatch_count": 0,
                        "uids": 1,
                        "source_groups": 1,
                    }
                },
            }
        )
    )
    adjudicator_prompt = tmp_path / "adjudicator.txt"
    verifier_prompt = tmp_path / "verifier.txt"
    adjudicator_prompt.write_text("adjudicator\n")
    verifier_prompt.write_text("verifier\n")
    predeclaration_path = tmp_path / "predeclaration.json"
    predeclaration_path.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-task-local-gepa-predeclaration-v1",
                "status": "FROZEN_AND_EXECUTION_AUTHORIZED",
                "candidate_k": 2,
                "split": {
                    "gepa_seed": 7,
                    "gepa_dev_percent": 25,
                    "minimum_prompt_train_rows": 2,
                    "minimum_prompt_dev_rows": 2,
                },
                "selection_gate": {
                    "minimum_point_precision": 0.9,
                    "minimum_wilson_95_lower": 0.8,
                    "minimum_retained_support": 2,
                },
                "api": {
                    "base_url": "https://openrouter.ai/api/v1",
                    "model": "google/gemma-4-31b-it",
                    "maximum_total_logical_requests_per_task": 1000,
                    "implicit_transport_retries": 0,
                },
                "direct_batch": {
                    "model": "local-gemma",
                    "batch_size": 16,
                    "gpu_memory_utilization": 0.8,
                },
                "tasks": {
                    task: {
                        "execution_authorized": True,
                        "blocker": None,
                        "execution_evidence": {
                            "optimize_truth_release_sha256": sha256_file(
                                Path(optimize["truth_release"])
                            ),
                            "select_truth_release_sha256": sha256_file(
                                Path(select["truth_release"])
                            ),
                            "complete_exclusion_union_sha256": sha256_file(
                                exclusion_path
                            ),
                            "exclusion_inventory_sha256": sha256_file(inventory_path),
                        },
                        "adjudicator_variants": [
                            {
                                "name": "r0",
                                "combined_prompt_sha256": hashlib.sha256(
                                    b"adjudicator\n"
                                ).hexdigest(),
                            }
                        ],
                        "verifier_variants": [
                            {
                                "name": "v0",
                                "combined_prompt_sha256": hashlib.sha256(
                                    b"verifier\n"
                                ).hexdigest(),
                            }
                        ],
                    }
                },
            }
        )
    )
    activation = json.loads(predeclaration_path.read_text())
    parent = copy.deepcopy(activation)
    parent["created_date"] = "2026-07-12"
    parent["status"] = "PREDECLARED_PENDING_CANONICAL_PACKS_AND_COMPLETE_EXCLUSIONS"
    parent_task = parent["tasks"][task]
    parent_task["execution_authorized"] = False
    parent_task["blocker"] = "canonical clean role releases and exclusions pending"
    parent_task.pop("execution_evidence")
    parent_path = tmp_path / "parent_predeclaration.json"
    parent_path.write_text(json.dumps(parent))
    activation["parent_predeclaration"] = {
        "path": str(parent_path),
        "sha256": sha256_file(parent_path),
        "variants_gates_models_and_budgets_changed": False,
    }
    activation["activation_scope"] = "clean explicit-role artifacts only"
    predeclaration_path.write_text(json.dumps(activation))
    return Namespace(
        task=task,
        predeclaration=str(predeclaration_path),
        manifest=str(manifest_path),
        optimize_truth=optimize["truth"],
        optimize_candidates=optimize["candidates"],
        optimize_freeze=optimize["freeze"],
        optimize_identities=optimize["identities"],
        optimize_truth_release=optimize["truth_release"],
        optimize_candidate_audit=optimize["candidate_audit"],
        select_truth=select["truth"],
        select_candidates=select["candidates"],
        select_freeze=select["freeze"],
        select_identities=select["identities"],
        select_truth_release=select["truth_release"],
        select_candidate_audit=select["candidate_audit"],
        exclude_reference=[str(exclusion_path)],
        adjudicator_variant=[f"r0={adjudicator_prompt}"],
        verifier_variant=[f"v0={verifier_prompt}"],
        output_root=str(tmp_path / "plan"),
        candidate_k=2,
        minimum_train=2,
        minimum_dev=2,
        minimum_point_precision=0.9,
        minimum_wilson_lower=0.8,
        minimum_retained=2,
        max_total_api_requests=100,
        api_base_url="https://openrouter.ai/api/v1",
        api_key_file="~/.openrouter-api-key.txt",
        model="google/gemma-4-31b-it",
        concurrency=2,
        direct_model="local-gemma",
        direct_batch_size=16,
        gpu_memory_utilization=0.8,
    )


def test_explicit_role_plan_happy_path_binds_roles_and_commands(tmp_path):
    args = _fixture(tmp_path)
    norms_path = Path(
        json.loads(Path(args.manifest).read_text())["corpora"]["c"]["path"]
    )
    norms = {
        row["norm_uid"]: row
        for row in map(json.loads, norms_path.read_text().splitlines())
    }
    first_uid = json.loads(Path(args.optimize_truth).read_text().splitlines()[0])[
        "norm_uid"
    ]
    # Regression guard: authoritative Nemotron-train provenance is allowed even
    # when the unrelated legacy calibration hash assigns another role.
    assert split_for(split_group_for(norms[first_uid])) != "train"
    result = freeze(args)
    plan = json.loads((Path(args.output_root) / "COMMAND_PLAN.json").read_text())
    assert result["maximum_total_api_requests"] == 40
    assert set(plan["inputs"]["explicit_roles"]) == {"optimize", "select"}
    assert plan["roles"]["optimize"]["count"] == 2
    assert plan["roles"]["select"]["count"] == 2
    inference = [
        row for row in plan["commands"] if row["stage"] in {"adjudicator", "verifier"}
    ]
    assert inference and all("direct_batch_command" in row for row in inference)
    assert all("--max-api-requests" in row["command"]["argv"] for row in inference)


def test_explicit_role_plan_rejects_legacy_transcript_release(tmp_path):
    args = _fixture(tmp_path)
    release_path = Path(args.optimize_truth_release)
    release = json.loads(release_path.read_text())
    release["passes"][0]["transcript_audit"] = {
        "mode": "legacy_sandboxed_transcript_leakage_audit",
        "complete": True,
    }
    release_path.write_text(json.dumps(release))
    activation_path = Path(args.predeclaration)
    activation = json.loads(activation_path.read_text())
    activation["tasks"][args.task]["execution_evidence"][
        "optimize_truth_release_sha256"
    ] = sha256_file(release_path)
    activation_path.write_text(json.dumps(activation))
    with pytest.raises(ValueError, match="strict fully bound transcript audit"):
        freeze(args)


def test_explicit_role_plan_rejects_cross_role_source_group(tmp_path):
    args = _fixture(tmp_path)
    norms_path = Path(
        json.loads(Path(args.manifest).read_text())["corpora"]["c"]["path"]
    )
    norms = [json.loads(line) for line in norms_path.read_text().splitlines()]
    optimize_uid = json.loads(Path(args.optimize_truth).read_text().splitlines()[0])[
        "norm_uid"
    ]
    select_rows = [
        json.loads(line) for line in Path(args.select_truth).read_text().splitlines()
    ]
    optimize_row = next(row for row in norms if row["norm_uid"] == optimize_uid)
    target = next(row for row in norms if row["norm_uid"] == select_rows[0]["norm_uid"])
    target["source_id"] = optimize_row["source_id"]
    write_jsonl(norms_path, norms)
    group = split_group_for(target)
    select_rows[0]["source_group"] = group
    write_jsonl(Path(args.select_truth), select_rows)
    identities = [
        json.loads(line)
        for line in Path(args.select_identities).read_text().splitlines()
    ]
    identities[0]["source_group"] = group
    write_jsonl(Path(args.select_identities), identities)
    upstream_rows = [
        json.loads(line)
        for line in Path(args.select_candidates)
        .parent.joinpath("roles.jsonl")
        .read_text()
        .splitlines()
    ]
    upstream = next(
        row for row in upstream_rows if row["norm_uid"] == target["norm_uid"]
    )
    upstream["source_group"] = group
    upstream["retriever_source_group"] = source_group_key(target)
    upstream["split"] = split_source_group(upstream["retriever_source_group"])
    assert upstream["split"] == "train"
    write_jsonl(Path(args.select_candidates).parent / "roles.jsonl", upstream_rows)
    _refresh_role_bindings(
        {
            "freeze": args.select_freeze,
            "identities": args.select_identities,
            "truth": args.select_truth,
            "truth_release": args.select_truth_release,
            "candidate_audit": args.select_candidate_audit,
        }
    )
    with pytest.raises(ValueError, match="overlap by canonical source group"):
        freeze(args)


def test_explicit_role_plan_rejects_role_freeze_drift(tmp_path):
    args = _fixture(tmp_path)
    payload = json.loads(Path(args.optimize_freeze).read_text())
    payload["selected_count"] += 1
    Path(args.optimize_freeze).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="role freeze"):
        freeze(args)


def test_explicit_role_plan_rejects_truth_label_drift(tmp_path):
    args = _fixture(tmp_path)
    rows = [
        json.loads(line) for line in Path(args.optimize_truth).read_text().splitlines()
    ]
    rows[0]["metric_id"] = "a1"
    write_jsonl(Path(args.optimize_truth), rows)
    with pytest.raises(ValueError, match="truth release"):
        freeze(args)


def test_explicit_role_plan_rejects_candidate_audit_or_bank_drift(tmp_path):
    args = _fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in Path(args.optimize_candidates).read_text().splitlines()
    ]
    rows[0]["bank_source_sha256"] = "wrong-bank"
    write_jsonl(Path(args.optimize_candidates), rows)
    with pytest.raises(ValueError, match="candidate audit"):
        freeze(args)


def test_explicit_role_plan_rejects_weakened_predeclaration_status(tmp_path):
    args = _fixture(tmp_path)
    lock = json.loads(Path(args.predeclaration).read_text())
    lock["status"] = "PREDECLARED_PENDING_CANONICAL_PACKS_AND_COMPLETE_EXCLUSIONS"
    Path(args.predeclaration).write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="not execution-authorized"):
        freeze(args)


def test_explicit_role_plan_rejects_nontrain_authoritative_upstream_role(tmp_path):
    args = _fixture(tmp_path)
    truth = [
        json.loads(line) for line in Path(args.optimize_truth).read_text().splitlines()
    ]
    uid = truth[0]["norm_uid"]
    roles_path = Path(args.optimize_candidates).parent / "roles.jsonl"
    roles = [json.loads(line) for line in roles_path.read_text().splitlines()]
    next(row for row in roles if row["norm_uid"] == uid)["split"] = "dev"
    write_jsonl(roles_path, roles)
    _refresh_role_bindings(
        {
            "freeze": args.optimize_freeze,
            "identities": args.optimize_identities,
            "truth": args.optimize_truth,
            "truth_release": args.optimize_truth_release,
            "candidate_audit": args.optimize_candidate_audit,
        }
    )
    with pytest.raises(ValueError, match="non-train"):
        freeze(args)


def test_explicit_role_plan_rejects_parent_predeclaration_drift(tmp_path):
    args = _fixture(tmp_path)
    lock = json.loads(Path(args.predeclaration).read_text())
    parent_path = Path(lock["parent_predeclaration"]["path"])
    parent = json.loads(parent_path.read_text())
    parent["selection_gate"]["minimum_point_precision"] = 0.89
    parent_path.write_text(json.dumps(parent))
    with pytest.raises(ValueError, match="parent task-local GEPA predeclaration"):
        freeze(args)


def test_explicit_role_plan_rejects_unapproved_activation_field(tmp_path):
    args = _fixture(tmp_path)
    lock = json.loads(Path(args.predeclaration).read_text())
    lock["tasks"][args.task]["unapproved_scientific_change"] = True
    Path(args.predeclaration).write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="changed fields beyond"):
        freeze(args)


def test_explicit_role_plan_rejects_unrelated_complete_exclusion_union(tmp_path):
    args = _fixture(tmp_path)
    manifest = json.loads(Path(args.manifest).read_text())
    norms_path = Path(manifest["corpora"]["c"]["path"])
    norms = [json.loads(line) for line in norms_path.read_text().splitlines()]
    used = {
        json.loads(line)["norm_uid"]
        for path in (
            args.optimize_identities,
            args.select_identities,
            args.exclude_reference[0],
        )
        for line in Path(path).read_text().splitlines()
    }
    replacement = next(row for row in norms if row["norm_uid"] not in used)
    union_path = Path(args.exclude_reference[0])
    replacement_row = {
        "schema_version": "silver-match-v3-gepa-exclusion-identity-v1",
        "norm_uid": replacement["norm_uid"],
        "task": args.task,
        "corpus": replacement["corpus"],
        "source_group": split_group_for(replacement),
        "upstream_split": "train",
    }
    write_jsonl(union_path, [replacement_row])
    inventory_path = union_path.parent / "EXCLUSION_INVENTORY.json"
    inventory = json.loads(inventory_path.read_text())
    inventory["identity_union"].update(
        {
            "sha256": sha256_file(union_path),
            "by_corpus": {replacement["corpus"]: 1},
            "by_upstream_split": {"train": 1},
        }
    )
    inventory_path.write_text(json.dumps(inventory))
    lock_path = Path(args.predeclaration)
    lock = json.loads(lock_path.read_text())
    evidence = lock["tasks"][args.task]["execution_evidence"]
    evidence["complete_exclusion_union_sha256"] = sha256_file(union_path)
    evidence["exclusion_inventory_sha256"] = sha256_file(inventory_path)
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="omits inventoried source"):
        freeze(args)


def test_explicit_role_plan_rejects_exclusion_inventory_hash_drift(tmp_path):
    args = _fixture(tmp_path)
    inventory_path = Path(args.exclude_reference[0]).parent / "EXCLUSION_INVENTORY.json"
    inventory = json.loads(inventory_path.read_text())
    inventory["content_contract"]["parsed_sources_used_only_identity_fields"] = False
    inventory_path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="inventory is missing or hash-drifted"):
        freeze(args)


def test_explicit_role_plan_rejects_exclusion_inventory_union_ref_drift(tmp_path):
    args = _fixture(tmp_path)
    inventory_path = Path(args.exclude_reference[0]).parent / "EXCLUSION_INVENTORY.json"
    inventory = json.loads(inventory_path.read_text())
    inventory["identity_union"]["sha256"] = "0" * 64
    inventory_path.write_text(json.dumps(inventory))
    lock_path = Path(args.predeclaration)
    lock = json.loads(lock_path.read_text())
    lock["tasks"][args.task]["execution_evidence"]["exclusion_inventory_sha256"] = (
        sha256_file(inventory_path)
    )
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="inventory contract is invalid"):
        freeze(args)


def test_explicit_role_plan_rejects_claimed_but_unsourced_inventory_category(tmp_path):
    args = _fixture(tmp_path)
    inventory_path = Path(args.exclude_reference[0]).parent / "EXCLUSION_INVENTORY.json"
    inventory = json.loads(inventory_path.read_text())
    inventory["required_categories"].append("missing_category")
    inventory["observed_categories"].append("missing_category")
    inventory_path.write_text(json.dumps(inventory))
    lock_path = Path(args.predeclaration)
    lock = json.loads(lock_path.read_text())
    lock["tasks"][args.task]["execution_evidence"]["exclusion_inventory_sha256"] = (
        sha256_file(inventory_path)
    )
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="source categories differ"):
        freeze(args)


def test_explicit_role_plan_fails_closed_on_api_budget(tmp_path):
    args = _fixture(tmp_path)
    args.max_total_api_requests = 39
    with pytest.raises(ValueError, match="exceeds budget"):
        freeze(args)
