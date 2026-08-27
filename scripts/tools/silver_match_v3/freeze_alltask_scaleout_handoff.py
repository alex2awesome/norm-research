#!/usr/bin/env python3
"""Freeze the shortest fail-closed handoff from current evidence to all-task v3.

This program does not discover data, launch jobs, or promote a model.  It binds
the exact current eight-task scope and the audited truth/retrieval/CE artifacts
supplied by the caller, then emits an actionable task-by-task readiness matrix.

The Humor CE recipe is carried forward only as a hyperparameter seed.  Every
task must retain its own source-disjoint splits, LoRA weights, thresholds, and
release audits.  The legacy K50 universe is diagnostic only.  Humor is
retrieval-staged only when its coverage-preserving K200 primary, K285 rescue,
structural audits, capture audit, and exact 77,378-row pair universe are bound.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import sha256_file


SCHEMA = "silver-match-v3-alltask-scaleout-handoff-v2"
TASK_ORDER = (
    "humor",
    "code-review",
    "creative-writing",
    "legal-outcome-prediction",
    "math-stackexchange",
    "peer-review",
    "press-releases",
    "notice-and-comment",
)
NONHUMOR_ORDER = tuple(task for task in TASK_ORDER if task != "humor")
PILOT_ORDER = tuple(
    task for task in NONHUMOR_ORDER if task != "notice-and-comment"
)
EXPECTED_TASKS = 8
EXPECTED_CORPORA = 23
EXPECTED_NORMS = 1_732_515
THREE_WAY = {"EXACT", "FAMILY", "REJECT"}
HUMOR_K200_CANDIDATE_SHA256 = (
    "aee3619be5a22b7c65e6db4bd9bcf6f246e5ad31638619de1049f2870a828b24"
)
HUMOR_K200_META_SHA256 = (
    "4e775c0cfa5788a226084a08613db8f2d6145b2c8f6c91c235a108bb4d7e65da"
)
HUMOR_K200_AUDIT_SHA256 = (
    "29daace7a867efc3cb832d491763367baa6b0f39b2cf245c85a1b2183f583e8d"
)
HUMOR_K200_CAPTURE_SHA256 = (
    "2db7a82999d1ae77daf646752d2fcc16dc265922a9585572475e70d4cdf70b6d"
)
HUMOR_K200_PAIRS_META_SHA256 = (
    "bc10e6b995d157cd4d5bec55692cc4909f4f22c48e3564b1b978a180dc789e37"
)
HUMOR_K200_PAIRS_SHA256 = (
    "f90c19bd3c06bcabffd52b165526aa88366ba1def1c21df3904af77edaf2b84a"
)
HUMOR_UNIVERSE_SHA256 = (
    "b066e6d7a58e70d45a05b7dd6bb9e8088acf44c5bfb44173814d6a3fb135b9ed"
)
HUMOR_FULL285_CANDIDATE_SHA256 = (
    "356314b34ef0eac8fd48ec53e4600d8cbad8b7a6e8895610d7d9731ea6b7712e"
)
HUMOR_FULL285_META_SHA256 = (
    "3e6ff84254a0ca3de51982b1960574211ccd8e3f74ce945b4e14e715df1c8260"
)
HUMOR_FULL285_AUDIT_SHA256 = (
    "ad94eaf16ab54b61ba3819e3b29a15fdd4dc0ea2c866b542cd8fc13e740d5206"
)


def _load(path: Path) -> dict[str, Any]:
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _bound_artifact(binding: dict[str, Any], *, label: str) -> Path:
    path = Path(str(binding.get("path") or "")).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    expected = str(binding.get("sha256") or "")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} hash mismatch: {path}")
    return path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _task_scope(coverage: dict[str, Any]) -> dict[str, dict[str, Any]]:
    manifest = coverage.get("manifest") or {}
    extractions = coverage.get("extractions") or {}
    corpora = extractions.get("corpora") or {}
    _require(extractions.get("complete") is True, "canonical extractions incomplete")
    _require(
        manifest.get("total_tasks") == EXPECTED_TASKS
        and manifest.get("total_corpora") == EXPECTED_CORPORA
        and manifest.get("total_norms") == EXPECTED_NORMS,
        "coverage is not the exact 8-task/23-corpus/1,732,515-norm scope",
    )
    _require(len(corpora) == EXPECTED_CORPORA, "coverage corpus count mismatch")
    _require(
        sum(int(meta.get("count", -1)) for meta in corpora.values())
        == EXPECTED_NORMS,
        "coverage norm count mismatch",
    )
    by_task: dict[str, dict[str, Any]] = {}
    for task in TASK_ORDER:
        selected = {
            corpus: {
                "count": int(meta["count"]),
                "path": meta["path"],
                "sha256": meta["sha256"],
            }
            for corpus, meta in sorted(corpora.items())
            if meta.get("task") == task
        }
        _require(bool(selected), f"canonical task has no corpora: {task}")
        by_task[task] = {
            "corpora": selected,
            "corpus_count": len(selected),
            "norm_count": sum(row["count"] for row in selected.values()),
        }
    observed_tasks = {str(meta.get("task")) for meta in corpora.values()}
    _require(observed_tasks == set(TASK_ORDER), "canonical task set mismatch")
    selections = coverage.get("retriever_selections") or {}
    _require(
        selections.get("complete") is True
        and set((selections.get("tasks") or {})) == set(TASK_ORDER),
        "task-local retriever selections are incomplete",
    )
    return by_task


def _validate_final_coverage(final: dict[str, Any]) -> None:
    summary = final.get("summary") or {}
    _require(
        summary.get("expected_tasks") == EXPECTED_TASKS
        and summary.get("expected_corpora") == EXPECTED_CORPORA
        and summary.get("expected_count") == EXPECTED_NORMS,
        "final-coverage scope mismatch",
    )
    _require(
        summary.get("canonical_final_ready_tasks") == 0
        and summary.get("canonical_final_ready_corpora") == 0,
        "handoff must be refreshed because canonical finals now exist",
    )
    _require(set(final.get("tasks") or {}) == set(TASK_ORDER), "final task set mismatch")


def _load_truth_queues(
    truth_freeze: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    _require(
        tuple(truth_freeze.get("task_order") or ()) == NONHUMOR_ORDER,
        "truth queue order mismatch",
    )
    _require(
        truth_freeze.get("executable_task_count") == 0
        and truth_freeze.get("release_ready") is False,
        "truth audit unexpectedly claims execution or release readiness",
    )
    queues = truth_freeze.get("queues") or {}
    _require(set(queues) == set(NONHUMOR_ORDER), "truth queue task set mismatch")
    result: dict[str, dict[str, Any]] = {}
    for task in NONHUMOR_ORDER:
        path = _bound_artifact(queues[task], label=f"{task} truth queue")
        queue = _load(path)
        _require(queue.get("task") == task, f"truth queue task mismatch: {task}")
        _require(not queue.get("leakage_failures"), f"truth leakage failure: {task}")
        role = queue.get("authoritative_role_map") or {}
        role_path = _bound_artifact(role, label=f"{task} role map")
        _require(
            role_path.stat().st_size == int(role.get("bytes", -1)),
            f"role-map size mismatch: {task}",
        )
        result[task] = queue
    return result


def _load_ce_seed(
    ce_freeze: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    _require(
        tuple(ce_freeze.get("task_order") or ()) == NONHUMOR_ORDER,
        "CE seed task order mismatch",
    )
    _require(
        ce_freeze.get("all_tasks_have_three_way_train_and_dev") is True
        and ce_freeze.get("all_tasks_source_group_disjoint") is True
        and ce_freeze.get("cross_task_truth_borrowed") is False
        and ce_freeze.get("blind_or_test_rows_read") == 0,
        "CE seed isolation contracts failed",
    )
    records = {row["task"]: row for row in ce_freeze.get("task_reports") or []}
    _require(set(records) == set(NONHUMOR_ORDER), "CE seed task set mismatch")
    result: dict[str, dict[str, Any]] = {}
    for task in NONHUMOR_ORDER:
        record = records[task]
        report_path = _bound_artifact(record["report"], label=f"{task} CE report")
        queue_path = _bound_artifact(
            record["train_queue"], label=f"{task} CE train queue"
        )
        report = _load(report_path)
        queue = _load(queue_path)
        _require(report.get("task") == task, f"CE report task mismatch: {task}")
        _require(queue.get("task") == task, f"CE queue task mismatch: {task}")
        contracts = report.get("contracts") or {}
        source = report.get("source_coverage") or {}
        _require(
            contracts.get("source_group_disjoint") is True
            and contracts.get("cross_task_truth_borrowed") is False
            and contracts.get("no_blind_or_test_rows_read") is True
            and source.get("source_group_overlap") == 0
            and source.get("norm_uid_overlap") == 0,
            f"CE report isolation failed: {task}",
        )
        for split in ("train", "dev"):
            counts = (report.get("class_counts") or {}).get(split) or {}
            _require(
                set(counts) == THREE_WAY and all(int(value) > 0 for value in counts.values()),
                f"CE {split} lacks three-way support: {task}",
            )
        _require(
            int(record.get("train_rows", -1))
            == sum(report["class_counts"]["train"].values())
            and int(record.get("dev_rows", -1))
            == sum(report["class_counts"]["dev"].values()),
            f"CE row-count mismatch: {task}",
        )
        result[task] = {"record": record, "report": report, "queue": queue}
    return result


def _load_teacher_packs(
    teacher_freeze: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    _require(
        tuple(teacher_freeze.get("task_order") or ()) == NONHUMOR_ORDER,
        "teacher-pack task order mismatch",
    )
    _require(
        teacher_freeze.get("labels_collected") == 0
        and teacher_freeze.get("release_ready") is False,
        "teacher pack is not an unlabeled train-only freeze",
    )
    records = {row["task"]: row for row in teacher_freeze.get("task_records") or []}
    _require(set(records) == set(NONHUMOR_ORDER), "teacher task set mismatch")
    for task, record in records.items():
        report_path = _bound_artifact(
            record["report"], label=f"{task} teacher report"
        )
        report = _load(report_path)
        _require(report.get("task") == task, f"teacher report task mismatch: {task}")
        contracts = report.get("contracts") or {}
        _require(
            contracts.get("cross_task_borrowing") is False
            and contracts.get("eval_heldout_blind_labels_used_as_truth") is False
            and contracts.get("full_current_task_bank_presented") is True
            and contracts.get("source_group_firewall") is True
            and contracts.get("typed_abstention_enabled") is True,
            f"teacher contract failure: {task}",
        )
        record["_report_payload"] = report
    expected = (teacher_freeze.get("global_attrition_audit") or {}).get(
        "unique_train_only_recovery_queries"
    )
    _require(
        sum(int(row.get("eligible_unique_queries", -1)) for row in records.values())
        == int(expected),
        "teacher-query total mismatch",
    )
    return records


def _validate_pilot(pilot: dict[str, Any]) -> None:
    _require(tuple(pilot.get("task_order") or ()) == PILOT_ORDER, "pilot task order mismatch")
    _require(
        pilot.get("tier") == "pilot"
        and pilot.get("rows_per_task") == 256
        and pilot.get("notice_and_comment_launched") is False
        and pilot.get("core_or_scale_launched") is False
        and pilot.get("private_selection_ledger_staged") is False
        and pilot.get("teacher_visible_prior_labels_or_proposals") is False,
        "pilot isolation contract failed",
    )


def _validate_humor(
    *,
    truth_collection: dict[str, Any],
    primary_meta: dict[str, Any],
    primary_audit: dict[str, Any],
    primary_capture: dict[str, Any],
    primary_pairs: dict[str, Any],
    rescue_meta: dict[str, Any],
    rescue_audit: dict[str, Any],
    recipe: dict[str, Any],
) -> dict[str, Any]:
    _require(
        truth_collection.get("task") == "humor"
        and truth_collection.get("total_count") == 6600
        and truth_collection.get("role_counts")
        == {"blind": 1000, "dev": 600, "train": 5000}
        and truth_collection.get("bank_metric_count") == 285,
        "Humor truth-collection scope mismatch",
    )
    _require(
        primary_meta.get("task") == "humor"
        and primary_meta.get("corpus") == "humor_multi"
        and primary_meta.get("input_count") == 77378
        and primary_meta.get("new_count") == 77378
        and primary_meta.get("output_k") == 200
        and primary_meta.get("output_sha256") == HUMOR_K200_CANDIDATE_SHA256
        and primary_meta.get("manifest_sha256")
        == "b614e345a07123f9fe79d9521351886107476d34cf2b09daa50efce71dc1356f",
        "Humor K200 candidate meta mismatch",
    )
    _require(
        primary_audit.get("task") == "humor"
        and primary_audit.get("corpus") == "humor_multi"
        and primary_audit.get("complete") is True
        and primary_audit.get("observed_count") == 77378
        and primary_audit.get("materialized_k") == 200
        and primary_audit.get("bank_count") == 285,
        "Humor K200 structural audit mismatch",
    )
    primary_inputs = primary_audit.get("candidate_inputs") or {}
    _require(len(primary_inputs) == 1, "Humor K200 audit must bind one candidate")
    primary_input = next(iter(primary_inputs.values()))
    _require(
        primary_input.get("sha256") == HUMOR_K200_CANDIDATE_SHA256
        and primary_input.get("meta_sha256") == HUMOR_K200_META_SHA256
        and primary_input.get("count") == 77378,
        "Humor K200 audit/meta binding mismatch",
    )
    capture = primary_capture.get("overall") or {}
    _require(
        capture.get("gold_matches") == 549
        and capture.get("under_target_supported") is True
        and float(capture.get("union_capture_rate", 0.0)) > 0.97
        and float(capture.get("union_miss_upper_bound", 1.0)) < 0.05,
        "Humor K200 capture gate is not supported",
    )
    _require(
        set((primary_capture.get("candidate_inputs") or {}).values())
        == {HUMOR_K200_CANDIDATE_SHA256},
        "Humor K200 capture input mismatch",
    )
    _require(
        primary_pairs.get("task") == "humor"
        and primary_pairs.get("norm_count") == 77378
        and primary_pairs.get("candidate_depth") == 200
        and primary_pairs.get("pair_count") == 15_475_600
        and primary_pairs.get("labels_present") is False
        and primary_pairs.get("release_ready") is False
        and (primary_pairs.get("pairs") or {}).get("sha256")
        == HUMOR_K200_PAIRS_SHA256
        and (primary_pairs.get("norm_universe") or {}).get("sha256")
        == HUMOR_UNIVERSE_SHA256,
        "Humor K200 CE pair universe mismatch",
    )
    pair_corpus = (primary_pairs.get("corpora") or {}).get("humor_multi") or {}
    pair_union = pair_corpus.get("candidate_union") or {}
    _require(
        pair_corpus.get("pair_count") == 15_475_600
        and pair_union.get("sha256") == HUMOR_K200_CANDIDATE_SHA256
        and pair_union.get("meta_sha256") == HUMOR_K200_META_SHA256
        and pair_union.get("output_k") == 200
        and set(pair_union.get("complete_bank_lane_names") or ()) == {"bge", "human"},
        "Humor K200 pair/candidate binding mismatch",
    )
    _require(
        rescue_meta.get("task") == "humor"
        and rescue_meta.get("corpus") == "humor_multi"
        and rescue_meta.get("input_count") == 77378
        and rescue_meta.get("new_count") == 77378
        and rescue_meta.get("output_k") == 285
        and rescue_meta.get("output_sha256") == HUMOR_FULL285_CANDIDATE_SHA256
        and rescue_meta.get("bank_source_sha256")
        == "1b4a29d34b4ef4d999e0cb0b2d1125286372349ff6dfa21a6adc5bc8e76f0de9",
        "Humor K285 rescue meta mismatch",
    )
    _require(
        rescue_audit.get("task") == "humor"
        and rescue_audit.get("corpus") == "humor_multi"
        and rescue_audit.get("complete") is True
        and rescue_audit.get("observed_count") == 77378
        and rescue_audit.get("materialized_k") == 285
        and rescue_audit.get("bank_count") == 285,
        "Humor K285 rescue audit mismatch",
    )
    rescue_inputs = rescue_audit.get("candidate_inputs") or {}
    _require(len(rescue_inputs) == 1, "Humor K285 audit must bind one candidate")
    rescue_input = next(iter(rescue_inputs.values()))
    _require(
        rescue_input.get("sha256") == HUMOR_FULL285_CANDIDATE_SHA256
        and rescue_input.get("meta_sha256") == HUMOR_FULL285_META_SHA256
        and rescue_input.get("count") == 77378,
        "Humor K285 audit/meta binding mismatch",
    )
    _require(
        recipe.get("schema_version")
        == "silver-match-v3-nemotron-bidirectional-cross-encoder-v1"
        and recipe.get("labels") == ["EXACT", "FAMILY", "REJECT"]
        and recipe.get("bidirectional_concatenation") is True
        and recipe.get("lora")
        == {
            "alpha": 64,
            "dropout": 0.05,
            "rank": 32,
            "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        }
        and recipe.get("lora_learning_rate") == 0.00005,
        "Humor r32 recipe mismatch",
    )
    return {
        "primary_k": 200,
        "primary_candidate_sha256": HUMOR_K200_CANDIDATE_SHA256,
        "primary_capture_rate": capture["union_capture_rate"],
        "primary_miss_upper_bound": capture["union_miss_upper_bound"],
        "primary_pair_count": 15_475_600,
        "primary_pairs_sha256": HUMOR_K200_PAIRS_SHA256,
        "norm_universe_sha256": HUMOR_UNIVERSE_SHA256,
        "rescue_k": 285,
        "rescue_candidate_sha256": HUMOR_FULL285_CANDIDATE_SHA256,
        "primary_structurally_complete": True,
        "full_bank_rescue_structurally_complete": True,
    }


def freeze_alltask_scaleout_handoff(
    *,
    coverage_path: Path,
    final_coverage_path: Path,
    truth_audit_freeze_path: Path,
    ce_seed_freeze_path: Path,
    teacher_pack_freeze_path: Path,
    pilot_freeze_path: Path,
    humor_truth_collection_path: Path,
    humor_k200_meta_path: Path,
    humor_k200_audit_path: Path,
    humor_k200_capture_path: Path,
    humor_k200_pairs_path: Path,
    humor_full285_meta_path: Path,
    humor_full285_audit_path: Path,
    humor_recipe_path: Path,
) -> dict[str, Any]:
    inputs = {
        "coverage": coverage_path,
        "final_coverage": final_coverage_path,
        "truth_audit_freeze": truth_audit_freeze_path,
        "ce_seed_freeze": ce_seed_freeze_path,
        "teacher_pack_freeze": teacher_pack_freeze_path,
        "pilot_freeze": pilot_freeze_path,
        "humor_truth_collection": humor_truth_collection_path,
        "humor_k200_meta": humor_k200_meta_path,
        "humor_k200_audit": humor_k200_audit_path,
        "humor_k200_capture": humor_k200_capture_path,
        "humor_k200_pairs": humor_k200_pairs_path,
        "humor_full285_meta": humor_full285_meta_path,
        "humor_full285_audit": humor_full285_audit_path,
        "humor_recipe": humor_recipe_path,
    }
    payloads = {name: _load(path) for name, path in inputs.items()}
    artifacts = {name: _artifact(path) for name, path in inputs.items()}
    expected_humor_artifact_hashes = {
        "humor_k200_meta": HUMOR_K200_META_SHA256,
        "humor_k200_audit": HUMOR_K200_AUDIT_SHA256,
        "humor_k200_capture": HUMOR_K200_CAPTURE_SHA256,
        "humor_k200_pairs": HUMOR_K200_PAIRS_META_SHA256,
        "humor_full285_meta": HUMOR_FULL285_META_SHA256,
        "humor_full285_audit": HUMOR_FULL285_AUDIT_SHA256,
    }
    for name, expected in expected_humor_artifact_hashes.items():
        _require(
            artifacts[name]["sha256"] == expected,
            f"exact Humor evidence hash mismatch: {name}",
        )

    scope = _task_scope(payloads["coverage"])
    _validate_final_coverage(payloads["final_coverage"])
    truth = _load_truth_queues(payloads["truth_audit_freeze"])
    ce_seed = _load_ce_seed(payloads["ce_seed_freeze"])
    teacher = _load_teacher_packs(payloads["teacher_pack_freeze"])
    _validate_pilot(payloads["pilot_freeze"])
    humor_capture = _validate_humor(
        truth_collection=payloads["humor_truth_collection"],
        primary_meta=payloads["humor_k200_meta"],
        primary_audit=payloads["humor_k200_audit"],
        primary_capture=payloads["humor_k200_capture"],
        primary_pairs=payloads["humor_k200_pairs"],
        rescue_meta=payloads["humor_full285_meta"],
        rescue_audit=payloads["humor_full285_audit"],
        recipe=payloads["humor_recipe"],
    )

    coverage = payloads["coverage"]
    existing_candidates = (coverage.get("candidate_retrieval") or {}).get("corpora") or {}
    selections = (coverage.get("retriever_selections") or {}).get("tasks") or {}
    recipe = payloads["humor_recipe"]
    tasks: dict[str, Any] = {}
    total_train = 0
    total_dev = 0
    total_queries = 0

    for priority, task in enumerate(TASK_ORDER, start=1):
        task_scope = scope[task]
        corpora = set(task_scope["corpora"])
        legacy_complete = sorted(corpora & set(existing_candidates))
        missing_legacy = sorted(corpora - set(existing_candidates))
        if task == "humor":
            bank_count = 285
            truth_stage = {
                "status": "EXACT_TWO_PASS_CONSENSUS_IN_PROGRESS",
                "frozen_role_counts": {"train": 5000, "dev": 600, "blind": 1000},
                "blockers": [
                    "finish both independent full-bank passes and resolver rounds",
                    "freeze source-disjoint train/dev/blind consensus artifacts",
                ],
            }
            ce_stage = {
                "status": "K200_UNLABELED_PAIR_UNIVERSE_STAGED_TRUTH_BLOCKED",
                "seed_recipe": "humor-r32-alpha64-lr5e-5",
                "candidate_depth": 200,
                "staged_pair_count": 15_475_600,
                "staged_pairs_sha256": HUMOR_K200_PAIRS_SHA256,
                "staged_norm_universe_sha256": HUMOR_UNIVERSE_SHA256,
                "pair_universe_structurally_complete": True,
                "adjudication_authorized": False,
                "blockers": [
                    "finish exact truth and source-disjoint final two-seed selection",
                    "meet frozen precision/Wilson/minimum-prediction gate",
                    "run K200 primary and K285 rescue adjudication only after model promotion",
                ],
            }
            teacher_stage = {
                "status": "DEDICATED_EXACT_TRUTH_COLLECTION_IN_PROGRESS",
                "eligible_unique_queries": 6600,
            }
        else:
            record = ce_seed[task]["record"]
            queue = truth[task]
            teacher_record = teacher[task]
            report = teacher_record.pop("_report_payload")
            bank_count = int((report.get("bank") or {})["metric_count"])
            total_train += int(record["train_rows"])
            total_dev += int(record["dev_rows"])
            total_queries += int(teacher_record["eligible_unique_queries"])
            role = queue["authoritative_role_map"]
            truth_stage = {
                "status": "PAIR_SEED_READY_NORM_TRUTH_INCOMPLETE",
                "role_map_rows": int(role["row_count"]),
                "missing_required_roles": queue["missing_required_roles"],
                "teacher_recovery_queries": int(
                    teacher_record["eligible_unique_queries"]
                ),
                "pilot_rows_frozen": 256 if task in PILOT_ORDER else 0,
                "blockers": [
                    "complete independent teacher labeling and stratified pilot audit",
                    "freeze task-local norm train/dev/blind roles without source overlap",
                ],
            }
            if task == "notice-and-comment":
                truth_stage["status"] = "NO_NORM_TRUTH_ROLE_MAP_NC_LAST"
                truth_stage["blockers"].insert(
                    0, "authoritative norm role map is empty (0 rows)"
                )
            ce_stage = {
                "status": "TASK_LOCAL_THREE_WAY_SEED_QUEUE_READY",
                "train_rows": int(record["train_rows"]),
                "dev_rows": int(record["dev_rows"]),
                "class_counts": record["class_counts"],
                "low_exact_dev_support": bool(record["low_exact_dev_support"]),
                "launch_authorized": False,
                "blockers": [
                    "merge only audited task-local teacher labels",
                    "freeze fresh task-local source-disjoint split",
                    "train and select an independent task-local LoRA and thresholds",
                ],
            }
            teacher_stage = {
                "status": "UNLABELED_TRAIN_ONLY_RECOVERY_PACK_FROZEN",
                "eligible_unique_queries": int(
                    teacher_record["eligible_unique_queries"]
                ),
                "tier_query_counts": teacher_record["tier_query_counts"],
            }

        primary_k = min(200, bank_count)
        retrieval_status = (
            "PRIMARY_K200_AND_FULLBANK_RESCUE_STRUCTURALLY_COMPLETE"
            if task == "humor"
            else (
                "LEGACY_SINGLE_LANE_COMPLETE_REBUILD_REQUIRED"
                if not missing_legacy
                else "MISSING_CANONICAL_CORPUS_RETRIEVAL"
            )
        )
        next_action = (
            "finish Humor dev consensus and two-seed model gate; then adjudicate staged K200 with K285 rescue"
            if task == "humor"
            else (
                "materialize two complete-bank lanes for every corpus and freeze union/rescue"
                if task != "notice-and-comment"
                else "after all other tasks: build norm truth, then refresh retrieval and final stack"
            )
        )
        tasks[task] = {
            "priority": priority,
            "task": task,
            "scope": task_scope,
            "bank_metric_count": bank_count,
            "selected_retriever": selections[task],
            "extraction": {
                "status": "CANONICAL_EXTRACTION_REUSED_COMPLETE",
                "reextraction_required": False,
                "reextraction_authorized": False,
            },
            "truth": truth_stage,
            "teacher": teacher_stage,
            "retrieval": {
                "status": retrieval_status,
                "legacy_complete_corpora": legacy_complete,
                "legacy_missing_corpora": missing_legacy,
                "legacy_k50_production_eligible": False,
                "required_primary_k": primary_k,
                "required_full_bank_rescue_k": bank_count,
                "minimum_independent_complete_bank_lanes": 2,
                "full_bank_rescue_required": True,
                "primary_structurally_complete": task == "humor",
                "full_bank_rescue_structurally_complete": task == "humor",
                "primary_pair_universe_structurally_complete": task == "humor",
                "rescue_adjudication_pending": task == "humor",
                "production_ready": False,
            },
            "ce_training": ce_stage,
            "typed_gemma": {
                "status": "BLOCKED_ON_CE_AND_SOURCE_DISJOINT_TRUTH",
                "task_local_lora_required": True,
                "shared_cross_task_weights_allowed": False,
                "production_ready": False,
            },
            "canonical_final": {
                "status": "NOT_MATERIALIZED",
                "required_decisions_per_norm": 1,
                "typed_abstention_noise_required": True,
                "independent_blind_audit_required": True,
                "mi_join_authorized": False,
            },
            "next_action": next_action,
            "launch_authorized": False,
            "release_ready": False,
        }

    _require(total_train == 13_272, "non-Humor CE train total drifted")
    _require(total_dev == 2_884, "non-Humor CE dev total drifted")
    _require(total_queries == 8_548, "teacher-recovery query total drifted")
    _require(list(tasks) == list(TASK_ORDER), "task order drifted")

    return {
        "schema_version": SCHEMA,
        "status": "FROZEN_FAIL_CLOSED_ALLTASK_SCALEOUT_HANDOFF",
        "release_ready": False,
        "launch_authorized": False,
        "mi_correlation_authorized": False,
        "task_order": list(TASK_ORDER),
        "inputs": artifacts,
        "scope": {
            "manifest": coverage["manifest"],
            "tasks": EXPECTED_TASKS,
            "corpora": EXPECTED_CORPORA,
            "norms": EXPECTED_NORMS,
            "canonical_finals": 0,
            "existing_extractions_reused": True,
            "reextraction_required": False,
        },
        "validated_humor_evidence": {
            **humor_capture,
            "retrieval_and_pair_stage_ready": True,
            "production_model_promoted": False,
            "recipe_role": "hyperparameter_seed_only_not_shared_weights",
        },
        "recipe_seed": {
            "model": recipe["model"],
            "max_length": recipe["max_length"],
            "bidirectional_concatenation": True,
            "pooling": recipe["pooling"],
            "lora": recipe["lora"],
            "lora_learning_rate": recipe["lora_learning_rate"],
            "head_learning_rate": recipe["head_learning_rate"],
            "sampler_weights": recipe["sampler_weights"],
            "exposure_budgets": recipe["exposure_budgets"],
            "labels": recipe["labels"],
            "reuse_humor_weights_across_tasks": False,
            "fresh_task_local_lora_required": True,
            "fresh_task_local_split_required": True,
            "fresh_task_local_threshold_selection_required": True,
        },
        "global_contracts": {
            "humor_first": True,
            "notice_and_comment_last": True,
            "task_local_splits_and_loras": True,
            "cross_task_truth_or_weights_forbidden": True,
            "minimum_complete_bank_retrieval_lanes": 2,
            "coverage_preserving_primary_k": "min(200, bank_metric_count)",
            "separate_full_bank_rescue": True,
            "legacy_k50_is_diagnostic_only": True,
            "typed_abstention_noise_required": True,
            "independent_blind_release_audit_required": True,
            "no_gpu_host_or_device_authorized_by_this_artifact": True,
            "sk3_gpu_indices_forbidden": [1, 2, 3, 4],
        },
        "summary": {
            "canonical_final_ready_tasks": 0,
            "retrieval_production_ready_tasks": 0,
            "retrieval_structurally_staged_tasks": 1,
            "primary_pair_universe_structurally_staged_tasks": 1,
            "full_bank_rescue_structurally_staged_tasks": 1,
            "typed_gemma_production_ready_tasks": 0,
            "nonhumor_three_way_seed_train_rows": total_train,
            "nonhumor_three_way_seed_dev_rows": total_dev,
            "nonhumor_train_only_teacher_queries": total_queries,
            "first_six_pilot_rows_frozen": len(PILOT_ORDER) * 256,
            "notice_and_comment_norm_role_rows": 0,
        },
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "coverage",
        "final-coverage",
        "truth-audit-freeze",
        "ce-seed-freeze",
        "teacher-pack-freeze",
        "pilot-freeze",
        "humor-truth-collection",
        "humor-k200-meta",
        "humor-k200-audit",
        "humor-k200-capture",
        "humor-k200-pairs",
        "humor-full285-meta",
        "humor-full285-audit",
        "humor-recipe",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    result = freeze_alltask_scaleout_handoff(
        coverage_path=Path(args.coverage),
        final_coverage_path=Path(args.final_coverage),
        truth_audit_freeze_path=Path(args.truth_audit_freeze),
        ce_seed_freeze_path=Path(args.ce_seed_freeze),
        teacher_pack_freeze_path=Path(args.teacher_pack_freeze),
        pilot_freeze_path=Path(args.pilot_freeze),
        humor_truth_collection_path=Path(args.humor_truth_collection),
        humor_k200_meta_path=Path(args.humor_k200_meta),
        humor_k200_audit_path=Path(args.humor_k200_audit),
        humor_k200_capture_path=Path(args.humor_k200_capture),
        humor_k200_pairs_path=Path(args.humor_k200_pairs),
        humor_full285_meta_path=Path(args.humor_full285_meta),
        humor_full285_audit_path=Path(args.humor_full285_audit),
        humor_recipe_path=Path(args.humor_recipe),
    )
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
