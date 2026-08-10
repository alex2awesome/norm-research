#!/usr/bin/env python3
"""Audit and freeze non-Humor silver-match truth-source preparation queues.

This is deliberately a truth *inventory*, not a label merger.  It validates
only final, hash-bound releases; intermediate labeling passes are never
promoted.  Permanent-blind sources, if added later, are represented in the
outputs only by an opaque seal (source id, hash, and count).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

from .common import normalize_space, sha256_file


CATALOG_SCHEMA = "silver-match-v3-truth-source-catalog-v1"
INVENTORY_SCHEMA = "silver-match-v3-alltask-truth-source-inventory-v1"
OVERLAP_SCHEMA = "silver-match-v3-alltask-truth-source-overlap-v1"
QUEUE_SCHEMA = "silver-match-v3-task-truth-preparation-queue-v1"
FREEZE_SCHEMA = "silver-match-v3-alltask-truth-preparation-freeze-v1"

CLASSIFICATIONS = {
    "TRAIN_ELIGIBLE",
    "DEV_ONLY",
    "BLIND_ONLY",
    "FAMILY_ONLY",
    "REJECT",
}
DECISIONS = {
    "MATCH",
    "MATCH_FAMILY_ONLY",
    "NO_CANDIDATE_FITS",
    "NO_EXPLICIT_CRITERION",
    "GENERIC_VERDICT",
    "CONTEXT_NEEDED",
    "NOISE",
}
TASK_ORDER = [
    "code-review",
    "creative-writing",
    "legal-outcome-prediction",
    "math-stackexchange",
    "peer-review",
    "press-releases",
    "notice-and-comment",
]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL row: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row: {path}:{line_number}")
            yield line_number, value


def _resolve(raw: Any, *, anchor: Path, root: Path) -> Path:
    value = Path(str(raw or ""))
    if not str(value):
        raise ValueError("empty artifact path")
    if not value.is_absolute():
        anchored = (anchor.parent / value).resolve()
        if anchored.exists():
            return anchored
        return (root / value).resolve()
    if value.exists():
        return value.resolve()
    parts = value.parts
    if root.name in parts:
        candidate = root.joinpath(*parts[parts.index(root.name) + 1 :])
        if candidate.exists():
            return candidate.resolve()
    return value


def _check_binding(binding: Mapping[str, Any], *, anchor: Path, root: Path) -> Path:
    path = _resolve(binding.get("path"), anchor=anchor, root=root)
    expected = normalize_space(binding.get("sha256"))
    if not path.is_file() or not expected or sha256_file(path) != expected:
        raise ValueError(f"missing or hash-drifted artifact: {path}")
    return path


def _binding(path: Path, *, redact_path: bool = False) -> dict[str, Any]:
    value = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    if not redact_path:
        value["path"] = str(path)
    return value


def _annotators(row: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for prediction in (row.get("source_predictions") or {}).values():
        if isinstance(prediction, Mapping):
            annotator = normalize_space(prediction.get("annotator"))
            if annotator:
                values.append(annotator)
    return values


def _audit_norm_rows(
    *,
    source_id: str,
    task: str,
    role: str,
    classification: str,
    truth_path: Path,
    expected_count: int,
    bank_hash: str,
    freezer_path: Path,
    freezer_schema: str,
) -> tuple[dict[str, Any], set[str], set[str], list[dict[str, Any]]]:
    uids: set[str] = set()
    groups: set[str] = set()
    corpus_counts: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    label_sources: Counter[str] = Counter()
    annotators: Counter[str] = Counter()
    row_contracts: Counter[str] = Counter()
    metric_rows = family_rows = 0
    role_map_rows: list[dict[str, Any]] = []
    for line_number, row in _rows(truth_path):
        uid = normalize_space(row.get("norm_uid"))
        corpus = normalize_space(row.get("corpus"))
        group = str(row.get("source_group") or "").strip()
        decision = normalize_space(row.get("decision"))
        row_bank = normalize_space(
            row.get("current_bank_source_sha256") or row.get("bank_source_sha256")
        )
        metric_id = normalize_space(row.get("metric_id"))
        if (
            not uid
            or uid in uids
            or not corpus
            or not group
            or row.get("task") != task
            or decision not in DECISIONS
            or row_bank != bank_hash
        ):
            raise ValueError(f"invalid/duplicate norm truth row: {source_id}:{line_number}")
        if decision == "MATCH" and not metric_id:
            raise ValueError(f"MATCH lacks metric_id: {source_id}:{line_number}")
        if decision not in {"MATCH", "MATCH_FAMILY_ONLY"} and metric_id:
            raise ValueError(f"typed nonmatch carries metric_id: {source_id}:{line_number}")
        uids.add(uid)
        groups.add(group)
        corpus_counts[corpus] += 1
        decisions[decision] += 1
        metric_rows += int(decision == "MATCH")
        family_rows += int(decision == "MATCH_FAMILY_ONLY")
        label_sources[normalize_space(row.get("label_source")) or "UNKNOWN"] += 1
        annotators.update(_annotators(row))
        handoff_role = {
            "TRAIN_ELIGIBLE": "train",
            "DEV_ONLY": "dev",
            "BLIND_ONLY": "blind",
        }.get(classification)
        if handoff_role:
            role_map_rows.append(
                {
                    "schema_version": "silver-match-v3-task-truth-role-map-v1",
                    "task": task,
                    "corpus": corpus,
                    "norm_uid": uid,
                    "source_group": group,
                    "role": handoff_role,
                    "permanent_blind": handoff_role == "blind",
                    "current_bank_source_sha256": bank_hash,
                    "truth_source_id": source_id,
                    "source_role": role,
                }
            )
        for field in (
            "training_eligible",
            "prompt_gradient_eligible",
            "prompt_selection_eligible",
            "evaluation_only",
        ):
            if field in row:
                row_contracts[f"{field}={str(row[field]).lower()}"] += 1
    if len(uids) != expected_count:
        raise ValueError(
            f"truth count differs from freezer: {source_id}/{len(uids)}/{expected_count}"
        )
    supervised_training_allowed = classification == "TRAIN_ELIGIBLE" and not (
        row_contracts["training_eligible=false"] == expected_count
    )
    allowed_uses = []
    if classification == "TRAIN_ELIGIBLE":
        allowed_uses.append("PROMPT_OPTIMIZATION")
        if supervised_training_allowed:
            allowed_uses.append("SUPERVISED_MODEL_TRAINING")
    elif classification == "DEV_ONLY":
        allowed_uses.append("MODEL_OR_PROMPT_SELECTION_ONLY")
    elif classification == "BLIND_ONLY":
        allowed_uses.append("SEALED_FINAL_EVALUATION_ONLY")
    record = {
        "source_id": source_id,
        "format": "NORM_TO_METRIC_TRUTH",
        "task": task,
        "role": role,
        "classification": classification,
        "availability": "LOCAL_HASH_VALIDATED",
        "status": "VALIDATED_EXACT_FINAL_TRUTH",
        "row_count": expected_count,
        "unique_uid_count": len(uids),
        "unique_source_group_count": len(groups),
        "corpus_counts": dict(sorted(corpus_counts.items())),
        "label_type_counts": dict(sorted(decisions.items())),
        "exact_metric_match_rows": metric_rows,
        "family_only_rows": family_rows,
        "label_source_counts": dict(sorted(label_sources.items())),
        "annotator_counts": dict(sorted(annotators.items())),
        "row_contract_counts": dict(sorted(row_contracts.items())),
        "allowed_uses": allowed_uses,
        "supervised_model_training_allowed": supervised_training_allowed,
        "bank_source_sha256": bank_hash,
        "freezer_schema": freezer_schema,
        "freezer": _binding(freezer_path),
        "truth": _binding(truth_path, redact_path=classification == "BLIND_ONLY"),
        "blind_labels_exposed": False,
    }
    role_map_rows.sort(key=lambda row: str(row["norm_uid"]))
    return record, uids, groups, role_map_rows


def _audit_content_release(
    spec: Mapping[str, Any], *, root: Path
) -> tuple[dict[str, Any], set[str], set[str], list[dict[str, Any]]]:
    freezer_path = (root / str(spec["freezer"])).resolve()
    freeze = _json(freezer_path)
    task = str(spec["task"])
    role = str(spec["role"])
    classification = str(spec["classification"])
    if (
        freeze.get("schema_version") != "silver-match-v3-content-truth-release-freeze-v1"
        or freeze.get("status") != "FROZEN_COMPLETE_EXACT_TRUTH"
        or freeze.get("task") != task
        or freeze.get("role") != role
        or int((freeze.get("contracts") or {}).get("unresolved_count", -1)) != 0
        or freeze.get("contracts", {}).get("exact_source_coverage") is not True
    ):
        raise ValueError(f"content truth release contract failed: {spec['source_id']}")
    artifacts = freeze.get("artifacts") or {}
    for name in (
        "truth",
        "resolution_report",
        "role_freeze",
        "source_bank",
        "source_items",
        "source_pack_validation",
    ):
        if not isinstance(artifacts.get(name), Mapping):
            raise ValueError(f"content release lacks binding {name}: {spec['source_id']}")
        _check_binding(artifacts[name], anchor=freezer_path, root=root)
    truth_path = _check_binding(artifacts["truth"], anchor=freezer_path, root=root)
    return _audit_norm_rows(
        source_id=str(spec["source_id"]),
        task=task,
        role=role,
        classification=classification,
        truth_path=truth_path,
        expected_count=int(freeze["count"]),
        bank_hash=normalize_space(freeze.get("bank_source_sha256")),
        freezer_path=freezer_path,
        freezer_schema=str(freeze["schema_version"]),
    )


def _audit_exact_consensus(
    spec: Mapping[str, Any], *, root: Path
) -> tuple[dict[str, Any], set[str], set[str], list[dict[str, Any]]]:
    report_path = (root / str(spec["report"])).resolve()
    role_freeze_path = (root / str(spec["role_freeze"])).resolve()
    report = _json(report_path)
    role_freeze = _json(role_freeze_path)
    task = str(spec["task"])
    role = str(spec["role"])
    if (
        report.get("schema_version") != "silver-match-v3-exact-multi-pass-truth-report-v1"
        or report.get("complete") is not True
        or report.get("task") != task
        or report.get("gepa_role") != role
        or int(report.get("unresolved_count", -1)) != 0
        or int(report.get("resolved_count", -1)) != int(report.get("source_count", -2))
        or int(report.get("permanent_blind_rows_in_source", -1)) != 0
        or role_freeze.get("schema_version")
        != "silver-match-v3-clean-gepa-panel-freeze-v1"
        or role_freeze.get("status") != "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES"
        or role_freeze.get("task") != task
        or role_freeze.get("role") != role
        or int(role_freeze.get("selected_count", -1)) != int(report["source_count"])
    ):
        raise ValueError(f"exact consensus contract failed: {spec['source_id']}")
    outputs = report.get("outputs") or {}
    truth_path = _check_binding(outputs["resolved"], anchor=report_path, root=root)
    unresolved_path = _check_binding(outputs["unresolved"], anchor=report_path, root=root)
    if unresolved_path.stat().st_size:
        raise ValueError(f"exact consensus unresolved artifact is non-empty: {spec['source_id']}")
    bank_hashes = {
        normalize_space(row.get("current_bank_source_sha256"))
        for _, row in _rows(truth_path)
    }
    if len(bank_hashes) != 1 or "" in bank_hashes:
        raise ValueError(f"exact consensus bank is not singular: {spec['source_id']}")
    return _audit_norm_rows(
        source_id=str(spec["source_id"]),
        task=task,
        role=role,
        classification=str(spec["classification"]),
        truth_path=truth_path,
        expected_count=int(report["source_count"]),
        bank_hash=next(iter(bank_hashes)),
        freezer_path=report_path,
        freezer_schema=str(report["schema_version"]),
    )


def _audit_final_dev_freeze(
    spec: Mapping[str, Any], *, root: Path
) -> tuple[dict[str, Any], set[str], set[str], list[dict[str, Any]]]:
    freezer_path = (root / str(spec["freezer"])).resolve()
    freeze = _json(freezer_path)
    task = str(spec["task"])
    counts = freeze.get("counts") or {}
    contract = freeze.get("scientific_contract") or {}
    if (
        freeze.get("schema_version")
        != "silver-match-v3-math-fresh-dev-final-truth-freeze-v1"
        or freeze.get("status") != "FROZEN_COMPLETE_DEV_TRUTH"
        or freeze.get("task") != task
        or int(counts.get("unresolved_rows", -1)) != 0
        or int(counts.get("resolved_rows", -1)) != int(counts.get("source_rows", -2))
        or contract.get("may_train_cross_encoder") is not False
        or contract.get("external_test_remains_unconsumed") is not True
        or contract.get("blind_production_audits_remain_unconsumed") is not True
    ):
        raise ValueError(f"final dev truth contract failed: {spec['source_id']}")
    truth_path = _check_binding(
        freeze["truth_release"]["resolved"], anchor=freezer_path, root=root
    )
    bank_hashes = {
        normalize_space(row.get("current_bank_source_sha256"))
        for _, row in _rows(truth_path)
    }
    if len(bank_hashes) != 1 or "" in bank_hashes:
        raise ValueError(f"final dev bank is not singular: {spec['source_id']}")
    return _audit_norm_rows(
        source_id=str(spec["source_id"]),
        task=task,
        role=str(spec["role"]),
        classification=str(spec["classification"]),
        truth_path=truth_path,
        expected_count=int(counts["source_rows"]),
        bank_hash=next(iter(bank_hashes)),
        freezer_path=freezer_path,
        freezer_schema=str(freeze["schema_version"]),
    )


def _audit_pair_verdicts(
    spec: Mapping[str, Any], *, root: Path
) -> tuple[list[dict[str, Any]], dict[str, set[tuple[str, str]]], dict[str, set[str]]]:
    path = (root / str(spec["path"])).resolve()
    wanted = set(spec["tasks"])
    pair_sets: dict[str, set[tuple[str, str]]] = defaultdict(set)
    rubric_sets: dict[str, set[str]] = defaultdict(set)
    counts: Counter[tuple[str, str]] = Counter()
    labels: Counter[tuple[str, str, str]] = Counter()
    invalid: Counter[tuple[str, str]] = Counter()
    for line_number, row in _rows(path):
        task = normalize_space(row.get("task"))
        if task not in wanted:
            continue
        split = normalize_space(row.get("split"))
        key_a = str(row.get("key_a") or "").strip()
        key_b = str(row.get("key_b") or "").strip()
        if not key_a or not key_b or key_a == key_b:
            raise ValueError(f"invalid pair identity: {path}:{line_number}")
        logical = f"{task}/{split}"
        pair = tuple(sorted((key_a, key_b)))
        if pair in pair_sets[logical]:
            raise ValueError(f"duplicate pair verdict: {path}:{line_number}")
        pair_sets[logical].add(pair)
        rubric_sets[logical].update(pair)
        counts[(task, split)] += 1
        score = row.get("score")
        if score not in (0, 1, 2):
            invalid[(task, split)] += 1
        else:
            labels[(task, split, str(score))] += 1
    records: list[dict[str, Any]] = []
    physical = _binding(path)
    for task in TASK_ORDER:
        if task not in wanted:
            continue
        for split, classification in (("train", "TRAIN_ELIGIBLE"), ("eval", "DEV_ONLY")):
            total = counts[(task, split)]
            if not total:
                continue
            source_id = f"{spec['source_id']}/{task}/{split}"
            good = total - invalid[(task, split)]
            records.append(
                {
                    "source_id": source_id,
                    "format": "RUBRIC_PAIR_SIMILARITY",
                    "task": task,
                    "role": split,
                    "classification": classification,
                    "availability": "LOCAL_HASH_VALIDATED",
                    "status": "VALIDATED_WITH_FAIL_CLOSED_ROW_EXCLUSIONS"
                    if good != total
                    else "VALIDATED_EXACT_PAIR_TRUTH",
                    "row_count": total,
                    "eligible_row_count": good,
                    "rejected_row_count": total - good,
                    "unique_pair_count": total,
                    "unique_rubric_key_count": len(rubric_sets[f"{task}/{split}"]),
                    "corpus_counts": {"NOT_APPLICABLE": total},
                    "label_type_counts": {
                        score: labels[(task, split, score)] for score in ("0", "1", "2")
                    },
                    "allowed_uses": ["SUPERVISED_MODEL_TRAINING"]
                    if classification == "TRAIN_ELIGIBLE"
                    else ["MODEL_SELECTION_ONLY"],
                    "physical_artifact": physical,
                    "logical_filter": {"task": task, "split": split},
                    "blind_labels_exposed": False,
                }
            )
    return records, pair_sets, rubric_sets


def _audit_family_adjudications(
    spec: Mapping[str, Any], *, root: Path
) -> list[dict[str, Any]]:
    path = (root / str(spec["path"])).resolve()
    wanted = set(spec["tasks"])
    counts: Counter[str] = Counter()
    labels: Counter[tuple[str, str]] = Counter()
    confirmed: Counter[tuple[str, str]] = Counter()
    invalid: Counter[str] = Counter()
    seen: set[tuple[str, Any, Any]] = set()
    for line_number, row in _rows(path):
        task = normalize_space(row.get("task"))
        if task not in wanted:
            continue
        key = (task, row.get("absorbed"), row.get("target"))
        if key in seen:
            raise ValueError(f"duplicate family adjudication: {path}:{line_number}")
        seen.add(key)
        counts[task] += 1
        majority = row.get("majority")
        if majority not in (0, 1, 2):
            invalid[task] += 1
        else:
            labels[(task, str(majority))] += 1
        confirmed[(task, str(bool(row.get("confirmed"))).lower())] += 1
    physical = _binding(path)
    return [
        {
            "source_id": f"{spec['source_id']}/{task}",
            "format": "METRIC_FAMILY_ADJUDICATION",
            "task": task,
            "role": "family",
            "classification": "FAMILY_ONLY",
            "availability": "LOCAL_HASH_VALIDATED",
            "status": "VALIDATED_WITH_FAIL_CLOSED_ROW_EXCLUSIONS"
            if invalid[task]
            else "VALIDATED_EXACT_FAMILY_TRUTH",
            "row_count": counts[task],
            "eligible_row_count": counts[task] - invalid[task],
            "rejected_row_count": invalid[task],
            "unique_family_pair_count": counts[task],
            "corpus_counts": {"NOT_APPLICABLE": counts[task]},
            "label_type_counts": {
                score: labels[(task, score)] for score in ("0", "1", "2")
            },
            "confirmed_counts": {
                value: confirmed[(task, value)] for value in ("false", "true")
            },
            "allowed_uses": ["METRIC_FAMILY_STRUCTURE_ONLY"],
            "physical_artifact": physical,
            "blind_labels_exposed": False,
        }
        for task in TASK_ORDER
        if counts[task]
    ]


def _remote_record(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_id": str(spec["source_id"]),
        "format": str(spec["format"]),
        "task": str(spec["task"]),
        "role": str(spec.get("role") or "unknown"),
        "classification": "REJECT",
        "intended_classification": str(spec.get("intended_classification") or "UNKNOWN"),
        "availability": "REMOTE_INACCESSIBLE_NOT_AUDITED",
        "status": "REJECT_UNTIL_LOCAL_HASH_AND_CONTENT_VALIDATION",
        "remote_path": str(spec["remote_path"]),
        "expected_row_count_from_memory": spec.get("expected_row_count"),
        "expected_sha256_from_memory": spec.get("expected_sha256"),
        "evidence": spec.get("evidence"),
        "allowed_uses": [],
        "blind_labels_exposed": False,
    }


def _overlap_report(
    norm_sets: Mapping[str, tuple[str, str, set[str], set[str]]],
    pair_sets: Mapping[str, set[tuple[str, str]]],
    rubric_sets: Mapping[str, set[str]],
) -> dict[str, Any]:
    norm_overlaps: list[dict[str, Any]] = []
    leakage_failures = 0
    by_task: dict[str, list[str]] = defaultdict(list)
    for source_id, (task, _classification, _uids, _groups) in norm_sets.items():
        by_task[task].append(source_id)
    for task, sources in sorted(by_task.items()):
        for left, right in combinations(sorted(sources), 2):
            _, left_class, left_uids, left_groups = norm_sets[left]
            _, right_class, right_uids, right_groups = norm_sets[right]
            uid_overlap = len(left_uids & right_uids)
            group_overlap = len(left_groups & right_groups)
            firewall = left_class != right_class and {
                left_class,
                right_class,
            } <= {"TRAIN_ELIGIBLE", "DEV_ONLY", "BLIND_ONLY"}
            failure = firewall and (uid_overlap > 0 or group_overlap > 0)
            leakage_failures += int(failure)
            norm_overlaps.append(
                {
                    "task": task,
                    "left": left,
                    "right": right,
                    "left_classification": left_class,
                    "right_classification": right_class,
                    "uid_overlap": uid_overlap,
                    "source_group_overlap": group_overlap,
                    "cross_role_firewall": firewall,
                    "leakage_failure": failure,
                }
            )
    pair_overlaps: list[dict[str, Any]] = []
    pair_by_task: dict[str, list[str]] = defaultdict(list)
    for logical in pair_sets:
        task, _ = logical.rsplit("/", 1)
        pair_by_task[task].append(logical)
    for task, logicals in sorted(pair_by_task.items()):
        if len(logicals) < 2:
            continue
        for left, right in combinations(sorted(logicals), 2):
            pair_overlap = len(pair_sets[left] & pair_sets[right])
            pair_failure = pair_overlap > 0
            leakage_failures += int(pair_failure)
            pair_overlaps.append(
                {
                    "task": task,
                    "left": left,
                    "right": right,
                    "exact_pair_overlap": pair_overlap,
                    "rubric_key_overlap": len(rubric_sets[left] & rubric_sets[right]),
                    "rubric_key_overlap_is_expected": True,
                    "leakage_failure": pair_failure,
                }
            )
    return {
        "schema_version": OVERLAP_SCHEMA,
        "status": "PASS_NO_CROSS_ROLE_IDENTITY_LEAKAGE"
        if not leakage_failures
        else "FAIL_CROSS_ROLE_IDENTITY_LEAKAGE",
        "leakage_failure_count": leakage_failures,
        "norm_truth_overlaps": norm_overlaps,
        "pair_truth_overlaps": pair_overlaps,
        "cross_format_overlap": {
            "computed": False,
            "reason": "norm_uid/source_group and rubric source keys are different identity namespaces",
        },
    }


def _queue(task: str, records: list[dict[str, Any]], overlap: Mapping[str, Any]) -> dict[str, Any]:
    sources = [row for row in records if row.get("task") == task]
    norm = [row for row in sources if row.get("format") == "NORM_TO_METRIC_TRUTH"]
    pairs = [row for row in sources if row.get("format") == "RUBRIC_PAIR_SIMILARITY"]
    family = [row for row in sources if row.get("classification") == "FAMILY_ONLY"]
    rejected = [row for row in sources if row.get("classification") == "REJECT"]
    leakage = [
        row
        for row in overlap["norm_truth_overlaps"] + overlap["pair_truth_overlaps"]
        if row.get("task") == task and row.get("leakage_failure")
    ]

    def ids(rows: Iterable[Mapping[str, Any]], classification: str) -> list[str]:
        return sorted(
            str(row["source_id"])
            for row in rows
            if row.get("classification") == classification
        )

    blind = ids(norm, "BLIND_ONLY")
    blind_seals = [
        {
            "source_id": row["source_id"],
            "sha256": (row.get("truth") or {}).get("sha256"),
            "row_count": row.get("row_count"),
        }
        for row in norm
        if row.get("classification") == "BLIND_ONLY"
    ]
    missing = [
        role
        for role, available in (
            ("norm_train", ids(norm, "TRAIN_ELIGIBLE")),
            ("norm_dev", ids(norm, "DEV_ONLY")),
            ("norm_blind", blind),
            ("pair_train", ids(pairs, "TRAIN_ELIGIBLE")),
            ("pair_dev", ids(pairs, "DEV_ONLY")),
        )
        if not available
    ]
    executable = not leakage and not missing
    return {
        "schema_version": QUEUE_SCHEMA,
        "status": "FROZEN_EXECUTABLE_TRUTH_PREPARATION_QUEUE"
        if executable
        else "FROZEN_INCOMPLETE_INPUTS_NOT_EXECUTABLE",
        "task": task,
        "norm_metric_truth": {
            "train": ids(norm, "TRAIN_ELIGIBLE"),
            "dev": ids(norm, "DEV_ONLY"),
            "blind_source_ids": blind,
            "blind_seals": blind_seals,
            "blind_truth_paths_emitted": False,
        },
        "rubric_pair_truth": {
            "train": ids(pairs, "TRAIN_ELIGIBLE"),
            "dev": ids(pairs, "DEV_ONLY"),
        },
        "metric_family_truth": sorted(str(row["source_id"]) for row in family),
        "rejected_or_inaccessible_sources": sorted(str(row["source_id"]) for row in rejected),
        "missing_required_roles": missing,
        "leakage_failures": leakage,
        "downstream_ce_production_materialization": {
            "builder": "scripts/tools/silver_match_v3/materialize_nemotron_ce_production_pairs.py",
            "status": "PREREQUISITE_NOT_BOUND_BY_THIS_TRUTH_QUEUE",
            "requires_exact_manifest_norm_universe": True,
            "requires_hash_bound_complete_bank_candidate_union_per_corpus": True,
            "minimum_independent_retrieval_lanes": 2,
            "labels_materialized": False,
        },
        "release_ready": False,
        "mi_correlation_ready": False,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with temp.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with temp.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def build(*, catalog_path: Path, output_dir: Path, root: Path) -> dict[str, Any]:
    catalog_path = catalog_path.resolve()
    output_dir = output_dir.resolve()
    root = root.resolve()
    catalog = _json(catalog_path)
    if catalog.get("schema_version") != CATALOG_SCHEMA:
        raise ValueError("truth source catalog schema mismatch")
    if catalog.get("task_order") != TASK_ORDER:
        raise ValueError("truth source catalog task order differs; notice-and-comment must remain last")
    specs = catalog.get("sources")
    if not isinstance(specs, list) or not specs:
        raise ValueError("truth source catalog is empty")
    source_ids = [normalize_space(row.get("source_id")) for row in specs]
    if not all(source_ids) or len(source_ids) != len(set(source_ids)):
        raise ValueError("truth source catalog has missing/duplicate source IDs")
    for spec in specs:
        classification = str(spec.get("classification") or "REJECT")
        if classification not in CLASSIFICATIONS:
            raise ValueError(f"invalid source classification: {spec.get('source_id')}")

    records: list[dict[str, Any]] = []
    norm_sets: dict[str, tuple[str, str, set[str], set[str]]] = {}
    role_rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pair_sets: dict[str, set[tuple[str, str]]] = {}
    rubric_sets: dict[str, set[str]] = {}
    for spec in specs:
        kind = spec.get("kind")
        if kind == "content_truth_release":
            record, uids, groups, role_rows = _audit_content_release(spec, root=root)
            records.append(record)
            role_rows_by_task[record["task"]].extend(role_rows)
            norm_sets[record["source_id"]] = (
                record["task"],
                record["classification"],
                uids,
                groups,
            )
        elif kind == "exact_consensus":
            record, uids, groups, role_rows = _audit_exact_consensus(spec, root=root)
            records.append(record)
            role_rows_by_task[record["task"]].extend(role_rows)
            norm_sets[record["source_id"]] = (
                record["task"],
                record["classification"],
                uids,
                groups,
            )
        elif kind == "final_dev_freeze":
            record, uids, groups, role_rows = _audit_final_dev_freeze(spec, root=root)
            records.append(record)
            role_rows_by_task[record["task"]].extend(role_rows)
            norm_sets[record["source_id"]] = (
                record["task"],
                record["classification"],
                uids,
                groups,
            )
        elif kind == "pair_verdicts":
            pair_records, physical_pairs, physical_rubrics = _audit_pair_verdicts(
                spec, root=root
            )
            records.extend(pair_records)
            pair_sets.update(physical_pairs)
            rubric_sets.update(physical_rubrics)
        elif kind == "family_adjudications":
            records.extend(_audit_family_adjudications(spec, root=root))
        elif kind == "remote_reference":
            records.append(_remote_record(spec))
        else:
            raise ValueError(f"unknown truth source kind: {kind}")

    records.sort(key=lambda row: (TASK_ORDER.index(str(row["task"])), str(row["source_id"])))
    overlap = _overlap_report(norm_sets, pair_sets, rubric_sets)
    classification_counts = Counter(str(row["classification"]) for row in records)
    availability_counts = Counter(str(row["availability"]) for row in records)
    task_counts = Counter(str(row["task"]) for row in records)
    inventory = {
        "schema_version": INVENTORY_SCHEMA,
        "status": "FROZEN_VALIDATED_LOCAL_AND_EXPLICITLY_QUARANTINED_REMOTE_SOURCES"
        if overlap["leakage_failure_count"] == 0
        else "FAILED_LEAKAGE_AUDIT",
        "catalog": {"path": str(catalog_path), "sha256": sha256_file(catalog_path)},
        "task_order": TASK_ORDER,
        "source_count": len(records),
        "classification_counts": dict(sorted(classification_counts.items())),
        "availability_counts": dict(sorted(availability_counts.items())),
        "task_source_counts": {task: task_counts[task] for task in TASK_ORDER},
        "sources": records,
        "blind_label_values_emitted": False,
        "raw_intermediate_passes_promoted": False,
    }
    queues = {task: _queue(task, records, overlap) for task in TASK_ORDER}
    frozen_role_rows: dict[str, list[dict[str, Any]]] = {}
    for task in TASK_ORDER:
        role_rows = sorted(role_rows_by_task[task], key=lambda row: str(row["norm_uid"]))
        role_uids = [str(row["norm_uid"]) for row in role_rows]
        if len(role_uids) != len(set(role_uids)):
            raise ValueError(f"duplicate UID in task truth role map: {task}")
        frozen_role_rows[task] = role_rows
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite truth audit output: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "queues").mkdir()
    _atomic_json(output_dir / "inventory.json", inventory)
    _atomic_json(output_dir / "overlap_report.json", overlap)
    queue_refs: dict[str, Any] = {}
    for task in TASK_ORDER:
        role_rows = frozen_role_rows[task]
        role_map_path = output_dir / "queues" / f"{task}.role_map.jsonl"
        _atomic_jsonl(role_map_path, role_rows)
        queues[task]["authoritative_role_map"] = {
            **_binding(role_map_path),
            "row_count": len(role_rows),
            "labels_included": False,
            "blind_label_values_included": False,
        }
        queue_path = output_dir / "queues" / f"{task}.json"
        _atomic_json(queue_path, queues[task])
        queue_refs[task] = _binding(queue_path)
    freeze = {
        "schema_version": FREEZE_SCHEMA,
        "status": "FROZEN_PARTIAL_ALLTASK_TRUTH_PREPARATION",
        "task_order": TASK_ORDER,
        "inventory": _binding(output_dir / "inventory.json"),
        "overlap_report": _binding(output_dir / "overlap_report.json"),
        "queues": queue_refs,
        "executable_task_count": sum(
            queue["status"] == "FROZEN_EXECUTABLE_TRUTH_PREPARATION_QUEUE"
            for queue in queues.values()
        ),
        "release_ready": False,
        "mi_correlation_ready": False,
    }
    _atomic_json(output_dir / "FREEZE.json", freeze)
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    freeze = build(catalog_path=args.catalog, output_dir=args.output_dir, root=args.root)
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
