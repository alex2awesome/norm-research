#!/usr/bin/env python3
"""Build strict task-local Nemotron CE datasets from audited v6 pair truth.

The historical v6 labels compare two source rubrics.  This builder maps both
rubric keys through the *current* R2 bank, expands each compatible pair in both
directions, and emits the norm/metric pair schema consumed by
``train_nemotron_cross_encoder.py``.  It never treats historical cluster IDs
as current bank IDs.

The original eval split has priority.  Every train row touching an eval source
document on either side is quarantined before directional expansion, yielding
an exact source-group firewall.  Missing/ambiguous current-bank mappings,
bank/label contradictions, malformed labels, and conflicting deduplications
are also quarantined rather than guessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .common import normalize_space, sha256_file
from .train_nemotron_cross_encoder import DEFAULT_NEMOTRON, load_pair_examples


SCHEMA = "silver-match-v3-v6-current-bank-ce-pair-v1"
REPORT_SCHEMA = "silver-match-v3-v6-current-bank-ce-dataset-report-v1"
QUEUE_SCHEMA = "silver-match-v3-task-local-nemotron-ce-training-queue-v1"
FREEZE_SCHEMA = "silver-match-v3-v6-pair-ce-alltask-freeze-v1"
QUARANTINE_SCHEMA = "silver-match-v3-v6-pair-ce-quarantine-v1"
AUDIT_FREEZE_SCHEMA = "silver-match-v3-alltask-truth-preparation-freeze-v1"
AUDIT_INVENTORY_SCHEMA = "silver-match-v3-alltask-truth-source-inventory-v1"

TASK_ORDER = [
    "code-review",
    "creative-writing",
    "legal-outcome-prediction",
    "math-stackexchange",
    "peer-review",
    "press-releases",
    "notice-and-comment",
]
SCORE_TO_RELATION = {2: "EXACT", 1: "FAMILY", 0: "REJECT"}
HARD_NEGATIVE_COSINE = 0.65


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
        raise ValueError("empty path")
    if value.is_absolute() and value.exists():
        return value.resolve()
    if not value.is_absolute():
        anchored = (anchor.parent / value).resolve()
        if anchored.exists():
            return anchored
        return (root / value).resolve()
    parts = value.parts
    if root.name in parts:
        relocated = root.joinpath(*parts[parts.index(root.name) + 1 :])
        if relocated.exists():
            return relocated.resolve()
    return value


def _binding(path: Path, *, published_path: Path | None = None) -> dict[str, Any]:
    return {
        "path": str(published_path or path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def _source_group(key: str) -> str:
    parts = key.rsplit("::", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"malformed rubric key: {key!r}")
    return parts[0]


def _norm_uid(task: str, key: str) -> str:
    return hashlib.sha256(f"v6-rubric-query\0{task}\0{key}".encode()).hexdigest()


def _pair_identity(task: str, key_a: str, key_b: str) -> str:
    left, right = sorted((key_a, key_b))
    return hashlib.sha256(f"{task}\0{left}\0{right}".encode()).hexdigest()


def _metric_card(metric: Mapping[str, Any]) -> str:
    name = normalize_space(metric.get("merged_name") or metric.get("name"))
    description = normalize_space(
        metric.get("merged_description") or metric.get("description")
    )
    if not name or not description:
        raise ValueError("current bank metric lacks name or description")
    return f"{name}. Definition: {description}"


def _load_audited_pair_source(
    *, audit_freeze_path: Path, root: Path
) -> tuple[Path, str, dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    freeze = _json(audit_freeze_path)
    inventory_binding = freeze.get("inventory") or {}
    if freeze.get("schema_version") != AUDIT_FREEZE_SCHEMA:
        raise ValueError("truth audit freeze schema mismatch")
    inventory_path = _resolve(
        inventory_binding.get("path"), anchor=audit_freeze_path, root=root
    )
    if sha256_file(inventory_path) != normalize_space(inventory_binding.get("sha256")):
        raise ValueError("truth audit inventory hash drift")
    inventory = _json(inventory_path)
    if (
        inventory.get("schema_version") != AUDIT_INVENTORY_SCHEMA
        or inventory.get("status")
        != "FROZEN_VALIDATED_LOCAL_AND_EXPLICITLY_QUARANTINED_REMOTE_SOURCES"
        or inventory.get("blind_label_values_emitted") is not False
    ):
        raise ValueError("truth audit inventory contract failed")
    records: dict[tuple[str, str], dict[str, Any]] = {}
    paths: set[Path] = set()
    hashes: set[str] = set()
    for row in inventory.get("sources") or []:
        if not isinstance(row, dict) or row.get("format") != "RUBRIC_PAIR_SIMILARITY":
            continue
        task = normalize_space(row.get("task"))
        role = normalize_space(row.get("role"))
        expected_class = "TRAIN_ELIGIBLE" if role == "train" else "DEV_ONLY"
        if (
            task not in TASK_ORDER
            or role not in {"train", "eval"}
            or row.get("classification") != expected_class
            or row.get("availability") != "LOCAL_HASH_VALIDATED"
        ):
            raise ValueError(f"audited pair source role/class contract failed: {task}/{role}")
        physical = row.get("physical_artifact") or {}
        path = _resolve(physical.get("path"), anchor=inventory_path, root=root)
        digest = normalize_space(physical.get("sha256"))
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"audited pair source hash drift: {task}/{role}")
        paths.add(path)
        hashes.add(digest)
        records[(task, role)] = row
    expected = {(task, role) for task in TASK_ORDER for role in ("train", "eval")}
    if set(records) != expected or len(paths) != 1 or len(hashes) != 1:
        raise ValueError("audited pair sources are incomplete or physically inconsistent")
    norm_sources = [
        row
        for row in inventory.get("sources") or []
        if isinstance(row, dict)
        and row.get("format") == "NORM_TO_METRIC_TRUTH"
        and row.get("task") in TASK_ORDER
    ]
    unsupported = [
        row.get("source_id")
        for row in norm_sources
        if row.get("classification") == "TRAIN_ELIGIBLE"
        and row.get("supervised_model_training_allowed") is True
    ]
    if unsupported:
        raise ValueError(
            "independently train-eligible norm truth requires an explicit pair materializer: "
            f"{unsupported}"
        )
    norm_merge_audit = {
        "audited_norm_truth_sources": len(norm_sources),
        "independently_supervised_train_eligible_sources": 0,
        "merged_norm_truth_pair_rows": 0,
        "excluded_sources": [
            {
                "source_id": row.get("source_id"),
                "task": row.get("task"),
                "classification": row.get("classification"),
                "row_count": row.get("row_count"),
                "reason": "source_contract_forbids_supervised_model_training"
                if row.get("classification") == "TRAIN_ELIGIBLE"
                else "selection_or_evaluation_only_source",
            }
            for row in norm_sources
        ],
    }
    return next(iter(paths)), next(iter(hashes)), records, norm_merge_audit


def _load_current_banks(
    *, manifest_path: Path, hierarchy_root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    manifest = _json(manifest_path)
    if manifest.get("source_mode") not in (None, "canonical"):
        raise ValueError("current-bank manifest must be canonical")
    bank_meta = manifest.get("banks") or {}
    banks: dict[str, dict[str, Any]] = {}
    bindings: dict[str, Any] = {}
    for task in TASK_ORDER:
        declared = bank_meta.get(task)
        if not isinstance(declared, dict):
            raise ValueError(f"manifest lacks task bank: {task}")
        path = (hierarchy_root / f"{task}_general_r2_expanded.json").resolve()
        digest = sha256_file(path)
        groups = _json(path).get("merged_groups")
        if (
            digest != normalize_space(declared.get("source_sha256"))
            or not isinstance(groups, list)
            or len(groups) != int(declared.get("count", -1))
        ):
            raise ValueError(f"current hierarchy bank hash/count drift: {task}")
        key_to_ids: dict[str, set[str]] = defaultdict(set)
        metrics: dict[str, dict[str, Any]] = {}
        for index, group in enumerate(groups):
            if not isinstance(group, dict):
                raise ValueError(f"non-object current bank metric: {task}/{index}")
            metric_id = f"a{index}"
            metrics[metric_id] = {
                "metric_id": metric_id,
                "metric_card": _metric_card(group),
            }
            leaves = group.get("all_leaves") or []
            if not isinstance(leaves, list):
                raise ValueError(f"current bank all_leaves is not a list: {task}/{metric_id}")
            for leaf in leaves:
                if not isinstance(leaf, dict):
                    raise ValueError(f"current bank leaf is not an object: {task}/{metric_id}")
                key = str(leaf.get("key") or "").strip()
                if key:
                    key_to_ids[key].add(metric_id)
        banks[task] = {
            "bank_source_sha256": digest,
            "metrics": metrics,
            "key_to_ids": key_to_ids,
        }
        bindings[task] = {
            **_binding(path),
            "metric_count": len(metrics),
            "mapped_rubric_keys": len(key_to_ids),
            "unambiguous_rubric_keys": sum(len(ids) == 1 for ids in key_to_ids.values()),
            "ambiguous_rubric_keys": sum(len(ids) > 1 for ids in key_to_ids.values()),
        }
    return banks, {
        "manifest": _binding(manifest_path),
        "banks": bindings,
    }


def _quarantine(
    *, line_number: int, row: Mapping[str, Any], reason: str
) -> dict[str, Any]:
    task = normalize_space(row.get("task"))
    key_a = str(row.get("key_a") or "")
    key_b = str(row.get("key_b") or "")
    return {
        "schema_version": QUARANTINE_SCHEMA,
        "task": task,
        "source_line": line_number,
        "source_split": row.get("split"),
        "source_pair_id": _pair_identity(task, key_a, key_b)
        if key_a and key_b
        else None,
        "key_a": key_a or None,
        "key_b": key_b or None,
        "score": row.get("score"),
        "reason": reason,
    }


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {name: None for name in ("min", "p25", "median", "p75", "max")}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        return ordered[min(int(math.floor((len(ordered) - 1) * fraction)), len(ordered) - 1)]

    return {
        "min": ordered[0],
        "p25": at(0.25),
        "median": at(0.50),
        "p75": at(0.75),
        "max": ordered[-1],
    }


def _adaptive_exposure_budgets(train_rows: int) -> list[int]:
    """Checkpoint at 25x/50x/100x unique rows without exceeding 400K."""

    if train_rows < 1:
        raise ValueError("train_rows must be positive")
    values = []
    for multiplier in (25, 50, 100):
        raw = min(400_000, max(10_000, train_rows * multiplier))
        rounded = int(math.ceil(raw / 8.0) * 8)
        values.append(rounded)
    return sorted(set(values))


def _build_task(
    *,
    task: str,
    pair_path: Path,
    pair_hash: str,
    audit_records: Mapping[tuple[str, str], Mapping[str, Any]],
    bank: Mapping[str, Any],
    eval_groups: set[str],
    output_dir: Path,
    published_output_dir: Path,
    norm_merge_audit: Mapping[str, Any],
) -> dict[str, Any]:
    key_to_ids = bank["key_to_ids"]
    metrics = bank["metrics"]
    drafts: dict[tuple[str, str, str], dict[str, Any]] = {}
    quarantine_rows: list[dict[str, Any]] = []
    quarantine_counts: Counter[str] = Counter()
    observed_split_counts: Counter[str] = Counter()
    compatible_raw = directional_drafts = duplicate_drafts = 0

    def reject(line_number: int, row: Mapping[str, Any], reason: str) -> None:
        quarantine_counts[reason] += 1
        quarantine_rows.append(_quarantine(line_number=line_number, row=row, reason=reason))

    for line_number, row in _rows(pair_path):
        if row.get("task") != task:
            continue
        split = normalize_space(row.get("split"))
        observed_split_counts[split] += 1
        if split not in {"train", "eval"}:
            raise ValueError(f"blind/test/unknown split in v6 pair source: {task}/{split}")
        key_a = str(row.get("key_a") or "").strip()
        key_b = str(row.get("key_b") or "").strip()
        canonical_a = normalize_space(row.get("canonical_a"))
        canonical_b = normalize_space(row.get("canonical_b"))
        if not key_a or not key_b or key_a == key_b or not canonical_a or not canonical_b:
            reject(line_number, row, "malformed_pair_identity_or_text")
            continue
        try:
            group_a = _source_group(key_a)
            group_b = _source_group(key_b)
        except ValueError:
            reject(line_number, row, "malformed_source_group")
            continue
        score = row.get("score")
        if score not in SCORE_TO_RELATION:
            reject(line_number, row, "malformed_or_null_v6_score")
            continue
        try:
            cosine = float(row.get("cos"))
        except (TypeError, ValueError):
            reject(line_number, row, "malformed_cosine_provenance")
            continue
        if not math.isfinite(cosine) or not -1.0 <= cosine <= 1.0:
            reject(line_number, row, "malformed_cosine_provenance")
            continue
        ids_a = key_to_ids.get(key_a) or set()
        ids_b = key_to_ids.get(key_b) or set()
        if not ids_a or not ids_b:
            reject(line_number, row, "rubric_outside_current_bank_coverage")
            continue
        if len(ids_a) != 1 or len(ids_b) != 1:
            reject(line_number, row, "ambiguous_rubric_to_current_bank_mapping")
            continue
        metric_a = next(iter(ids_a))
        metric_b = next(iter(ids_b))
        if score == 2 and metric_a != metric_b:
            reject(line_number, row, "same_rule_crosses_current_bank_metrics")
            continue
        if score in {0, 1} and metric_a == metric_b:
            reject(line_number, row, "nonexact_label_inside_same_current_bank_metric")
            continue
        if split == "train" and ({group_a, group_b} & eval_groups):
            reject(line_number, row, "train_pair_touches_dev_source_group")
            continue
        compatible_raw += 1
        role = "dev" if split == "eval" else "train"
        relation = SCORE_TO_RELATION[score]
        pair_id = _pair_identity(task, key_a, key_b)
        for key, text, source_group, candidate_metric in (
            (key_a, canonical_a, group_a, metric_b),
            (key_b, canonical_b, group_b, metric_a),
        ):
            directional_drafts += 1
            uid = _norm_uid(task, key)
            draft_key = (role, uid, candidate_metric)
            existing = drafts.get(draft_key)
            provenance = {
                "source_line": line_number,
                "source_pair_id": pair_id,
                "v6_score": score,
                "cosine": cosine,
            }
            if existing is None:
                drafts[draft_key] = {
                    "task": task,
                    "split": role,
                    "norm_uid": uid,
                    "source_group": source_group,
                    "query": text,
                    "metric_id": candidate_metric,
                    "relation_votes": {relation},
                    "provenance": [provenance],
                }
            else:
                if existing["source_group"] != source_group or existing["query"] != text:
                    existing["relation_votes"].add("TEXT_OR_GROUP_CONFLICT")
                existing["relation_votes"].add(relation)
                existing["provenance"].append(provenance)
                duplicate_drafts += 1

    for role in ("train", "eval"):
        audit = audit_records[(task, role)]
        if observed_split_counts[role] != int(audit.get("row_count", -1)):
            raise ValueError(f"pair source count differs from audit: {task}/{role}")
        null_count = sum(
            1
            for row in quarantine_rows
            if row["source_split"] == role and row["reason"] == "malformed_or_null_v6_score"
        )
        if null_count != int(audit.get("rejected_row_count", -1)):
            raise ValueError(f"malformed pair count differs from audit: {task}/{role}")

    output_rows: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    conflict_count = 0
    for (role, uid, metric_id), draft in sorted(drafts.items()):
        votes = draft.pop("relation_votes")
        provenance = draft.pop("provenance")
        if len(votes) != 1 or "TEXT_OR_GROUP_CONFLICT" in votes:
            conflict_count += 1
            quarantine_counts["conflicting_directional_deduplication"] += 1
            quarantine_rows.append(
                {
                    "schema_version": QUARANTINE_SCHEMA,
                    "task": task,
                    "source_line": None,
                    "source_split": role,
                    "source_pair_id": None,
                    "key_a": None,
                    "key_b": None,
                    "score": None,
                    "reason": "conflicting_directional_deduplication",
                    "norm_uid": uid,
                    "metric_id": metric_id,
                    "relation_votes": sorted(votes),
                    "source_lines": sorted({int(row["source_line"]) for row in provenance}),
                }
            )
            continue
        relation = next(iter(votes))
        cosines = [float(row["cosine"]) for row in provenance]
        row = {
            "schema_version": SCHEMA,
            **draft,
            "relation": relation,
            "metric_card": metrics[metric_id]["metric_card"],
            "current_bank_source_sha256": bank["bank_source_sha256"],
            "label_source": "audited_norm_embed_v6_rubric_pair_judge",
            "pair_source_sha256": pair_hash,
            "source_pair_count": len(provenance),
            "source_pair_ids_sha256": hashlib.sha256(
                "\n".join(sorted(row["source_pair_id"] for row in provenance)).encode()
            ).hexdigest(),
            "v6_cosine_min": min(cosines),
            "v6_cosine_max": max(cosines),
            "realistic_hard_negative": relation == "REJECT"
            and max(cosines) >= HARD_NEGATIVE_COSINE,
            "negative_provenance": {
                "candidate_pool": "historical_semantic_neighbor_pair_pool",
                "v6_score_semantics": "0=unrelated,1=related,2=same_rule",
                "hard_negative_cosine_threshold": HARD_NEGATIVE_COSINE,
                "current_bank_compatibility_checked": True,
            },
        }
        output_rows[role].append(row)

    for role in output_rows:
        output_rows[role].sort(key=lambda row: (str(row["norm_uid"]), str(row["metric_id"])))
    train_groups = {str(row["source_group"]) for row in output_rows["train"]}
    dev_groups = {str(row["source_group"]) for row in output_rows["dev"]}
    train_uids = {str(row["norm_uid"]) for row in output_rows["train"]}
    dev_uids = {str(row["norm_uid"]) for row in output_rows["dev"]}
    if train_groups & dev_groups or train_uids & dev_uids:
        raise ValueError(f"train/dev source identity leakage after build: {task}")
    for role in ("train", "dev"):
        classes = {str(row["relation"]) for row in output_rows[role]}
        if classes != {"EXACT", "FAMILY", "REJECT"}:
            raise ValueError(f"task/split lacks a three-way CE class: {task}/{role}/{classes}")

    task_dir = output_dir / task
    published_task_dir = published_output_dir / task
    task_dir.mkdir()
    train_path = task_dir / "train.jsonl"
    dev_path = task_dir / "dev.jsonl"
    quarantine_path = task_dir / "quarantine.jsonl"
    _write_jsonl(train_path, output_rows["train"])
    _write_jsonl(dev_path, output_rows["dev"])
    quarantine_rows.sort(
        key=lambda row: (
            int(row["source_line"]) if row.get("source_line") is not None else 10**12,
            str(row.get("reason")),
        )
    )
    _write_jsonl(quarantine_path, quarantine_rows)

    # Exercise the exact production loader, including duplicate and source-group checks.
    train_examples = load_pair_examples([train_path])
    dev_examples = load_pair_examples([dev_path])
    if {row.source_group for row in train_examples} & {
        row.source_group for row in dev_examples
    }:
        raise ValueError(f"production loader found source-group leakage: {task}")

    class_counts = {
        role: dict(sorted(Counter(str(row["relation"]) for row in rows).items()))
        for role, rows in output_rows.items()
    }
    metric_coverage = {
        role: sorted({str(row["metric_id"]) for row in rows})
        for role, rows in output_rows.items()
    }
    reject_cosines = {
        role: [
            float(row["v6_cosine_max"])
            for row in rows
            if row["relation"] == "REJECT"
        ]
        for role, rows in output_rows.items()
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "FROZEN_TASK_LOCAL_THREE_WAY_CURRENT_BANK_CE_DATASET",
        "task": task,
        "inputs": {
            "audited_pair_source": {
                "path": str(pair_path),
                "sha256": pair_hash,
                "train_logical_source": audit_records[(task, "train")]["source_id"],
                "dev_logical_source": audit_records[(task, "eval")]["source_id"],
            },
            "current_bank_source_sha256": bank["bank_source_sha256"],
        },
        "outputs": {
            "train": {
                **_binding(train_path, published_path=published_task_dir / "train.jsonl"),
                "count": len(output_rows["train"]),
            },
            "dev": {
                **_binding(dev_path, published_path=published_task_dir / "dev.jsonl"),
                "count": len(output_rows["dev"]),
            },
            "quarantine": {
                **_binding(
                    quarantine_path,
                    published_path=published_task_dir / "quarantine.jsonl",
                ),
                "count": len(quarantine_rows),
            },
        },
        "class_counts": class_counts,
        "source_coverage": {
            "input_split_counts": dict(sorted(observed_split_counts.items())),
            "compatible_raw_pairs": compatible_raw,
            "directional_drafts": directional_drafts,
            "compatible_duplicate_drafts_collapsed": duplicate_drafts,
            "conflicting_deduplications_quarantined": conflict_count,
            "train_source_groups": len(train_groups),
            "dev_source_groups": len(dev_groups),
            "source_group_overlap": 0,
            "norm_uid_overlap": 0,
        },
        "metric_coverage": {
            role: {
                "covered_count": len(ids),
                "bank_metric_count": len(metrics),
                "covered_fraction": len(ids) / len(metrics),
                "covered_metric_ids": ids,
                "uncovered_metric_ids": sorted(set(metrics) - set(ids)),
            }
            for role, ids in metric_coverage.items()
        },
        "realistic_hard_negative_provenance": {
            "threshold": HARD_NEGATIVE_COSINE,
            "candidate_pool": "historical_semantic_neighbor_pair_pool",
            "family_near_miss_counts": {
                role: class_counts[role].get("FAMILY", 0) for role in ("train", "dev")
            },
            "reject": {
                role: {
                    "count": len(values),
                    "hard_count": sum(value >= HARD_NEGATIVE_COSINE for value in values),
                    "hard_fraction": (
                        sum(value >= HARD_NEGATIVE_COSINE for value in values) / len(values)
                        if values
                        else None
                    ),
                    "cosine_quantiles": _quantiles(values),
                }
                for role, values in reject_cosines.items()
            },
        },
        "quarantine_reason_counts": dict(sorted(quarantine_counts.items())),
        "norm_truth_pair_merge": {
            "audited_norm_truth_sources": sum(
                row.get("task") == task for row in norm_merge_audit["excluded_sources"]
            ),
            "independently_supervised_train_eligible_sources": 0,
            "merged_norm_truth_pair_rows": 0,
            "excluded_sources": [
                row
                for row in norm_merge_audit["excluded_sources"]
                if row.get("task") == task
            ],
        },
        "contracts": {
            "source_group_disjoint": True,
            "no_blind_or_test_rows_read": True,
            "current_bank_ids_and_hash_bound": True,
            "three_way_semantics": {
                "EXACT": "v6 score=2 and both rubric keys map to the same current-bank metric",
                "FAMILY": "v6 score=1 and rubric keys map to distinct current-bank metrics",
                "REJECT": "v6 score=0 and rubric keys map to distinct current-bank metrics",
            },
            "cross_task_truth_borrowed": False,
        },
        "release_ready": False,
    }
    report_path = task_dir / "report.json"
    _atomic_json(report_path, report)
    low_dev_exact = class_counts["dev"].get("EXACT", 0) < 20
    exposure_budgets = _adaptive_exposure_budgets(len(output_rows["train"]))
    commands = []
    for seed in (20260713, 20260714):
        command = [
            "python",
            "-m",
            "scripts.tools.silver_match_v3.train_nemotron_cross_encoder",
            "--train-pairs",
            str(published_task_dir / "train.jsonl"),
            "--dev-pairs",
            str(published_task_dir / "dev.jsonl"),
            "--model",
            DEFAULT_NEMOTRON,
            "--output",
            str(published_task_dir / f"nemotron_ce_seed{seed}"),
            "--seed",
            str(seed),
        ]
        for budget in exposure_budgets:
            command.extend(("--exposure-budget", str(budget)))
        commands.append(command)
    queue = {
        "schema_version": QUEUE_SCHEMA,
        "status": "READY_FOR_TASK_LOCAL_TRAINING_LOW_EXACT_DEV_SUPPORT"
        if low_dev_exact
        else "READY_FOR_TASK_LOCAL_TRAINING",
        "task": task,
        "train_pairs": _binding(
            train_path, published_path=published_task_dir / "train.jsonl"
        ),
        "dev_pairs": _binding(dev_path, published_path=published_task_dir / "dev.jsonl"),
        "dataset_report": _binding(
            report_path, published_path=published_task_dir / "report.json"
        ),
        "commands": commands,
        "recommended_exposure_budgets": exposure_budgets,
        "budget_policy": "25x/50x/100x unique train rows, floor 10K, ceiling 400K, rounded to 8",
        "task_local_lora_only": True,
        "cross_task_pooling": False,
        "low_exact_dev_support": low_dev_exact,
        "exact_dev_rows": class_counts["dev"].get("EXACT", 0),
        "blind_or_test_inputs": [],
        "release_ready": False,
    }
    queue_path = task_dir / "TRAIN_QUEUE.json"
    _atomic_json(queue_path, queue)
    return {
        "task": task,
        "report": _binding(
            report_path, published_path=published_task_dir / "report.json"
        ),
        "train_queue": _binding(
            queue_path, published_path=published_task_dir / "TRAIN_QUEUE.json"
        ),
        "train_rows": len(output_rows["train"]),
        "dev_rows": len(output_rows["dev"]),
        "class_counts": class_counts,
        "low_exact_dev_support": low_dev_exact,
        "release_ready": False,
    }


def build(
    *,
    audit_freeze_path: Path,
    manifest_path: Path,
    hierarchy_root: Path,
    output_dir: Path,
    root: Path,
) -> dict[str, Any]:
    audit_freeze_path = audit_freeze_path.resolve()
    manifest_path = manifest_path.resolve()
    hierarchy_root = hierarchy_root.resolve()
    output_dir = output_dir.resolve()
    root = root.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite CE dataset freeze: {output_dir}")
    pair_path, pair_hash, audit_records, norm_merge_audit = _load_audited_pair_source(
        audit_freeze_path=audit_freeze_path, root=root
    )
    banks, bank_audit = _load_current_banks(
        manifest_path=manifest_path, hierarchy_root=hierarchy_root
    )

    eval_groups: dict[str, set[str]] = defaultdict(set)
    physical_counts: Counter[tuple[str, str]] = Counter()
    for _line_number, row in _rows(pair_path):
        task = normalize_space(row.get("task"))
        if task not in TASK_ORDER:
            continue
        split = normalize_space(row.get("split"))
        physical_counts[(task, split)] += 1
        if split == "eval":
            for field in ("key_a", "key_b"):
                key = str(row.get(field) or "").strip()
                if key:
                    try:
                        eval_groups[task].add(_source_group(key))
                    except ValueError:
                        pass
    for task in TASK_ORDER:
        for split in ("train", "eval"):
            expected = int(audit_records[(task, split)].get("row_count", -1))
            if physical_counts[(task, split)] != expected:
                raise ValueError(f"pair source changed since truth audit: {task}/{split}")

    stage = output_dir.with_name(output_dir.name + f".tmp.{os.getpid()}")
    if stage.exists():
        raise FileExistsError(stage)
    stage.mkdir(parents=True)
    try:
        task_reports = []
        for task in TASK_ORDER:
            task_reports.append(
                _build_task(
                    task=task,
                    pair_path=pair_path,
                    pair_hash=pair_hash,
                    audit_records=audit_records,
                    bank=banks[task],
                    eval_groups=eval_groups[task],
                    output_dir=stage,
                    published_output_dir=output_dir,
                    norm_merge_audit=norm_merge_audit,
                )
            )
        freeze = {
            "schema_version": FREEZE_SCHEMA,
            "status": "FROZEN_SEVEN_TASK_CURRENT_BANK_CE_DATASETS",
            "task_order": TASK_ORDER,
            "truth_audit_freeze": _binding(audit_freeze_path),
            "audited_pair_source": _binding(pair_path),
            "current_bank_audit": bank_audit,
            "task_reports": task_reports,
            "task_count": len(task_reports),
            "all_tasks_have_three_way_train_and_dev": True,
            "all_tasks_source_group_disjoint": True,
            "blind_or_test_rows_read": 0,
            "cross_task_truth_borrowed": False,
            "norm_truth_pair_rows_merged": 0,
            "notice_and_comment_processed_last": True,
            "release_ready": False,
        }
        _atomic_json(stage / "FREEZE.json", freeze)
        stage.replace(output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return _json(output_dir / "FREEZE.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-freeze", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--hierarchy-root", type=Path, default=Path("outputs/hierarchy"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = build(
        audit_freeze_path=args.audit_freeze,
        manifest_path=args.manifest,
        hierarchy_root=args.hierarchy_root,
        output_dir=args.output_dir,
        root=args.root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
