#!/usr/bin/env python3
"""Freeze train-only full-bank teacher packs for v6 scale recovery.

The strict v6-to-current-bank conversion deliberately quarantines historical
rubric pairs whenever either endpoint is absent from, or maps ambiguously into,
the current task bank.  Pair counts overstate this loss because the same rubric
appears in many pairs.  This builder deduplicates those endpoints into rubric
queries, reserves every source document observed in v6 eval before selection,
and freezes nested full-bank teacher-label budgets.

Historical pair scores and mapping candidates are used only for aggregate
auditing and hidden sampling balance.  They are never written to a
teacher-visible item.  No eval, held-out, blind, or cross-task label is used as
teacher truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping

from .build_v6_pair_ce_datasets import (
    FREEZE_SCHEMA as UPSTREAM_FREEZE_SCHEMA,
    SCORE_TO_RELATION,
    TASK_ORDER,
    _load_current_banks,
    _rows,
    _source_group,
)
from .common import normalize_space, sha256_file


SCHEMA = "silver-match-v3-v6-scale-recovery-teacher-pack-v1"
ITEM_SCHEMA = "silver-match-v3-v6-scale-recovery-teacher-item-v1"
LEDGER_SCHEMA = "silver-match-v3-v6-scale-recovery-private-ledger-v1"
REPORT_SCHEMA = "silver-match-v3-v6-scale-recovery-task-report-v1"
QUEUE_SCHEMA = "silver-match-v3-v6-scale-recovery-label-queue-v1"
TEACHER_LABEL_SCHEMA = "silver-match-v3-v6-scale-recovery-teacher-label-v1"
TARGET_REASONS = {
    "rubric_outside_current_bank_coverage",
    "ambiguous_rubric_to_current_bank_mapping",
}
TIER_LIMITS = (("pilot", 256), ("core", 1024), ("scale", None))
CHUNK_SIZE = 20
ABSTENTIONS = {
    "NO_EXPLICIT_CRITERION",
    "CONTEXT_NEEDED",
    "GENERIC_VERDICT",
    "NO_CANDIDATE_FITS",
    "NOISE",
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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
    if root.name in value.parts:
        relocated = root.joinpath(*value.parts[value.parts.index(root.name) + 1 :])
        if relocated.exists():
            return relocated.resolve()
    return value


def _binding(path: Path, *, published: Path | None = None) -> dict[str, Any]:
    return {
        "path": str(published or path),
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
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def _uid(task: str, key: str) -> str:
    return hashlib.sha256(f"v6-scale-recovery-query\0{task}\0{key}".encode()).hexdigest()


def _stable(seed: str, *values: str) -> tuple[str, ...]:
    payload = "\x1f".join((seed, *values))
    return (hashlib.sha256(payload.encode()).hexdigest(), *values)


def _quantiles(values: list[int]) -> dict[str, int | None]:
    if not values:
        return {key: None for key in ("min", "p25", "median", "p75", "max")}
    ordered = sorted(values)

    def at(fraction: float) -> int:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "min": ordered[0],
        "p25": at(0.25),
        "median": at(0.5),
        "p75": at(0.75),
        "max": ordered[-1],
    }


def _teacher_schema(chunk_size: int = CHUNK_SIZE) -> dict[str, Any]:
    decisions = ["EXACT", "FAMILY", *sorted(ABSTENTIONS)]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": TEACHER_LABEL_SCHEMA,
        "type": "object",
        "additionalProperties": False,
        "required": ["task", "chunk_id", "labels"],
        "properties": {
            "task": {"type": "string"},
            "chunk_id": {"type": "string"},
            "labels": {
                "type": "array",
                "minItems": 1,
                "maxItems": chunk_size,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "norm_uid",
                        "decision",
                        "primary_metric_id",
                        "pair_labels",
                        "confidence",
                        "reason",
                    ],
                    "properties": {
                        "norm_uid": {
                            "type": "string",
                            "pattern": "^[0-9a-f]{64}$",
                        },
                        "decision": {"type": "string", "enum": decisions},
                        "primary_metric_id": {
                            "anyOf": [
                                {"type": "string", "pattern": "^a[0-9]+$"},
                                {"type": "null"},
                            ]
                        },
                        "pair_labels": {
                            "type": "array",
                            "maxItems": 10,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["metric_id", "relation"],
                                "properties": {
                                    "metric_id": {
                                        "type": "string",
                                        "pattern": "^a[0-9]+$",
                                    },
                                    "relation": {
                                        "type": "string",
                                        "enum": ["EXACT", "FAMILY", "REJECT"],
                                    },
                                },
                            },
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "reason": {"type": "string", "minLength": 8, "maxLength": 600},
                    },
                },
            },
        },
    }


def validate_teacher_label(
    label: Mapping[str, Any], *, valid_metric_ids: set[str]
) -> None:
    """Validate the semantic contract not fully expressible in JSON Schema."""

    required = {
        "norm_uid",
        "decision",
        "primary_metric_id",
        "pair_labels",
        "confidence",
        "reason",
    }
    if set(label) != required:
        raise ValueError("teacher label fields differ from the frozen contract")
    uid = label.get("norm_uid")
    if not isinstance(uid, str) or len(uid) != 64 or any(c not in "0123456789abcdef" for c in uid):
        raise ValueError("invalid norm_uid")
    decision = label.get("decision")
    if decision not in {"EXACT", "FAMILY", *ABSTENTIONS}:
        raise ValueError("invalid teacher decision")
    if label.get("confidence") not in {"high", "medium", "low"}:
        raise ValueError("invalid confidence")
    reason = label.get("reason")
    if not isinstance(reason, str) or not 8 <= len(reason) <= 600:
        raise ValueError("invalid contrastive reason")
    pairs = label.get("pair_labels")
    if not isinstance(pairs, list) or len(pairs) > 10:
        raise ValueError("invalid pair_labels")
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    for pair in pairs:
        if not isinstance(pair, dict) or set(pair) != {"metric_id", "relation"}:
            raise ValueError("invalid pair label fields")
        metric_id = pair.get("metric_id")
        relation = pair.get("relation")
        if metric_id not in valid_metric_ids or metric_id in seen:
            raise ValueError("unknown or duplicate pair-label metric")
        if relation not in SCORE_TO_RELATION.values():
            raise ValueError("invalid pair relation")
        seen.add(str(metric_id))
        counts[str(relation)] += 1
    primary = label.get("primary_metric_id")
    exact_ids = [pair["metric_id"] for pair in pairs if pair["relation"] == "EXACT"]
    if decision == "EXACT":
        if primary not in valid_metric_ids or exact_ids != [primary]:
            raise ValueError("EXACT requires one matching primary metric")
        if not 2 <= counts["REJECT"] <= 5 or counts["FAMILY"] > 3:
            raise ValueError("EXACT requires 2-5 hard rejects and at most 3 family labels")
    elif decision == "FAMILY":
        if primary is not None or counts["EXACT"] or not 1 <= counts["FAMILY"] <= 4:
            raise ValueError("FAMILY requires 1-4 family labels and no exact primary")
        if not 2 <= counts["REJECT"] <= 5:
            raise ValueError("FAMILY requires 2-5 hard rejects")
    else:
        if primary is not None or counts["EXACT"] or counts["FAMILY"]:
            raise ValueError("typed abstentions cannot assert exact/family metrics")
        if counts["REJECT"] > 5:
            raise ValueError("typed abstentions permit at most 5 explicit rejects")


def _instructions() -> str:
    return """# Full-bank teacher labeling for v6 scale recovery

Each item is a standalone train-only rubric criterion. Compare it with every
metric in that task's frozen `bank.json`; there is deliberately no top-k slate
or inherited proposal.

Use query-level `EXACT` only when one metric directly captures the same
actionable criterion and beats every sibling. Return that metric as
`primary_metric_id` and as the single `EXACT` pair label. Add zero to three
genuinely related but broader, narrower, or sibling metrics as `FAMILY`, plus
two to five plausible-but-wrong hard negatives as `REJECT`.

Use query-level `FAMILY` when the construct belongs in the bank but no one leaf
is an exact winner. Return one to four `FAMILY` pair labels, two to five hard
`REJECT` labels, and a null primary metric.

Otherwise use the most specific typed abstention:

- `NO_CANDIDATE_FITS`: a clear criterion is expressed, but the bank lacks it.
- `NO_EXPLICIT_CRITERION`: content without an evaluative or prescriptive norm.
- `CONTEXT_NEEDED`: the criterion cannot be resolved without missing context.
- `GENERIC_VERDICT`: undifferentiated praise or blame.
- `NOISE`: garbled or meaningless language.

An abstention is a valid positive result. It has a null primary metric and no
EXACT/FAMILY pair labels; up to five explicit hard rejects are optional. Never
force yield. Every metric ID may appear at most once. The reason must briefly
contrast the selected interpretation with its nearest alternative. Preserve
each `norm_uid` verbatim and return every item exactly once using
`teacher_label.schema.json`.

The hidden selection ledger contains sampling/audit information and must never
be included in a teacher prompt. Teacher-visible inputs are only a chunk from
the chosen tier, this instruction file, the schema, and that task's full bank.
"""


def _interleave_groups(rows: list[dict[str, Any]], *, seed: str) -> list[dict[str, Any]]:
    by_group: dict[str, deque[dict[str, Any]]] = {}
    for group, values in _group_rows(rows, "source_group").items():
        values.sort(
            key=lambda row: (
                -int(row["supporting_pair_count"]),
                *_stable(seed, group, str(row["norm_uid"])),
            )
        )
        by_group[group] = deque(values)
    groups = sorted(by_group, key=lambda group: _stable(seed, group))
    output: list[dict[str, Any]] = []
    while groups:
        remaining: list[str] = []
        for group in groups:
            output.append(by_group[group].popleft())
            if by_group[group]:
                remaining.append(group)
        groups = remaining
    return output


def _group_rows(
    rows: Iterable[dict[str, Any]], field: str
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return grouped


def rank_candidates(rows: list[dict[str, Any]], *, task: str) -> list[dict[str, Any]]:
    """Coverage-first deterministic ranking, interleaved by source group."""

    stratum_sizes = Counter(
        metric_id
        for row in rows
        for metric_id in set(row.get("hidden_balance_metric_ids") or [])
    )
    prepared: list[dict[str, Any]] = []
    for row in rows:
        hints = sorted(set(row.get("hidden_balance_metric_ids") or []))
        if hints:
            balance = min(
                hints,
                key=lambda value: (
                    stratum_sizes[value],
                    *_stable(f"{task}:stratum", str(row["norm_uid"]), value),
                ),
            )
        else:
            balance = "__UNSTRATIFIED__"
        prepared.append({**row, "balance_stratum": balance})

    # Distinguish outside-bank and ambiguous-map recovery so the smaller but
    # especially informative ambiguous pool is not drowned out.
    buckets: dict[str, deque[dict[str, Any]]] = {}
    for bucket, values in _group_rows(
        [
            {
                **row,
                "balance_bucket": f"{row['recovery_reason']}::{row['balance_stratum']}",
            }
            for row in prepared
        ],
        "balance_bucket",
    ).items():
        buckets[bucket] = deque(
            _interleave_groups(values, seed=f"v6-recovery:{task}:{bucket}")
        )
    bucket_order = sorted(buckets, key=lambda value: _stable(f"{task}:buckets", value))
    ranked: list[dict[str, Any]] = []
    while bucket_order:
        remaining = []
        for bucket in bucket_order:
            ranked.append(buckets[bucket].popleft())
            if buckets[bucket]:
                remaining.append(bucket)
        bucket_order = remaining
    return [{**row, "budget_rank": rank} for rank, row in enumerate(ranked, 1)]


def _visible_item(row: Mapping[str, Any], *, bank_hash: str, bank_count: int) -> dict[str, Any]:
    return {
        "schema_version": ITEM_SCHEMA,
        "task": row["task"],
        "norm_uid": row["norm_uid"],
        "query": row["query"],
        "current_bank_source_sha256": bank_hash,
        "full_bank_metric_count": bank_count,
        "full_bank_required": True,
        "truth_hidden": True,
    }


def _tier_audit(rows: list[dict[str, Any]], bank_count: int) -> dict[str, Any]:
    strata = {str(row["balance_stratum"]) for row in rows}
    hinted = strata - {"__UNSTRATIFIED__"}
    reasons = Counter(str(row["recovery_reason"]) for row in rows)
    return {
        "query_count": len(rows),
        "source_group_count": len({str(row["source_group"]) for row in rows}),
        "recovery_reason_counts": dict(sorted(reasons.items())),
        "hidden_balance_metric_strata_covered": len(hinted),
        "current_bank_metric_count": bank_count,
        "hidden_balance_metric_coverage_fraction": len(hinted) / bank_count,
        "unstratified_query_count": sum(
            row["balance_stratum"] == "__UNSTRATIFIED__" for row in rows
        ),
        "supporting_pair_count": sum(int(row["supporting_pair_count"]) for row in rows),
        "potential_pair_labels_if_all_nonabstain": {
            "minimum": len(rows) * 3,
            "maximum": len(rows) * 10,
            "note": "actual count depends on typed abstention rate and teacher decisions",
        },
    }


def _load_upstream(
    *, freeze_path: Path, root: Path
) -> tuple[dict[str, Any], Path, dict[str, dict[str, Any]]]:
    freeze = _json(freeze_path)
    if (
        freeze.get("schema_version") != UPSTREAM_FREEZE_SCHEMA
        or freeze.get("status") != "FROZEN_SEVEN_TASK_CURRENT_BANK_CE_DATASETS"
        or freeze.get("task_order") != TASK_ORDER
        or freeze.get("blind_or_test_rows_read") != 0
        or freeze.get("cross_task_truth_borrowed") is not False
        or freeze.get("all_tasks_source_group_disjoint") is not True
    ):
        raise ValueError("upstream strict CE freeze contract failed")
    pair_binding = freeze.get("audited_pair_source") or {}
    pair_path = _resolve(pair_binding.get("path"), anchor=freeze_path, root=root)
    if sha256_file(pair_path) != normalize_space(pair_binding.get("sha256")):
        raise ValueError("audited v6 pair source hash drift")
    audit_binding = freeze.get("truth_audit_freeze") or {}
    audit_path = _resolve(audit_binding.get("path"), anchor=freeze_path, root=root)
    if sha256_file(audit_path) != normalize_space(audit_binding.get("sha256")):
        raise ValueError("truth audit freeze hash drift")
    reports: dict[str, dict[str, Any]] = {}
    for task_record in freeze.get("task_reports") or []:
        task = normalize_space(task_record.get("task"))
        binding = task_record.get("report") or {}
        path = _resolve(binding.get("path"), anchor=freeze_path, root=root)
        if task not in TASK_ORDER or sha256_file(path) != normalize_space(binding.get("sha256")):
            raise ValueError("upstream task report hash/task drift")
        reports[task] = _json(path)
    if set(reports) != set(TASK_ORDER):
        raise ValueError("upstream strict CE freeze lacks a task report")
    return freeze, pair_path, reports


def _collect_candidates(
    *,
    pair_path: Path,
    banks: Mapping[str, Mapping[str, Any]],
    reports: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    eval_groups: dict[str, set[str]] = defaultdict(set)
    physical_counts: Counter[tuple[str, str]] = Counter()
    physical_task_counts: Counter[str] = Counter()
    physical_total = 0
    for _line, row in _rows(pair_path):
        physical_total += 1
        task = normalize_space(row.get("task"))
        physical_task_counts[task or "__MISSING_TASK__"] += 1
        if task not in TASK_ORDER:
            continue
        split = normalize_space(row.get("split"))
        physical_counts[(task, split)] += 1
        if split == "eval":
            for field in ("key_a", "key_b"):
                key = str(row.get(field) or "").strip()
                if key:
                    eval_groups[task].add(_source_group(key))

    raw: dict[str, dict[str, dict[str, Any]]] = {task: {} for task in TASK_ORDER}
    counts: dict[str, Counter[str]] = {task: Counter() for task in TASK_ORDER}
    for _line, row in _rows(pair_path):
        task = normalize_space(row.get("task"))
        if task not in TASK_ORDER or normalize_space(row.get("split")) != "train":
            continue
        counts[task]["train_rows_scanned"] += 1
        key_a = str(row.get("key_a") or "").strip()
        key_b = str(row.get("key_b") or "").strip()
        text_a = normalize_space(row.get("canonical_a"))
        text_b = normalize_space(row.get("canonical_b"))
        if not key_a or not key_b or not text_a or not text_b or key_a == key_b:
            counts[task]["malformed_rows_excluded"] += 1
            continue
        try:
            group_a, group_b = _source_group(key_a), _source_group(key_b)
        except ValueError:
            counts[task]["malformed_rows_excluded"] += 1
            continue
        if {group_a, group_b} & eval_groups[task]:
            counts[task]["train_rows_excluded_by_eval_source_firewall"] += 1
            continue
        key_to_ids = banks[task]["key_to_ids"]
        ids_a = set(key_to_ids.get(key_a) or [])
        ids_b = set(key_to_ids.get(key_b) or [])
        if (not ids_a or len(ids_a) > 1) or (not ids_b or len(ids_b) > 1):
            counts[task]["eligible_supporting_pair_rows"] += 1
        for key, text, group, own_ids, partner_ids in (
            (key_a, text_a, group_a, ids_a, ids_b),
            (key_b, text_b, group_b, ids_b, ids_a),
        ):
            if len(own_ids) == 1:
                continue
            reason = (
                "ambiguous_rubric_to_current_bank_mapping"
                if own_ids
                else "rubric_outside_current_bank_coverage"
            )
            uid = _uid(task, key)
            existing = raw[task].get(key)
            if existing is None:
                existing = {
                    "schema_version": LEDGER_SCHEMA,
                    "task": task,
                    "norm_uid": uid,
                    "rubric_key": key,
                    "source_group": group,
                    "query": text,
                    "recovery_reason": reason,
                    "supporting_pair_count": 0,
                    "supporting_v6_score_counts": Counter(),
                    "hidden_balance_metric_ids": set(own_ids),
                    "partner_source_groups": set(),
                }
                raw[task][key] = existing
            elif (
                existing["query"] != text
                or existing["source_group"] != group
                or existing["recovery_reason"] != reason
            ):
                existing["identity_conflict"] = True
            existing["supporting_pair_count"] += 1
            existing["supporting_v6_score_counts"][str(row.get("score"))] += 1
            existing["partner_source_groups"].add(group_b if key == key_a else group_a)
            if len(partner_ids) == 1:
                existing["hidden_balance_metric_ids"].update(partner_ids)

    candidates: dict[str, list[dict[str, Any]]] = {}
    audit: dict[str, Any] = {}
    for task in TASK_ORDER:
        expected_counts = reports[task].get("source_coverage", {}).get("input_split_counts", {})
        for split in ("train", "eval"):
            if physical_counts[(task, split)] != int(expected_counts.get(split, -1)):
                raise ValueError(f"pair source count differs from upstream report: {task}/{split}")
        values = []
        conflicts = 0
        for value in raw[task].values():
            if value.pop("identity_conflict", False):
                conflicts += 1
                continue
            value["supporting_v6_score_counts"] = dict(
                sorted(value["supporting_v6_score_counts"].items())
            )
            value["hidden_balance_metric_ids"] = sorted(value["hidden_balance_metric_ids"])
            value["partner_source_group_count"] = len(value.pop("partner_source_groups"))
            values.append(value)
        ranked = rank_candidates(values, task=task)
        candidates[task] = ranked
        reason_counts = Counter(str(row["recovery_reason"]) for row in ranked)
        audit[task] = {
            "physical_pair_rows": {
                split: physical_counts[(task, split)] for split in ("train", "eval")
            },
            "strict_quarantine_pair_reason_counts": {
                reason: int(reports[task].get("quarantine_reason_counts", {}).get(reason, 0))
                for reason in sorted(TARGET_REASONS)
            },
            "train_rows_excluded_by_eval_source_firewall": counts[task][
                "train_rows_excluded_by_eval_source_firewall"
            ],
            "eligible_supporting_pair_rows": counts[task]["eligible_supporting_pair_rows"],
            "unique_train_only_recovery_queries": len(ranked),
            "unique_recovery_reason_counts": dict(sorted(reason_counts.items())),
            "unique_source_groups": len({row["source_group"] for row in ranked}),
            "identity_conflicts_quarantined": conflicts,
            "supporting_pairs_per_unique_query": _quantiles(
                [int(row["supporting_pair_count"]) for row in ranked]
            ),
            "queries_with_hidden_metric_balance_stratum": sum(
                bool(row["hidden_balance_metric_ids"]) for row in ranked
            ),
            "eval_source_group_count": len(eval_groups[task]),
            "recovery_query_overlap_with_eval_source_groups": len(
                {str(row["source_group"]) for row in ranked} & eval_groups[task]
            ),
        }
        if audit[task]["recovery_query_overlap_with_eval_source_groups"]:
            raise ValueError(f"eval source-group leakage in recovery pack: {task}")
    in_scope_total = sum(
        physical_counts[(task, split)]
        for task in TASK_ORDER
        for split in ("train", "eval")
    )
    source_census = {
        "physical_rows_all_tasks": physical_total,
        "physical_rows_seven_current_tasks": in_scope_total,
        "physical_rows_outside_seven_task_scope": physical_total - in_scope_total,
        "physical_rows_by_task": dict(sorted(physical_task_counts.items())),
    }
    return candidates, audit, source_census


def build(
    *,
    upstream_freeze_path: Path,
    output_dir: Path,
    root: Path,
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, Any]:
    upstream_freeze_path = upstream_freeze_path.resolve()
    output_dir = output_dir.resolve()
    root = root.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite recovery freeze: {output_dir}")
    if chunk_size < 1 or chunk_size > 40:
        raise ValueError("chunk_size must be between 1 and 40")
    upstream, pair_path, reports = _load_upstream(
        freeze_path=upstream_freeze_path, root=root
    )
    manifest_binding = upstream.get("current_bank_audit", {}).get("manifest") or {}
    manifest_path = _resolve(
        manifest_binding.get("path"), anchor=upstream_freeze_path, root=root
    )
    if sha256_file(manifest_path) != normalize_space(manifest_binding.get("sha256")):
        raise ValueError("current-bank manifest hash drift")
    banks, bank_audit = _load_current_banks(
        manifest_path=manifest_path, hierarchy_root=root / "outputs/hierarchy"
    )
    if bank_audit != upstream.get("current_bank_audit"):
        raise ValueError("current-bank audit differs from upstream strict freeze")
    candidates, attrition, source_census = _collect_candidates(
        pair_path=pair_path, banks=banks, reports=reports
    )

    stage = output_dir.with_name(output_dir.name + f".tmp.{os.getpid()}")
    if stage.exists():
        raise FileExistsError(stage)
    stage.mkdir(parents=True)
    try:
        instructions_path = stage / "LABELING_INSTRUCTIONS.md"
        instructions_path.write_text(_instructions(), encoding="utf-8")
        schema_path = stage / "teacher_label.schema.json"
        _atomic_json(schema_path, _teacher_schema(chunk_size))
        task_records = []
        for task in TASK_ORDER:
            task_dir = stage / task
            published_task_dir = output_dir / task
            task_dir.mkdir()
            metrics = banks[task]["metrics"]
            bank_rows = [metrics[metric_id] for metric_id in sorted(metrics, key=lambda x: int(x[1:]))]
            bank_path = task_dir / "bank.json"
            _atomic_json(
                bank_path,
                {
                    "schema_version": SCHEMA,
                    "task": task,
                    "current_bank_source_sha256": banks[task]["bank_source_sha256"],
                    "metric_count": len(bank_rows),
                    "metrics": bank_rows,
                },
            )
            ranked = candidates[task]
            ledger_path = task_dir / "PRIVATE_SELECTION_LEDGER.jsonl"
            _write_jsonl(ledger_path, ranked)
            tiers: dict[str, Any] = {}
            for tier, raw_limit in TIER_LIMITS:
                limit = len(ranked) if raw_limit is None else min(raw_limit, len(ranked))
                selected = ranked[:limit]
                visible = [
                    _visible_item(
                        row,
                        bank_hash=str(banks[task]["bank_source_sha256"]),
                        bank_count=len(bank_rows),
                    )
                    for row in selected
                ]
                tier_dir = task_dir / f"tier_{tier}"
                items_path = tier_dir / "items.jsonl"
                _write_jsonl(items_path, visible)
                chunks: list[Path] = []
                for start in range(0, len(visible), chunk_size):
                    chunk = tier_dir / "chunks" / f"part-{start // chunk_size:04d}.jsonl"
                    _write_jsonl(chunk, visible[start : start + chunk_size])
                    chunks.append(chunk)
                tiers[tier] = {
                    "requested_query_budget": raw_limit if raw_limit is not None else "ALL_ELIGIBLE",
                    "effective_query_budget": len(selected),
                    "nested_prefix_through_budget_rank": len(selected),
                    "items": _binding(
                        items_path,
                        published=published_task_dir / f"tier_{tier}" / "items.jsonl",
                    ),
                    "chunks": {
                        str(output_dir / task / f"tier_{tier}" / "chunks" / path.name): sha256_file(path)
                        for path in chunks
                    },
                    "audit": _tier_audit(selected, len(bank_rows)),
                }
            if not (
                tiers["pilot"]["effective_query_budget"]
                <= tiers["core"]["effective_query_budget"]
                <= tiers["scale"]["effective_query_budget"]
            ):
                raise AssertionError("teacher budgets are not nested")
            report = {
                "schema_version": REPORT_SCHEMA,
                "status": "FROZEN_TRAIN_ONLY_FULL_BANK_TEACHER_PACKS",
                "task": task,
                "attrition_audit": attrition[task],
                "bank": {
                    **_binding(bank_path, published=published_task_dir / "bank.json"),
                    "source_sha256": banks[task]["bank_source_sha256"],
                    "metric_count": len(bank_rows),
                },
                "private_selection_ledger": {
                    **_binding(
                        ledger_path,
                        published=published_task_dir / "PRIVATE_SELECTION_LEDGER.jsonl",
                    ),
                    "teacher_visible": False,
                },
                "tiers": tiers,
                "contracts": {
                    "source_group_firewall": True,
                    "eval_heldout_blind_labels_used_as_truth": False,
                    "historical_pair_labels_teacher_visible": False,
                    "historical_mapping_candidates_teacher_visible": False,
                    "full_current_task_bank_presented": True,
                    "teacher_output_relations": ["EXACT", "FAMILY", "REJECT"],
                    "typed_abstention_enabled": True,
                    "cross_task_borrowing": False,
                    "metric_coverage_balanced_order": True,
                    "tiers_are_nested_prefixes": True,
                },
                "release_ready": False,
            }
            report_path = task_dir / "report.json"
            _atomic_json(report_path, report)
            queue = {
                "schema_version": QUEUE_SCHEMA,
                "status": "READY_FOR_PILOT_TEACHER_LABELING",
                "task": task,
                "recommended_sequence": ["pilot", "core", "scale"],
                "quality_gate": (
                    "Validate every response with teacher_label.schema.json and the semantic "
                    "contract; manually adjudicate a stratified pilot sample before expanding."
                ),
                "teacher_visible_inputs_per_call": [
                    str(output_dir / "LABELING_INSTRUCTIONS.md"),
                    str(output_dir / "teacher_label.schema.json"),
                    str(output_dir / task / "bank.json"),
                    str(output_dir / task / "tier_<tier>" / "chunks" / "part-<id>.jsonl"),
                ],
                "forbidden_teacher_input": str(
                    output_dir / task / "PRIVATE_SELECTION_LEDGER.jsonl"
                ),
                "tiers": {
                    tier: {
                        "query_count": value["effective_query_budget"],
                        "chunk_count": len(value["chunks"]),
                    }
                    for tier, value in tiers.items()
                },
                "gpu_required": False,
                "labels_collected": 0,
                "release_ready": False,
            }
            queue_path = task_dir / "LABEL_QUEUE.json"
            _atomic_json(queue_path, queue)
            task_records.append(
                {
                    "task": task,
                    "eligible_unique_queries": len(ranked),
                    "report": _binding(
                        report_path, published=published_task_dir / "report.json"
                    ),
                    "label_queue": _binding(
                        queue_path, published=published_task_dir / "LABEL_QUEUE.json"
                    ),
                    "tier_query_counts": {
                        tier: value["effective_query_budget"] for tier, value in tiers.items()
                    },
                    "release_ready": False,
                }
            )
        freeze = {
            "schema_version": SCHEMA,
            "status": "FROZEN_SEVEN_TASK_TRAIN_ONLY_SCALE_RECOVERY_PACKS",
            "task_order": TASK_ORDER,
            "task_records": task_records,
            "task_count": len(task_records),
            "global_attrition_audit": {
                **source_census,
                "unique_train_only_recovery_queries": sum(
                    record["eligible_unique_queries"] for record in task_records
                ),
                "target_reasons": sorted(TARGET_REASONS),
                "unit_of_teacher_labeling": "unique_rubric_query_not_repeated_pair",
            },
            "inputs": {
                "upstream_strict_ce_freeze": _binding(upstream_freeze_path),
                "audited_pair_source": _binding(pair_path),
                "current_bank_audit": bank_audit,
            },
            "outputs": {
                "instructions": _binding(
                    instructions_path, published=output_dir / "LABELING_INSTRUCTIONS.md"
                ),
                "teacher_schema": _binding(
                    schema_path, published=output_dir / "teacher_label.schema.json"
                ),
            },
            "contracts": {
                "teacher_pack_contains_only_train_rows": True,
                "all_eval_source_groups_reserved_before_selection": True,
                "eval_heldout_blind_labels_used_as_truth": False,
                "all_banks_hash_bound_and_complete": True,
                "all_tiers_nested_and_deterministic": True,
                "private_balance_hints_hidden_from_teacher": True,
                "cross_task_borrowing": False,
                "notice_and_comment_processed_last": True,
                "gpu_jobs_launched": False,
            },
            "labels_collected": 0,
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
    parser.add_argument("--upstream-freeze", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()
    result = build(
        upstream_freeze_path=args.upstream_freeze,
        output_dir=args.output_dir,
        root=args.root,
        chunk_size=args.chunk_size,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
