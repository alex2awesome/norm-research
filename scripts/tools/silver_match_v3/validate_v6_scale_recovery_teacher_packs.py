#!/usr/bin/env python3
"""Independently validate a frozen v6 scale-recovery teacher pack."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .build_v6_pair_ce_datasets import TASK_ORDER, _rows, _source_group
from .common import normalize_space, sha256_file
from .prepare_v6_scale_recovery_teacher_packs import (
    ITEM_SCHEMA,
    LEDGER_SCHEMA,
    SCHEMA,
)


VISIBLE_FIELDS = {
    "schema_version",
    "task",
    "norm_uid",
    "query",
    "current_bank_source_sha256",
    "full_bank_metric_count",
    "full_bank_required",
    "truth_hidden",
}
PRIVATE_ONLY_FIELDS = {
    "rubric_key",
    "source_group",
    "recovery_reason",
    "supporting_pair_count",
    "supporting_v6_score_counts",
    "hidden_balance_metric_ids",
    "balance_stratum",
    "balance_bucket",
    "budget_rank",
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [row for _line, row in _rows(path)]


def _resolve(raw: Any, *, anchor: Path, root: Path) -> Path:
    value = Path(str(raw or ""))
    if not str(value):
        raise ValueError("empty path")
    if value.is_absolute() and value.exists():
        return value.resolve()
    if not value.is_absolute():
        anchored = (anchor / value).resolve()
        if anchored.exists():
            return anchored
        return (root / value).resolve()
    if root.name in value.parts:
        relocated = root.joinpath(*value.parts[value.parts.index(root.name) + 1 :])
        if relocated.exists():
            return relocated.resolve()
    return value


def _check_binding(
    binding: Mapping[str, Any], *, anchor: Path, root: Path
) -> Path:
    path = _resolve(binding.get("path"), anchor=anchor, root=root)
    if not path.is_file() or sha256_file(path) != normalize_space(binding.get("sha256")):
        raise ValueError(f"binding hash/path drift: {path}")
    if "bytes" in binding and path.stat().st_size != int(binding["bytes"]):
        raise ValueError(f"binding byte-count drift: {path}")
    return path


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


def validate(*, pack_dir: Path, root: Path) -> dict[str, Any]:
    pack_dir = pack_dir.resolve()
    root = root.resolve()
    freeze_path = pack_dir / "FREEZE.json"
    freeze = _json(freeze_path)
    if (
        freeze.get("schema_version") != SCHEMA
        or freeze.get("status") != "FROZEN_SEVEN_TASK_TRAIN_ONLY_SCALE_RECOVERY_PACKS"
        or freeze.get("task_order") != TASK_ORDER
        or freeze.get("task_count") != len(TASK_ORDER)
        or freeze.get("labels_collected") != 0
        or freeze.get("release_ready") is not False
    ):
        raise ValueError("recovery freeze top-level contract failed")
    contracts = freeze.get("contracts") or {}
    required_true = {
        "teacher_pack_contains_only_train_rows",
        "all_eval_source_groups_reserved_before_selection",
        "all_banks_hash_bound_and_complete",
        "all_tiers_nested_and_deterministic",
        "private_balance_hints_hidden_from_teacher",
        "notice_and_comment_processed_last",
    }
    if any(contracts.get(key) is not True for key in required_true):
        raise ValueError("required recovery freeze contract is not true")
    if (
        contracts.get("eval_heldout_blind_labels_used_as_truth") is not False
        or contracts.get("cross_task_borrowing") is not False
        or contracts.get("gpu_jobs_launched") is not False
    ):
        raise ValueError("forbidden recovery freeze behavior recorded")

    _check_binding(freeze["outputs"]["instructions"], anchor=pack_dir, root=root)
    schema_path = _check_binding(
        freeze["outputs"]["teacher_schema"], anchor=pack_dir, root=root
    )
    schema = _json(schema_path)
    schema_max = int(schema["properties"]["labels"]["maxItems"])
    pair_path = _check_binding(
        freeze["inputs"]["audited_pair_source"], anchor=pack_dir, root=root
    )

    eval_groups: dict[str, set[str]] = defaultdict(set)
    physical_total = 0
    physical_tasks: Counter[str] = Counter()
    for _line, row in _rows(pair_path):
        physical_total += 1
        task = normalize_space(row.get("task")) or "__MISSING_TASK__"
        physical_tasks[task] += 1
        if task in TASK_ORDER and normalize_space(row.get("split")) == "eval":
            for field in ("key_a", "key_b"):
                key = str(row.get(field) or "").strip()
                if key:
                    eval_groups[task].add(_source_group(key))
    global_audit = freeze.get("global_attrition_audit") or {}
    if physical_total != int(global_audit.get("physical_rows_all_tasks", -1)):
        raise ValueError("full physical source census mismatch")
    if dict(sorted(physical_tasks.items())) != global_audit.get("physical_rows_by_task"):
        raise ValueError("physical per-task source census mismatch")

    records = freeze.get("task_records") or []
    if [record.get("task") for record in records] != TASK_ORDER:
        raise ValueError("task records are missing, reordered, or cross-task")
    total_queries = 0
    task_results = []
    for record in records:
        task = str(record["task"])
        report_path = _check_binding(record["report"], anchor=pack_dir, root=root)
        queue_path = _check_binding(record["label_queue"], anchor=pack_dir, root=root)
        report = _json(report_path)
        queue = _json(queue_path)
        if report.get("task") != task or queue.get("task") != task:
            raise ValueError(f"task report/queue mismatch: {task}")
        bank_path = _check_binding(report["bank"], anchor=pack_dir, root=root)
        bank = _json(bank_path)
        metric_ids = [str(row.get("metric_id") or "") for row in bank.get("metrics") or []]
        if (
            bank.get("task") != task
            or bank.get("schema_version") != SCHEMA
            or not metric_ids
            or "" in metric_ids
            or len(metric_ids) != len(set(metric_ids))
            or len(metric_ids) != int(bank.get("metric_count", -1))
            or len(metric_ids) != int(report["bank"].get("metric_count", -1))
        ):
            raise ValueError(f"bank completeness contract failed: {task}")
        ledger_path = _check_binding(
            report["private_selection_ledger"], anchor=pack_dir, root=root
        )
        if report["private_selection_ledger"].get("teacher_visible") is not False:
            raise ValueError(f"private ledger marked teacher-visible: {task}")
        ledger = _jsonl(ledger_path)
        if any(row.get("schema_version") != LEDGER_SCHEMA for row in ledger):
            raise ValueError(f"private ledger schema mismatch: {task}")
        ledger_uids = [str(row.get("norm_uid") or "") for row in ledger]
        if (
            not ledger
            or "" in ledger_uids
            or len(ledger_uids) != len(set(ledger_uids))
            or {str(row.get("task")) for row in ledger} != {task}
        ):
            raise ValueError(f"private ledger identity failure: {task}")
        groups = {str(row.get("source_group") or "") for row in ledger}
        if "" in groups or groups & eval_groups[task]:
            raise ValueError(f"eval source-group leakage: {task}")
        ranks = [int(row.get("budget_rank", -1)) for row in ledger]
        if ranks != list(range(1, len(ledger) + 1)):
            raise ValueError(f"non-contiguous deterministic budget ranks: {task}")

        previous_uids: list[str] = []
        tier_counts: dict[str, int] = {}
        for tier in ("pilot", "core", "scale"):
            tier_meta = report["tiers"][tier]
            items_path = _check_binding(tier_meta["items"], anchor=pack_dir, root=root)
            items = _jsonl(items_path)
            uids = [str(row.get("norm_uid") or "") for row in items]
            if uids != ledger_uids[: len(items)] or uids[: len(previous_uids)] != previous_uids:
                raise ValueError(f"tier is not a nested ledger prefix: {task}/{tier}")
            if any(set(row) != VISIBLE_FIELDS for row in items):
                raise ValueError(f"teacher item fields differ from contract: {task}/{tier}")
            if any(PRIVATE_ONLY_FIELDS & set(row) for row in items):
                raise ValueError(f"private hint leaked to teacher item: {task}/{tier}")
            if any(
                row.get("schema_version") != ITEM_SCHEMA
                or row.get("task") != task
                or row.get("current_bank_source_sha256") != report["bank"]["source_sha256"]
                or row.get("full_bank_metric_count") != len(metric_ids)
                or row.get("full_bank_required") is not True
                or row.get("truth_hidden") is not True
                for row in items
            ):
                raise ValueError(f"visible full-bank binding failed: {task}/{tier}")
            chunks: list[dict[str, Any]] = []
            for raw_path, digest in sorted(tier_meta["chunks"].items()):
                path = _resolve(raw_path, anchor=pack_dir, root=root)
                if sha256_file(path) != digest:
                    raise ValueError(f"chunk hash drift: {path}")
                values = _jsonl(path)
                if len(values) > schema_max:
                    raise ValueError(f"chunk exceeds teacher schema maximum: {path}")
                chunks.extend(values)
            if chunks != items:
                raise ValueError(f"chunks do not reconstruct tier items: {task}/{tier}")
            previous_uids = uids
            tier_counts[tier] = len(items)
        if previous_uids != ledger_uids:
            raise ValueError(f"scale tier does not cover every eligible query: {task}")
        if tier_counts != record.get("tier_query_counts"):
            raise ValueError(f"recorded tier counts drift: {task}")
        total_queries += len(ledger)
        task_results.append(
            {
                "task": task,
                "queries": len(ledger),
                "source_groups": len(groups),
                "bank_metrics": len(metric_ids),
                "tier_counts": tier_counts,
                "eval_source_group_overlap": 0,
            }
        )
    if total_queries != int(global_audit.get("unique_train_only_recovery_queries", -1)):
        raise ValueError("global unique recovery query count mismatch")
    return {
        "schema_version": "silver-match-v3-v6-scale-recovery-validation-v1",
        "status": "VALIDATED",
        "pack_freeze": {"path": str(freeze_path), "sha256": sha256_file(freeze_path)},
        "physical_pair_source_rows": physical_total,
        "unique_train_only_recovery_queries": total_queries,
        "tasks": task_results,
        "contracts": {
            "all_declared_hashes_valid": True,
            "full_source_census_reproduced": True,
            "all_eval_source_groups_absent": True,
            "all_task_banks_complete_and_hash_bound": True,
            "all_tiers_nested_and_chunk_complete": True,
            "private_hints_absent_from_teacher_items": True,
            "notice_and_comment_last": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(pack_dir=args.pack_dir, root=args.root)
    if args.output:
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(output)
        _atomic_json(output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
