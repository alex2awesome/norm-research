#!/usr/bin/env python3
"""Validate append-only Codex outputs from v6 scale-recovery pilots."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .build_v6_pair_ce_datasets import TASK_ORDER
from .common import normalize_space, sha256_file
from .run_v6_scale_recovery_codex_pilots import (
    DEFAULT_TASKS,
    REQUEST_SCHEMA,
    RUN_SCHEMA,
    _load_bank,
    _rows,
    validate_payload,
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json_x(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _check(path: Path, expected: str, expected_bytes: int | None = None) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"hash/path drift: {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ValueError(f"byte-count drift: {path}")


def validate(*, output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    freeze_path = output_dir / "RUN_FREEZE.json"
    freeze = _json(freeze_path)
    if (
        freeze.get("schema_version") != RUN_SCHEMA
        or freeze.get("status") != "FROZEN_BEFORE_ANY_CODEX_PILOT_REQUEST"
        or freeze.get("task_order") != DEFAULT_TASKS
        or freeze.get("tier") != "pilot"
        or freeze.get("rows_per_task") != 256
        or freeze.get("model") != "gpt-5.6-sol"
        or freeze.get("reasoning_effort") != "high"
        or freeze.get("runner_concurrency") != 1
        or freeze.get("private_selection_ledger_staged") is not False
        or freeze.get("teacher_visible_prior_labels_or_proposals") is not False
        or freeze.get("notice_and_comment_launched") is not False
        or freeze.get("core_or_scale_launched") is not False
    ):
        raise ValueError("pilot run freeze contract failed")
    if DEFAULT_TASKS != TASK_ORDER[:-1] or TASK_ORDER[-1] != "notice-and-comment":
        raise AssertionError("N&C-last task contract drift")
    staged = freeze.get("staged_inputs") or {}
    workspace = Path(str(staged.get("workspace") or "")).resolve()
    if not workspace.is_dir():
        raise ValueError("minimal labeling workspace is missing")
    public_inputs = staged.get("public_inputs") or []
    declared_staged: set[Path] = set()
    for record in public_inputs:
        source = record.get("source") or {}
        destination = record.get("staged") or {}
        source_path = Path(str(source.get("path") or "")).resolve()
        staged_path = Path(str(destination.get("path") or "")).resolve()
        if workspace not in staged_path.parents:
            raise ValueError("declared staged input escapes minimal workspace")
        _check(source_path, str(source.get("sha256") or ""), int(source.get("bytes", -1)))
        _check(
            staged_path,
            str(destination.get("sha256") or ""),
            int(destination.get("bytes", -1)),
        )
        if sha256_file(source_path) != sha256_file(staged_path):
            raise ValueError("staged public input differs from source")
        declared_staged.add(staged_path)
    observed_staged = {path.resolve() for path in workspace.rglob("*") if path.is_file()}
    if observed_staged != declared_staged:
        raise ValueError("minimal workspace contains undeclared evidence")
    serialized = json.dumps(freeze, sort_keys=True).lower()
    if "private_selection_ledger.jsonl" in serialized:
        raise ValueError("private ledger path leaked into run freeze")
    if any("ledger" in path.name.lower() or "private" in path.name.lower() for path in observed_staged):
        raise ValueError("private/ledger-like file is staged")

    task_results = []
    total_valid_rows = total_expected_rows = 0
    global_decisions: Counter[str] = Counter()
    global_relations: Counter[str] = Counter()
    for task_record in staged.get("tasks") or []:
        task = normalize_space(task_record.get("task"))
        if task not in DEFAULT_TASKS:
            raise ValueError(f"unexpected pilot task: {task}")
        _bank, metric_ids = _load_bank(workspace / task / "bank.json")
        task_valid = task_expected = 0
        decision_counts: Counter[str] = Counter()
        relation_counts: Counter[str] = Counter()
        completed_chunks = 0
        for chunk_record in task_record.get("chunks") or []:
            chunk_id = normalize_space(chunk_record.get("chunk_id"))
            chunk_path = Path(str(chunk_record.get("staged_path") or "")).resolve()
            if chunk_path not in declared_staged:
                raise ValueError(f"undeclared staged chunk: {task}/{chunk_id}")
            chunk = _rows(chunk_path)
            expected_uids = [str(row.get("norm_uid") or "") for row in chunk]
            task_expected += len(expected_uids)
            request_path = output_dir / "requests" / task / f"{chunk_id}.json"
            accepted_path = output_dir / "valid_labels" / task / f"{chunk_id}.json"
            if request_path.exists():
                request = _json(request_path)
                if (
                    request.get("schema_version") != REQUEST_SCHEMA
                    or request.get("status") != "FROZEN_BEFORE_REQUEST"
                    or request.get("task") != task
                    or request.get("chunk_id") != chunk_id
                    or request.get("model") != "gpt-5.6-sol"
                    or request.get("reasoning_effort") != "high"
                    or request.get("working_directory") != str(workspace)
                    or request.get("private_selection_ledger_available") is not False
                    or request.get("prompt_sha256")
                    != __import__("hashlib").sha256(str(request.get("prompt") or "").encode()).hexdigest()
                ):
                    raise ValueError(f"request contract drift: {task}/{chunk_id}")
                scrubbed = dict(request)
                scrubbed.pop("private_selection_ledger_available", None)
                request_text = json.dumps(scrubbed, sort_keys=True).lower()
                if "private_selection_ledger.jsonl" in request_text:
                    raise ValueError(f"private ledger path exposed in request: {task}/{chunk_id}")
            if not accepted_path.exists():
                continue
            if not request_path.exists():
                raise ValueError(f"accepted label lacks frozen request: {task}/{chunk_id}")
            summary = validate_payload(
                _json(accepted_path),
                task=task,
                chunk_id=chunk_id,
                expected_uids=expected_uids,
                metric_ids=metric_ids,
            )
            completed_chunks += 1
            task_valid += int(summary["row_count"])
            decision_counts.update(summary["decision_counts"])
            relation_counts.update(summary["pair_relation_counts"])
        if task_expected != int(task_record.get("row_count", -1)) or task_expected != 256:
            raise ValueError(f"expected pilot universe drift: {task}")
        total_expected_rows += task_expected
        total_valid_rows += task_valid
        global_decisions.update(decision_counts)
        global_relations.update(relation_counts)
        task_results.append(
            {
                "task": task,
                "expected_rows": task_expected,
                "valid_rows": task_valid,
                "expected_chunks": len(task_record.get("chunks") or []),
                "completed_chunks": completed_chunks,
                "decision_counts": dict(sorted(decision_counts.items())),
                "pair_relation_counts": dict(sorted(relation_counts.items())),
            }
        )
    if [row["task"] for row in task_results] != DEFAULT_TASKS:
        raise ValueError("task execution priority/order drift")
    status = "COMPLETE_VALIDATED" if total_valid_rows == total_expected_rows else "PARTIAL_VALIDATED"
    return {
        "schema_version": "silver-match-v3-v6-scale-recovery-codex-pilot-validation-v1",
        "status": status,
        "run_freeze": {"path": str(freeze_path), "sha256": sha256_file(freeze_path)},
        "model": "gpt-5.6-sol",
        "reasoning_effort": "high",
        "task_order": DEFAULT_TASKS,
        "expected_rows": total_expected_rows,
        "valid_rows": total_valid_rows,
        "tasks": task_results,
        "decision_counts": dict(sorted(global_decisions.items())),
        "pair_relation_counts": dict(sorted(global_relations.items())),
        "contracts": {
            "schema_and_semantic_validation_passed_for_every_accepted_label": True,
            "minimal_workspace_contains_only_declared_public_inputs": True,
            "full_bank_present_for_every_task": True,
            "private_selection_ledger_absent": True,
            "first_six_tasks_only": True,
            "notice_and_comment_not_launched": True,
            "core_and_scale_not_launched": True,
        },
        "release_ready": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(output_dir=args.output_dir)
    if args.output:
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(output)
        _atomic_json_x(output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
