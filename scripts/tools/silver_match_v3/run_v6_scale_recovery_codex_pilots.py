#!/usr/bin/env python3
"""Run fail-closed gpt-5.6-sol labeling over frozen recovery pilots.

The Codex worker runs from a minimal workspace containing only the public
labeling guide, output schema, complete task banks, and selected pilot chunks.
The private selection ledger and all prior labels remain outside the staged
workspace.  Requests, attempts, and accepted labels are append-only.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .build_v6_pair_ce_datasets import TASK_ORDER
from .common import normalize_space, sha256_file
from .prepare_v6_scale_recovery_teacher_packs import (
    SCHEMA as PACK_SCHEMA,
    validate_teacher_label,
)
from .validate_v6_scale_recovery_teacher_packs import validate as validate_pack


RUN_SCHEMA = "silver-match-v3-v6-scale-recovery-codex-pilot-run-v1"
REQUEST_SCHEMA = "silver-match-v3-v6-scale-recovery-codex-request-v1"
EVENT_SCHEMA = "silver-match-v3-v6-scale-recovery-codex-event-v1"
DEFAULT_TASKS = TASK_ORDER[:-1]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL row: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row: {path}:{line_number}")
            values.append(value)
    return values


def _binding(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


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


def _copy_x(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())


def _append_event(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _event(kind: str, **values: Any) -> dict[str, Any]:
    return {
        "schema_version": EVENT_SCHEMA,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": kind,
        **values,
    }


def _workspace_id(output_dir: Path, pack_hash: str) -> str:
    return hashlib.sha256(f"{output_dir.resolve()}\0{pack_hash}".encode()).hexdigest()[:20]


def _load_bank(path: Path) -> tuple[dict[str, Any], set[str]]:
    bank = _json(path)
    metrics = list(bank.get("metrics") or [])
    ids = {normalize_space(row.get("metric_id")) for row in metrics if isinstance(row, dict)}
    if (
        bank.get("schema_version") != PACK_SCHEMA
        or not ids
        or "" in ids
        or len(ids) != len(metrics)
        or len(ids) != int(bank.get("metric_count", -1))
    ):
        raise ValueError(f"invalid complete task bank: {path}")
    return bank, ids


def validate_payload(
    payload: Mapping[str, Any],
    *,
    task: str,
    chunk_id: str,
    expected_uids: list[str],
    metric_ids: set[str],
) -> dict[str, Any]:
    if set(payload) != {"task", "chunk_id", "labels"}:
        raise ValueError("label envelope fields differ from frozen schema")
    labels = payload.get("labels")
    if payload.get("task") != task or payload.get("chunk_id") != chunk_id or not isinstance(labels, list):
        raise ValueError("label envelope task/chunk/type drift")
    observed = [str(row.get("norm_uid") or "") for row in labels if isinstance(row, dict)]
    if (
        len(observed) != len(expected_uids)
        or len(observed) != len(set(observed))
        or set(observed) != set(expected_uids)
    ):
        raise ValueError("label envelope does not cover every chunk UID exactly once")
    decisions: Counter[str] = Counter()
    relations: Counter[str] = Counter()
    for label in labels:
        if not isinstance(label, dict):
            raise ValueError("teacher label is not an object")
        validate_teacher_label(label, valid_metric_ids=metric_ids)
        decisions[str(label["decision"])] += 1
        relations.update(str(pair["relation"]) for pair in label["pair_labels"])
    return {
        "row_count": len(labels),
        "decision_counts": dict(sorted(decisions.items())),
        "pair_relation_counts": dict(sorted(relations.items())),
    }


def _active_codex_exec() -> int:
    completed = subprocess.run(
        ["ps", "-axo", "command="],
        check=True,
        capture_output=True,
        text=True,
    )
    return sum(
        line.strip().startswith("codex exec ")
        for line in completed.stdout.splitlines()
    )


def _wait_for_capacity(cap: int, poll_seconds: int, events: Path) -> int:
    last_reported: int | None = None
    while True:
        active = _active_codex_exec()
        if active < cap:
            return active
        if active != last_reported:
            _append_event(
                events,
                _event(
                    "WAITING_FOR_EXTERNAL_CODEX_CAPACITY",
                    active_codex_exec=active,
                    external_codex_cap=cap,
                ),
            )
            last_reported = active
        time.sleep(poll_seconds)


def _prompt(task: str, chunk_id: str, count: int) -> str:
    return (
        "Act as an independent high-precision teacher for silver norm-to-metric "
        "scale recovery. The working directory is a deliberately minimal evidence "
        "workspace. Do not inspect any path outside it and do not search for prior "
        "labels, mappings, proposals, audits, hidden artifacts, MI, or outcomes. Read "
        "LABELING_INSTRUCTIONS.md and teacher_label.schema.json. Read the complete "
        f"current {task} metric bank at {task}/bank.json, then label all {count} "
        f"queries in {task}/chunks/{chunk_id}.jsonl independently. Consider every bank "
        "metric for every query. Use query-level EXACT, FAMILY, or the most specific "
        "typed abstention, and provide schema-valid pair-level EXACT/FAMILY/REJECT hard "
        "contrasts. Never force yield. Set the envelope task to "
        f"{task} and chunk_id to {chunk_id}. Return only the schema-conforming JSON object."
    )


def _stage_inputs(
    *, pack_dir: Path, output_dir: Path, tasks: list[str]
) -> tuple[Path, dict[str, Any]]:
    pack_freeze_path = pack_dir / "FREEZE.json"
    pack_hash = sha256_file(pack_freeze_path)
    workspace = Path("/private/tmp") / (
        "silver_match_v3_v6_recovery_codex_" + _workspace_id(output_dir, pack_hash)
    )
    sources: list[tuple[Path, Path]] = [
        (pack_dir / "LABELING_INSTRUCTIONS.md", workspace / "LABELING_INSTRUCTIONS.md"),
        (pack_dir / "teacher_label.schema.json", workspace / "teacher_label.schema.json"),
    ]
    task_records = []
    for task in tasks:
        source_task = pack_dir / task
        chunks = sorted((source_task / "tier_pilot" / "chunks").glob("part-*.jsonl"))
        if not chunks:
            raise ValueError(f"pilot pack has no chunks: {task}")
        sources.append((source_task / "bank.json", workspace / task / "bank.json"))
        staged_chunks = []
        expected_count = 0
        for source in chunks:
            destination = workspace / task / "chunks" / source.name
            sources.append((source, destination))
            rows = _rows(source)
            expected_count += len(rows)
            staged_chunks.append(
                {
                    "chunk_id": source.stem,
                    "row_count": len(rows),
                    "source": _binding(source),
                    "staged_path": str(destination),
                    "sha256": sha256_file(source),
                }
            )
        if expected_count != 256:
            raise ValueError(f"pilot tier is not exactly 256 rows: {task}/{expected_count}")
        task_records.append(
            {
                "task": task,
                "row_count": expected_count,
                "bank_source": _binding(source_task / "bank.json"),
                "chunks": staged_chunks,
            }
        )

    if workspace.exists():
        allowed = {str(destination.relative_to(workspace)) for _source, destination in sources}
        observed = {
            str(path.relative_to(workspace))
            for path in workspace.rglob("*")
            if path.is_file()
        }
        if observed != allowed:
            raise ValueError("minimal workspace contains undeclared files")
        for source, destination in sources:
            if sha256_file(source) != sha256_file(destination):
                raise ValueError(f"staged public input hash drift: {destination}")
    else:
        workspace.mkdir(parents=True)
        for source, destination in sources:
            _copy_x(source, destination)
    if any("PRIVATE" in path.name.upper() or "LEDGER" in path.name.upper() for path in workspace.rglob("*")):
        raise ValueError("private ledger-like file present in minimal workspace")
    return workspace, {
        "workspace": str(workspace),
        "pack_freeze": _binding(pack_freeze_path),
        "public_inputs": [
            {"source": _binding(source), "staged": _binding(destination)}
            for source, destination in sources
        ],
        "tasks": task_records,
        "private_selection_ledger_staged": False,
    }


def _validate_existing_run(
    *,
    freeze: Mapping[str, Any],
    output_dir: Path,
    model: str,
    effort: str,
    external_codex_cap: int,
    timeout_seconds: int,
    chunk_attempts: int,
) -> None:
    if (
        freeze.get("schema_version") != RUN_SCHEMA
        or freeze.get("status") != "FROZEN_BEFORE_ANY_CODEX_PILOT_REQUEST"
        or freeze.get("task_order") != DEFAULT_TASKS
        or freeze.get("tier") != "pilot"
        or freeze.get("model") != model
        or freeze.get("reasoning_effort") != effort
        or freeze.get("external_codex_cap") != external_codex_cap
        or freeze.get("timeout_seconds") != timeout_seconds
        or freeze.get("chunk_attempts") != chunk_attempts
        or freeze.get("private_selection_ledger_staged") is not False
        or freeze.get("notice_and_comment_launched") is not False
    ):
        raise ValueError(f"existing run freeze contract drift: {output_dir}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    pack_dir = Path(args.pack_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    repo = Path(args.root).resolve()
    if args.model != "gpt-5.6-sol" or args.reasoning_effort != "high":
        raise ValueError("this frozen pilot requires gpt-5.6-sol with high reasoning")
    if args.external_codex_cap < 1 or args.poll_seconds < 1:
        raise ValueError("external cap and poll interval must be positive")
    if args.chunk_attempts < 1 or args.timeout_seconds < 1:
        raise ValueError("attempt count and timeout must be positive")
    if DEFAULT_TASKS != TASK_ORDER[:-1] or TASK_ORDER[-1] != "notice-and-comment":
        raise AssertionError("task priority/N&C-last contract drift")
    pack_validation = validate_pack(pack_dir=pack_dir, root=repo)
    if pack_validation.get("status") != "VALIDATED":
        raise ValueError("source teacher pack did not independently validate")
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / "runtime.lock"
    lock = lock_path.open("a", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another recovery pilot runner owns {lock_path}") from exc
    lock.write(f"{datetime.now(timezone.utc).isoformat()} pid={os.getpid()}\n")
    lock.flush()
    os.fsync(lock.fileno())
    if not (output_dir / "RUN_FREEZE.json").exists():
        unexpected = {
            path.name for path in output_dir.iterdir() if path.name != "runtime.lock"
        }
        if unexpected:
            raise ValueError(f"unfrozen output directory is not empty: {sorted(unexpected)}")

    workspace, staged = _stage_inputs(
        pack_dir=pack_dir, output_dir=output_dir, tasks=DEFAULT_TASKS
    )
    freeze_path = output_dir / "RUN_FREEZE.json"
    if freeze_path.exists():
        freeze = _json(freeze_path)
        _validate_existing_run(
            freeze=freeze,
            output_dir=output_dir,
            model=args.model,
            effort=args.reasoning_effort,
            external_codex_cap=args.external_codex_cap,
            timeout_seconds=args.timeout_seconds,
            chunk_attempts=args.chunk_attempts,
        )
        if freeze.get("staged_inputs") != staged:
            raise ValueError("staged input bytes differ from existing run freeze")
    else:
        freeze = {
            "schema_version": RUN_SCHEMA,
            "status": "FROZEN_BEFORE_ANY_CODEX_PILOT_REQUEST",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "task_order": DEFAULT_TASKS,
            "tier": "pilot",
            "rows_per_task": 256,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "runner_concurrency": 1,
            "external_codex_cap": args.external_codex_cap,
            "timeout_seconds": args.timeout_seconds,
            "chunk_attempts": args.chunk_attempts,
            "staged_inputs": staged,
            "source_pack_validation": pack_validation["pack_freeze"],
            "private_selection_ledger_staged": False,
            "teacher_visible_prior_labels_or_proposals": False,
            "notice_and_comment_launched": False,
            "core_or_scale_launched": False,
            "release_ready": False,
        }
        _atomic_json_x(freeze_path, freeze)

    codex = shutil.which("codex")
    if not codex:
        raise FileNotFoundError("codex")
    schema_path = workspace / "teacher_label.schema.json"
    events = output_dir / "events.jsonl"
    failures = []
    accepted = skipped = 0
    started = time.time()
    for task_record in staged["tasks"]:
        task = str(task_record["task"])
        _bank, metric_ids = _load_bank(workspace / task / "bank.json")
        for chunk_record in task_record["chunks"]:
            chunk_id = str(chunk_record["chunk_id"])
            chunk_path = Path(str(chunk_record["staged_path"]))
            chunk_rows = _rows(chunk_path)
            expected_uids = [str(row.get("norm_uid") or "") for row in chunk_rows]
            if (
                "" in expected_uids
                or len(expected_uids) != len(set(expected_uids))
                or len(expected_uids) != int(chunk_record["row_count"])
            ):
                raise ValueError(f"staged chunk identity failure: {task}/{chunk_id}")
            accepted_path = output_dir / "valid_labels" / task / f"{chunk_id}.json"
            if accepted_path.exists():
                summary = validate_payload(
                    _json(accepted_path),
                    task=task,
                    chunk_id=chunk_id,
                    expected_uids=expected_uids,
                    metric_ids=metric_ids,
                )
                skipped += 1
                _append_event(events, _event("SKIPPED_VALID", task=task, chunk_id=chunk_id, **summary))
                continue
            prompt = _prompt(task, chunk_id, len(expected_uids))
            request_path = output_dir / "requests" / task / f"{chunk_id}.json"
            request = {
                "schema_version": REQUEST_SCHEMA,
                "status": "FROZEN_BEFORE_REQUEST",
                "task": task,
                "chunk_id": chunk_id,
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "working_directory": str(workspace),
                "inputs": {
                    "instructions": _binding(workspace / "LABELING_INSTRUCTIONS.md"),
                    "schema": _binding(schema_path),
                    "complete_bank": _binding(workspace / task / "bank.json"),
                    "chunk": _binding(chunk_path),
                },
                "private_selection_ledger_available": False,
            }
            if request_path.exists():
                if _json(request_path) != request:
                    raise ValueError(f"existing request drift: {request_path}")
            else:
                _atomic_json_x(request_path, request)
            success = False
            last_error = ""
            for attempt in range(1, args.chunk_attempts + 1):
                attempt_dir = output_dir / "attempts" / task / chunk_id
                output_path = attempt_dir / f"attempt-{attempt:03d}.output.json"
                log_path = attempt_dir / f"attempt-{attempt:03d}.log"
                if output_path.exists() or log_path.exists():
                    # Existing attempts are immutable. Re-accept if a previously valid
                    # response was not copied before interruption; otherwise continue.
                    if output_path.exists():
                        try:
                            summary = validate_payload(
                                _json(output_path),
                                task=task,
                                chunk_id=chunk_id,
                                expected_uids=expected_uids,
                                metric_ids=metric_ids,
                            )
                        except (ValueError, json.JSONDecodeError):
                            continue
                        _copy_x(output_path, accepted_path)
                        accepted += 1
                        _append_event(
                            events,
                            _event(
                                "RECOVERED_VALID_ATTEMPT",
                                task=task,
                                chunk_id=chunk_id,
                                attempt=attempt,
                                **summary,
                            ),
                        )
                        success = True
                        break
                    continue
                active_before = _wait_for_capacity(
                    args.external_codex_cap, args.poll_seconds, events
                )
                attempt_dir.mkdir(parents=True, exist_ok=True)
                command = [
                    codex,
                    "exec",
                    "--skip-git-repo-check",
                    "--ephemeral",
                    "--sandbox",
                    "read-only",
                    "--dangerously-bypass-hook-trust",
                    "--ignore-rules",
                    "-m",
                    args.model,
                    "-c",
                    f'model_reasoning_effort="{args.reasoning_effort}"',
                    "--output-schema",
                    str(schema_path),
                    "-o",
                    str(output_path),
                    prompt,
                ]
                child_env = os.environ.copy()
                child_env.pop("PYTHONPATH", None)
                child_env.pop("OLDPWD", None)
                child_env["PWD"] = str(workspace)
                timed_out = False
                _append_event(
                    events,
                    _event(
                        "ATTEMPT_STARTED",
                        task=task,
                        chunk_id=chunk_id,
                        attempt=attempt,
                        active_codex_exec_before_launch=active_before,
                    ),
                )
                attempt_started = time.time()
                with log_path.open("xb") as log:
                    try:
                        completed = subprocess.run(
                            command,
                            cwd=workspace,
                            env=child_env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=args.timeout_seconds,
                            check=False,
                        )
                    except subprocess.TimeoutExpired:
                        completed = None
                        timed_out = True
                try:
                    if timed_out:
                        raise ValueError("Codex attempt timed out")
                    if completed is None or completed.returncode != 0:
                        raise ValueError(
                            f"Codex exit code {None if completed is None else completed.returncode}"
                        )
                    if not output_path.is_file():
                        raise ValueError("Codex produced no structured output")
                    summary = validate_payload(
                        _json(output_path),
                        task=task,
                        chunk_id=chunk_id,
                        expected_uids=expected_uids,
                        metric_ids=metric_ids,
                    )
                    _copy_x(output_path, accepted_path)
                    accepted += 1
                    _append_event(
                        events,
                        _event(
                            "ATTEMPT_ACCEPTED",
                            task=task,
                            chunk_id=chunk_id,
                            attempt=attempt,
                            elapsed_seconds=time.time() - attempt_started,
                            output_sha256=sha256_file(output_path),
                            **summary,
                        ),
                    )
                    success = True
                    break
                except (ValueError, json.JSONDecodeError) as exc:
                    last_error = str(exc)
                    _append_event(
                        events,
                        _event(
                            "ATTEMPT_REJECTED",
                            task=task,
                            chunk_id=chunk_id,
                            attempt=attempt,
                            elapsed_seconds=time.time() - attempt_started,
                            error=last_error,
                            output_sha256=sha256_file(output_path)
                            if output_path.is_file()
                            else None,
                        ),
                    )
            if not success:
                failures.append({"task": task, "chunk_id": chunk_id, "error": last_error})
                _append_event(
                    events,
                    _event("CHUNK_FAILED_CLOSED", task=task, chunk_id=chunk_id, error=last_error),
                )

    summary = {
        "schema_version": "silver-match-v3-v6-scale-recovery-codex-run-summary-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_VALIDATED_CHUNKS" if not failures else "PARTIAL_FAILED_CLOSED",
        "run_freeze": _binding(freeze_path),
        "accepted_this_invocation": accepted,
        "skipped_existing_valid": skipped,
        "failed_chunks": failures,
        "elapsed_seconds": time.time() - started,
        "notice_and_comment_launched": False,
        "core_or_scale_launched": False,
        "release_ready": False,
    }
    summary_path = output_dir / "summaries" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".json"
    )
    _atomic_json_x(summary_path, summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    if failures:
        raise SystemExit(1)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root", default=str(Path.cwd()))
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--external-codex-cap", type=int, default=12)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--chunk-attempts", type=int, default=3)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
