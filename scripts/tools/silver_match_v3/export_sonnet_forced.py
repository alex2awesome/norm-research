#!/usr/bin/env python3
"""Recover the June 2026 forced-top-3 Sonnet matching workflow safely.

This workflow is useful distillation data, but it is *not* adjudicated silver:
the prompt required three choices for every norm, exposed no abstention option,
and used an older, broader aspect catalog.  Consequently this exporter emits
only weak, train-only positive pairs.  It never turns the labels into an
evaluation set or calibration target.

The bridge is intentionally strict:

* every workflow batch must be present exactly once and cover exactly its 100
  expected integer IDs (or the final partial batch);
* the journal result must agree with the agent transcript's structured output;
* the transcript's actual batch-file read must agree with a deterministic
  reconstruction of the sampled anchor input;
* an old aspect is retained only when its normalized name has one unambiguous
  exact match in the frozen current task bank;
* an anchor is joined to the production universe by its original anchor-file
  physical row, exact normalized source ID, and exact normalized norm text.

All obsolete/ambiguous aspect choices are written to an append-only rejection
ledger.  Structural corruption raises ``IntegrityError`` instead of silently
producing a partial teacher set.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import SCHEMA_VERSION
from .common import normalize_name, normalize_space, read_jsonl, sha256_file, write_jsonl
from .config import DEFAULT_OUTPUT_ROOT


CLAUDE_PROJECT = (
    Path.home()
    / ".claude/projects/-Users-spangher-Projects-stanford-research-norm-research"
)
SOURCE_SESSION_ID = "18d23164-0d0a-4d95-838c-0189393e21c5"
SOURCE_RUN_ID = "wf_33b7e713-ffa"
DEFAULT_WORKFLOW_ROOT = (
    CLAUDE_PROJECT / SOURCE_SESSION_ID / "subagents/workflows" / SOURCE_RUN_ID
)
DEFAULT_JOURNAL = DEFAULT_WORKFLOW_ROOT / "journal.jsonl"

SK3_HOME = Path("/lfs/skampere3/0/alexspan")
DEFAULT_ASPECT_FILES = {
    "humor": SK3_HOME / "norm-research/runs/validity_full/v2/humor/aspects.json",
    "press_releases": SK3_HOME
    / "norm-research/runs/validity_full/v2/press_releases/aspects.json",
    "math": SK3_HOME / "norm-research/runs/validity_full/v2/math/aspects.json",
    "code_review": SK3_HOME
    / "norm-research/runs/validity_full/v2/code_review/aspects.json",
}
DEFAULT_ANCHOR_FILES = {
    "humor": SK3_HOME / "data/humor/standup_multi/gepa/anchors_round3.jsonl",
    "press_releases": SK3_HOME / "data/press_releases/gepa/anchors_round2.jsonl",
    "math": SK3_HOME / "data/math_se/gepa/anchors_round3.jsonl",
    "code_review": SK3_HOME / "data/crse/gepa/anchors_round1.jsonl",
}

# Order is part of the historical sampling algorithm: a single Random(42)
# instance was shared by the four iterations.
TASK_ORDER = ("humor", "press_releases", "math", "code_review")
TASK_CONFIG = {
    "humor": {"corpus": "humor_multi", "task": "humor", "source_key": "thread_id"},
    "press_releases": {
        "corpus": "press_releases",
        "task": "press-releases",
        "source_key": "pair_id",
    },
    "math": {"corpus": "math_se", "task": "math-stackexchange", "source_key": "unit_id"},
    "code_review": {"corpus": "crse", "task": "code-review", "source_key": "unit_id"},
}
SAMPLE_SIZE = 20_000
BATCH_SIZE = 100
PROMPT_BATCH_RE = re.compile(
    r"/match_(humor|press_releases|math|code_review)/batches/batch_([0-9]{3})\.txt"
)
READ_LINE_RE = re.compile(r"^[0-9]+\t")


class IntegrityError(ValueError):
    """Raised when the historical workflow cannot be reconstructed exactly."""


@dataclass(frozen=True)
class SampledAnchor:
    task_key: str
    sample_id: int
    anchor_row: int
    source_id: str
    norm: str
    prompt_text: str
    passage: str
    reason: str


@dataclass(frozen=True)
class Transcript:
    agent_id: str
    task_key: str
    batch_index: int
    models: tuple[str, ...]
    batch_read: str
    structured_outputs: tuple[tuple[dict[str, Any], ...], ...]
    path: Path


@dataclass(frozen=True)
class CurrentMetric:
    metric_id: str
    name: str
    ambiguous: bool


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _path_map(values: Sequence[str] | None, defaults: Mapping[str, Path]) -> dict[str, Path]:
    result = {key: Path(value) for key, value in defaults.items()}
    for value in values or ():
        if "=" not in value:
            raise ValueError(f"expected TASK=PATH, got {value!r}")
        key, raw_path = value.split("=", 1)
        if key not in TASK_CONFIG:
            raise ValueError(f"unknown task key {key!r}")
        result[key] = Path(raw_path)
    return result


def _message_blocks(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    message = record.get("message")
    if not isinstance(message, Mapping):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, Mapping)]


def _message_text(record: Mapping[str, Any]) -> str:
    message = record.get("message")
    if not isinstance(message, Mapping):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text") or "")
            for block in content
            if isinstance(block, Mapping) and block.get("type") == "text"
        )
    return ""


def _strip_read_line_numbers(content: str) -> str:
    lines = [READ_LINE_RE.sub("", line) for line in content.splitlines()]
    # Claude Read adds one numbered empty sentinel after the final source line.
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + ("\n" if lines else "")


def parse_transcript(path: Path) -> Transcript:
    records = list(read_jsonl(path))
    if not records:
        raise IntegrityError(f"empty transcript: {path}")
    agent_ids = {normalize_space(row.get("agentId")) for row in records if row.get("agentId")}
    if len(agent_ids) != 1:
        raise IntegrityError(f"ambiguous agent ID in {path}: {sorted(agent_ids)}")
    agent_id = next(iter(agent_ids))

    prompts = [
        match
        for row in records
        if (match := PROMPT_BATCH_RE.search(_message_text(row))) is not None
    ]
    prompt_pairs = {(match.group(1), int(match.group(2))) for match in prompts}
    if len(prompt_pairs) != 1:
        raise IntegrityError(f"ambiguous task/batch prompt in {path}: {prompt_pairs}")
    task_key, batch_index = next(iter(prompt_pairs))

    models = sorted(
        {
            str(row["message"]["model"])
            for row in records
            if isinstance(row.get("message"), Mapping)
            and row["message"].get("role") == "assistant"
            and row["message"].get("model")
        }
    )
    if not models or any("sonnet" not in model.lower() for model in models):
        raise IntegrityError(f"non-Sonnet or missing model provenance in {path}: {models}")

    pending_reads: dict[str, str] = {}
    batch_reads: list[str] = []
    structured: list[tuple[dict[str, Any], ...]] = []
    for row in records:
        for block in _message_blocks(row):
            if block.get("type") == "tool_use" and block.get("name") == "Read":
                tool_id = normalize_space(block.get("id"))
                payload = block.get("input") or {}
                if tool_id and isinstance(payload, Mapping):
                    pending_reads[tool_id] = normalize_space(payload.get("file_path"))
            elif block.get("type") == "tool_use" and block.get("name") == "StructuredOutput":
                payload = block.get("input") or {}
                matches = payload.get("matches") if isinstance(payload, Mapping) else None
                if isinstance(matches, list) and all(isinstance(item, dict) for item in matches):
                    structured.append(tuple(matches))
            elif block.get("type") == "tool_result":
                tool_id = normalize_space(block.get("tool_use_id"))
                read_path = pending_reads.get(tool_id, "")
                if PROMPT_BATCH_RE.search(read_path):
                    content = block.get("content")
                    if isinstance(content, str):
                        batch_reads.append(_strip_read_line_numbers(content))
    if len(set(batch_reads)) != 1:
        raise IntegrityError(f"missing or conflicting batch Read result in {path}")
    if not structured:
        raise IntegrityError(f"missing StructuredOutput in {path}")
    return Transcript(
        agent_id=agent_id,
        task_key=task_key,
        batch_index=batch_index,
        models=tuple(models),
        batch_read=batch_reads[0],
        structured_outputs=tuple(structured),
        path=path,
    )


def load_transcripts(workflow_root: Path) -> dict[str, Transcript]:
    transcripts: dict[str, Transcript] = {}
    for path in sorted(workflow_root.glob("agent-*.jsonl")):
        transcript = parse_transcript(path)
        if transcript.agent_id in transcripts:
            raise IntegrityError(f"duplicate transcript for {transcript.agent_id}")
        transcripts[transcript.agent_id] = transcript
    if not transcripts:
        raise IntegrityError(f"no agent transcripts under {workflow_root}")
    return transcripts


def load_journal(journal_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    results: dict[str, dict[str, Any]] = {}
    started: dict[str, str] = {}
    for row in read_jsonl(journal_path):
        event_type = row.get("type")
        key = normalize_space(row.get("key"))
        agent_id = normalize_space(row.get("agentId"))
        if event_type == "started":
            if key in started and started[key] != agent_id:
                raise IntegrityError(f"workflow key started by multiple agents: {key}")
            started[key] = agent_id
        elif event_type == "result":
            payload = row.get("result") or {}
            matches = payload.get("matches") if isinstance(payload, Mapping) else None
            if not key or not agent_id or not isinstance(matches, list):
                raise IntegrityError(f"malformed result event for key {key!r}")
            if key in results:
                raise IntegrityError(f"duplicate result event for key {key}")
            results[key] = {"agent_id": agent_id, "matches": matches}
    if set(started) != set(results):
        missing = sorted(set(started) - set(results))
        extra = sorted(set(results) - set(started))
        raise IntegrityError(f"started/result mismatch; missing={missing[:5]}, extra={extra[:5]}")
    for key, event in results.items():
        if started[key] != event["agent_id"]:
            raise IntegrityError(f"agent mismatch for workflow key {key}")
    return results, started


def load_sampled_anchors(anchor_files: Mapping[str, Path]) -> dict[str, list[SampledAnchor]]:
    rng = random.Random(42)
    sampled: dict[str, list[SampledAnchor]] = {}
    for task_key in TASK_ORDER:
        path = anchor_files[task_key]
        raw = list(read_jsonl(path))
        indices = rng.sample(range(len(raw)), SAMPLE_SIZE) if len(raw) > SAMPLE_SIZE else list(range(len(raw)))
        config = TASK_CONFIG[task_key]
        values = []
        for sample_id, anchor_row in enumerate(indices):
            row = raw[anchor_row]
            if int(row.get("faithful") or 0) != 1 or int(row.get("valid") or 0) != 1:
                raise IntegrityError(f"non-approved anchor at {path}:{anchor_row + 1}")
            norm = str(row.get("signal_text") or "")
            source_id = normalize_space(row.get(config["source_key"]))
            if not normalize_space(norm) or not source_id:
                raise IntegrityError(f"anchor lacks norm/source identity at {path}:{anchor_row + 1}")
            values.append(
                SampledAnchor(
                    task_key=task_key,
                    sample_id=sample_id,
                    anchor_row=anchor_row,
                    source_id=source_id,
                    norm=normalize_space(norm),
                    prompt_text=norm[:250],
                    passage=normalize_space(row.get("passage_text")),
                    reason=normalize_space(row.get("reason")),
                )
            )
        sampled[task_key] = values
    return sampled


def load_old_aspects(aspect_files: Mapping[str, Path]) -> dict[str, dict[str, dict[str, Any]]]:
    result = {}
    for task_key, path in aspect_files.items():
        payload = _load_json(path)
        if not isinstance(payload, list) or not payload:
            raise IntegrityError(f"invalid aspect catalog: {path}")
        by_id = {}
        for row in payload:
            if not isinstance(row, dict):
                raise IntegrityError(f"non-object aspect in {path}")
            aspect_id = normalize_space(row.get("aspect_id"))
            name = normalize_space(row.get("name"))
            if not aspect_id or not name or aspect_id in by_id:
                raise IntegrityError(f"invalid/duplicate aspect {aspect_id!r} in {path}")
            by_id[aspect_id] = row
        result[task_key] = by_id
    return result


def load_current_banks(manifest_root: Path) -> tuple[
    dict[str, dict[str, tuple[CurrentMetric, ...]]], dict[str, str]
]:
    by_task: dict[str, dict[str, tuple[CurrentMetric, ...]]] = {}
    hashes = {}
    for config in TASK_CONFIG.values():
        task = str(config["task"])
        if task in by_task:
            continue
        path = manifest_root / "banks" / f"{task}.json"
        payload = _load_json(path)
        grouped: dict[str, list[CurrentMetric]] = defaultdict(list)
        for row in payload.get("metrics") or []:
            name = normalize_space(row.get("name"))
            key = normalize_name(row.get("name_key") or name)
            grouped[key].append(
                CurrentMetric(
                    metric_id=str(row["metric_id"]),
                    name=name,
                    ambiguous=bool(row.get("name_ambiguous")),
                )
            )
        by_task[task] = {
            key: tuple(
                CurrentMetric(metric.metric_id, metric.name, metric.ambiguous or len(values) > 1)
                for metric in values
            )
            for key, values in grouped.items()
        }
        hashes[task] = str(payload.get("source_sha256") or "")
        if not hashes[task]:
            raise IntegrityError(f"frozen bank lacks source hash: {path}")
    return by_task, hashes


def load_production_norms(manifest_root: Path) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for config in TASK_CONFIG.values():
        corpus = str(config["corpus"])
        path = manifest_root / "norms" / f"{corpus}.jsonl"
        rows = list(read_jsonl(path))
        for physical_row, row in enumerate(rows):
            if int(row.get("row", -1)) != physical_row:
                raise IntegrityError(f"non-contiguous canonical rows in {path}:{physical_row + 1}")
            if row.get("corpus") != corpus or row.get("task") != config["task"]:
                raise IntegrityError(f"canonical routing mismatch in {path}:{physical_row + 1}")
        result[corpus] = rows
    return result


def _expected_batch_text(anchors: Sequence[SampledAnchor]) -> str:
    return "".join(f"{anchor.sample_id}: {anchor.prompt_text}\n" for anchor in anchors)


def validate_workflow(
    journal_path: Path,
    workflow_root: Path,
    sampled: Mapping[str, Sequence[SampledAnchor]],
    aspects: Mapping[str, Mapping[str, dict[str, Any]]],
) -> tuple[list[tuple[Transcript, str, list[dict[str, Any]]]], list[dict[str, Any]]]:
    transcripts = load_transcripts(workflow_root)
    results, _ = load_journal(journal_path)
    seen_batches: dict[tuple[str, int], str] = {}
    validated = []
    rejected: list[dict[str, Any]] = []
    for workflow_key, event in results.items():
        agent_id = event["agent_id"]
        transcript = transcripts.get(agent_id)
        if transcript is None:
            raise IntegrityError(f"journal agent lacks transcript: {agent_id}")
        batch_key = (transcript.task_key, transcript.batch_index)
        if batch_key in seen_batches:
            raise IntegrityError(f"duplicate workflow batch {batch_key}")
        seen_batches[batch_key] = workflow_key
        task_anchors = sampled[transcript.task_key]
        lo = transcript.batch_index * BATCH_SIZE
        hi = min(lo + BATCH_SIZE, len(task_anchors))
        batch = list(task_anchors[lo:hi])
        if not batch:
            raise IntegrityError(f"out-of-range workflow batch {batch_key}")
        if normalize_space(transcript.batch_read) != normalize_space(_expected_batch_text(batch)):
            raise IntegrityError(f"transcript batch content differs from reconstructed input: {batch_key}")

        journal_matches = event["matches"]
        journal_signature = json.dumps(journal_matches, sort_keys=True)
        transcript_signatures = {
            json.dumps(list(output), sort_keys=True)
            for output in transcript.structured_outputs
        }
        if journal_signature not in transcript_signatures:
            raise IntegrityError(f"journal/transcript output mismatch: {batch_key}")
        expected_ids = {anchor.sample_id for anchor in batch}
        actual_ids: list[int] = []
        structurally_valid: list[dict[str, Any]] = []
        for row in journal_matches:
            if not isinstance(row, dict) or not isinstance(row.get("id"), int):
                raise IntegrityError(f"malformed match in {batch_key}")
            actual_ids.append(row["id"])
            choices = row.get("aspects")
            if not isinstance(choices, list) or len(choices) != 3 or len(set(choices)) != 3:
                rejected.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "reason": "invalid_forced_choice_cardinality",
                        "label_source": "sonnet_forced_top3",
                        "source_run_id": SOURCE_RUN_ID,
                        "task_key": transcript.task_key,
                        "task": TASK_CONFIG[transcript.task_key]["task"],
                        "corpus": TASK_CONFIG[transcript.task_key]["corpus"],
                        "sample_id": row["id"],
                        "workflow_key": workflow_key,
                        "details": {"aspects": choices},
                    }
                )
                continue
            # Unknown aspect IDs are a label-level hallucination, not evidence
            # that the input/output batch alignment itself is corrupt.  They
            # are rejected explicitly by ``export`` while the two remaining
            # choices can still contribute weak positives.
            if row["id"] not in expected_ids:
                rejected.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "reason": "cross_batch_output_id_leakage",
                        "label_source": "sonnet_forced_top3",
                        "source_run_id": SOURCE_RUN_ID,
                        "task_key": transcript.task_key,
                        "task": TASK_CONFIG[transcript.task_key]["task"],
                        "corpus": TASK_CONFIG[transcript.task_key]["corpus"],
                        "sample_id": row["id"],
                        "workflow_key": workflow_key,
                        "details": {"expected_min": lo, "expected_max": hi - 1},
                    }
                )
                continue
            structurally_valid.append(row)
        counts = Counter(actual_ids)
        duplicate_ids = {sample_id for sample_id, count in counts.items() if count > 1}
        if duplicate_ids:
            structurally_valid = [row for row in structurally_valid if row["id"] not in duplicate_ids]
            for sample_id in sorted(duplicate_ids):
                rejected.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "reason": "duplicate_workflow_output_id",
                        "label_source": "sonnet_forced_top3",
                        "source_run_id": SOURCE_RUN_ID,
                        "task_key": transcript.task_key,
                        "task": TASK_CONFIG[transcript.task_key]["task"],
                        "corpus": TASK_CONFIG[transcript.task_key]["corpus"],
                        "sample_id": sample_id,
                        "workflow_key": workflow_key,
                        "details": {"count": counts[sample_id]},
                    }
                )
        missing_ids = expected_ids - set(actual_ids)
        for sample_id in sorted(missing_ids):
            rejected.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "reason": "missing_workflow_output_id",
                    "label_source": "sonnet_forced_top3",
                    "source_run_id": SOURCE_RUN_ID,
                    "task_key": transcript.task_key,
                    "task": TASK_CONFIG[transcript.task_key]["task"],
                    "corpus": TASK_CONFIG[transcript.task_key]["corpus"],
                    "sample_id": sample_id,
                    "workflow_key": workflow_key,
                }
            )
        validated.append((transcript, workflow_key, structurally_valid))

    expected_batches = {
        (task_key, batch_index)
        for task_key, anchors_for_task in sampled.items()
        for batch_index in range(math.ceil(len(anchors_for_task) / BATCH_SIZE))
    }
    if set(seen_batches) != expected_batches:
        missing = sorted(expected_batches - set(seen_batches))
        extra = sorted(set(seen_batches) - expected_batches)
        raise IntegrityError(f"incomplete batch universe; missing={missing[:5]}, extra={extra[:5]}")
    validated.sort(key=lambda value: (TASK_ORDER.index(value[0].task_key), value[0].batch_index))
    return validated, rejected


def _rejection(
    reason: str,
    *,
    anchor: SampledAnchor,
    aspect_id: str | None,
    aspect_name: str | None,
    rank: int | None,
    details: Any = None,
) -> dict[str, Any]:
    config = TASK_CONFIG[anchor.task_key]
    return {
        "schema_version": SCHEMA_VERSION,
        "reason": reason,
        "label_source": "sonnet_forced_top3",
        "source_run_id": SOURCE_RUN_ID,
        "task_key": anchor.task_key,
        "task": config["task"],
        "corpus": config["corpus"],
        "sample_id": anchor.sample_id,
        "anchor_source_row": anchor.anchor_row,
        "source_id": anchor.source_id,
        "legacy_aspect_id": aspect_id,
        "legacy_aspect_name": aspect_name,
        "forced_rank": rank,
        "details": details,
    }


def export(args: argparse.Namespace) -> dict[str, Any]:
    manifest_root = Path(args.manifest_root)
    output_root = Path(args.output_root)
    workflow_root = Path(args.workflow_root)
    journal_path = Path(args.journal)
    aspect_files = _path_map(getattr(args, "aspect_file", None), DEFAULT_ASPECT_FILES)
    anchor_files = _path_map(getattr(args, "anchor_file", None), DEFAULT_ANCHOR_FILES)

    aspects = load_old_aspects(aspect_files)
    sampled = load_sampled_anchors(anchor_files)
    banks, bank_hashes = load_current_banks(manifest_root)
    canonical = load_production_norms(manifest_root)
    batches, workflow_rejections = validate_workflow(
        journal_path, workflow_root, sampled, aspects
    )

    teachers: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = list(workflow_rejections)
    seen_norms: set[str] = set()
    for transcript, workflow_key, matches in batches:
        config = TASK_CONFIG[transcript.task_key]
        corpus = str(config["corpus"])
        task = str(config["task"])
        for match in matches:
            sample_id = int(match["id"])
            anchor = sampled[transcript.task_key][sample_id]
            if anchor.anchor_row >= len(canonical[corpus]):
                rejections.append(
                    _rejection(
                        "canonical_anchor_row_missing",
                        anchor=anchor,
                        aspect_id=None,
                        aspect_name=None,
                        rank=None,
                    )
                )
                continue
            norm = canonical[corpus][anchor.anchor_row]
            if (
                normalize_space(norm.get("source_id")) != anchor.source_id
                or normalize_space(norm.get("norm")) != anchor.norm
            ):
                rejections.append(
                    _rejection(
                        "canonical_anchor_identity_mismatch",
                        anchor=anchor,
                        aspect_id=None,
                        aspect_name=None,
                        rank=None,
                        details={
                            "canonical_source_id": norm.get("source_id"),
                            "canonical_norm": norm.get("norm"),
                        },
                    )
                )
                continue
            norm_uid = str(norm["norm_uid"])
            if norm_uid in seen_norms:
                raise IntegrityError(f"sampled anchors resolve to duplicate production UID: {norm_uid}")
            seen_norms.add(norm_uid)

            for rank, aspect_id in enumerate(match["aspects"], 1):
                if aspect_id not in aspects[transcript.task_key]:
                    rejections.append(
                        _rejection(
                            "unknown_legacy_aspect_id",
                            anchor=anchor,
                            aspect_id=aspect_id,
                            aspect_name=None,
                            rank=rank,
                        )
                    )
                    continue
                old = aspects[transcript.task_key][aspect_id]
                old_name = normalize_space(old.get("name"))
                candidates = banks[task].get(normalize_name(old_name), ())
                if not candidates:
                    rejections.append(
                        _rejection(
                            "obsolete_aspect_name_not_in_current_bank",
                            anchor=anchor,
                            aspect_id=aspect_id,
                            aspect_name=old_name,
                            rank=rank,
                        )
                    )
                    continue
                if len(candidates) != 1 or candidates[0].ambiguous:
                    rejections.append(
                        _rejection(
                            "ambiguous_current_bank_name",
                            anchor=anchor,
                            aspect_id=aspect_id,
                            aspect_name=old_name,
                            rank=rank,
                            details={"metric_ids": [value.metric_id for value in candidates]},
                        )
                    )
                    continue
                metric = candidates[0]
                teachers.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "norm_uid": norm_uid,
                        "corpus": corpus,
                        "task": task,
                        "row": int(norm["row"]),
                        "source_id": anchor.source_id,
                        "decision": "MATCH",
                        "metric_id": metric.metric_id,
                        "current_metric_name": metric.name,
                        "current_bank_source_sha256": bank_hashes[task],
                        "label_source": "sonnet_forced_top3",
                        "source_model": list(transcript.models),
                        "source_session_id": SOURCE_SESSION_ID,
                        "source_run_id": SOURCE_RUN_ID,
                        "workflow_key": workflow_key,
                        "agent_id": transcript.agent_id,
                        "sample_id": sample_id,
                        "anchor_source_row": anchor.anchor_row,
                        "legacy_aspect_id": aspect_id,
                        "legacy_aspect_name": old_name,
                        "forced_rank": rank,
                        "rank_semantics": "top3_order_not_calibrated",
                        "bridge_method": "anchor_row_source_norm_and_unique_current_name",
                        "supervision_strength": "weak_forced_positive",
                        "data_role": "train_only",
                        "eligible_for_training": True,
                        "eligible_for_evaluation": False,
                        "eligible_for_threshold_calibration": False,
                        "abstention_was_available": False,
                    }
                )

    # Duplicate metric IDs within a norm are impossible after the workflow's
    # distinct aspect-ID gate unless two legacy names collapse onto one current
    # name.  Reject those collapsed duplicates rather than silently overweight.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in teachers:
        grouped[(row["norm_uid"], row["metric_id"])].append(row)
    clean = []
    for (norm_uid, metric_id), rows in grouped.items():
        if len(rows) == 1:
            clean.append(rows[0])
            continue
        for row in rows:
            rejections.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "reason": "legacy_aspects_collapse_to_same_current_metric",
                    "label_source": "sonnet_forced_top3",
                    "source_run_id": SOURCE_RUN_ID,
                    "norm_uid": norm_uid,
                    "metric_id": metric_id,
                    "task": row["task"],
                    "corpus": row["corpus"],
                    "sample_id": row["sample_id"],
                    "legacy_aspect_id": row["legacy_aspect_id"],
                    "legacy_aspect_name": row["legacy_aspect_name"],
                    "forced_rank": row["forced_rank"],
                }
            )

    task_order_index = {
        str(TASK_CONFIG[key]["task"]): index for index, key in enumerate(TASK_ORDER)
    }
    clean.sort(
        key=lambda row: (
            task_order_index[row["task"]],
            row["row"],
            row["forced_rank"],
        )
    )
    rejections.sort(
        key=lambda row: (
            str(row.get("task") or ""),
            int(row.get("sample_id") or -1),
            int(row.get("forced_rank") or -1),
            str(row.get("reason") or ""),
        )
    )
    teacher_path = output_root / "teachers/sonnet_forced_top3.production.jsonl"
    rejection_path = output_root / "teachers/sonnet_forced_top3.rejections.jsonl"
    write_jsonl(teacher_path, clean)
    write_jsonl(rejection_path, rejections)

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "source_session_id": SOURCE_SESSION_ID,
        "source_run_id": SOURCE_RUN_ID,
        "workflow_journal": str(journal_path),
        "workflow_journal_sha256": sha256_file(journal_path),
        "aspect_files": {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in aspect_files.items()},
        "anchor_files": {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in anchor_files.items()},
        "teacher_path": str(teacher_path),
        "rejection_path": str(rejection_path),
        "policy": {
            "data_role": "train_only",
            "eligible_for_evaluation": False,
            "eligible_for_threshold_calibration": False,
            "reason": "historical prompt forced exactly three choices and offered no abstention",
        },
        "counts": {
            "sampled_norms": sum(map(len, sampled.values())),
            "batches": len(batches),
            "teachers": len(clean),
            "rejections": len(rejections),
            "teachers_by_task": dict(sorted(Counter(row["task"] for row in clean).items())),
            "rejections_by_reason": dict(sorted(Counter(row["reason"] for row in rejections).items())),
        },
    }
    provenance_path = output_root / "teachers/sonnet_forced_top3.provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = provenance_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(provenance_path)
    return {**provenance["counts"], "teacher_path": str(teacher_path), "rejection_path": str(rejection_path), "provenance_path": str(provenance_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--workflow-root", default=str(DEFAULT_WORKFLOW_ROOT))
    parser.add_argument("--journal", default=str(DEFAULT_JOURNAL))
    parser.add_argument("--aspect-file", action="append", help="Override as TASK=PATH")
    parser.add_argument("--anchor-file", action="append", help="Override as TASK=PATH")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(export(parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
