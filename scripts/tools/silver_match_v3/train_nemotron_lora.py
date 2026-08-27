#!/usr/bin/env python3
"""Train one audited llama-embed-nemotron-8b LoRA retriever per task.

This trainer deliberately does *not* use in-batch negatives: several bank
metrics are near-equivalent and would become silent false negatives.  Instead,
it mines explicit query and metric-sibling hard negatives from the frozen
current bank, excludes known acceptable/name-equivalent metrics, and optimizes
a cosine triplet objective.

The output is an adapter, not a second 8B model.  The script asserts that every
trainable parameter is a LoRA parameter and calls PEFT's adapter-only saver.
It also writes source-group split assignments, exact training triplets, input
hashes, and exhaustive whole-bank retrieval results.  Invoke it once per task;
that separation is intentional so no adapter shares gradients across tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import socket
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .common import (
    metric_card,
    norm_query,
    normalize_name,
    normalize_space,
    read_jsonl,
    sha256_file,
    write_jsonl,
)
from .config import DEFAULT_OUTPUT_ROOT


# Latest complete immutable SentenceTransformers snapshot currently cached on
# sk3.  It can be overridden, but recording it (and its config hashes) is part
# of every run report.
DEFAULT_NEMOTRON = (
    "/lfs/skampere3/0/shared_hf_cache/"
    "models--nvidia--llama-embed-nemotron-8b/snapshots/"
    "aa3b43a495a9b280d1bdb716da37c54bb495d630"
)
DEFAULT_MODEL_ROOT = Path(
    "/lfs/skampere3/0/alexspan/models/silver_match_v3_nemotron_lora"
)
RETRIEVAL_INSTRUCTION = (
    "Given a human evaluative statement, retrieve the rubric metric that best "
    "captures the criterion the human is invoking."
)
DEFAULT_KS = (1, 3, 5, 10, 16, 30, 50)


def epoch_selection_key(
    exact_metrics: Mapping[str, Any], selection_k: int, policy: str
) -> tuple[float, ...]:
    """Return a deterministic epoch key without trading away primary-depth recall."""
    primary = float(exact_metrics[f"recall_at_{selection_k}"])
    if policy == "single_k":
        return (primary,)
    if policy != "depth_lexicographic":
        raise ValueError(f"unknown epoch selection policy: {policy}")
    lower_depths = [depth for depth in reversed(DEFAULT_KS) if depth < selection_k]
    return (
        primary,
        float(exact_metrics["mrr"]),
        *(float(exact_metrics[f"recall_at_{depth}"]) for depth in lower_depths),
    )


def epoch_selection_key_names(selection_k: int, policy: str) -> tuple[str, ...]:
    if policy == "single_k":
        return (f"recall_at_{selection_k}",)
    if policy != "depth_lexicographic":
        raise ValueError(f"unknown epoch selection policy: {policy}")
    lower_depths = [depth for depth in reversed(DEFAULT_KS) if depth < selection_k]
    return (
        f"recall_at_{selection_k}",
        "mrr",
        *(f"recall_at_{depth}" for depth in lower_depths),
    )


def epoch_promotion_passes(
    before_key: Sequence[float],
    after_key: Sequence[float],
    *,
    policy: str,
    minimum_primary_gain: float,
) -> bool:
    if len(before_key) != len(after_key) or not before_key:
        raise ValueError("epoch selection keys must be nonempty and aligned")
    primary_gain = after_key[0] - before_key[0]
    if policy == "single_k":
        return primary_gain >= minimum_primary_gain
    if policy != "depth_lexicographic":
        raise ValueError(f"unknown epoch selection policy: {policy}")
    if primary_gain < 0:
        return False
    if primary_gain > 0:
        return primary_gain >= minimum_primary_gain
    return tuple(after_key[1:]) > tuple(before_key[1:])


@dataclass(frozen=True)
class LabeledNorm:
    norm_uid: str
    corpus: str
    task: str
    source_group: str
    split: str
    query: str
    metric_id: str
    acceptable_metric_ids: tuple[str, ...]
    teacher_sources: tuple[str, ...]
    teacher_norm_uids: tuple[str, ...] = ()
    supervision_strength: str = "strong"


@dataclass(frozen=True)
class TrainingUniverse:
    task: str
    bank: tuple[dict[str, Any], ...]
    labels: tuple[LabeledNorm, ...]
    manifest: dict[str, Any]
    manifest_path: Path
    bank_path: Path
    bank_source_sha256: str
    teacher_paths: tuple[Path, ...]
    teacher_audit: dict[str, Any]
    split_audit: dict[str, Any]


def hash_bucket(value: str, modulus: int = 100) -> int:
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16) % modulus


def source_group_key(norm: Mapping[str, Any]) -> str:
    """Return the strongest available document-level leakage boundary."""
    task = normalize_space(norm.get("task"))
    corpus = normalize_space(norm.get("corpus"))
    paper_id = normalize_space(norm.get("paper_id"))
    source_id = normalize_space(norm.get("source_id"))
    if paper_id:
        identity_type, identity = "paper", paper_id
    elif source_id:
        identity_type, identity = "source", source_id
    else:
        # Manifest rows should always have source_id.  This fallback remains
        # source-disjoint at the norm level and is surfaced in the audit.
        identity_type, identity = "norm", normalize_space(norm.get("norm_uid"))
    return "\x1f".join((task, corpus, identity_type, identity))


def split_source_group(
    source_group: str,
    seed: int = 73129,
    train_percent: int = 80,
    dev_percent: int = 10,
) -> str:
    if train_percent <= 0 or dev_percent <= 0 or train_percent + dev_percent >= 100:
        raise ValueError("split percentages must leave non-empty train/dev/test ranges")
    bucket = hash_bucket(f"{seed}\x1f{source_group}", 100)
    if bucket < train_percent:
        return "train"
    if bucket < train_percent + dev_percent:
        return "dev"
    return "test"


def format_query(query: str, instruction: str = RETRIEVAL_INSTRUCTION) -> str:
    """Apply Nemotron's documented query-only instruction template."""
    return f"Instruct: {normalize_space(instruction)}\nQuery: {normalize_space(query)}"


def _resolve_artifact(path: str | Path, manifest_path: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else manifest_path.parent / value


def _teacher_acceptables(row: Mapping[str, Any], metric_id: str) -> set[str]:
    values = {metric_id}
    for key in ("acceptable_metric_ids", "equivalent_metric_ids", "metric_ids"):
        raw = row.get(key)
        if isinstance(raw, str):
            values.add(normalize_space(raw))
        elif isinstance(raw, Sequence):
            values.update(normalize_space(value) for value in raw)
    values.discard("")
    return values


def merge_match_teachers(
    teacher_rows: Iterable[tuple[str, Mapping[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Merge agreeing labels and reject contradictory direct supervision."""
    decisions: Counter[str] = Counter()
    by_uid: dict[str, list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    malformed = 0
    for teacher_file, row in teacher_rows:
        decision = normalize_space(row.get("decision"))
        decisions[decision or "MISSING"] += 1
        uid = normalize_space(row.get("norm_uid"))
        if not uid:
            malformed += 1
            continue
        if (
            decision == "MATCH"
            and "gradient_eligible" in row
            and row.get("gradient_eligible") is not True
        ):
            raise ValueError(
                f"gradient-locked teacher passed to trainer: {uid} from {teacher_file}"
            )
        if decision == "MATCH":
            by_uid[uid].append((teacher_file, row))

    merged: dict[str, dict[str, Any]] = {}
    conflicts: dict[str, list[str]] = {}
    duplicate_agreements = 0
    weak_forced_groups = 0
    for uid, group in sorted(by_uid.items()):
        strong = [
            pair
            for pair in group
            if normalize_space(pair[1].get("supervision_strength"))
            != "weak_forced_positive"
            and normalize_space(pair[1].get("label_source")) != "sonnet_forced_top3"
        ]
        selected_group = strong or group
        metric_ids = {
            normalize_space(row.get("metric_id")) for _, row in selected_group
        }
        metric_ids.discard("")
        is_weak_forced = not strong
        if not is_weak_forced and len(metric_ids) != 1:
            conflicts[uid] = sorted(metric_ids)
            continue
        if is_weak_forced:
            ranked = sorted(
                selected_group,
                key=lambda pair: (
                    int(pair[1].get("forced_rank") or 10**9),
                    normalize_space(pair[1].get("metric_id")),
                ),
            )
            metric_id = normalize_space(ranked[0][1].get("metric_id"))
            if not metric_id or len(metric_ids) < 1:
                conflicts[uid] = sorted(metric_ids)
                continue
            weak_forced_groups += 1
        else:
            metric_id = next(iter(metric_ids))
        acceptable = {metric_id}
        sources = set()
        bank_hashes = set()
        task_values = set()
        split_values = set()
        for teacher_file, row in selected_group:
            acceptable.update(_teacher_acceptables(row, metric_id))
            if is_weak_forced:
                acceptable.add(normalize_space(row.get("metric_id")))
            sources.add(
                normalize_space(row.get("label_source")) or Path(teacher_file).name
            )
            bank_hash = normalize_space(row.get("current_bank_source_sha256"))
            if bank_hash:
                bank_hashes.add(bank_hash)
            task = normalize_space(row.get("task"))
            if task:
                task_values.add(task)
            split = normalize_space(row.get("split") or row.get("predeclared_split"))
            if split:
                split_values.add(split)
        if len(group) > 1:
            duplicate_agreements += len(group) - 1
        merged[uid] = {
            "norm_uid": uid,
            "metric_id": metric_id,
            "acceptable_metric_ids": sorted(acceptable),
            "teacher_sources": sorted(sources),
            "bank_hashes": sorted(bank_hashes),
            "teacher_tasks": sorted(task_values),
            "teacher_splits": sorted(split_values),
            "agreeing_rows": len(group),
            "supervision_strength": (
                "weak_forced_top3" if is_weak_forced else "strong"
            ),
        }
    audit = {
        "rows_by_decision": dict(sorted(decisions.items())),
        "malformed_missing_norm_uid": malformed,
        "match_unique_uids": len(merged),
        "duplicate_agreements": duplicate_agreements,
        "conflicting_match_uids": len(conflicts),
        "weak_forced_groups": weak_forced_groups,
        "conflicts": conflicts,
    }
    return merged, audit


def audit_source_splits(labels: Sequence[LabeledNorm]) -> dict[str, Any]:
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    rows_by_split: Counter[str] = Counter()
    corpus_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    texts_by_split: dict[str, set[str]] = defaultdict(set)
    metrics_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for label in labels:
        groups_by_split[label.split].add(label.source_group)
        rows_by_split[label.split] += 1
        corpus_by_split[label.split][label.corpus] += 1
        texts_by_split[label.split].add(normalize_name(label.query))
        metrics_by_split[label.split][label.metric_id] += 1
    overlap = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        shared = groups_by_split[left] & groups_by_split[right]
        if shared:
            overlap[f"{left}_{right}"] = sorted(shared)
    if overlap:
        raise ValueError(f"source-group split leakage: {overlap}")
    text_overlap = {
        f"{left}_{right}": len(texts_by_split[left] & texts_by_split[right])
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
    }
    return {
        "rows": {split: rows_by_split[split] for split in ("train", "dev", "test")},
        "source_groups": {
            split: len(groups_by_split[split]) for split in ("train", "dev", "test")
        },
        "corpora": {
            split: dict(sorted(corpus_by_split[split].items()))
            for split in ("train", "dev", "test")
        },
        "metric_coverage": {
            split: len(metrics_by_split[split]) for split in ("train", "dev", "test")
        },
        "rows_by_metric": {
            split: dict(sorted(metrics_by_split[split].items()))
            for split in ("train", "dev", "test")
        },
        "source_group_overlap": overlap,
        "normalized_query_overlap": text_overlap,
    }


def load_universe(
    manifest_path: Path,
    teacher_paths: Sequence[Path],
    task: str,
    *,
    split_seed: int,
    train_percent: int,
    dev_percent: int,
    require_bank_hash: bool,
    teacher_manifest_path: Path | None = None,
    respect_teacher_splits: bool = False,
) -> TrainingUniverse:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if task not in manifest.get("banks", {}):
        raise KeyError(f"task {task!r} is absent from {manifest_path}")
    bank_meta = manifest["banks"][task]
    bank_path = _resolve_artifact(bank_meta["path"], manifest_path)
    bank_payload = json.loads(bank_path.read_text(encoding="utf-8"))
    bank = tuple(bank_payload["metrics"])
    if not bank:
        raise ValueError(f"empty bank for {task}")
    bank_ids = [normalize_space(metric.get("metric_id")) for metric in bank]
    if len(bank_ids) != len(set(bank_ids)) or not all(bank_ids):
        raise ValueError(f"non-unique or empty metric IDs in {bank_path}")
    bank_source_sha = normalize_space(
        bank_meta.get("source_sha256") or bank_payload.get("source_sha256")
    )
    if not bank_source_sha:
        raise ValueError(f"missing hierarchy source hash for bank {task}")

    norms: dict[str, dict[str, Any]] = {}
    norms_by_corpus_text: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for corpus, meta in sorted(manifest.get("corpora", {}).items()):
        if meta.get("task") != task:
            continue
        norm_path = _resolve_artifact(meta["path"], manifest_path)
        for row in read_jsonl(norm_path):
            uid = normalize_space(row.get("norm_uid"))
            if not uid or uid in norms:
                raise ValueError(f"missing/duplicate norm_uid {uid!r} in {norm_path}")
            if row.get("task") != task or row.get("corpus") != corpus:
                raise ValueError(f"manifest routing mismatch for {uid}")
            norms[uid] = row
            norms_by_corpus_text[(corpus, normalize_space(row.get("norm")))].append(row)

    teacher_norms: dict[str, dict[str, Any]] = {}
    if teacher_manifest_path is not None:
        teacher_manifest = json.loads(teacher_manifest_path.read_text(encoding="utf-8"))
        for corpus, meta in sorted(teacher_manifest.get("corpora", {}).items()):
            if meta.get("task") != task:
                continue
            norm_path = _resolve_artifact(meta["path"], teacher_manifest_path)
            for row in read_jsonl(norm_path):
                uid = normalize_space(row.get("norm_uid"))
                if not uid or uid in teacher_norms:
                    raise ValueError(
                        f"missing/duplicate teacher norm_uid {uid!r} in {norm_path}"
                    )
                teacher_norms[uid] = row

    teacher_rows = (
        (str(path), row) for path in teacher_paths for row in read_jsonl(path)
    )
    merged, teacher_audit = merge_match_teachers(teacher_rows)
    if teacher_audit["conflicting_match_uids"]:
        sample = list(teacher_audit["conflicts"].items())[:5]
        raise ValueError(f"conflicting MATCH teachers (sample): {sample}")

    bank_id_set = set(bank_ids)
    bank_names = {
        metric["metric_id"]: normalize_name(metric["name"]) for metric in bank
    }
    selected_by_uid: dict[str, LabeledNorm] = {}
    rejection_counts: Counter[str] = Counter()
    bridge_counts: Counter[str] = Counter()
    unhashed = 0
    for uid, teacher in sorted(merged.items()):
        teacher_tasks = set(teacher["teacher_tasks"])
        if teacher_tasks and task not in teacher_tasks:
            rejection_counts["other_task_match_label"] += 1
            continue
        if len(teacher_tasks) > 1:
            raise ValueError(
                f"teacher has multiple tasks for {uid}: {sorted(teacher_tasks)}"
            )
        norm = norms.get(uid)
        if norm is None:
            old_norm = teacher_norms.get(uid)
            if old_norm is None:
                rejection_counts["norm_not_in_task_manifest"] += 1
                continue
            old_corpus = normalize_space(old_norm.get("corpus"))
            target_corpus = (
                normalize_space(manifest.get("aliases", {}).get(old_corpus))
                or old_corpus
            )
            candidates = norms_by_corpus_text.get(
                (target_corpus, normalize_space(old_norm.get("norm"))), []
            )
            if len(candidates) == 0:
                rejection_counts["exact_quote_bridge_missing"] += 1
                continue
            if len(candidates) > 1:
                rejection_counts["exact_quote_bridge_ambiguous"] += 1
                continue
            norm = candidates[0]
            bridge_counts["unique_exact_quote"] += 1
        else:
            bridge_counts["direct_uid"] += 1
        hashes = set(teacher["bank_hashes"])
        if not hashes:
            unhashed += 1
            if require_bank_hash:
                rejection_counts["missing_bank_hash"] += 1
                continue
        elif hashes != {bank_source_sha}:
            rejection_counts["stale_bank_hash"] += 1
            continue
        metric_id = teacher["metric_id"]
        if metric_id not in bank_id_set:
            rejection_counts["metric_not_in_current_bank"] += 1
            continue
        acceptable = set(teacher["acceptable_metric_ids"]) & bank_id_set
        # Same-name bank entries are indistinguishable at the name level and
        # must never be mined as negatives against one another.
        positive_name = bank_names[metric_id]
        acceptable.update(
            mid for mid, name in bank_names.items() if name == positive_name
        )
        group = source_group_key(norm)
        canonical_uid = normalize_space(norm["norm_uid"])
        if respect_teacher_splits:
            teacher_splits = set(teacher["teacher_splits"])
            if len(teacher_splits) != 1 or not teacher_splits <= {
                "train",
                "dev",
                "test",
            }:
                raise ValueError(
                    f"teacher has missing, conflicting, or invalid explicit split "
                    f"for {uid}: {sorted(teacher_splits)}"
                )
            split = next(iter(teacher_splits))
        else:
            split = split_source_group(group, split_seed, train_percent, dev_percent)
        proposed = LabeledNorm(
            norm_uid=canonical_uid,
            corpus=norm["corpus"],
            task=task,
            source_group=group,
            split=split,
            query=format_query(norm_query(norm)),
            metric_id=metric_id,
            acceptable_metric_ids=tuple(sorted(acceptable)),
            teacher_sources=tuple(teacher["teacher_sources"]),
            teacher_norm_uids=(uid,),
            supervision_strength=teacher["supervision_strength"],
        )
        existing = selected_by_uid.get(canonical_uid)
        if existing is None:
            selected_by_uid[canonical_uid] = proposed
        elif existing.metric_id != proposed.metric_id:
            raise ValueError(
                f"teachers bridged to conflicting metrics for {canonical_uid}: "
                f"{existing.metric_id} != {proposed.metric_id}"
            )
        elif existing.split != proposed.split:
            raise ValueError(
                f"teachers bridged to conflicting splits for {canonical_uid}: "
                f"{existing.split} != {proposed.split}"
            )
        else:
            selected_by_uid[canonical_uid] = LabeledNorm(
                norm_uid=canonical_uid,
                corpus=existing.corpus,
                task=task,
                source_group=existing.source_group,
                split=existing.split,
                query=existing.query,
                metric_id=existing.metric_id,
                acceptable_metric_ids=tuple(
                    sorted(
                        set(existing.acceptable_metric_ids)
                        | set(proposed.acceptable_metric_ids)
                    )
                ),
                teacher_sources=tuple(
                    sorted(
                        set(existing.teacher_sources) | set(proposed.teacher_sources)
                    )
                ),
                teacher_norm_uids=tuple(
                    sorted(
                        set(existing.teacher_norm_uids)
                        | set(proposed.teacher_norm_uids)
                    )
                ),
                supervision_strength=(
                    "strong"
                    if "strong"
                    in {existing.supervision_strength, proposed.supervision_strength}
                    else "weak_forced_top3"
                ),
            )
    selected = sorted(selected_by_uid.values(), key=lambda label: label.norm_uid)
    teacher_audit.update(
        {
            "task_manifest_norms": len(norms),
            "selected_match_labels": len(selected),
            "unhashed_match_labels": unhashed,
            "bridge": dict(sorted(bridge_counts.items())),
            "rejections": dict(sorted(rejection_counts.items())),
            "split_mode": (
                "explicit_teacher_role" if respect_teacher_splits else "source_hash"
            ),
        }
    )
    if not selected:
        raise ValueError(f"no usable MATCH teachers for task {task}")
    split_audit = audit_source_splits(selected)
    empty = [split for split, count in split_audit["rows"].items() if count == 0]
    if empty:
        raise ValueError(f"empty source-disjoint splits for {task}: {empty}")
    return TrainingUniverse(
        task=task,
        bank=bank,
        labels=tuple(selected),
        manifest=manifest,
        manifest_path=manifest_path,
        bank_path=bank_path,
        bank_source_sha256=bank_source_sha,
        teacher_paths=tuple(teacher_paths),
        teacher_audit=teacher_audit,
        split_audit=split_audit,
    )


def stable_rank(scores: Sequence[float]) -> list[int]:
    values = np.asarray(scores, dtype=np.float64)
    return np.lexsort((np.arange(len(values)), -values)).tolist()


def select_hard_negatives(
    query_scores: Sequence[float],
    sibling_scores: Sequence[float],
    bank_ids: Sequence[str],
    excluded_ids: set[str],
    *,
    pool_size: int,
    count: int,
) -> list[dict[str, Any]]:
    """Interleave query confusions and positive-metric siblings deterministically."""
    if count <= 0 or pool_size <= 0:
        raise ValueError("hard-negative count and pool_size must be positive")
    if len(query_scores) != len(bank_ids) or len(sibling_scores) != len(bank_ids):
        raise ValueError("score and bank dimensions differ")
    lanes = []
    for strategy, scores in (
        ("query_hard", query_scores),
        ("metric_sibling", sibling_scores),
    ):
        lane = [idx for idx in stable_rank(scores) if bank_ids[idx] not in excluded_ids]
        lanes.append((strategy, lane[:pool_size], scores))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    cursor = 0
    while len(selected) < count and any(cursor < len(lane) for _, lane, _ in lanes):
        for strategy, lane, scores in lanes:
            if cursor >= len(lane):
                continue
            idx = lane[cursor]
            metric_id = bank_ids[idx]
            if metric_id not in seen:
                selected.append(
                    {
                        "metric_id": metric_id,
                        "negative_strategy": strategy,
                        "lane_rank": cursor + 1,
                        "base_cosine": float(scores[idx]),
                    }
                )
                seen.add(metric_id)
                if len(selected) == count:
                    break
        cursor += 1
    if len(selected) < count:
        raise ValueError(
            f"bank has only {len(selected)} usable negatives; requested {count}"
        )
    return selected


def build_triplets(
    labels: Sequence[LabeledNorm],
    bank: Sequence[Mapping[str, Any]],
    query_embeddings: np.ndarray,
    bank_embeddings: np.ndarray,
    *,
    pool_size: int,
    negatives_per_positive: int,
) -> list[dict[str, Any]]:
    if query_embeddings.shape[0] != len(labels):
        raise ValueError("query embedding row count differs from labels")
    if bank_embeddings.shape[0] != len(bank):
        raise ValueError("bank embedding row count differs from bank")
    bank_ids = [str(metric["metric_id"]) for metric in bank]
    bank_index = {metric_id: idx for idx, metric_id in enumerate(bank_ids)}
    query_bank_scores = np.asarray(query_embeddings) @ np.asarray(bank_embeddings).T
    sibling_scores = np.asarray(bank_embeddings) @ np.asarray(bank_embeddings).T
    output: list[dict[str, Any]] = []
    for row_idx, label in enumerate(labels):
        if label.split != "train":
            continue
        positive_idx = bank_index[label.metric_id]
        negative_count = (
            1
            if label.supervision_strength == "weak_forced_top3"
            else negatives_per_positive
        )
        negatives = select_hard_negatives(
            query_bank_scores[row_idx],
            sibling_scores[positive_idx],
            bank_ids,
            set(label.acceptable_metric_ids),
            pool_size=pool_size,
            count=negative_count,
        )
        for negative in negatives:
            negative_idx = bank_index[negative["metric_id"]]
            output.append(
                {
                    "norm_uid": label.norm_uid,
                    "corpus": label.corpus,
                    "task": label.task,
                    "source_group": label.source_group,
                    "split": "train",
                    "query": label.query,
                    "positive_metric_id": label.metric_id,
                    "positive": metric_card(dict(bank[positive_idx])),
                    "negative_metric_id": negative["metric_id"],
                    "negative": metric_card(dict(bank[negative_idx])),
                    "acceptable_metric_ids": list(label.acceptable_metric_ids),
                    "teacher_sources": list(label.teacher_sources),
                    "teacher_norm_uids": list(label.teacher_norm_uids),
                    "supervision_strength": label.supervision_strength,
                    "negative_strategy": negative["negative_strategy"],
                    "lane_rank": negative["lane_rank"],
                    "base_cosine": negative["base_cosine"],
                }
            )
    return output


def retrieval_metrics(
    scores: np.ndarray,
    gold_ids: Sequence[str],
    bank_ids: Sequence[str],
    family_by_id: Mapping[str, str],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, Any]:
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape != (len(gold_ids), len(bank_ids)):
        raise ValueError("retrieval score matrix has the wrong shape")
    if not gold_ids:
        return {"n": 0}
    index = {metric_id: idx for idx, metric_id in enumerate(bank_ids)}
    exact_ranks = []
    family_ranks = []
    for row_idx, gold_id in enumerate(gold_ids):
        if gold_id not in index:
            raise KeyError(f"gold metric not in bank: {gold_id}")
        order = stable_rank(values[row_idx])
        exact_ranks.append(order.index(index[gold_id]) + 1)
        family = family_by_id[gold_id]
        family_ranks.append(
            next(
                rank
                for rank, idx in enumerate(order, 1)
                if family_by_id[bank_ids[idx]] == family
            )
        )

    def summarize(ranks: Sequence[int], groups: Sequence[str]) -> dict[str, Any]:
        array = np.asarray(ranks, dtype=np.float64)
        group_indices: dict[str, list[int]] = defaultdict(list)
        for idx, group in enumerate(groups):
            group_indices[group].append(idx)
        return {
            **{
                f"recall_at_{k}": float(np.mean(array <= min(k, len(bank_ids))))
                for k in ks
            },
            **{
                f"macro_recall_at_{k}": float(
                    np.mean(
                        [
                            np.mean(array[indices] <= min(k, len(bank_ids)))
                            for indices in group_indices.values()
                        ]
                    )
                )
                for k in ks
            },
            "mrr": float(np.mean(1.0 / array)),
            "mean_rank": float(np.mean(array)),
            "median_rank": float(np.median(array)),
            "groups": len(group_indices),
        }

    return {
        "n": len(gold_ids),
        "exact": summarize(exact_ranks, gold_ids),
        "name_family": summarize(
            family_ranks, [family_by_id[metric_id] for metric_id in gold_ids]
        ),
    }


def evaluate_embeddings(
    query_embeddings: np.ndarray,
    bank_embeddings: np.ndarray,
    labels: Sequence[LabeledNorm],
    bank: Sequence[Mapping[str, Any]],
    splits: Sequence[str],
) -> dict[str, Any]:
    bank_ids = [str(metric["metric_id"]) for metric in bank]
    family_by_id = {
        str(metric["metric_id"]): normalize_name(metric["name"]) for metric in bank
    }
    all_scores = np.asarray(query_embeddings) @ np.asarray(bank_embeddings).T
    output: dict[str, Any] = {}
    for split in splits:
        indices = [idx for idx, label in enumerate(labels) if label.split == split]
        if not indices:
            output[split] = {"all": {"n": 0}, "by_corpus": {}}
            continue
        split_scores = all_scores[indices]
        split_labels = [labels[idx] for idx in indices]
        by_corpus = {}
        for corpus in sorted({label.corpus for label in split_labels}):
            corpus_indices = [
                i for i, label in enumerate(split_labels) if label.corpus == corpus
            ]
            by_corpus[corpus] = retrieval_metrics(
                split_scores[corpus_indices],
                [split_labels[i].metric_id for i in corpus_indices],
                bank_ids,
                family_by_id,
            )
        output[split] = {
            "all": retrieval_metrics(
                split_scores,
                [label.metric_id for label in split_labels],
                bank_ids,
                family_by_id,
            ),
            "by_corpus": by_corpus,
        }
    return output


def _encode(model: Any, texts: Sequence[str], batch_size: int) -> np.ndarray:
    return np.asarray(
        model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )


def _library_versions() -> dict[str, str]:
    import peft
    import sentence_transformers
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "peft": peft.__version__,
    }


def _set_determinism(seed: int) -> None:
    import torch

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _model_source_hashes(model_path: Path) -> dict[str, str]:
    hashes = {}
    for name in (
        "config.json",
        "config_sentence_transformers.json",
        "modules.json",
        "sentence_bert_config.json",
    ):
        path = model_path / name
        if path.exists():
            hashes[name] = sha256_file(path.resolve())
    return hashes


def validate_adapter_artifact(adapter_dir: Path) -> dict[str, Any]:
    """Prove the saved artifact contains LoRA tensors, not copied base weights."""
    from safetensors import safe_open

    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise RuntimeError(f"missing PEFT config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA":
        raise RuntimeError(f"saved artifact is not LoRA: {config.get('peft_type')!r}")
    targets = set(config.get("target_modules") or [])
    expected_targets = {"q_proj", "k_proj", "v_proj", "o_proj"}
    if targets != expected_targets:
        raise RuntimeError(f"saved LoRA targets differ: {sorted(targets)}")
    full_weight_names = {
        "pytorch_model.bin",
        "model.safetensors",
        "model.safetensors.index.json",
    }
    copied_base = sorted(
        path.name for path in adapter_dir.iterdir() if path.name in full_weight_names
    )
    if copied_base:
        raise RuntimeError(
            f"base-model weights leaked into adapter output: {copied_base}"
        )
    weight_paths = sorted(adapter_dir.glob("adapter_model*.safetensors"))
    if len(weight_paths) != 1:
        raise RuntimeError(f"expected one adapter safetensor, found {weight_paths}")
    with safe_open(weight_paths[0], framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if not keys or any("lora_" not in key for key in keys):
            bad = [key for key in keys if "lora_" not in key]
            raise RuntimeError(f"non-LoRA tensors in adapter: {bad[:20]}")
        tensor_shapes = {key: list(handle.get_slice(key).get_shape()) for key in keys}
    return {
        "peft_type": config["peft_type"],
        "base_model_name_or_path": config.get("base_model_name_or_path"),
        "target_modules": sorted(targets),
        "tensor_count": len(keys),
        "parameter_count_from_shapes": int(
            sum(math.prod(shape) for shape in tensor_shapes.values())
        ),
        "weight_file": weight_paths[0].name,
        "weight_bytes": weight_paths[0].stat().st_size,
    }


def load_nemotron_adapter(
    base_model: str | Path,
    adapter_dir: str | Path,
    *,
    device: str = "cuda",
    attention: str = "eager",
    max_seq_length: int = 512,
) -> Any:
    """Load a saved adapter into the frozen SentenceTransformers base model."""
    os.environ.setdefault(
        "HF_MODULES_CACHE", "/lfs/skampere3/0/alexspan/.cache/huggingface/modules"
    )
    import torch
    from peft import PeftModel
    from sentence_transformers import SentenceTransformer

    model_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
    if attention != "auto":
        model_kwargs["attn_implementation"] = attention
    model = SentenceTransformer(
        str(base_model),
        device=device,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"padding_side": "left"},
    )
    model.max_seq_length = max_seq_length
    model[0].auto_model = PeftModel.from_pretrained(
        model[0].auto_model, str(adapter_dir), is_trainable=False
    )
    model.eval()
    return model


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def train(args: argparse.Namespace) -> dict[str, Any]:
    # The immutable shared HF model cache is read-only on sk3.  Trusted-code
    # models still need a writable location for their generated Python module.
    os.environ.setdefault(
        "HF_MODULES_CACHE", str(Path.home() / ".cache" / "huggingface" / "modules")
    )
    import torch
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from sentence_transformers.util import batch_to_device
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    _set_determinism(args.seed)
    manifest_path = Path(args.manifest).resolve()
    teacher_paths = tuple(Path(path).resolve() for path in args.teachers)
    teacher_manifest_path = (
        Path(args.teacher_manifest).resolve() if args.teacher_manifest else None
    )
    universe = load_universe(
        manifest_path,
        teacher_paths,
        args.task,
        split_seed=args.split_seed,
        train_percent=args.train_percent,
        dev_percent=args.dev_percent,
        require_bank_hash=not args.allow_unhashed_teachers,
        teacher_manifest_path=teacher_manifest_path,
        respect_teacher_splits=args.respect_teacher_splits,
    )
    output = Path(args.output_root).resolve() / args.task
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty adapter run directory: {output}; "
            "choose a new versioned --output-root"
        )
    output.mkdir(parents=True, exist_ok=True)
    split_rows = [
        {
            "norm_uid": label.norm_uid,
            "task": label.task,
            "corpus": label.corpus,
            "source_group": label.source_group,
            "split": label.split,
            "metric_id": label.metric_id,
            "acceptable_metric_ids": list(label.acceptable_metric_ids),
            "teacher_sources": list(label.teacher_sources),
            "teacher_norm_uids": list(label.teacher_norm_uids),
            "supervision_strength": label.supervision_strength,
        }
        for label in universe.labels
    ]
    split_path = output / "split_assignments.jsonl"
    write_jsonl(split_path, split_rows)

    model_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
    if args.attention != "auto":
        model_kwargs["attn_implementation"] = args.attention
    model = SentenceTransformer(
        args.model,
        device=args.device,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"padding_side": "left"},
    )
    model.max_seq_length = args.max_seq_length
    bank_cards = [metric_card(dict(metric)) for metric in universe.bank]
    queries = [label.query for label in universe.labels]
    # Mine and score against the untouched base model before attaching LoRA.
    base_bank_embeddings = _encode(model, bank_cards, args.eval_batch_size)
    base_query_embeddings = _encode(model, queries, args.eval_batch_size)
    before = evaluate_embeddings(
        base_query_embeddings,
        base_bank_embeddings,
        universe.labels,
        universe.bank,
        ("dev", "test"),
    )
    triplets = build_triplets(
        universe.labels,
        universe.bank,
        base_query_embeddings,
        base_bank_embeddings,
        pool_size=args.hard_negative_pool,
        negatives_per_positive=args.negatives_per_positive,
    )
    if not triplets:
        raise ValueError("no train triplets after source-disjoint split")
    triplet_path = output / "training_triplets.jsonl"
    write_jsonl(triplet_path, triplets)

    backbone = model[0].auto_model
    if hasattr(backbone, "gradient_checkpointing_enable"):
        try:
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            backbone.gradient_checkpointing_enable()
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()
    if hasattr(backbone, "config"):
        backbone.config.use_cache = False
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    peft_model = get_peft_model(backbone, lora_config)
    model[0].auto_model = peft_model
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    non_lora = [name for name, _ in trainable if "lora_" not in name]
    if not trainable or non_lora:
        raise RuntimeError(
            f"LoRA isolation failed: trainable={len(trainable)}, non_lora={non_lora[:20]}"
        )
    trainable_parameters = sum(parameter.numel() for _, parameter in trainable)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    examples = [
        InputExample(texts=[row["query"], row["positive"], row["negative"]])
        for row in triplets
    ]
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        examples,
        shuffle=True,
        batch_size=args.batch_size,
        generator=generator,
        num_workers=0,
        collate_fn=model.smart_batching_collate,
    )
    loss_fn = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=args.margin,
    )
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = math.ceil(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)

    # LoRA B matrices initialize to zero, so epoch 0 is the exact base-model
    # promotion baseline.  Dev selects an epoch; test remains untouched until
    # the selected adapter is restored.
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in get_peft_model_state_dict(peft_model).items()
    }
    best_epoch = 0
    selection_metric = f"recall_at_{args.selection_k}"
    selection_key_names = epoch_selection_key_names(
        args.selection_k, args.epoch_selection_policy
    )
    before_key = epoch_selection_key(
        before["dev"]["all"]["exact"],
        args.selection_k,
        args.epoch_selection_policy,
    )
    best_key = before_key
    epoch_reports = []
    global_update = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        for batch_idx, (features, labels) in enumerate(loader, 1):
            features = [batch_to_device(feature, model.device) for feature in features]
            labels = labels.to(model.device)
            loss = loss_fn(features, labels)
            (loss / args.gradient_accumulation_steps).backward()
            running_loss += float(loss.detach().cpu())
            batches += 1
            should_update = (
                batch_idx % args.gradient_accumulation_steps == 0
                or batch_idx == len(loader)
            )
            if should_update:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for _, parameter in trainable], args.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1
        dev_indices = [
            i for i, label in enumerate(universe.labels) if label.split == "dev"
        ]
        dev_queries = [queries[i] for i in dev_indices]
        trained_bank_embeddings = _encode(model, bank_cards, args.eval_batch_size)
        trained_dev_embeddings = _encode(model, dev_queries, args.eval_batch_size)
        dev_labels = [universe.labels[i] for i in dev_indices]
        dev_eval = evaluate_embeddings(
            trained_dev_embeddings,
            trained_bank_embeddings,
            dev_labels,
            universe.bank,
            ("dev",),
        )["dev"]
        dev_key = epoch_selection_key(
            dev_eval["all"]["exact"],
            args.selection_k,
            args.epoch_selection_policy,
        )
        epoch_reports.append(
            {
                "epoch": epoch,
                "mean_triplet_loss": running_loss / max(batches, 1),
                "optimizer_updates": global_update,
                "selection_key_names": list(selection_key_names),
                "selection_key": list(dev_key),
                "dev": dev_eval,
            }
        )
        if dev_key > best_key:
            best_key = dev_key
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in get_peft_model_state_dict(peft_model).items()
            }

    set_result = set_peft_model_state_dict(peft_model, best_state)
    unexpected = list(getattr(set_result, "unexpected_keys", []) or [])
    if unexpected:
        raise RuntimeError(f"unexpected keys restoring best adapter: {unexpected[:20]}")
    after_bank_embeddings = _encode(model, bank_cards, args.eval_batch_size)
    eval_indices = [
        i for i, label in enumerate(universe.labels) if label.split in {"dev", "test"}
    ]
    after_query_embeddings = _encode(
        model, [queries[i] for i in eval_indices], args.eval_batch_size
    )
    after_labels = [universe.labels[i] for i in eval_indices]
    after = evaluate_embeddings(
        after_query_embeddings,
        after_bank_embeddings,
        after_labels,
        universe.bank,
        ("dev", "test"),
    )

    adapter_dir = output / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(adapter_dir, safe_serialization=True)
    adapter_files = sorted(
        path.name for path in adapter_dir.iterdir() if path.is_file()
    )
    if "adapter_config.json" not in adapter_files or not any(
        name.startswith("adapter_model") for name in adapter_files
    ):
        raise RuntimeError(f"PEFT adapter save incomplete: {adapter_files}")
    adapter_validation = validate_adapter_artifact(adapter_dir)

    before_dev = before["dev"]["all"]["exact"][selection_metric]
    after_dev = after["dev"]["all"]["exact"][selection_metric]
    after_key = epoch_selection_key(
        after["dev"]["all"]["exact"],
        args.selection_k,
        args.epoch_selection_policy,
    )
    promotion_pass = epoch_promotion_passes(
        before_key,
        after_key,
        policy=args.epoch_selection_policy,
        minimum_primary_gain=args.min_dev_recall_gain,
    )
    label_counts_by_metric = Counter(label.metric_id for label in universe.labels)
    bank_ids = {str(metric["metric_id"]) for metric in universe.bank}
    run_config = {
        "task": args.task,
        "base_model": str(Path(args.model).resolve()),
        "query_instruction": RETRIEVAL_INSTRUCTION,
        "manifest": str(manifest_path),
        "teachers": [str(path) for path in teacher_paths],
        "teacher_manifest": (
            str(teacher_manifest_path) if teacher_manifest_path else None
        ),
        "split_seed": args.split_seed,
        "train_percent": args.train_percent,
        "dev_percent": args.dev_percent,
        "test_percent": 100 - args.train_percent - args.dev_percent,
        "respect_teacher_splits": args.respect_teacher_splits,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "margin": args.margin,
        "hard_negative_pool": args.hard_negative_pool,
        "negatives_per_positive": args.negatives_per_positive,
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "bias": "none",
        },
        "max_seq_length": args.max_seq_length,
        "attention": args.attention,
        "device": args.device,
        "eval_batch_size": args.eval_batch_size,
        "min_dev_recall_gain": args.min_dev_recall_gain,
        "selection_k": args.selection_k,
        "epoch_selection_policy": args.epoch_selection_policy,
    }
    config_path = output / "run_config.json"
    _save_json(config_path, run_config)
    report = {
        "status": "PROMOTABLE" if promotion_pass else "REJECTED_NO_DEV_GAIN",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "task": args.task,
        "adapter_path": str(adapter_dir),
        "adapter_files": adapter_files,
        "adapter_validation": adapter_validation,
        "base_model": str(Path(args.model).resolve()),
        "model_source_hashes": _model_source_hashes(Path(args.model)),
        "libraries": _library_versions(),
        "input_hashes": {
            "manifest": sha256_file(manifest_path),
            "bank_artifact": sha256_file(universe.bank_path),
            "bank_hierarchy_source": universe.bank_source_sha256,
            "teachers": {str(path): sha256_file(path) for path in teacher_paths},
            "teacher_manifest": (
                sha256_file(teacher_manifest_path) if teacher_manifest_path else None
            ),
            "trainer": sha256_file(Path(__file__).resolve()),
        },
        "generated_hashes": {
            "split_assignments": sha256_file(split_path),
            "training_triplets": sha256_file(triplet_path),
            "run_config": sha256_file(config_path),
            "adapter": {
                path.name: sha256_file(path)
                for path in sorted(adapter_dir.iterdir())
                if path.is_file()
            },
        },
        "bank_metrics": len(universe.bank),
        "teacher_metric_coverage": {
            "covered": len(label_counts_by_metric),
            "fraction": len(label_counts_by_metric) / len(universe.bank),
            "rows_by_metric": dict(sorted(label_counts_by_metric.items())),
            "uncovered_metric_ids": sorted(bank_ids - set(label_counts_by_metric)),
        },
        "teacher_audit": universe.teacher_audit,
        "split_audit": universe.split_audit,
        "training_triplets": len(triplets),
        "negative_strategy_counts": dict(
            sorted(Counter(row["negative_strategy"] for row in triplets).items())
        ),
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "optimizer_updates": global_update,
        "warmup_steps": warmup_steps,
        "best_epoch": best_epoch,
        "epoch_reports": epoch_reports,
        "before": before,
        "after": after,
        "promotion_gate": {
            "metric": f"dev.exact.{selection_metric}",
            "before": before_dev,
            "after": after_dev,
            "minimum_gain": args.min_dev_recall_gain,
            "epoch_selection_policy": args.epoch_selection_policy,
            "selection_key_names": list(selection_key_names),
            "before_selection_key": list(before_key),
            "after_selection_key": list(after_key),
            "primary_depth_non_degradation": after_key[0] >= before_key[0],
            "passed": promotion_pass,
        },
    }
    report_path = output / "training_report.json"
    _save_json(report_path, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    if not promotion_pass and args.enforce_promotion_gate:
        raise SystemExit(2)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train exactly one task-specific Nemotron 8B LoRA adapter."
    )
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--manifest", default=str(DEFAULT_OUTPUT_ROOT / "manifest.json")
    )
    parser.add_argument(
        "--teachers",
        action="append",
        required=True,
        help="Clean teacher JSONL; repeat to combine independently produced label files.",
    )
    parser.add_argument(
        "--teacher-manifest",
        help=(
            "Manifest defining teacher norm_uids when it differs from --manifest; "
            "only unique exact-quote corpus/alias bridges are accepted."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_NEMOTRON)
    parser.add_argument("--output-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--attention", choices=("auto", "eager", "sdpa"), default="eager"
    )
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hard-negative-pool", type=int, default=16)
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--split-seed", type=int, default=73129)
    parser.add_argument("--train-percent", type=int, default=80)
    parser.add_argument("--dev-percent", type=int, default=10)
    parser.add_argument("--allow-unhashed-teachers", action="store_true")
    parser.add_argument(
        "--respect-teacher-splits",
        action="store_true",
        help=(
            "Use each MATCH teacher's explicit train/dev/test role and fail closed "
            "on missing or conflicting roles instead of re-hashing source groups."
        ),
    )
    parser.add_argument("--min-dev-recall-gain", type=float, default=0.0)
    parser.add_argument(
        "--selection-k",
        type=int,
        choices=DEFAULT_KS,
        default=50,
        help="Source-disjoint dev recall depth used for epoch selection and promotion.",
    )
    parser.add_argument(
        "--epoch-selection-policy",
        choices=("single_k", "depth_lexicographic"),
        default="single_k",
        help=(
            "single_k preserves legacy behavior; depth_lexicographic forbids recall loss "
            "at --selection-k, then selects on MRR and successively lower depths."
        ),
    )
    parser.add_argument(
        "--no-enforce-promotion-gate",
        dest="enforce_promotion_gate",
        action="store_false",
        help="Write a rejected adapter but return zero (for diagnostics only).",
    )
    parser.set_defaults(enforce_promotion_gate=True)
    args = parser.parse_args()
    positive_ints = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "eval_batch_size": args.eval_batch_size,
        "hard_negative_pool": args.hard_negative_pool,
        "negatives_per_positive": args.negatives_per_positive,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "max_seq_length": args.max_seq_length,
    }
    bad = {name: value for name, value in positive_ints.items() if value <= 0}
    if bad:
        parser.error(f"arguments must be positive: {bad}")
    if not 0 <= args.warmup_ratio < 1:
        parser.error("--warmup-ratio must be in [0,1)")
    if args.min_dev_recall_gain < 0:
        parser.error("--min-dev-recall-gain must be non-negative")
    return args


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
