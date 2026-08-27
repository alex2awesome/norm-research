#!/usr/bin/env python3
"""Build a frozen, compact, three-way Nemotron CE pair universe.

The builder joins canonical norms to independently frozen truth and the union
of one or more truth-blind candidate lanes.  Exact targets may be inserted
only for train rows.  Frozen-hierarchy siblings are FAMILY rather than false
negatives, and acceptable metric IDs are never emitted as REJECT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import (
    metric_card,
    norm_query,
    normalize_space,
    read_jsonl,
    sha256_file,
    write_jsonl,
)
from .train_nemotron_lora import source_group_key


SCHEMA = "silver-match-v3-nemotron-ce-pair-v1"
REPORT_SCHEMA = "silver-match-v3-nemotron-ce-pair-report-v1"
RELATIONS = ("EXACT", "FAMILY", "REJECT")
RELATION_ID = {"REJECT": 0, "FAMILY": 1, "EXACT": 2}
DECISIONS = {
    "MATCH",
    "MATCH_FAMILY_ONLY",
    "NO_EXPLICIT_CRITERION",
    "CONTEXT_NEEDED",
    "GENERIC_VERDICT",
    "NO_CANDIDATE_FITS",
    "NOISE",
}
SPLITS = {"train", "dev", "test", "blind"}


@dataclass(frozen=True)
class CandidateSource:
    lane: str
    artifact_sha256: str
    rank: int | None
    retrieval_lane: str | None
    score: float | None


@dataclass(frozen=True)
class Draft:
    norm_uid: str
    metric_id: str
    relation: str
    sources: tuple[CandidateSource, ...]


def _stable_digest(seed: int, *parts: Any) -> str:
    payload = "\x1f".join((str(seed), *(normalize_space(part) for part in parts)))
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()


def _canonical_source_group(value: Any) -> str:
    """Preserve structural delimiters while accepting legacy colon rendering."""

    raw = str(value or "").strip()
    if not raw:
        return ""
    if "\x1f" in raw:
        parts = raw.split("\x1f")
    elif raw.count(":") >= 3:
        parts = raw.split(":", 3)
    else:
        return raw
    if len(parts) != 4 or any(not part.strip() for part in parts):
        raise ValueError(f"malformed canonical source_group: {raw!r}")
    return "\x1f".join(part.strip() for part in parts)


def _ref(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _resolve(path: str | Path, relative_to: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (relative_to / value).resolve()


def _index_unique(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    uids = [normalize_space(row.get("norm_uid")) for row in rows]
    if not rows or "" in uids or len(uids) != len(set(uids)):
        raise ValueError(
            f"{label} is empty or has missing/duplicate norm_uid values: {path}"
        )
    return rows


def _acceptable_ids(row: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in ("acceptable_metric_ids", "equivalent_metric_ids"):
        raw = row.get(key)
        if isinstance(raw, str):
            values.add(normalize_space(raw))
        elif isinstance(raw, (list, tuple, set)):
            values.update(normalize_space(value) for value in raw)
        elif raw is not None:
            raise ValueError(
                f"{key} must be a string or sequence: {row.get('norm_uid')}"
            )
    values.discard("")
    return values


def _family_anchors(row: Mapping[str, Any]) -> set[str]:
    anchors = _acceptable_ids(row)
    raw = row.get("family_metric_ids")
    if isinstance(raw, str):
        anchors.add(normalize_space(raw))
    elif isinstance(raw, (list, tuple, set)):
        anchors.update(normalize_space(value) for value in raw)
    elif raw is not None:
        raise ValueError(
            f"family_metric_ids must be a string or sequence: {row.get('norm_uid')}"
        )
    metric_id = normalize_space(row.get("metric_id"))
    if metric_id:
        anchors.add(metric_id)
    anchors.discard("")
    return anchors


def _load_bank(
    manifest: Mapping[str, Any], bank_path: Path, task: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], str]:
    bank_meta = (manifest.get("banks") or {}).get(task)
    if not isinstance(bank_meta, dict):
        raise ValueError(f"task absent from manifest banks: {task}")
    bank_hash = normalize_space(bank_meta.get("source_sha256"))
    if not bank_hash:
        raise ValueError("manifest bank lacks source_sha256")
    payload = json.loads(bank_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("bank must be a JSON object")
    if (
        payload.get("task") != task
        or normalize_space(payload.get("source_sha256")) != bank_hash
    ):
        raise ValueError("bank task/source hash differs from canonical manifest")
    metrics = payload.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("bank contains no metrics")
    metric_ids = [normalize_space(metric.get("metric_id")) for metric in metrics]
    if "" in metric_ids or len(metric_ids) != len(set(metric_ids)):
        raise ValueError("bank has missing or duplicate metric IDs")
    expected_count = int(bank_meta.get("count", len(metrics)))
    if len(metrics) != expected_count:
        raise ValueError("bank metric count differs from canonical manifest")
    for metric in metrics:
        if metric.get("task") not in (None, task):
            raise ValueError(f"foreign-task bank metric: {metric.get('metric_id')}")
    return metrics, dict(zip(metric_ids, metrics)), bank_hash


def _load_split_assignments(
    path: Path | None, *, task: str, bank_hash: str
) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    if path is None:
        return {}, {"assignment_rows": 0}
    rows = _index_unique(path, label="split assignments")
    assignments: dict[str, dict[str, str]] = {}
    for row in rows:
        uid = normalize_space(row.get("norm_uid"))
        supplied_task = normalize_space(row.get("task"))
        if supplied_task and supplied_task != task:
            raise ValueError(f"split assignment contains foreign task: {uid}")
        supplied_hash = normalize_space(
            row.get("current_bank_source_sha256") or row.get("bank_source_sha256")
        )
        if supplied_hash and supplied_hash != bank_hash:
            raise ValueError(f"split assignment bank hash mismatch: {uid}")
        split = normalize_space(row.get("split"))
        if split not in SPLITS:
            raise ValueError(f"split assignment has invalid split: {uid}")
        assignments[uid] = {
            "split": split,
            "source_group": _canonical_source_group(row.get("source_group")),
        }
    return assignments, {"assignment_rows": len(rows)}


def _load_truth(
    paths: Sequence[Path],
    *,
    task: str,
    bank_hash: str,
    bank_ids: set[str],
    split_assignments: Mapping[str, Mapping[str, str]],
    allow_unanchored_family_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, int]]:
    if not paths:
        raise ValueError("at least one truth input is required")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    group_split: dict[str, str] = {}
    for path in paths:
        for row in _index_unique(path, label="truth"):
            uid = normalize_space(row.get("norm_uid"))
            if uid in seen:
                raise ValueError(f"truth norm_uid appears in multiple inputs: {uid}")
            seen.add(uid)
            decision = normalize_space(row.get("decision"))
            split = normalize_space(row.get("split"))
            source_group = _canonical_source_group(row.get("source_group"))
            assignment = split_assignments.get(uid)
            if split_assignments and assignment is None:
                raise ValueError(
                    f"truth UID absent from authoritative split assignments: {uid}"
                )
            if assignment is not None:
                assigned_split = normalize_space(assignment.get("split"))
                assigned_group = _canonical_source_group(assignment.get("source_group"))
                if split and split != assigned_split:
                    raise ValueError(f"truth/split-assignment split mismatch: {uid}")
                if source_group and assigned_group and source_group != assigned_group:
                    raise ValueError(
                        f"truth/split-assignment source_group mismatch: {uid}"
                    )
                split = assigned_split
                source_group = source_group or assigned_group
            supplied_hash = normalize_space(
                row.get("current_bank_source_sha256") or row.get("bank_source_sha256")
            )
            if row.get("task") != task or decision not in DECISIONS:
                raise ValueError(f"truth task/decision is invalid: {uid}")
            if supplied_hash != bank_hash:
                raise ValueError(f"truth current-bank hash mismatch: {uid}")
            if split not in SPLITS or not source_group:
                raise ValueError(f"truth split/source_group is invalid: {uid}")
            prior_split = group_split.setdefault(source_group, split)
            if prior_split != split:
                raise ValueError(f"source group crosses truth splits: {source_group}")
            metric_id = normalize_space(row.get("metric_id"))
            acceptable = _acceptable_ids(row)
            if decision == "MATCH":
                if not metric_id or metric_id not in bank_ids:
                    raise ValueError(f"MATCH truth lacks an in-bank metric: {uid}")
                acceptable.add(metric_id)
            elif decision == "MATCH_FAMILY_ONLY":
                if not _family_anchors(row) and not allow_unanchored_family_only:
                    raise ValueError(
                        f"MATCH_FAMILY_ONLY truth lacks family anchors: {uid}"
                    )
            elif metric_id or acceptable:
                raise ValueError(f"typed nonmatch carries metric targets: {uid}")
            unknown = (
                _family_anchors(row) if decision == "MATCH_FAMILY_ONLY" else acceptable
            ) - bank_ids
            if unknown:
                raise ValueError(
                    f"truth references metrics outside bank for {uid}: {sorted(unknown)}"
                )
            rendered = dict(row)
            rendered["split"] = split
            rendered["source_group"] = source_group
            rendered["acceptable_metric_ids"] = sorted(acceptable)
            rows.append(rendered)
    rows.sort(key=lambda row: normalize_space(row["norm_uid"]))
    return (
        rows,
        {normalize_space(row["norm_uid"]): row for row in rows},
        {
            "truth_rows_joined_to_assignments": len(rows) if split_assignments else 0,
            "assignment_rows_outside_truth": len(set(split_assignments) - seen),
        },
    )


def _load_canonical_norms(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    task: str,
    truth: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    wanted = set(truth)
    norms: dict[str, dict[str, Any]] = {}
    corpora = manifest.get("corpora") or {}
    for corpus, meta in sorted(corpora.items()):
        if not isinstance(meta, dict) or meta.get("task") != task:
            continue
        path = _resolve(meta.get("path") or "", manifest_path.parent)
        for row in read_jsonl(path):
            uid = normalize_space(row.get("norm_uid"))
            if uid not in wanted:
                continue
            if uid in norms:
                raise ValueError(
                    f"canonical norm_uid is duplicated across corpora: {uid}"
                )
            if row.get("task") != task or row.get("corpus") != corpus:
                raise ValueError(f"canonical task/corpus routing mismatch: {uid}")
            norms[uid] = row
    missing = sorted(wanted - set(norms))
    if missing:
        raise ValueError(
            f"truth UIDs absent from canonical manifest norms: {missing[:5]}"
        )
    split_groups: dict[str, str] = {}
    for uid, truth_row in truth.items():
        norm = norms[uid]
        truth_group = _canonical_source_group(truth_row.get("source_group"))
        canonical_group = source_group_key(norm)
        if not canonical_group or truth_group != canonical_group:
            raise ValueError(f"truth/canonical source_group mismatch: {uid}")
        if truth_row.get("corpus") != norm.get("corpus"):
            raise ValueError(f"truth/canonical corpus mismatch: {uid}")
        split = normalize_space(truth_row.get("split"))
        prior = split_groups.setdefault(canonical_group, split)
        if prior != split:
            raise ValueError(
                f"canonical source group crosses splits: {canonical_group}"
            )
        if not normalize_space(norm.get("norm")):
            raise ValueError(f"canonical norm has no evaluative statement: {uid}")
    return norms


def _hierarchy_ids(group: Mapping[str, Any]) -> tuple[list[str], list[str] | None]:
    for key in ("metric_ids", "source_metric_ids", "source_r2_metric_ids"):
        raw = group.get(key)
        if raw is not None:
            if not isinstance(raw, list):
                raise ValueError(f"hierarchy {key} must be a list")
            return [normalize_space(value) for value in raw], None
    raw = group.get("source_r2_cluster_ids")
    if not isinstance(raw, list):
        raise ValueError("hierarchy group lacks metric IDs")
    try:
        ids = [f"a{int(value)}" for value in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "source_r2_cluster_ids are not integer metric suffixes"
        ) from exc
    names = group.get("source_r2_cluster_names")
    if names is not None and not isinstance(names, list):
        raise ValueError("source_r2_cluster_names must be a list")
    return ids, (
        [normalize_space(value) for value in names] if names is not None else None
    )


def _load_families(
    hierarchy_path: Path,
    *,
    task: str,
    metric_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, frozenset[str]]:
    payload = json.loads(hierarchy_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("task") != task:
        raise ValueError("frozen hierarchy task mismatch")
    expected = payload.get("n_r2_clusters_in")
    if expected is not None and int(expected) != len(metric_by_id):
        raise ValueError("frozen hierarchy input count differs from current bank")
    groups = payload.get("merged_groups")
    if not isinstance(groups, list):
        raise ValueError("frozen hierarchy lacks merged_groups")
    if payload.get("n_merged_groups") is not None and int(
        payload["n_merged_groups"]
    ) != len(groups):
        raise ValueError("frozen hierarchy merged-group count is inconsistent")
    families: dict[str, frozenset[str]] = {}
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError(f"hierarchy group {index} is not an object")
        ids, names = _hierarchy_ids(group)
        if len(ids) < 2 or "" in ids or len(ids) != len(set(ids)):
            raise ValueError(
                f"hierarchy group {index} is not a unique multi-member family"
            )
        unknown = set(ids) - set(metric_by_id)
        if unknown:
            raise ValueError(
                f"hierarchy group {index} references out-of-bank metrics: {sorted(unknown)}"
            )
        if names is not None:
            if len(names) != len(ids):
                raise ValueError(f"hierarchy group {index} name/ID counts differ")
            for metric_id, expected_name in zip(ids, names):
                if (
                    normalize_space(metric_by_id[metric_id].get("name"))
                    != expected_name
                ):
                    raise ValueError(f"hierarchy metric name drift: {metric_id}")
        family = frozenset(ids)
        for metric_id in ids:
            if metric_id in families:
                raise ValueError(
                    f"metric occurs in multiple frozen families: {metric_id}"
                )
            families[metric_id] = family
    return families


def _parse_candidate_specs(specs: Sequence[str], base: Path) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    lanes: set[str] = set()
    for spec in specs:
        lane, separator, raw_path = spec.partition("=")
        lane = normalize_space(lane)
        if not separator or not lane or not raw_path:
            raise ValueError("candidate inputs must use LANE=PATH")
        if lane in lanes:
            raise ValueError(f"duplicate candidate lane: {lane}")
        lanes.add(lane)
        parsed.append((lane, _resolve(raw_path, base)))
    return parsed


def _optional_score(candidate: Mapping[str, Any]) -> float | None:
    for key in ("score", "reranker_score", "rrf_score", "dense_score"):
        value = candidate.get(key)
        if value is None:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"candidate {key} is not numeric") from exc
        if not math.isfinite(score):
            raise ValueError(f"candidate {key} is non-finite")
        return score
    return None


def _load_candidate_union(
    specs: Sequence[tuple[str, Path]],
    *,
    truth: Mapping[str, Mapping[str, Any]],
    task: str,
    bank_hash: str,
    bank_ids: set[str],
) -> tuple[
    dict[str, dict[str, list[CandidateSource]]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    union: dict[str, dict[str, list[CandidateSource]]] = defaultdict(
        lambda: defaultdict(list)
    )
    refs: dict[str, dict[str, Any]] = {}
    lane_audits: dict[str, dict[str, int]] = {}
    for lane, path in specs:
        artifact_hash = sha256_file(path)
        refs[lane] = _ref(path)
        rows = _index_unique(path, label=f"candidate lane {lane}")
        audit: Counter[str] = Counter()
        for row in rows:
            uid = normalize_space(row.get("norm_uid"))
            if (
                row.get("task") != task
                or normalize_space(row.get("bank_source_sha256")) != bank_hash
            ):
                raise ValueError(f"candidate lane {lane} task/bank mismatch: {uid}")
            audit["input_rows"] += 1
            joined = uid in truth
            audit["joined_truth_rows" if joined else "outside_truth_rows_ignored"] += 1
            if joined:
                candidate_group = _canonical_source_group(row.get("source_group"))
                if candidate_group and candidate_group != _canonical_source_group(
                    truth[uid].get("source_group")
                ):
                    raise ValueError(
                        f"candidate lane {lane} source_group mismatch: {uid}"
                    )
                candidate_corpus = normalize_space(row.get("corpus"))
                if candidate_corpus and candidate_corpus != normalize_space(
                    truth[uid].get("corpus")
                ):
                    raise ValueError(f"candidate lane {lane} corpus mismatch: {uid}")
            candidates = row.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"candidate lane {lane} has no candidates: {uid}")
            seen: set[str] = set()
            for position, candidate in enumerate(candidates, 1):
                if not isinstance(candidate, dict):
                    raise ValueError(
                        f"candidate lane {lane} contains a non-object: {uid}"
                    )
                metric_id = normalize_space(candidate.get("metric_id"))
                if not metric_id or metric_id in seen or metric_id not in bank_ids:
                    raise ValueError(
                        f"candidate lane {lane} has duplicate/out-of-bank metric: {uid}"
                    )
                seen.add(metric_id)
                raw_rank = candidate.get("rank", position)
                try:
                    rank = int(raw_rank)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"candidate lane {lane} has invalid rank: {uid}"
                    ) from exc
                if rank < 1:
                    raise ValueError(
                        f"candidate lane {lane} has non-positive rank: {uid}"
                    )
                retrieval_lane = (
                    normalize_space(
                        candidate.get("retrieval_lane")
                        or candidate.get("candidate_lane")
                    )
                    or None
                )
                score = _optional_score(candidate)
                audit["candidate_rows_validated"] += 1
                if joined:
                    union[uid][metric_id].append(
                        CandidateSource(
                            lane=lane,
                            artifact_sha256=artifact_hash,
                            rank=rank,
                            retrieval_lane=retrieval_lane,
                            score=score,
                        )
                    )
        lane_audits[lane] = dict(sorted(audit.items()))
    missing = sorted(set(truth) - set(union))
    if missing:
        raise ValueError(
            f"candidate lane union misses {len(missing)} truth UIDs: {missing[:5]}"
        )
    return (
        union,
        refs,
        {
            "per_lane": lane_audits,
            "truth_uid_coverage": len(union),
            "missing_truth_uids": 0,
            "outside_truth_rows_ignored": sum(
                audit.get("outside_truth_rows_ignored", 0)
                for audit in lane_audits.values()
            ),
        },
    )


def _relation(
    truth_row: Mapping[str, Any], metric_id: str, families: Mapping[str, frozenset[str]]
) -> str:
    decision = normalize_space(truth_row.get("decision"))
    acceptable = _acceptable_ids(truth_row)
    primary = normalize_space(truth_row.get("metric_id"))
    if decision == "MATCH":
        acceptable.add(primary)
        if metric_id in acceptable:
            return "EXACT"
        anchors = acceptable
    elif decision == "MATCH_FAMILY_ONLY":
        anchors = _family_anchors(truth_row)
        if metric_id in anchors:
            return "FAMILY"
    else:
        return "REJECT"
    if any(
        metric_id in families.get(anchor, frozenset((anchor,))) for anchor in anchors
    ):
        return "FAMILY"
    return "REJECT"


def _add_source(
    union: dict[str, dict[str, list[CandidateSource]]],
    uid: str,
    metric_id: str,
    lane: str,
) -> None:
    if metric_id in union[uid]:
        return
    union[uid][metric_id].append(
        CandidateSource(
            lane=lane,
            artifact_sha256="derived",
            rank=None,
            retrieval_lane=None,
            score=None,
        )
    )


def _augment_train_candidates(
    union: dict[str, dict[str, list[CandidateSource]]],
    truth: Mapping[str, Mapping[str, Any]],
    families: Mapping[str, frozenset[str]],
    bank_ids: Sequence[str],
    *,
    seed: int,
    global_negatives_per_norm: int,
) -> dict[str, int]:
    counts: Counter[str] = Counter(
        metric_id for candidates in union.values() for metric_id in candidates
    )
    audits: Counter[str] = Counter()
    train_uids = sorted(
        (
            uid
            for uid, row in truth.items()
            if normalize_space(row.get("split")) == "train"
        ),
        key=lambda uid: (_stable_digest(seed, "train-order", uid), uid),
    )
    for uid in train_uids:
        row = truth[uid]
        decision = normalize_space(row.get("decision"))
        anchors = _family_anchors(row)
        if decision == "MATCH":
            for metric_id in sorted(
                _acceptable_ids(row) | {normalize_space(row.get("metric_id"))}
            ):
                if metric_id not in union[uid]:
                    _add_source(union, uid, metric_id, "train_gold_injection")
                    counts[metric_id] += 1
                    audits["train_exact_injections"] += 1
        if decision in {"MATCH", "MATCH_FAMILY_ONLY"}:
            siblings = (
                set().union(*(families.get(anchor, frozenset()) for anchor in anchors))
                - anchors
            )
            for metric_id in sorted(siblings):
                if metric_id not in union[uid]:
                    _add_source(union, uid, metric_id, "frozen_hierarchy_sibling")
                    counts[metric_id] += 1
                    audits["train_family_injections"] += 1
        if global_negatives_per_norm < 1:
            continue
        forbidden = set(union[uid]) | anchors
        forbidden.update(
            set().union(*(families.get(anchor, frozenset()) for anchor in anchors))
        )
        eligible = [metric_id for metric_id in bank_ids if metric_id not in forbidden]
        chosen = sorted(
            eligible,
            key=lambda metric_id: (
                counts[metric_id],
                _stable_digest(seed, "global-negative", uid, metric_id),
                metric_id,
            ),
        )[:global_negatives_per_norm]
        for metric_id in chosen:
            _add_source(union, uid, metric_id, "global_balanced_negative")
            counts[metric_id] += 1
            audits["global_balanced_negative_injections"] += 1
    return dict(sorted(audits.items()))


def _balanced_order(rows: Iterable[Draft], *, seed: int, relation: str) -> list[Draft]:
    groups: dict[str, deque[Draft]] = {}
    staged: dict[str, list[Draft]] = defaultdict(list)
    for row in rows:
        staged[row.metric_id].append(row)
    for metric_id, values in staged.items():
        groups[metric_id] = deque(
            sorted(
                values,
                key=lambda row: (
                    _stable_digest(seed, relation, metric_id, row.norm_uid),
                    row.norm_uid,
                ),
            )
        )
    output: list[Draft] = []
    active = sorted(groups)
    while active:
        next_active: list[str] = []
        for metric_id in active:
            output.append(groups[metric_id].popleft())
            if groups[metric_id]:
                next_active.append(metric_id)
        active = next_active
    return output


def _cap_drafts(
    drafts: Sequence[Draft], *, maximum_pairs: int, seed: int
) -> list[Draft]:
    if maximum_pairs < 1:
        raise ValueError("maximum_pairs must be positive")
    buckets = {
        relation: [row for row in drafts if row.relation == relation]
        for relation in RELATIONS
    }
    ordered = {
        relation: _balanced_order(rows, seed=seed, relation=relation)
        for relation, rows in buckets.items()
    }
    if len(ordered["EXACT"]) > maximum_pairs:
        raise ValueError("pair cap is smaller than the mandatory exact-pair universe")
    selected = list(ordered["EXACT"])
    used = {"EXACT": len(selected), "FAMILY": 0, "REJECT": 0}
    remaining = maximum_pairs - len(selected)
    family_goal = min(len(ordered["FAMILY"]), maximum_pairs // 4, remaining)
    selected.extend(ordered["FAMILY"][:family_goal])
    used["FAMILY"] = family_goal
    remaining -= family_goal
    reject_goal = min(len(ordered["REJECT"]), maximum_pairs // 2, remaining)
    selected.extend(ordered["REJECT"][:reject_goal])
    used["REJECT"] = reject_goal
    remaining -= reject_goal
    # Fill spare capacity deterministically, preferring REJECT then FAMILY.
    for relation in ("REJECT", "FAMILY"):
        if remaining < 1:
            break
        extra = ordered[relation][used[relation] : used[relation] + remaining]
        selected.extend(extra)
        used[relation] += len(extra)
        remaining -= len(extra)
    return sorted(
        selected,
        key=lambda row: (
            _stable_digest(seed, "output", row.norm_uid, row.metric_id),
            row.norm_uid,
            row.metric_id,
        ),
    )


def build(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = Path(args.manifest).resolve()
    bank_path = Path(args.bank).resolve()
    hierarchy_path = Path(args.hierarchy).resolve()
    truth_paths = tuple(Path(path).resolve() for path in args.truth)
    split_assignments_path = (
        Path(args.split_assignments).resolve()
        if getattr(args, "split_assignments", None)
        else None
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    metrics, metric_by_id, bank_hash = _load_bank(manifest, bank_path, args.task)
    bank_ids = set(metric_by_id)
    split_assignments, split_assignment_audit = _load_split_assignments(
        split_assignments_path, task=args.task, bank_hash=bank_hash
    )
    truth_rows, truth, truth_join_audit = _load_truth(
        truth_paths,
        task=args.task,
        bank_hash=bank_hash,
        bank_ids=bank_ids,
        split_assignments=split_assignments,
    )
    norms = _load_canonical_norms(manifest, manifest_path, args.task, truth)
    families = _load_families(hierarchy_path, task=args.task, metric_by_id=metric_by_id)
    candidate_specs = _parse_candidate_specs(args.candidates, manifest_path.parent)
    union, candidate_refs, candidate_audit = _load_candidate_union(
        candidate_specs,
        truth=truth,
        task=args.task,
        bank_hash=bank_hash,
        bank_ids=bank_ids,
    )
    augmentation = _augment_train_candidates(
        union,
        truth,
        families,
        sorted(bank_ids),
        seed=args.seed,
        global_negatives_per_norm=args.global_negatives_per_norm,
    )

    drafts: list[Draft] = []
    for uid in sorted(truth):
        for metric_id, sources in sorted(union.get(uid, {}).items()):
            relation = _relation(truth[uid], metric_id, families)
            if relation == "REJECT" and metric_id in _acceptable_ids(truth[uid]):
                raise AssertionError(
                    f"acceptable metric became REJECT: {uid}/{metric_id}"
                )
            drafts.append(
                Draft(
                    norm_uid=uid,
                    metric_id=metric_id,
                    relation=relation,
                    sources=tuple(
                        sorted(
                            sources,
                            key=lambda source: (
                                source.lane,
                                source.rank if source.rank is not None else 10**9,
                                source.retrieval_lane or "",
                            ),
                        )
                    ),
                )
            )
    if not drafts:
        raise ValueError("candidate union produced no CE pairs")
    selected = _cap_drafts(drafts, maximum_pairs=args.maximum_pairs, seed=args.seed)

    rows: list[dict[str, Any]] = []
    outside_train_injections = 0
    for draft in selected:
        truth_row = truth[draft.norm_uid]
        norm = norms[draft.norm_uid]
        split = normalize_space(truth_row.get("split"))
        derived_lanes = {
            source.lane
            for source in draft.sources
            if source.artifact_sha256 == "derived"
        }
        if split != "train" and derived_lanes:
            outside_train_injections += 1
            raise AssertionError(
                f"derived candidate leaked outside train: {draft.norm_uid}"
            )
        provenances = [
            {
                "lane": source.lane,
                "artifact_sha256": source.artifact_sha256,
                **({"rank": source.rank} if source.rank is not None else {}),
                **(
                    {"retrieval_lane": source.retrieval_lane}
                    if source.retrieval_lane
                    else {}
                ),
                **({"score": source.score} if source.score is not None else {}),
            }
            for source in draft.sources
        ]
        rows.append(
            {
                "schema_version": SCHEMA,
                "task": args.task,
                "corpus": str(norm["corpus"]),
                "norm_uid": draft.norm_uid,
                "source_group": source_group_key(norm),
                "split": split,
                "decision": str(truth_row["decision"]),
                "query": norm_query(norm, context_chars=args.context_chars),
                "metric_id": draft.metric_id,
                "metric_card": metric_card(metric_by_id[draft.metric_id]),
                "relation": draft.relation,
                "relation_id": RELATION_ID[draft.relation],
                "acceptable_metric_ids": sorted(_acceptable_ids(truth_row)),
                "candidate_lanes": sorted({source.lane for source in draft.sources}),
                "candidate_provenance": provenances,
                "current_bank_source_sha256": bank_hash,
                "frozen_hierarchy_sha256": sha256_file(hierarchy_path),
                "gradient_eligible": split == "train"
                and truth_row.get("gradient_eligible") is not False,
            }
        )

    full_counts = Counter(row.relation for row in drafts)
    selected_counts = Counter(row["relation"] for row in rows)
    metric_relation_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        metric_relation_counts[str(row["metric_id"])][str(row["relation"])] += 1
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "FROZEN_COMPACT_CE_PAIRS_READY",
        "task": args.task,
        "seed": args.seed,
        "maximum_pairs": args.maximum_pairs,
        "global_negatives_per_train_norm": args.global_negatives_per_norm,
        "bank_source_sha256": bank_hash,
        "bank_metric_count": len(metrics),
        "frozen_hierarchy_sha256": sha256_file(hierarchy_path),
        "frozen_family_count": len(set(families.values())),
        "truth_rows": len(truth_rows),
        "truth_split_counts": dict(
            sorted(Counter(str(row["split"]) for row in truth_rows).items())
        ),
        "truth_decision_counts": dict(
            sorted(Counter(str(row["decision"]) for row in truth_rows).items())
        ),
        "candidate_lane_count": len(candidate_specs),
        "candidate_union_audit": candidate_audit,
        "full_pair_count": len(drafts),
        "full_relation_counts": {key: full_counts[key] for key in RELATIONS},
        "selected_pair_count": len(rows),
        "selected_relation_counts": {key: selected_counts[key] for key in RELATIONS},
        "selected_split_counts": dict(
            sorted(Counter(str(row["split"]) for row in rows).items())
        ),
        "selected_metric_count": len({str(row["metric_id"]) for row in rows}),
        "per_metric_relation_counts": {
            metric_id: {relation: counts[relation] for relation in RELATIONS}
            for metric_id, counts in sorted(metric_relation_counts.items())
        },
        "augmentation_counts": augmentation,
        "split_assignment_audit": {
            **split_assignment_audit,
            **truth_join_audit,
        },
        "acceptable_as_reject_count": 0,
        "derived_candidate_rows_outside_train": outside_train_injections,
        "inputs": {
            "manifest": _ref(manifest_path),
            "bank": _ref(bank_path),
            "hierarchy": _ref(hierarchy_path),
            "truth": [_ref(path) for path in truth_paths],
            "split_assignments": (
                _ref(split_assignments_path) if split_assignments_path else None
            ),
            "candidate_lanes": candidate_refs,
            "builder": _ref(Path(__file__).resolve()),
        },
    }
    return rows, report


def write_release(args: argparse.Namespace) -> dict[str, Any]:
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    if output_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite frozen CE pair output/report")
    rows, report = build(args)
    write_jsonl(output_path, rows)
    report["output"] = {**_ref(output_path), "count": len(rows)}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--truth", action="append", required=True)
    parser.add_argument(
        "--split-assignments",
        help="optional authoritative norm_uid/source_group/split JSONL join",
    )
    parser.add_argument(
        "--candidate",
        dest="candidates",
        action="append",
        default=[],
        metavar="LANE=PATH",
        help="truth-blind candidate artifact; repeat for each retrieval lane",
    )
    parser.add_argument("--hierarchy", required=True)
    parser.add_argument("--maximum-pairs", type=int, default=400_000)
    parser.add_argument("--global-negatives-per-norm", type=int, default=4)
    parser.add_argument("--context-chars", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if args.global_negatives_per_norm < 0 or args.context_chars < 1:
        parser.error(
            "global-negatives-per-norm must be nonnegative and context-chars positive"
        )
    report = write_release(args)
    print(
        json.dumps(
            {**report, "report_sha256": sha256_file(Path(args.report).resolve())},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
