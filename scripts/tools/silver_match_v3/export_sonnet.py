#!/usr/bin/env python3
"""Export the legacy Sonnet matcher labels into the frozen v3 universe.

The July 2026 Sonnet runs used names from the old per-corpus catalogs.  This
module deliberately does *not* treat those names as current metric IDs.  A
metric label is retained only when its normalized name has one unambiguous
match in the frozen task bank.  Ordinary full/pilot decisions must also be a
candidate belonging to that same norm; this is the guard against the observed
multi-item prompt leakage.  Audit and rescue runs are allowed to name a metric
outside the top ten because those prompts explicitly exposed the full bank.

One best label is emitted per norm, with precedence:

    audit > rescue > pilot > full

The source artifacts are append-only inputs.  The canonical teacher and its
rejection ledger are written atomically under ``teachers/``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from . import SCHEMA_VERSION
from .common import normalize_name, normalize_space, read_jsonl, stable_uid, write_jsonl
from .config import DEFAULT_OUTPUT_ROOT


REPO_ROOT = Path(__file__).resolve().parents[3]
CLAUDE_PROJECT_ROOT = (
    Path.home()
    / ".claude/projects/-Users-spangher-Projects-stanford-research-norm-research"
)
LEGACY_SESSION = "ec784e30-9de7-4e8e-b613-b98a31a4919d"
DEFAULT_WORKFLOW_ROOT = CLAUDE_PROJECT_ROOT / LEGACY_SESSION / "subagents/workflows"
DEFAULT_SCRATCH_ROOT = (
    Path("/private/tmp/claude-502")
    / "-Users-spangher-Projects-stanford-research-norm-research"
    / LEGACY_SESSION
    / "scratchpad"
)
DEFAULT_V1_ROOT = REPO_ROOT / "notebooks/data/silver_20260704"

DEFAULT_JOURNALS = {
    "sonnet_full": DEFAULT_WORKFLOW_ROOT / "wf_e4731a71-903/journal.jsonl",
    "sonnet_pilot": DEFAULT_WORKFLOW_ROOT / "wf_4542c656-99d/journal.jsonl",
    "sonnet_audit": DEFAULT_WORKFLOW_ROOT / "wf_2b1aaf05-237/journal.jsonl",
    "sonnet_rescue": DEFAULT_WORKFLOW_ROOT / "wf_ef201696-750/journal.jsonl",
}
SOURCE_PRIORITY = {
    "sonnet_full": 1,
    "sonnet_pilot": 2,
    "sonnet_rescue": 3,
    "sonnet_audit": 4,
}
REAL_IDX_RE = re.compile(r"^(?P<corpus>.+)-(?P<row>[0-9]+)$")


@dataclass(frozen=True)
class LegacyItem:
    corpus: str
    row: int
    norm: str
    candidate_names: tuple[str, ...]
    candidate_ids: tuple[str, ...]

    @property
    def candidate_keys(self) -> frozenset[str]:
        return frozenset(normalize_name(name) for name in self.candidate_names)


@dataclass(frozen=True)
class CanonicalNorm:
    norm_uid: str
    corpus: str
    task: str
    row: int
    source_id: str
    norm: str


@dataclass(frozen=True)
class BankMetric:
    metric_id: str
    name: str
    name_key: str
    ambiguous: bool


@dataclass
class RawLabel:
    corpus: str
    legacy_row: int
    decision: str
    legacy_choice: str | None
    confidence: str | None
    label_source: str
    candidate_valid: bool | None
    notes: dict[str, Any] = field(default_factory=dict)
    metric_id: str | None = None
    bridge_method: str | None = None
    canonical: CanonicalNorm | None = None
    bank_sha256: str | None = None


@dataclass(frozen=True)
class BankBridge:
    task: str
    source_sha256: str
    by_name: Mapping[str, tuple[BankMetric, ...]]

    def resolve(self, choice: str) -> tuple[BankMetric | None, str | None]:
        matches = self.by_name.get(normalize_name(choice), ())
        if not matches:
            return None, "name_not_in_current_bank"
        if len(matches) != 1 or matches[0].ambiguous:
            return None, "ambiguous_current_bank_name"
        return matches[0], None


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _candidate_pairs(record: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    named = record.get("top10_names")
    if isinstance(named, list) and named:
        names, ids = [], []
        for rank, value in enumerate(named):
            if isinstance(value, dict):
                name = normalize_space(value.get("name"))
                metric_id = normalize_space(value.get("id"))
            else:
                name = normalize_space(value)
                metric_id = ""
            if not name:
                raise ValueError(f"empty candidate at rank {rank + 1}")
            names.append(name)
            ids.append(metric_id)
        return tuple(names), tuple(ids)

    names = record.get("top10")
    if not isinstance(names, list) or not names:
        raise ValueError("legacy row has no top10_names/top10")
    clean_names = tuple(normalize_space(value) for value in names)
    ids = record.get("top10_ids") or ()
    clean_ids = tuple(normalize_space(value) for value in ids)
    if clean_ids and len(clean_ids) != len(clean_names):
        raise ValueError("candidate ID/name length mismatch")
    if not clean_ids:
        clean_ids = tuple("" for _ in clean_names)
    return clean_names, clean_ids


def load_legacy_items(v1_root: Path, corpora: Iterable[str]) -> dict[tuple[str, int], LegacyItem]:
    items: dict[tuple[str, int], LegacyItem] = {}
    for corpus in sorted(corpora):
        path = v1_root / f"matches_joined_{corpus}.jsonl"
        if not path.exists():
            continue
        for line_no, record in enumerate(read_jsonl(path), 1):
            try:
                row = int(record.get("row"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid row in {path}:{line_no}") from exc
            names, ids = _candidate_pairs(record)
            item = LegacyItem(
                corpus=corpus,
                row=row,
                norm=normalize_space(record.get("norm")),
                candidate_names=names,
                candidate_ids=ids,
            )
            key = (corpus, row)
            if key in items:
                raise ValueError(f"duplicate legacy key {key}")
            items[key] = item
    return items


def load_manifest_universe(
    manifest_root: Path,
) -> tuple[dict[str, list[CanonicalNorm]], dict[str, BankBridge], dict[str, str]]:
    manifest = _load_json(manifest_root / "manifest.json")
    routing = {str(k): str(v) for k, v in (manifest.get("routing") or {}).items()}
    canonical: dict[str, list[CanonicalNorm]] = {}
    for corpus, meta in (manifest.get("corpora") or {}).items():
        path = manifest_root / "norms" / f"{corpus}.jsonl"
        if not path.exists() and isinstance(meta, dict) and meta.get("path"):
            path = Path(meta["path"])
        rows = []
        for record in read_jsonl(path):
            frozen_corpus = str(record["corpus"])
            frozen_row = int(record["row"])
            frozen_norm = normalize_space(record.get("norm"))
            frozen_uid = str(record["norm_uid"])
            expected_uid = stable_uid(frozen_corpus, frozen_row, frozen_norm)
            if frozen_uid != expected_uid:
                raise ValueError(
                    f"stale norm_uid in {path}: row {frozen_row}; "
                    "expected stable_uid(corpus, physical_row, norm)"
                )
            rows.append(
                CanonicalNorm(
                    norm_uid=frozen_uid,
                    corpus=frozen_corpus,
                    task=str(record["task"]),
                    row=frozen_row,
                    source_id=normalize_space(record.get("source_id")),
                    norm=frozen_norm,
                )
            )
        canonical[str(corpus)] = rows

    bridges: dict[str, BankBridge] = {}
    for task, meta in (manifest.get("banks") or {}).items():
        path = manifest_root / "banks" / f"{task}.json"
        if not path.exists() and isinstance(meta, dict) and meta.get("path"):
            path = Path(meta["path"])
        bank = _load_json(path)
        by_name: dict[str, list[BankMetric]] = defaultdict(list)
        for record in bank.get("metrics") or []:
            name = normalize_space(record.get("name"))
            name_key = normalize_name(record.get("name_key") or name)
            metric = BankMetric(
                metric_id=str(record["metric_id"]),
                name=name,
                name_key=name_key,
                ambiguous=bool(record.get("name_ambiguous")),
            )
            by_name[name_key].append(metric)
        # Never trust a stale/missing name_ambiguous flag.
        frozen = {
            key: tuple(
                BankMetric(m.metric_id, m.name, m.name_key, m.ambiguous or len(values) > 1)
                for m in values
            )
            for key, values in by_name.items()
        }
        bridges[str(task)] = BankBridge(
            task=str(task),
            source_sha256=str(bank.get("source_sha256") or meta.get("source_sha256") or ""),
            by_name=frozen,
        )
    return canonical, bridges, routing


def resolve_canonical_norm(item: LegacyItem, rows: Sequence[CanonicalNorm]) -> CanonicalNorm | None:
    """Resolve legacy 1-based/ad-hoc rows without silently accepting an offset.

    The norm text is always required.  ``source_id == legacy row`` is the
    strongest tiebreaker, followed by exact and one-off physical row matches.
    """

    norm = normalize_space(item.norm)
    matches = [row for row in rows if normalize_space(row.norm) == norm]
    if not matches:
        return None

    def score(row: CanonicalNorm) -> tuple[int, int]:
        value = 0
        if row.source_id == str(item.row):
            value += 8
        if row.row == item.row:
            value += 4
        if row.row == item.row - 1:
            value += 2
        if row.row == item.row + 1:
            value += 1
        return value, -abs(row.row - item.row)

    ranked = sorted(matches, key=score, reverse=True)
    if len(ranked) == 1:
        return ranked[0]
    if score(ranked[0]) > score(ranked[1]):
        return ranked[0]
    return None


def _real_idx(idx: Any, corpora: Iterable[str]) -> tuple[str, int] | None:
    text = normalize_space(idx)
    if text.startswith("anchor-"):
        return None
    match = REAL_IDX_RE.fullmatch(text)
    if not match or match.group("corpus") not in set(corpora):
        return None
    return match.group("corpus"), int(match.group("row"))


def _journal_result_groups(path: Path) -> Iterator[tuple[dict[str, Any], list[dict[str, Any]]]]:
    if not path.exists():
        return
    for event in read_jsonl(path):
        if event.get("type") != "result" or not isinstance(event.get("result"), dict):
            continue
        results = event["result"].get("results")
        if isinstance(results, list) and results and all(isinstance(row, dict) for row in results):
            yield event, results


def load_anchor_choices(batch_dir: Path) -> dict[tuple[str, str], str]:
    """Load exact planted choices for full/pilot batch gates."""

    anchors: dict[tuple[str, str], str] = {}
    if not batch_dir.exists():
        return anchors
    # ``rematch_full`` also contains a JSON-array manifest; only task batch
    # files have the ``<corpus>_bNNNN.json`` shape.
    for path in sorted(batch_dir.glob("*_b*.json")):
        payload = _load_json(path)
        corpus = str(payload.get("task") or "")
        for item in payload.get("items") or []:
            idx = normalize_space(item.get("idx"))
            if not idx.startswith("anchor-"):
                continue
            if idx.startswith("anchor-noise"):
                expected = "NOISE"
            elif idx.startswith("anchor-good"):
                candidates = item.get("top10") or []
                expected = normalize_space(candidates[0]) if candidates else ""
            else:
                continue
            key = (corpus, idx)
            if key in anchors and normalize_name(anchors[key]) != normalize_name(expected):
                raise ValueError(f"conflicting anchor definition for {key}")
            anchors[key] = expected
    return anchors


def load_rescue_anchors(batch_dir: Path) -> dict[tuple[str, str], str]:
    anchors: dict[tuple[str, str], str] = {}
    if not batch_dir.exists():
        return anchors
    prefix = "what really sank it for me was the "
    suffix = " — just not up to par"
    for path in sorted(batch_dir.glob("*.json")):
        payload = _load_json(path)
        corpus = str(payload.get("task") or "humor")
        for item in payload.get("items") or []:
            idx = normalize_space(item.get("idx"))
            if idx.startswith("anchor-gap"):
                anchors[(corpus, idx)] = "BANK_GAP"
            elif idx.startswith("anchor-found"):
                norm = normalize_space(item.get("norm"))
                if norm.lower().startswith(prefix) and norm.endswith(suffix):
                    anchors[(corpus, idx)] = norm[len(prefix) : -len(suffix)]
    return anchors


def _infer_group_corpus(results: Sequence[Mapping[str, Any]], corpora: Iterable[str]) -> str | None:
    found = {
        parsed[0]
        for row in results
        if (parsed := _real_idx(row.get("idx"), corpora)) is not None
    }
    return next(iter(found)) if len(found) == 1 else None


def _choice_anchor_gate(
    results: Sequence[Mapping[str, Any]],
    corpus: str,
    expected: Mapping[tuple[str, str], str],
) -> tuple[bool, dict[str, Any]]:
    good, noise, missing = 0, 0, []
    for row in results:
        idx = normalize_space(row.get("idx"))
        if not idx.startswith("anchor-"):
            continue
        target = expected.get((corpus, idx))
        if not target:
            missing.append(idx)
            continue
        choice = normalize_space(row.get("choice"))
        if idx.startswith("anchor-good") and normalize_name(choice) == normalize_name(target):
            good += 1
        if idx.startswith("anchor-noise") and choice == "NOISE":
            noise += 1
    return good >= 2 and noise >= 1 and not missing, {
        "good_exact": good,
        "noise_exact": noise,
        "missing_anchors": missing,
    }


def _audit_anchor_gate(results: Sequence[Mapping[str, Any]]) -> tuple[bool, dict[str, Any]]:
    good = [row for row in results if normalize_space(row.get("idx")).startswith("anchor-good")]
    wrong = [row for row in results if normalize_space(row.get("idx")).startswith("anchor-wrong")]
    good_ok = sum(row.get("top1_fit") == "exact" and row.get("best_rank") == 1 for row in good)
    wrong_ok = sum(row.get("top1_fit") == "wrong" and row.get("best_rank") != 1 for row in wrong)
    return len(good) >= 2 and len(wrong) >= 2 and good_ok >= 2 and wrong_ok >= 2, {
        "good_exact": good_ok,
        "wrong_exact": wrong_ok,
    }


def _rescue_anchor_gate(
    results: Sequence[Mapping[str, Any]],
    corpus: str,
    expected: Mapping[tuple[str, str], str],
) -> tuple[bool, dict[str, Any]]:
    found_ok = gap_ok = 0
    missing = []
    for row in results:
        idx = normalize_space(row.get("idx"))
        if not idx.startswith("anchor-"):
            continue
        target = expected.get((corpus, idx))
        if not target:
            missing.append(idx)
            continue
        if idx.startswith("anchor-found"):
            if row.get("verdict") == "found" and normalize_name(row.get("metric")) == normalize_name(target):
                found_ok += 1
        elif idx.startswith("anchor-gap") and row.get("verdict") == "not_in_bank":
            gap_ok += 1
    return found_ok >= 1 and gap_ok >= 1 and not missing, {
        "found_exact": found_ok,
        "gap_exact": gap_ok,
        "missing_anchors": missing,
    }


def _rejection(
    reason: str,
    source: str,
    *,
    idx: str | None = None,
    choice: str | None = None,
    details: Any = None,
) -> dict[str, Any]:
    parsed = REAL_IDX_RE.fullmatch(idx or "")
    return {
        "schema_version": SCHEMA_VERSION,
        "reason": reason,
        "label_source": source,
        "corpus": parsed.group("corpus") if parsed else None,
        "legacy_row": int(parsed.group("row")) if parsed else None,
        "idx": idx,
        "legacy_choice": choice,
        "details": details,
    }


def parse_choice_journal(
    path: Path,
    source: str,
    legacy: Mapping[tuple[str, int], LegacyItem],
    anchor_choices: Mapping[tuple[str, str], str],
) -> tuple[list[RawLabel], list[dict[str, Any]]]:
    accepted: list[RawLabel] = []
    rejected: list[dict[str, Any]] = []
    corpora = {key[0] for key in legacy}
    for event, results in _journal_result_groups(path):
        corpus = _infer_group_corpus(results, corpora)
        if corpus is None:
            rejected.append(_rejection("mixed_or_missing_group_corpus", source, details=event.get("key")))
            continue
        passed, gate = _choice_anchor_gate(results, corpus, anchor_choices)
        if not passed:
            rejected.append(
                _rejection(
                    "anchor_gate_failed",
                    source,
                    details={"event_key": event.get("key"), **gate},
                )
            )
            continue
        for row in results:
            parsed = _real_idx(row.get("idx"), corpora)
            if parsed is None:
                continue
            idx = normalize_space(row.get("idx"))
            item = legacy.get(parsed)
            if item is None:
                rejected.append(_rejection("legacy_item_missing", source, idx=idx))
                continue
            choice = normalize_space(row.get("choice"))
            confidence = normalize_space(row.get("confidence")) or None
            if choice == "ABSTAIN":
                accepted.append(
                    RawLabel(
                        *parsed,
                        decision="ABSTAIN_LEGACY",
                        legacy_choice=choice,
                        confidence=confidence,
                        label_source=source,
                        candidate_valid=None,
                        notes={
                            "event_key": event.get("key"),
                            "agent_id": event.get("agentId"),
                            "anchor_gate": gate,
                        },
                    )
                )
            elif choice == "NOISE":
                accepted.append(
                    RawLabel(
                        *parsed,
                        decision="NOISE",
                        legacy_choice=choice,
                        confidence=confidence,
                        label_source=source,
                        candidate_valid=None,
                        notes={
                            "event_key": event.get("key"),
                            "agent_id": event.get("agentId"),
                            "anchor_gate": gate,
                        },
                    )
                )
            else:
                valid = normalize_name(choice) in item.candidate_keys
                if not valid:
                    rejected.append(
                        _rejection(
                            "cross_item_candidate_leakage",
                            source,
                            idx=idx,
                            choice=choice,
                            details={"own_candidates": list(item.candidate_names)},
                        )
                    )
                    continue
                accepted.append(
                    RawLabel(
                        *parsed,
                        decision="MATCH",
                        legacy_choice=choice,
                        confidence=confidence,
                        label_source=source,
                        candidate_valid=True,
                        notes={
                            "event_key": event.get("key"),
                            "agent_id": event.get("agentId"),
                            "anchor_gate": gate,
                        },
                    )
                )
    return accepted, rejected


def parse_audit_journal(
    path: Path,
    legacy: Mapping[tuple[str, int], LegacyItem],
) -> tuple[list[RawLabel], list[dict[str, Any]]]:
    source = "sonnet_audit"
    accepted: list[RawLabel] = []
    rejected: list[dict[str, Any]] = []
    corpora = {key[0] for key in legacy}
    for event, results in _journal_result_groups(path):
        corpus = _infer_group_corpus(results, corpora)
        if corpus is None:
            rejected.append(_rejection("mixed_or_missing_group_corpus", source, details=event.get("key")))
            continue
        passed, gate = _audit_anchor_gate(results)
        if not passed:
            rejected.append(
                _rejection(
                    "anchor_gate_failed",
                    source,
                    details={"event_key": event.get("key"), **gate},
                )
            )
            continue
        for row in results:
            parsed = _real_idx(row.get("idx"), corpora)
            if parsed is None:
                continue
            idx = normalize_space(row.get("idx"))
            item = legacy.get(parsed)
            if item is None:
                rejected.append(_rejection("legacy_item_missing", source, idx=idx))
                continue
            metadata = {
                "event_key": event.get("key"),
                "agent_id": event.get("agentId"),
                "anchor_gate": gate,
                "confidence_basis": "not_collected_by_audit_schema",
                "top1_fit": row.get("top1_fit"),
                "best_rank": row.get("best_rank"),
                "better_bank_metric": row.get("better_bank_metric"),
                "is_preference": row.get("is_preference"),
            }
            if row.get("is_preference") is False:
                accepted.append(
                    RawLabel(
                        *parsed,
                        decision="NOISE",
                        legacy_choice="NOISE",
                        confidence=None,
                        label_source=source,
                        candidate_valid=None,
                        notes=metadata,
                    )
                )
                continue

            better = normalize_space(row.get("better_bank_metric"))
            rank = row.get("best_rank")
            if better:
                choice = better
                candidate_valid = normalize_name(choice) in item.candidate_keys
                metadata["selection_basis"] = "audit_full_bank_correction"
                confidence = None
            elif isinstance(rank, int) and 1 <= rank <= len(item.candidate_names):
                choice = item.candidate_names[rank - 1]
                candidate_valid = True
                metadata["selection_basis"] = "audit_best_rank"
                confidence = None
            elif rank is None and row.get("top1_fit") == "wrong":
                accepted.append(
                    RawLabel(
                        *parsed,
                        decision="ABSTAIN_LEGACY",
                        legacy_choice="ABSTAIN",
                        confidence=None,
                        label_source=source,
                        candidate_valid=None,
                        notes={**metadata, "selection_basis": "audit_no_candidate_fit"},
                    )
                )
                continue
            else:
                rejected.append(
                    _rejection("inconsistent_audit_decision", source, idx=idx, details=metadata)
                )
                continue
            accepted.append(
                RawLabel(
                    *parsed,
                    decision="MATCH",
                    legacy_choice=choice,
                    confidence=confidence,
                    label_source=source,
                    candidate_valid=candidate_valid,
                    notes=metadata,
                )
            )
    return accepted, rejected


def parse_rescue_journal(
    path: Path,
    legacy: Mapping[tuple[str, int], LegacyItem],
    rescue_anchors: Mapping[tuple[str, str], str],
) -> tuple[list[RawLabel], list[dict[str, Any]]]:
    source = "sonnet_rescue"
    accepted: list[RawLabel] = []
    rejected: list[dict[str, Any]] = []
    corpora = {key[0] for key in legacy}
    for event, results in _journal_result_groups(path):
        corpus = _infer_group_corpus(results, corpora)
        if corpus is None:
            rejected.append(_rejection("mixed_or_missing_group_corpus", source, details=event.get("key")))
            continue
        passed, gate = _rescue_anchor_gate(results, corpus, rescue_anchors)
        if not passed:
            rejected.append(
                _rejection(
                    "anchor_gate_failed",
                    source,
                    details={"event_key": event.get("key"), **gate},
                )
            )
            continue
        for row in results:
            parsed = _real_idx(row.get("idx"), corpora)
            if parsed is None:
                continue
            idx = normalize_space(row.get("idx"))
            item = legacy.get(parsed)
            if item is None:
                rejected.append(_rejection("legacy_item_missing", source, idx=idx))
                continue
            verdict = row.get("verdict")
            metadata = {
                "event_key": event.get("key"),
                "agent_id": event.get("agentId"),
                "anchor_gate": gate,
                "confidence_basis": "not_collected_by_rescue_schema",
                "rescue_verdict": verdict,
                "proposed_name": row.get("proposed_name"),
            }
            if verdict == "found" and normalize_space(row.get("metric")):
                choice = normalize_space(row.get("metric"))
                accepted.append(
                    RawLabel(
                        *parsed,
                        decision="MATCH",
                        legacy_choice=choice,
                        confidence=None,
                        label_source=source,
                        candidate_valid=normalize_name(choice) in item.candidate_keys,
                        notes=metadata,
                    )
                )
            elif verdict == "not_in_bank" and not row.get("metric"):
                accepted.append(
                    RawLabel(
                        *parsed,
                        decision="BANK_GAP",
                        legacy_choice=None,
                        confidence=None,
                        label_source=source,
                        candidate_valid=None,
                        notes=metadata,
                    )
                )
            else:
                rejected.append(
                    _rejection("inconsistent_rescue_decision", source, idx=idx, details=metadata)
                )
    return accepted, rejected


def validate_and_bridge(
    labels: Iterable[RawLabel],
    legacy: Mapping[tuple[str, int], LegacyItem],
    canonical: Mapping[str, Sequence[CanonicalNorm]],
    bridges: Mapping[str, BankBridge],
) -> tuple[list[RawLabel], list[dict[str, Any]]]:
    accepted: list[RawLabel] = []
    rejected: list[dict[str, Any]] = []
    # Resolution still checks quote equality and row/source-ID tiebreakers, but
    # index the frozen universe once instead of scanning a 20K-row corpus for
    # every teacher label.
    canonical_by_text: dict[str, dict[str, list[CanonicalNorm]]] = {}
    for corpus, rows in canonical.items():
        by_text: dict[str, list[CanonicalNorm]] = defaultdict(list)
        for row in rows:
            by_text[normalize_space(row.norm)].append(row)
        canonical_by_text[corpus] = by_text
    for label in labels:
        idx = f"{label.corpus}-{label.legacy_row}"
        item = legacy[(label.corpus, label.legacy_row)]
        candidates = canonical_by_text.get(label.corpus, {}).get(normalize_space(item.norm), ())
        norm = resolve_canonical_norm(item, candidates)
        if norm is None:
            rejected.append(
                _rejection(
                    "canonical_norm_unresolved",
                    label.label_source,
                    idx=idx,
                    choice=label.legacy_choice,
                    details={"norm": item.norm},
                )
            )
            continue
        bridge = bridges.get(norm.task)
        if bridge is None:
            rejected.append(_rejection("current_bank_missing", label.label_source, idx=idx))
            continue
        label.canonical = norm
        label.bank_sha256 = bridge.source_sha256
        if label.decision == "MATCH":
            metric, error = bridge.resolve(label.legacy_choice or "")
            if error:
                details: dict[str, Any] = {"task": norm.task}
                if error == "ambiguous_current_bank_name":
                    details["metric_ids"] = [
                        metric.metric_id
                        for metric in bridge.by_name.get(normalize_name(label.legacy_choice), ())
                    ]
                rejected.append(
                    _rejection(
                        error,
                        label.label_source,
                        idx=idx,
                        choice=label.legacy_choice,
                        details=details,
                    )
                )
                continue
            assert metric is not None
            label.metric_id = metric.metric_id
            label.bridge_method = "normalized_current_name_unique"
            label.notes["current_metric_name"] = metric.name
        else:
            label.bridge_method = "not_applicable"
        label.notes["legacy_row"] = label.legacy_row
        accepted.append(label)
    return accepted, rejected


def _label_signature(label: RawLabel) -> tuple[str, str | None]:
    return label.decision, label.metric_id


def dedupe_and_select(
    labels: Iterable[RawLabel],
) -> tuple[list[RawLabel], list[dict[str, Any]]]:
    rejected: list[dict[str, Any]] = []
    per_source: dict[tuple[str, int, str], list[RawLabel]] = defaultdict(list)
    for label in labels:
        assert label.canonical is not None
        per_source[(label.corpus, label.canonical.row, label.label_source)].append(label)

    clean: list[RawLabel] = []
    for key, group in per_source.items():
        signatures = {_label_signature(label) for label in group}
        if len(signatures) > 1:
            corpus, row, source = key
            rejected.append(
                _rejection(
                    "conflicting_duplicate_labels",
                    source,
                    idx=f"{corpus}-{group[0].legacy_row}",
                    details={"canonical_row": row, "signatures": sorted(map(str, signatures))},
                )
            )
            continue
        # Preserve the strongest explicit confidence when duplicate workflow
        # events agree; duplicates never count as independent teacher votes.
        chosen = sorted(
            group,
            key=lambda label: {"high": 2, "low": 1, None: 0}.get(label.confidence, 0),
            reverse=True,
        )[0]
        chosen.notes["agreeing_duplicate_events"] = len(group)
        clean.append(chosen)

    by_norm: dict[str, list[RawLabel]] = defaultdict(list)
    for label in clean:
        assert label.canonical is not None
        by_norm[label.canonical.norm_uid].append(label)

    selected = []
    for group in by_norm.values():
        group.sort(
            key=lambda label: (
                SOURCE_PRIORITY[label.label_source],
                {"high": 2, "low": 1, None: 0}.get(label.confidence, 0),
            ),
            reverse=True,
        )
        winner = group[0]
        winner.notes["superseded_labels"] = [
            {
                "label_source": label.label_source,
                "decision": label.decision,
                "metric_id": label.metric_id,
                "confidence": label.confidence,
            }
            for label in group[1:]
        ]
        selected.append(winner)
    selected.sort(key=lambda label: (label.corpus, label.canonical.row if label.canonical else -1))
    return selected, rejected


def teacher_record(label: RawLabel) -> dict[str, Any]:
    assert label.canonical is not None
    return {
        "schema_version": SCHEMA_VERSION,
        "norm_uid": label.canonical.norm_uid,
        "corpus": label.corpus,
        "task": label.canonical.task,
        "row": label.canonical.row,
        "decision": label.decision,
        "metric_id": label.metric_id,
        "current_bank_source_sha256": label.bank_sha256,
        "confidence": label.confidence,
        "label_source": label.label_source,
        "legacy_choice": label.legacy_choice,
        "legacy_candidate_valid": label.candidate_valid,
        "bridge_method": label.bridge_method,
        "notes": label.notes,
    }


def export(args: argparse.Namespace) -> dict[str, Any]:
    manifest_root = Path(args.manifest_root)
    output_root = Path(args.output_root)
    v1_root = Path(args.v1_root)
    scratch_root = Path(args.scratch_root)
    canonical, bridges, routing = load_manifest_universe(manifest_root)
    legacy = load_legacy_items(v1_root, routing)
    if not legacy:
        raise ValueError(f"no legacy joined candidates found under {v1_root}")

    full_anchors = load_anchor_choices(scratch_root / "rematch_full")
    pilot_anchors = load_anchor_choices(scratch_root / "rematch_pilot")
    rescue_anchors = load_rescue_anchors(scratch_root / "abstain_rescue")
    sources = {
        "sonnet_full": Path(args.full_journal),
        "sonnet_pilot": Path(args.pilot_journal),
        "sonnet_audit": Path(args.audit_journal),
        "sonnet_rescue": Path(args.rescue_journal),
    }
    raw: list[RawLabel] = []
    rejections: list[dict[str, Any]] = []
    for source, anchors in (
        ("sonnet_full", full_anchors),
        ("sonnet_pilot", pilot_anchors),
    ):
        labels, bad = parse_choice_journal(sources[source], source, legacy, anchors)
        raw.extend(labels)
        rejections.extend(bad)
    labels, bad = parse_audit_journal(sources["sonnet_audit"], legacy)
    raw.extend(labels)
    rejections.extend(bad)
    labels, bad = parse_rescue_journal(sources["sonnet_rescue"], legacy, rescue_anchors)
    raw.extend(labels)
    rejections.extend(bad)

    bridged, bad = validate_and_bridge(raw, legacy, canonical, bridges)
    rejections.extend(bad)
    selected, bad = dedupe_and_select(bridged)
    rejections.extend(bad)

    teacher_path = output_root / "teachers/sonnet.jsonl"
    rejection_path = output_root / "teachers/sonnet_rejections.jsonl"
    write_jsonl(teacher_path, (teacher_record(label) for label in selected))
    rejections.sort(
        key=lambda row: (
            str(row.get("label_source") or ""),
            str(row.get("corpus") or ""),
            int(row.get("legacy_row") or -1),
            str(row.get("reason") or ""),
        )
    )
    write_jsonl(rejection_path, rejections)

    return {
        "teacher_path": str(teacher_path),
        "rejection_path": str(rejection_path),
        "raw_labels": len(raw),
        "bridged_labels": len(bridged),
        "teachers": len(selected),
        "by_source": dict(sorted(Counter(label.label_source for label in selected).items())),
        "by_decision": dict(sorted(Counter(label.decision for label in selected).items())),
        "rejections": len(rejections),
        "rejections_by_reason": dict(sorted(Counter(row["reason"] for row in rejections).items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--v1-root", default=str(DEFAULT_V1_ROOT))
    parser.add_argument("--scratch-root", default=str(DEFAULT_SCRATCH_ROOT))
    parser.add_argument("--full-journal", default=str(DEFAULT_JOURNALS["sonnet_full"]))
    parser.add_argument("--pilot-journal", default=str(DEFAULT_JOURNALS["sonnet_pilot"]))
    parser.add_argument("--audit-journal", default=str(DEFAULT_JOURNALS["sonnet_audit"]))
    parser.add_argument("--rescue-journal", default=str(DEFAULT_JOURNALS["sonnet_rescue"]))
    return parser.parse_args()


def main() -> None:
    print(json.dumps(export(parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
