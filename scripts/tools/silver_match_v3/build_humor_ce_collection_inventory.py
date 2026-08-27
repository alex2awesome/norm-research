#!/usr/bin/env python3
"""Build a truth-hidden priority inventory for expanded Humor CE labeling.

The output is a selector input, not a label artifact.  Legacy Sonnet decisions
are used only to enrich the train/diagnostic sampling pool and are deliberately
not copied into frozen label-pack items.  Blind-role selection must ignore all
priority fields and sample from the canonical norm universe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .common import read_jsonl, sha256_file, write_jsonl


TASK = "humor"
BANK_SOURCE_SHA256 = (
    "1b4a29d34b4ef4d999e0cb0b2d1125286372349ff6dfa21a6adc5bc8e76f0de9"
)
LEGACY_NONMATCH = {"BANK_GAP", "NOISE", "ABSTAIN_LEGACY"}


def canonical_source_group(row: dict[str, Any]) -> str:
    """Canonicalize the strongest document identity without delimiter drift."""

    task = str(row.get("task") or TASK).strip()
    corpus = str(row.get("corpus") or "").strip()
    if row.get("paper_id"):
        kind, identity = "paper", str(row["paper_id"]).strip()
    elif row.get("source_id"):
        kind, identity = "source", str(row["source_id"]).strip()
    else:
        kind, identity = "norm", str(row.get("norm_uid") or "").strip()
    if not corpus or not identity:
        raise ValueError("canonical norm lacks corpus/source identity")
    return ":".join((task, corpus, kind, identity))


def load_unique(path: Path, *, task_only: bool = True) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if task_only and row.get("task") != TASK:
            continue
        uid = str(row.get("norm_uid") or "")
        if not uid or uid in rows:
            raise ValueError(f"missing/duplicate norm_uid in {path}: {uid!r}")
        rows[uid] = row
    return rows


def candidate_top_ids(
    paths: Iterable[Path], expected_uids: set[str]
) -> tuple[dict[str, list[str]], dict[str, str]]:
    top_ids: dict[str, list[str]] = {uid: [] for uid in expected_uids}
    hashes: dict[str, str] = {}
    for path in paths:
        hashes[str(path)] = sha256_file(path)
        seen: set[str] = set()
        for row in read_jsonl(path):
            uid = str(row.get("norm_uid") or "")
            if uid not in expected_uids:
                continue
            if uid in seen:
                raise ValueError(f"duplicate candidate UID in {path}: {uid}")
            seen.add(uid)
            if row.get("bank_source_sha256") != BANK_SOURCE_SHA256:
                raise ValueError(f"candidate bank drift in {path}: {uid}")
            candidates = row.get("candidates") or []
            if not candidates:
                raise ValueError(f"candidate row is empty in {path}: {uid}")
            top_ids[uid].append(str(candidates[0]["metric_id"]))
        if seen != expected_uids:
            missing = sorted(expected_uids - seen)
            raise ValueError(f"candidate input misses canonical UIDs: {path}: {missing[:3]}")
    return top_ids, hashes


def priority_strata(
    legacy: dict[str, Any] | None,
    top_ids: list[str],
    uncovered: set[str],
) -> list[str]:
    strata: list[str] = []
    if legacy:
        decision = str(legacy.get("decision") or "")
        confidence = legacy.get("confidence")
        if decision in LEGACY_NONMATCH:
            strata.append("legacy_nonmatch_re_adjudication")
        if confidence in (None, "", "low"):
            strata.append("sonnet_low_or_null_confidence")
    if len(set(top_ids)) > 1:
        strata.append("retriever_top1_disagreement")
    if any(metric_id in uncovered for metric_id in top_ids):
        strata.append("uncovered_leaf_proxy")
    if not strata:
        strata.append("natural_background")
    return strata


def build(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    norms_path = Path(args.norms).resolve()
    sonnet_path = Path(args.sonnet).resolve()
    candidate_paths = [Path(value).resolve() for value in args.candidates]
    norms = load_unique(norms_path)
    if len(norms) != args.expected_norms:
        raise ValueError(f"expected {args.expected_norms} Humor norms, found {len(norms)}")
    sonnet = load_unique(sonnet_path)
    if any(
        row.get("current_bank_source_sha256") != BANK_SOURCE_SHA256
        for row in sonnet.values()
    ):
        raise ValueError("raw Sonnet inventory contains a foreign Humor bank")
    if set(sonnet) - set(norms):
        raise ValueError("raw Sonnet inventory contains noncanonical Humor UIDs")
    top_ids, candidate_hashes = candidate_top_ids(candidate_paths, set(norms))
    uncovered = {str(value) for value in args.uncovered_metric_id}

    rows: list[dict[str, Any]] = []
    strata_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    for uid in sorted(norms):
        norm = norms[uid]
        legacy = sonnet.get(uid)
        strata = priority_strata(legacy, top_ids[uid], uncovered)
        strata_counts.update(strata)
        primary_counts[strata[0]] += 1
        rows.append(
            {
                "schema_version": "silver-match-v3-humor-ce-collection-candidate-v1",
                "task": TASK,
                "corpus": str(norm["corpus"]),
                "norm_uid": uid,
                "source_group": canonical_source_group(norm),
                "priority_strata": strata,
                "primary_priority_stratum": strata[0],
                "retriever_top1_metric_ids": top_ids[uid],
                "retriever_top1_unique_count": len(set(top_ids[uid])),
                "legacy_inventory_present": legacy is not None,
                "legacy_decision_for_selection_only": (
                    str(legacy.get("decision") or "") if legacy else None
                ),
                "legacy_confidence_for_selection_only": (
                    legacy.get("confidence") if legacy else None
                ),
                "selection_tiebreak": hashlib.sha256(
                    f"humor-ce-collection-v1\x1f{uid}".encode()
                ).hexdigest(),
            }
        )
    report = {
        "schema_version": "silver-match-v3-humor-ce-collection-inventory-report-v1",
        "status": "READY_FOR_ROLE_FREEZE_NOT_LABEL_TRUTH",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": TASK,
        "bank_source_sha256": BANK_SOURCE_SHA256,
        "count": len(rows),
        "sonnet_rows": len(sonnet),
        "legacy_decisions": dict(
            sorted(Counter(str(row.get("decision") or "NULL") for row in sonnet.values()).items())
        ),
        "legacy_confidences": dict(
            sorted(Counter(str(row.get("confidence") or "NULL") for row in sonnet.values()).items())
        ),
        "priority_strata": dict(sorted(strata_counts.items())),
        "primary_priority_strata": dict(sorted(primary_counts.items())),
        "uncovered_metric_ids": sorted(uncovered),
        "blind_role_contract": "ignore every priority/legacy/retriever field; natural corpus sample only",
        "inputs": {
            "norms": {"path": str(norms_path), "sha256": sha256_file(norms_path)},
            "sonnet": {"path": str(sonnet_path), "sha256": sha256_file(sonnet_path)},
            "candidates": candidate_hashes,
        },
    }
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--norms", required=True)
    parser.add_argument("--sonnet", required=True)
    parser.add_argument("--candidates", action="append", required=True)
    parser.add_argument("--uncovered-metric-id", action="append", default=[])
    parser.add_argument("--expected-norms", type=int, default=77378)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    output, report_path = Path(args.output).resolve(), Path(args.report).resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite frozen collection inventory")
    rows, report = build(args)
    write_jsonl(output, rows)
    report["output"] = {
        "path": str(output),
        "sha256": sha256_file(output),
        "count": len(rows),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**report, "report_sha256": sha256_file(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
