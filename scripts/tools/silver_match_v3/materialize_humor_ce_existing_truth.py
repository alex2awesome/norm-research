#!/usr/bin/env python3
"""Freeze the existing Humor CE truth behind one canonical split contract.

The recovered MATCH teacher file contains a small number of stale embedded
``split`` values.  Its separately frozen source-disjoint assignment artifact
is authoritative.  The older typed bridge rows use a legacy three-part source
group rendering.  This materializer joins both sources to the canonical norm
corpus, applies the authoritative teacher assignments, and emits a single
append-only truth file that the generic CE pair builder can consume without a
partial split-assignment join.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .common import normalize_space, read_jsonl, sha256_file, write_jsonl
from .train_nemotron_lora import source_group_key


SCHEMA = "silver-match-v3-humor-ce-existing-truth-v1"
REPORT_SCHEMA = "silver-match-v3-humor-ce-existing-truth-report-v1"


def _source_group(value: Any) -> str:
    """Strip edges without collapsing the unit-separator structure."""

    return str(value or "").strip()


def _ref(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _index(path: Path, label: str) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    values: dict[str, dict[str, Any]] = {}
    for row in rows:
        uid = normalize_space(row.get("norm_uid"))
        if not uid or uid in values:
            raise ValueError(f"{label} has missing/duplicate norm_uid: {uid!r}")
        values[uid] = row
    if not values:
        raise ValueError(f"{label} is empty")
    return values


def _acceptable(row: Mapping[str, Any]) -> set[str]:
    raw = row.get("acceptable_metric_ids")
    if raw is None:
        values: set[str] = set()
    elif isinstance(raw, str):
        values = {normalize_space(raw)}
    elif isinstance(raw, (list, tuple, set)):
        values = {normalize_space(value) for value in raw}
    else:
        raise ValueError("acceptable_metric_ids must be a string or sequence")
    if normalize_space(row.get("decision")) == "MATCH":
        values.add(normalize_space(row.get("metric_id")))
    values.discard("")
    return values


def _load_norm_subset(
    path: Path, wanted: set[str]
) -> dict[str, dict[str, Any]]:
    norms: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        uid = normalize_space(row.get("norm_uid"))
        if uid not in wanted:
            continue
        if uid in norms:
            raise ValueError(f"canonical norms duplicate requested UID: {uid}")
        norms[uid] = row
    missing = sorted(wanted - set(norms))
    if missing:
        raise ValueError(f"canonical norms miss {len(missing)} UIDs: {missing[:5]}")
    return norms


def build(
    *,
    teacher_path: Path,
    assignment_path: Path,
    typed_paths: Iterable[Path],
    norms_path: Path,
    task: str,
    bank_hash: str,
    exclude_unanchored_family_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    teachers = _index(teacher_path, "teacher truth")
    assignments = _index(assignment_path, "teacher assignments")
    if set(teachers) != set(assignments):
        raise ValueError(
            "teacher truth and assignment UID universes differ: "
            f"truth_only={len(set(teachers)-set(assignments))}, "
            f"assignment_only={len(set(assignments)-set(teachers))}"
        )
    typed: dict[str, dict[str, Any]] = {}
    typed_refs: list[dict[str, Any]] = []
    for path in typed_paths:
        typed_refs.append(_ref(path))
        for uid, row in _index(path, f"typed truth {path.name}").items():
            if uid in typed or uid in teachers:
                raise ValueError(f"truth UID overlaps sources: {uid}")
            typed[uid] = row
    norms = _load_norm_subset(norms_path, set(teachers) | set(typed))

    rows: list[dict[str, Any]] = []
    stale_split_counts: Counter[str] = Counter()
    acceptable_expansions = 0
    for uid, teacher in teachers.items():
        assignment = assignments[uid]
        if teacher.get("task") != task or assignment.get("task") != task:
            raise ValueError(f"teacher task mismatch: {uid}")
        if normalize_space(teacher.get("metric_id")) != normalize_space(
            assignment.get("metric_id")
        ):
            raise ValueError(f"teacher/assignment metric mismatch: {uid}")
        supplied_hash = normalize_space(
            teacher.get("current_bank_source_sha256")
            or teacher.get("bank_source_sha256")
        )
        if supplied_hash != bank_hash:
            raise ValueError(f"teacher bank hash mismatch: {uid}")
        canonical_group = source_group_key(norms[uid])
        if _source_group(assignment.get("source_group")) != canonical_group:
            raise ValueError(f"assignment/canonical source group mismatch: {uid}")
        split = normalize_space(assignment.get("split"))
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"invalid teacher assignment split: {uid}")
        embedded_split = normalize_space(teacher.get("split"))
        if embedded_split and embedded_split != split:
            stale_split_counts[f"{embedded_split}->{split}"] += 1
        acceptable = _acceptable(teacher) | _acceptable(assignment)
        if acceptable != _acceptable(teacher):
            acceptable_expansions += 1
        rendered = dict(teacher)
        rendered.update(
            {
                "schema_version": SCHEMA,
                "task": task,
                "corpus": norms[uid]["corpus"],
                "decision": "MATCH",
                "metric_id": normalize_space(teacher.get("metric_id")),
                "acceptable_metric_ids": sorted(acceptable),
                "source_group": canonical_group,
                "split": split,
                "current_bank_source_sha256": bank_hash,
                "truth_source": "recovered_teacher_match",
            }
        )
        rows.append(rendered)

    excluded_unanchored_family_only = 0
    for uid, row in typed.items():
        if row.get("task") != task:
            raise ValueError(f"typed truth task mismatch: {uid}")
        supplied_hash = normalize_space(
            row.get("current_bank_source_sha256") or row.get("bank_source_sha256")
        )
        if supplied_hash != bank_hash:
            raise ValueError(f"typed truth bank hash mismatch: {uid}")
        split = normalize_space(row.get("split"))
        if split != "train":
            raise ValueError(f"existing typed truth is not train-only: {uid}")
        if normalize_space(row.get("decision")) == "MATCH_FAMILY_ONLY":
            anchors = _acceptable(row)
            raw_family = row.get("family_metric_ids")
            if isinstance(raw_family, str):
                anchors.add(normalize_space(raw_family))
            elif isinstance(raw_family, (list, tuple, set)):
                anchors.update(normalize_space(value) for value in raw_family)
            anchors.discard("")
            if not anchors and exclude_unanchored_family_only:
                excluded_unanchored_family_only += 1
                continue
        rendered = dict(row)
        rendered.update(
            {
                "schema_version": SCHEMA,
                "task": task,
                "corpus": norms[uid]["corpus"],
                "source_group": source_group_key(norms[uid]),
                "split": split,
                "current_bank_source_sha256": bank_hash,
                "truth_source": "strict_typed_bridge",
            }
        )
        rows.append(rendered)

    rows.sort(key=lambda row: normalize_space(row["norm_uid"]))
    uid_count = len({normalize_space(row["norm_uid"]) for row in rows})
    if uid_count != len(rows):
        raise AssertionError("combined truth contains duplicate UIDs")
    groups: dict[str, set[str]] = {}
    group_sources: dict[str, set[str]] = {}
    for row in rows:
        group = _source_group(row["source_group"])
        groups.setdefault(group, set()).add(normalize_space(row["split"]))
        group_sources.setdefault(group, set()).add(str(row["truth_source"]))
    crossed_splits = {group: splits for group, splits in groups.items() if len(splits) > 1}
    crossed_sources = {
        group: sources for group, sources in group_sources.items() if len(sources) > 1
    }
    if crossed_splits or crossed_sources:
        raise ValueError(
            "combined truth violates source-disjoint contract: "
            f"crossed_splits={len(crossed_splits)}, "
            f"teacher_typed_overlap={len(crossed_sources)}"
        )
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "CANONICAL_EXISTING_TRUTH_READY",
        "task": task,
        "bank_source_sha256": bank_hash,
        "truth_rows": len(rows),
        "teacher_rows": len(teachers),
        "typed_rows": len(typed),
        "typed_rows_emitted": len(typed) - excluded_unanchored_family_only,
        "typed_unanchored_family_only_excluded": excluded_unanchored_family_only,
        "unique_source_groups": len(groups),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "decision_counts": dict(
            sorted(Counter(row["decision"] for row in rows).items())
        ),
        "truth_source_counts": dict(
            sorted(Counter(row["truth_source"] for row in rows).items())
        ),
        "embedded_stale_split_overrides": dict(sorted(stale_split_counts.items())),
        "teacher_acceptable_set_expansions_from_assignments": acceptable_expansions,
        "source_groups_crossing_splits": 0,
        "source_groups_crossing_teacher_typed_sources": 0,
        "inputs": {
            "teacher": _ref(teacher_path),
            "assignments": _ref(assignment_path),
            "typed": typed_refs,
            "canonical_norms": _ref(norms_path),
        },
    }
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--typed", action="append", required=True)
    parser.add_argument("--norms", required=True)
    parser.add_argument("--task", default="humor")
    parser.add_argument("--bank-hash", required=True)
    parser.add_argument(
        "--exclude-unanchored-family-only",
        action="store_true",
        help=(
            "exclude family-only labels that carry no metric/family anchor; "
            "they cannot define FAMILY versus REJECT pairs"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    if output_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite frozen truth output/report")
    rows, report = build(
        teacher_path=Path(args.teacher).resolve(),
        assignment_path=Path(args.assignments).resolve(),
        typed_paths=[Path(path).resolve() for path in args.typed],
        norms_path=Path(args.norms).resolve(),
        task=args.task,
        bank_hash=args.bank_hash,
        exclude_unanchored_family_only=args.exclude_unanchored_family_only,
    )
    write_jsonl(output_path, rows)
    report["output"] = {**_ref(output_path), "count": len(rows)}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
