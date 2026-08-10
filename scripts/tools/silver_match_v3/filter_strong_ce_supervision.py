#!/usr/bin/env python3
"""Freeze strong exact/typed CE supervision and exclude evaluation groups.

Weak forced-top-k labels are useful retrieval hints, but they are not exact
metric truth.  This filter therefore permits only explicit non-weak ``MATCH``
rows with a current-bank leaf or explicit typed nonmatches.  It recomputes
canonical source groups from the frozen manifest and removes every source
group represented in the supplied future-dev references.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .adjudicate_gemma import DECISIONS
from .common import normalize_space, read_jsonl, sha256_file, write_jsonl
from .filter_teacher_train_by_reference_groups import (
    _validate_row,
    load_task_universe,
)


def is_weak_forced(row: Mapping[str, Any]) -> bool:
    return (
        normalize_space(row.get("supervision_strength"))
        in {"weak_forced_positive", "weak_forced_top3"}
        or normalize_space(row.get("label_source")) == "sonnet_forced_top3"
    )


def filter_strong_rows(
    *,
    teacher_inputs: Sequence[tuple[str, Iterable[Mapping[str, Any]]]],
    reference_inputs: Sequence[tuple[str, Iterable[Mapping[str, Any]]]],
    norms: Mapping[str, Mapping[str, Any]],
    task: str,
    bank_hash: str,
    bank_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    excluded_groups: set[str] = set()
    excluded_uids: set[str] = set()
    reference_counts: Counter[str] = Counter()
    for source, rows in reference_inputs:
        for row in rows:
            uid, group = _validate_row(
                row,
                source=source,
                task=task,
                norms=norms,
                bank_hash=bank_hash,
            )
            excluded_uids.add(uid)
            excluded_groups.add(group)
            reference_counts[source] += 1

    selected_by_uid: dict[str, dict[str, Any]] = {}
    selected_source: dict[str, str] = {}
    counts: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    label_sources: Counter[str] = Counter()
    for source, rows in teacher_inputs:
        for row in rows:
            counts["input_rows"] += 1
            uid, group = _validate_row(
                row,
                source=source,
                task=task,
                norms=norms,
                bank_hash=bank_hash,
            )
            if group in excluded_groups:
                counts["excluded_future_dev_source_group"] += 1
                continue
            if is_weak_forced(row):
                counts["excluded_weak_forced"] += 1
                continue
            decision = normalize_space(row.get("decision"))
            metric_id = normalize_space(row.get("metric_id"))
            if decision not in DECISIONS:
                raise ValueError(f"{source}: unsupported decision for {uid}: {decision!r}")
            if decision == "MATCH":
                if metric_id not in bank_ids:
                    raise ValueError(f"{source}: MATCH metric outside current bank: {uid}")
            elif metric_id:
                raise ValueError(f"{source}: typed nonmatch carries metric ID: {uid}")
            if uid in selected_by_uid:
                previous = selected_by_uid[uid]
                if (
                    normalize_space(previous.get("decision")) != decision
                    or normalize_space(previous.get("metric_id")) != metric_id
                ):
                    raise ValueError(
                        f"conflicting strong supervision for {uid}: "
                        f"{selected_source[uid]} vs {source}"
                    )
                counts["deduplicated_identical_strong"] += 1
                continue
            rendered = dict(row)
            rendered.update(
                {
                    "task": task,
                    "corpus": norms[uid]["corpus"],
                    "source_group": group,
                    "current_bank_source_sha256": bank_hash,
                    "ce_supervision_policy": "strong_exact_or_typed_nonmatch_only",
                    "ce_weak_forced_positive": False,
                }
            )
            selected_by_uid[uid] = rendered
            selected_source[uid] = source
            decisions[decision] += 1
            label_sources[normalize_space(row.get("label_source")) or "UNSPECIFIED"] += 1

    output = [selected_by_uid[uid] for uid in sorted(selected_by_uid)]
    if not output:
        raise ValueError("strong-supervision filter retained no rows")
    output_groups = {str(row["source_group"]) for row in output}
    overlap = output_groups & excluded_groups
    if overlap:
        raise AssertionError(f"future-dev source groups remain: {sorted(overlap)[:3]}")
    audit = {
        "input_rows": counts["input_rows"],
        "output_rows": len(output),
        "output_source_groups": len(output_groups),
        "output_metric_coverage": len(
            {
                str(row["metric_id"])
                for row in output
                if row.get("decision") == "MATCH"
            }
        ),
        "decision_counts": dict(sorted(decisions.items())),
        "label_source_counts": dict(sorted(label_sources.items())),
        "exclusions": dict(
            sorted(
                (key, value)
                for key, value in counts.items()
                if key not in {"input_rows"}
            )
        ),
        "reference_rows": sum(reference_counts.values()),
        "reference_source_groups": len(excluded_groups),
        "reference_unique_uids": len(excluded_uids),
        "output_reference_uid_overlap": len(set(selected_by_uid) & excluded_uids),
        "output_reference_source_group_overlap": len(overlap),
        "weak_forced_rows_used_as_exact_positives": 0,
    }
    return output, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--teacher", action="append", required=True)
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    teacher_paths = [Path(path).resolve() for path in args.teacher]
    reference_paths = [Path(path).resolve() for path in args.reference]
    output_path = Path(args.output).resolve()
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if output_path.exists() or meta_path.exists():
        raise FileExistsError(f"immutable output already exists: {output_path}")

    norms, bank_hash, bank_path = load_task_universe(manifest_path, args.task)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))["metrics"]
    rows, audit = filter_strong_rows(
        teacher_inputs=[(str(path), read_jsonl(path)) for path in teacher_paths],
        reference_inputs=[(str(path), read_jsonl(path)) for path in reference_paths],
        norms=norms,
        task=args.task,
        bank_hash=bank_hash,
        bank_ids={str(row["metric_id"]) for row in bank},
    )
    write_jsonl(output_path, rows)
    meta = {
        "schema_version": "silver-match-v3-strong-ce-supervision-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "bank": {
            "path": str(bank_path),
            "sha256": sha256_file(bank_path),
            "source_sha256": bank_hash,
        },
        "teachers": {str(path): sha256_file(path) for path in teacher_paths},
        "references": {str(path): sha256_file(path) for path in reference_paths},
        "audit": audit,
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
    }
    meta_path.write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({**meta, "meta_sha256": sha256_file(meta_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
