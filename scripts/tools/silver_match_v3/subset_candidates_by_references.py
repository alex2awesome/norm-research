#!/usr/bin/env python3
"""Freeze the exact candidate rows needed by multiple teacher/reference files.

Cross-encoder training consumes only the union of its explicit train and dev
UIDs.  Shipping or repeatedly parsing a task's full-corpus candidate artifact
is wasteful and creates room for accidental role drift.  This utility streams
the full artifact once, validates task/bank/K coverage, and writes only the
predeclared union with immutable provenance.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import normalize_space, read_jsonl, sha256_file, write_jsonl


def _resolve(path: str | Path, anchor: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (anchor.parent / value).resolve()


def _bank_hash(manifest_path: Path, task: str) -> tuple[str, Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    meta = manifest.get("banks", {}).get(task)
    if not isinstance(meta, dict):
        raise KeyError(f"task absent from manifest: {task}")
    value = normalize_space(meta.get("source_sha256"))
    if not value:
        raise ValueError(f"bank source hash missing for {task}")
    return value, _resolve(meta["path"], manifest_path)


def reference_uids(
    reference_inputs: Sequence[tuple[str, Iterable[Mapping[str, Any]]]],
    *,
    task: str,
    bank_hash: str,
) -> tuple[set[str], dict[str, int]]:
    wanted: set[str] = set()
    counts: dict[str, int] = {}
    owners: dict[str, str] = {}
    for source, rows in reference_inputs:
        count = 0
        for row in rows:
            uid = normalize_space(row.get("norm_uid"))
            if not uid:
                raise ValueError(f"{source}: reference row without norm_uid")
            previous = owners.get(uid)
            if previous is not None:
                raise ValueError(
                    f"reference UID appears in multiple inputs: {uid} ({previous}, {source})"
                )
            row_task = normalize_space(row.get("task"))
            if row_task and row_task != task:
                raise ValueError(f"{source}: task mismatch for {uid}")
            row_bank = normalize_space(
                row.get("current_bank_source_sha256")
                or row.get("candidate_bank_source_sha256")
                or row.get("bank_source_sha256")
            )
            if row_bank and row_bank != bank_hash:
                raise ValueError(f"{source}: bank hash mismatch for {uid}")
            owners[uid] = source
            wanted.add(uid)
            count += 1
        counts[source] = count
    if not wanted:
        raise ValueError("reference inputs contain no UIDs")
    return wanted, counts


def select_candidate_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    wanted: set[str],
    task: str,
    bank_hash: str,
    expected_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    observed_task_rows = 0
    source_rows = 0
    depths: Counter[int] = Counter()
    for row in rows:
        source_rows += 1
        uid = normalize_space(row.get("norm_uid"))
        if uid not in wanted:
            continue
        if uid in selected:
            raise ValueError(f"candidate source duplicates requested UID: {uid}")
        row_task = normalize_space(row.get("task"))
        if row_task != task:
            raise ValueError(f"candidate task mismatch for {uid}: {row_task!r}")
        row_bank = normalize_space(
            row.get("bank_source_sha256")
            or row.get("current_bank_source_sha256")
        )
        if row_bank != bank_hash:
            raise ValueError(f"candidate bank hash mismatch for {uid}")
        candidates = list(row.get("candidates") or [])
        if len(candidates) < expected_k:
            raise ValueError(
                f"candidate row shorter than K={expected_k}: {uid}/{len(candidates)}"
            )
        metric_ids = [
            normalize_space(value.get("metric_id") if isinstance(value, dict) else value)
            for value in candidates[:expected_k]
        ]
        if any(not value for value in metric_ids) or len(metric_ids) != len(set(metric_ids)):
            raise ValueError(f"candidate top-K is empty/duplicated for {uid}")
        selected[uid] = dict(row)
        observed_task_rows += 1
        depths[len(candidates)] += 1
    missing = sorted(wanted - set(selected))
    if missing:
        raise ValueError(f"candidate source misses {len(missing)} requested UIDs: {missing[:3]}")
    ordered = [selected[uid] for uid in sorted(selected)]
    audit = {
        "source_rows": source_rows,
        "requested_uids": len(wanted),
        "selected_rows": observed_task_rows,
        "candidate_depth_counts": {
            str(depth): count for depth, count in sorted(depths.items())
        },
        "missing_requested_uids": 0,
        "duplicate_requested_uids": 0,
        "all_selected_rows_have_at_least_expected_k": True,
    }
    return ordered, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--reference", action="append", required=True)
    parser.add_argument("--expected-k", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.expected_k < 1:
        parser.error("--expected-k must be positive")

    manifest_path = Path(args.manifest).resolve()
    candidate_path = Path(args.candidates).resolve()
    reference_paths = [Path(path).resolve() for path in args.reference]
    output_path = Path(args.output).resolve()
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if output_path.exists() or meta_path.exists():
        raise FileExistsError(f"immutable candidate subset already exists: {output_path}")

    bank_hash, bank_path = _bank_hash(manifest_path, args.task)
    wanted, reference_counts = reference_uids(
        [(str(path), read_jsonl(path)) for path in reference_paths],
        task=args.task,
        bank_hash=bank_hash,
    )
    rows, audit = select_candidate_rows(
        read_jsonl(candidate_path),
        wanted=wanted,
        task=args.task,
        bank_hash=bank_hash,
        expected_k=args.expected_k,
    )
    write_jsonl(output_path, rows)
    report = {
        "schema_version": "silver-match-v3-candidate-reference-union-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "expected_k": args.expected_k,
        "bank_source_sha256": bank_hash,
        "reference_counts": reference_counts,
        "audit": audit,
        "inputs": {
            "manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "bank": {"path": str(bank_path), "sha256": sha256_file(bank_path)},
            "candidates": {
                "path": str(candidate_path),
                "sha256": sha256_file(candidate_path),
            },
            "references": {
                str(path): sha256_file(path) for path in reference_paths
            },
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "count": len(rows),
        },
    }
    meta_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "meta_sha256": sha256_file(meta_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
