#!/usr/bin/env python3
"""Freeze a canonical teacher file after removing evaluation source groups.

Teacher artifacts produced before a frozen evaluation panel may omit their
``source_group`` field.  Filtering those rows by the fields carried in the
teacher file is therefore insufficient.  This utility treats the v3 manifest
as the source of truth, recomputes every document-level source group, removes
the complete group of every supplied reference row, and emits an immutable
artifact plus a provenance audit.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import normalize_space, read_jsonl, sha256_file, write_jsonl
from .train_nemotron_lora import source_group_key


def _resolve(path: str | Path, anchor: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (anchor.parent / value).resolve()


def load_task_universe(
    manifest_path: Path, task: str
) -> tuple[dict[str, dict[str, Any]], str, Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bank_meta = manifest.get("banks", {}).get(task)
    if not isinstance(bank_meta, dict):
        raise KeyError(f"task absent from manifest: {task}")
    bank_path = _resolve(bank_meta["path"], manifest_path)
    bank_hash = normalize_space(bank_meta.get("source_sha256"))
    if not bank_hash:
        raise ValueError(f"bank source hash missing for {task}")

    norms: dict[str, dict[str, Any]] = {}
    for corpus, meta in sorted(manifest.get("corpora", {}).items()):
        if meta.get("task") != task:
            continue
        for norm in read_jsonl(_resolve(meta["path"], manifest_path)):
            uid = normalize_space(norm.get("norm_uid"))
            if not uid or uid in norms:
                raise ValueError(f"missing/duplicate canonical UID: {uid!r}")
            if normalize_space(norm.get("task")) != task:
                raise ValueError(f"canonical task mismatch for {uid}")
            if normalize_space(norm.get("corpus")) != normalize_space(corpus):
                raise ValueError(f"canonical corpus mismatch for {uid}")
            norms[uid] = norm
    if not norms:
        raise ValueError(f"no canonical norms for task: {task}")
    return norms, bank_hash, bank_path


def _validate_row(
    row: Mapping[str, Any],
    *,
    source: str,
    task: str,
    norms: Mapping[str, Mapping[str, Any]],
    bank_hash: str,
) -> tuple[str, str]:
    uid = normalize_space(row.get("norm_uid"))
    if uid not in norms:
        raise ValueError(f"{source}: UID absent from task manifest: {uid!r}")
    row_task = normalize_space(row.get("task"))
    if row_task and row_task != task:
        raise ValueError(f"{source}: task mismatch for {uid}: {row_task!r}")
    row_bank = normalize_space(
        row.get("current_bank_source_sha256")
        or row.get("candidate_bank_source_sha256")
        or row.get("bank_source_sha256")
    )
    if row_bank and row_bank != bank_hash:
        raise ValueError(f"{source}: bank hash mismatch for {uid}")
    return uid, source_group_key(norms[uid])


def filter_rows(
    *,
    teacher_rows: Iterable[Mapping[str, Any]],
    reference_inputs: Sequence[tuple[str, Iterable[Mapping[str, Any]]]],
    norms: Mapping[str, Mapping[str, Any]],
    task: str,
    bank_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    excluded_uids: set[str] = set()
    excluded_groups: set[str] = set()
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

    output: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    supplied_group_counts: Counter[str] = Counter()
    excluded = Counter()
    input_count = 0
    for row in teacher_rows:
        input_count += 1
        uid, group = _validate_row(
            row,
            source="teacher",
            task=task,
            norms=norms,
            bank_hash=bank_hash,
        )
        if uid in seen_uids:
            raise ValueError(f"teacher: duplicate UID: {uid}")
        seen_uids.add(uid)
        supplied = normalize_space(row.get("source_group"))
        if not supplied:
            supplied_group_counts["missing"] += 1
        elif supplied == group:
            supplied_group_counts["canonical"] += 1
        else:
            supplied_group_counts["replaced_noncanonical"] += 1
        if group in excluded_groups:
            excluded["reference_source_group"] += 1
            if uid in excluded_uids:
                excluded["direct_reference_uid"] += 1
            continue
        rendered = dict(row)
        rendered.update(
            {
                "task": task,
                "corpus": norms[uid]["corpus"],
                "source_group": group,
                "current_bank_source_sha256": bank_hash,
                "ce_source_disjoint_filter": True,
            }
        )
        output.append(rendered)

    output.sort(key=lambda row: normalize_space(row["norm_uid"]))
    # ``source_group_key`` intentionally uses the unit separator.  It is a
    # structural delimiter, not presentation whitespace, so never pass these
    # keys through ``normalize_space`` during overlap checks.
    output_groups = {str(row["source_group"]) for row in output}
    overlap = output_groups & excluded_groups
    if overlap:
        raise AssertionError(f"reference source groups remain: {sorted(overlap)[:3]}")
    if not output:
        raise ValueError("source-group filtering removed every teacher row")
    audit = {
        "input_teacher_rows": input_count,
        "output_teacher_rows": len(output),
        "output_source_groups": len(output_groups),
        "reference_rows": sum(reference_counts.values()),
        "reference_source_groups": len(excluded_groups),
        "reference_unique_uids": len(excluded_uids),
        "excluded": dict(sorted(excluded.items())),
        "supplied_teacher_source_group": dict(sorted(supplied_group_counts.items())),
        "output_reference_uid_overlap": len(
            {normalize_space(row["norm_uid"]) for row in output} & excluded_uids
        ),
        "output_reference_source_group_overlap": len(overlap),
    }
    return output, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--reference", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    teacher_path = Path(args.teacher).resolve()
    reference_paths = [Path(path).resolve() for path in args.reference]
    output_path = Path(args.output).resolve()
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if output_path.exists() or meta_path.exists():
        raise FileExistsError(f"immutable output already exists: {output_path}")

    norms, bank_hash, bank_path = load_task_universe(manifest_path, args.task)
    rows, audit = filter_rows(
        teacher_rows=read_jsonl(teacher_path),
        reference_inputs=[(str(path), read_jsonl(path)) for path in reference_paths],
        norms=norms,
        task=args.task,
        bank_hash=bank_hash,
    )
    write_jsonl(output_path, rows)
    meta = {
        "schema_version": "silver-match-v3-ce-source-disjoint-teachers-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "manifest": str(manifest_path),
        "teacher": str(teacher_path),
        "references": [str(path) for path in reference_paths],
        "bank": str(bank_path),
        "bank_source_sha256": bank_hash,
        "input_hashes": {
            "manifest": sha256_file(manifest_path),
            "teacher": sha256_file(teacher_path),
            "references": {str(path): sha256_file(path) for path in reference_paths},
            "bank": sha256_file(bank_path),
        },
        "audit": audit,
        "output_sha256": sha256_file(output_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {**meta, "meta_sha256": sha256_file(meta_path)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
