#!/usr/bin/env python3
"""Build an immutable, source-disjoint train-only LoRA teacher set.

Calibration files are evaluation artifacts first.  This builder admits only
predeclared ``split=train`` exact MATCH labels and removes an entire source
group if that group appears in any supplied dev/test row.  The resulting file
can safely be re-split internally by the LoRA trainer without consuming either
the calibration dev set or the frozen calibration test set.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .common import normalize_space, read_jsonl, sha256_file, write_jsonl
from .train_nemotron_lora import source_group_key


def _resolve(path: str | Path, anchor: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else anchor.parent / value


def load_task_norms(
    manifest_path: Path, task: str
) -> tuple[dict[str, dict[str, Any]], str, Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if task not in manifest.get("banks", {}):
        raise KeyError(f"task absent from manifest: {task}")
    bank_meta = manifest["banks"][task]
    bank_path = _resolve(bank_meta["path"], manifest_path)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    bank_hash = normalize_space(
        bank_meta.get("source_sha256") or bank.get("source_sha256")
    )
    if not bank_hash:
        raise ValueError(f"bank source hash missing for {task}")
    norms: dict[str, dict[str, Any]] = {}
    for corpus, meta in sorted(manifest.get("corpora", {}).items()):
        if meta.get("task") != task:
            continue
        for row in read_jsonl(_resolve(meta["path"], manifest_path)):
            uid = normalize_space(row.get("norm_uid"))
            if not uid or uid in norms:
                raise ValueError(f"missing/duplicate norm_uid {uid!r}")
            norms[uid] = row
    return norms, bank_hash, bank_path


def build_teacher_rows(
    label_inputs: Iterable[tuple[str, Iterable[dict[str, Any]]]],
    norms: dict[str, dict[str, Any]],
    task: str,
    bank_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_rows: list[tuple[str, dict[str, Any], dict[str, Any], str]] = []
    input_counts: dict[str, Counter[str]] = defaultdict(Counter)
    forbidden_groups: set[str] = set()
    seen_input_uids: dict[str, tuple[str, str | None]] = {}

    for source, rows in label_inputs:
        for row in rows:
            if normalize_space(row.get("task")) != task:
                continue
            uid = normalize_space(row.get("norm_uid"))
            if uid not in norms:
                raise ValueError(f"{source}: norm_uid absent from task manifest: {uid}")
            norm = norms[uid]
            if normalize_space(row.get("corpus")) != normalize_space(norm.get("corpus")):
                raise ValueError(f"{source}: corpus mismatch for {uid}")
            split = normalize_space(row.get("split"))
            if split not in {"train", "dev", "test"}:
                raise ValueError(f"{source}: invalid/missing frozen split for {uid}: {split!r}")
            row_hash = normalize_space(
                row.get("current_bank_source_sha256")
                or row.get("candidate_bank_source_sha256")
            )
            if row_hash != bank_hash:
                raise ValueError(
                    f"{source}: bank hash mismatch for {uid}: {row_hash!r} != {bank_hash!r}"
                )
            group = source_group_key(norm)
            input_counts[source][f"split:{split}"] += 1
            input_counts[source][f"decision:{normalize_space(row.get('decision'))}"] += 1
            if split in {"dev", "test"}:
                forbidden_groups.add(group)
            all_rows.append((source, row, norm, group))

    selected: dict[str, dict[str, Any]] = {}
    excluded = Counter()
    for source, row, norm, group in all_rows:
        uid = normalize_space(row["norm_uid"])
        split = normalize_space(row["split"])
        decision = normalize_space(row.get("decision"))
        if split != "train":
            excluded[f"frozen_{split}"] += 1
            continue
        if group in forbidden_groups:
            excluded["source_group_touches_dev_or_test"] += 1
            continue
        if decision != "MATCH":
            excluded[f"decision:{decision or 'MISSING'}"] += 1
            continue
        metric_id = normalize_space(row.get("metric_id"))
        if not metric_id:
            raise ValueError(f"{source}: MATCH lacks metric_id for {uid}")
        prior = seen_input_uids.get(uid)
        if prior is not None and prior[1] != metric_id:
            raise ValueError(
                f"conflicting exact teachers for {uid}: {prior[1]} ({prior[0]}) != "
                f"{metric_id} ({source})"
            )
        seen_input_uids[uid] = (source, metric_id)
        out = dict(row)
        out.update(
            {
                "task": task,
                "corpus": norm["corpus"],
                "split": "train",
                "source_group": group,
                "current_bank_source_sha256": bank_hash,
                "training_role": "gradient_candidate",
                "teacher_input": source,
            }
        )
        selected[uid] = out

    result = [selected[uid] for uid in sorted(selected)]
    selected_groups = {row["source_group"] for row in result}
    leaked = selected_groups & forbidden_groups
    if leaked:
        raise AssertionError(f"selected teacher groups leak into frozen dev/test: {sorted(leaked)[:3]}")
    audit = {
        "task": task,
        "input_counts": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(input_counts.items())
        },
        "input_task_rows": len(all_rows),
        "forbidden_dev_test_source_groups": len(forbidden_groups),
        "excluded": dict(sorted(excluded.items())),
        "selected_match_rows": len(result),
        "selected_source_groups": len(selected_groups),
        "selected_metric_coverage": len({row["metric_id"] for row in result}),
        "selected_frozen_splits": dict(Counter(row["split"] for row in result)),
        "source_group_overlap_with_dev_test": len(leaked),
    }
    return result, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--labels", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    label_paths = [Path(path).resolve() for path in args.labels]
    output_path = Path(args.output).resolve()
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if output_path.exists() or meta_path.exists():
        raise FileExistsError(f"immutable teacher output already exists: {output_path}")
    norms, bank_hash, bank_path = load_task_norms(manifest_path, args.task)
    rows, audit = build_teacher_rows(
        ((str(path), read_jsonl(path)) for path in label_paths),
        norms,
        args.task,
        bank_hash,
    )
    if not rows:
        raise ValueError("no leakage-safe exact MATCH teachers selected")
    write_jsonl(output_path, rows)
    meta = {
        "schema_version": "silver-match-v3-lora-teachers-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "task": args.task,
        "labels": [str(path) for path in label_paths],
        "bank_path": str(bank_path),
        "bank_source_sha256": bank_hash,
        "audit": audit,
        "input_hashes": {
            "manifest": sha256_file(manifest_path),
            "bank": sha256_file(bank_path),
            "labels": {str(path): sha256_file(path) for path in label_paths},
        },
        "output_sha256": sha256_file(output_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, sort_keys=True))


if __name__ == "__main__":
    main()
