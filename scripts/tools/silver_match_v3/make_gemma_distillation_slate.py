#!/usr/bin/env python3
"""Freeze train-only full-bank candidates for Gemma teacher distillation.

Rows are selected only from source groups assigned to the predeclared
calibration ``train`` split.  Every source group appearing in any supplied
calibration label file is excluded, which keeps distillation independent of
human train/dev/test panels and ensures the frozen tests cannot enter adapter
gradients indirectly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import normalize_space, read_jsonl, sha256_file, write_jsonl
from .make_calibration import split_for, split_group_for


def _resolve(path: str | Path, anchor: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else anchor.parent / value


def rank_key(row: dict[str, Any]) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"gemma-distill-v1\0{row['split_group']}\0{row['norm_uid']}".encode("utf-8")
    ).hexdigest()
    return digest, row["norm_uid"]


def select_balanced(
    rows_by_corpus: dict[str, list[dict[str, Any]]], total: int
) -> list[dict[str, Any]]:
    corpora = sorted(rows_by_corpus)
    selected: list[dict[str, Any]] = []
    cursors = {corpus: 0 for corpus in corpora}
    for values in rows_by_corpus.values():
        values.sort(key=rank_key)
    while len(selected) < total:
        advanced = False
        for corpus in corpora:
            cursor = cursors[corpus]
            values = rows_by_corpus[corpus]
            if cursor >= len(values):
                continue
            selected.append(values[cursor])
            cursors[corpus] += 1
            advanced = True
            if len(selected) == total:
                break
        if not advanced:
            break
    return sorted(selected, key=lambda row: (row["corpus"], row["norm_uid"]))


def build_slate(
    manifest_path: Path,
    task: str,
    exclude_label_paths: list[Path],
    total: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bank_meta = manifest["banks"][task]
    bank_path = _resolve(bank_meta["path"], manifest_path)
    bank_payload = json.loads(bank_path.read_text(encoding="utf-8"))
    bank_ids = [normalize_space(row.get("metric_id")) for row in bank_payload["metrics"]]
    if not bank_ids or len(bank_ids) != len(set(bank_ids)):
        raise ValueError("empty or duplicate bank metric IDs")
    bank_hash = normalize_space(
        bank_meta.get("source_sha256") or bank_payload.get("source_sha256")
    )
    excluded_uids: set[str] = set()
    excluded_groups: set[str] = set()

    # Index only the task norms, streaming the very large manifest once.
    task_norms: dict[str, dict[str, Any]] = {}
    for corpus, meta in sorted(manifest["corpora"].items()):
        if meta.get("task") != task:
            continue
        for row in read_jsonl(_resolve(meta["path"], manifest_path)):
            uid = normalize_space(row.get("norm_uid"))
            if not uid or uid in task_norms:
                raise ValueError(f"missing/duplicate norm UID: {uid!r}")
            task_norms[uid] = row
    for label_path in exclude_label_paths:
        for row in read_jsonl(label_path):
            if normalize_space(row.get("task")) != task:
                continue
            uid = normalize_space(row.get("norm_uid"))
            if uid not in task_norms:
                raise ValueError(f"excluded label UID absent from manifest: {uid}")
            excluded_uids.add(uid)
            excluded_groups.add(split_group_for(task_norms[uid]))

    one_per_group: dict[str, dict[str, Any]] = {}
    scan_counts = Counter()
    for norm in task_norms.values():
        scan_counts["task_norms"] += 1
        uid = norm["norm_uid"]
        group = split_group_for(norm)
        if uid in excluded_uids or group in excluded_groups:
            scan_counts["excluded_calibration_group"] += 1
            continue
        if split_for(group) != "train":
            scan_counts["non_train_split"] += 1
            continue
        item = {**norm, "split_group": group, "split": "train"}
        existing = one_per_group.get(group)
        if existing is None or rank_key(item) < rank_key(existing):
            one_per_group[group] = item
    rows_by_corpus: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in one_per_group.values():
        rows_by_corpus[row["corpus"]].append(row)
    selected = select_balanced(rows_by_corpus, total)
    if len(selected) != total:
        raise ValueError(f"requested {total} rows but only {len(selected)} train groups available")

    candidates = [
        {
            "schema_version": manifest["schema_version"],
            "norm_uid": row["norm_uid"],
            "corpus": row["corpus"],
            "task": task,
            "row": row["row"],
            "bank_source_sha256": bank_hash,
            "candidates": [
                {"metric_id": metric_id, "rank": rank}
                for rank, metric_id in enumerate(bank_ids, 1)
            ],
        }
        for row in selected
    ]
    selected_groups = {row["split_group"] for row in selected}
    if selected_groups & excluded_groups:
        raise AssertionError("distillation slate overlaps a calibration source group")
    audit = {
        "task": task,
        "requested": total,
        "selected": len(selected),
        "selected_source_groups": len(selected_groups),
        "selected_by_corpus": dict(sorted(Counter(row["corpus"] for row in selected).items())),
        "selected_split_counts": dict(Counter(row["split"] for row in selected)),
        "bank_metrics_per_item": len(bank_ids),
        "excluded_calibration_uids": len(excluded_uids),
        "excluded_calibration_source_groups": len(excluded_groups),
        "source_group_overlap_with_calibration": len(selected_groups & excluded_groups),
        "scan_counts": dict(sorted(scan_counts.items())),
    }
    return candidates, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--exclude-labels", action="append", default=[])
    parser.add_argument("--total", type=int, default=1500)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.total < 1:
        parser.error("--total must be positive")
    manifest_path = Path(args.manifest).resolve()
    exclude_paths = [Path(path).resolve() for path in args.exclude_labels]
    output_path = Path(args.output).resolve()
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if output_path.exists() or meta_path.exists():
        raise FileExistsError(f"immutable slate already exists: {output_path}")
    candidates, audit = build_slate(manifest_path, args.task, exclude_paths, args.total)
    write_jsonl(output_path, candidates)
    meta = {
        "schema_version": "silver-match-v3-gemma-distillation-slate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "task": args.task,
        "exclude_labels": [str(path) for path in exclude_paths],
        "audit": audit,
        "input_hashes": {
            "manifest": sha256_file(manifest_path),
            "exclude_labels": {str(path): sha256_file(path) for path in exclude_paths},
        },
        "output_sha256": sha256_file(output_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, sort_keys=True))


if __name__ == "__main__":
    main()
