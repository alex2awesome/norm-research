#!/usr/bin/env python3
"""Freeze compact, leakage-safe full-bank packs for independent labelers."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .common import normalize_space, read_jsonl, sha256_file, write_jsonl
from .make_calibration import split_for, split_group_for


def resolve(path: str | Path, anchor: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (anchor.parent / value).resolve()


def balanced_sample(rows: Sequence[dict[str, Any]], total: int) -> list[dict[str, Any]]:
    by_corpus: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_corpus[str(row["corpus"])].append(row)
    for values in by_corpus.values():
        values.sort(
            key=lambda row: (
                hashlib.sha256(
                    f"independent-teacher-pack-v1\0{row['norm_uid']}".encode()
                ).hexdigest(),
                row["norm_uid"],
            )
        )
    output = []
    cursors = {corpus: 0 for corpus in by_corpus}
    while len(output) < total:
        advanced = False
        for corpus in sorted(by_corpus):
            index = cursors[corpus]
            if index >= len(by_corpus[corpus]):
                continue
            output.append(by_corpus[corpus][index])
            cursors[corpus] += 1
            advanced = True
            if len(output) == total:
                break
        if not advanced:
            break
    if len(output) != total:
        raise ValueError(f"requested {total} rows but only selected {len(output)}")
    return sorted(output, key=lambda row: (row["corpus"], row["norm_uid"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--candidate-meta")
    parser.add_argument("--external-label", action="append", default=[])
    parser.add_argument("--total", type=int, default=200)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    if args.total < 1 or args.chunk_size < 1:
        parser.error("--total and --chunk-size must be positive")

    manifest_path = Path(args.manifest).resolve()
    candidate_path = Path(args.candidates).resolve()
    candidate_meta_path = (
        Path(args.candidate_meta).resolve()
        if args.candidate_meta
        else candidate_path.with_suffix(candidate_path.suffix + ".meta.json")
    )
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite teacher pack: {output_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_meta = json.loads(candidate_meta_path.read_text(encoding="utf-8"))
    if candidate_meta.get("output_sha256") != sha256_file(candidate_path):
        raise ValueError("candidate metadata hash mismatch")
    if candidate_meta.get("task") != args.task:
        raise ValueError("candidate metadata task mismatch")
    if (candidate_meta.get("input_hashes") or {}).get("manifest") != sha256_file(
        manifest_path
    ):
        raise ValueError("candidate slate was built from a different manifest")

    bank_meta = manifest["banks"][args.task]
    bank_path = resolve(bank_meta["path"], manifest_path)
    bank_payload = json.loads(bank_path.read_text(encoding="utf-8"))
    bank_ids = [str(row["metric_id"]) for row in bank_payload["metrics"]]
    bank_hash = str(bank_meta["source_sha256"])
    if len(bank_ids) != len(set(bank_ids)):
        raise ValueError("bank contains duplicate metric IDs")

    candidate_rows = list(read_jsonl(candidate_path))
    candidate_by_uid = {}
    for row in candidate_rows:
        uid = str(row["norm_uid"])
        if uid in candidate_by_uid:
            raise ValueError(f"duplicate candidate UID: {uid}")
        if row.get("task") != args.task or row.get("bank_source_sha256") != bank_hash:
            raise ValueError(f"candidate task/bank hash mismatch: {uid}")
        ids = [str(value["metric_id"]) for value in row.get("candidates") or []]
        if ids != bank_ids:
            raise ValueError(f"candidate row is not the exact ordered full bank: {uid}")
        candidate_by_uid[uid] = row

    norms = {}
    for corpus, meta in manifest["corpora"].items():
        if meta.get("task") != args.task:
            continue
        for row in read_jsonl(resolve(meta["path"], manifest_path)):
            uid = str(row["norm_uid"])
            if uid in norms:
                raise ValueError(f"duplicate canonical norm UID: {uid}")
            norms[uid] = row
    missing = sorted(set(candidate_by_uid) - set(norms))
    if missing:
        raise KeyError(f"candidate UIDs absent from canonical norms: {missing[:3]}")

    external_paths = {
        Path(path).resolve()
        for path in [*(candidate_meta.get("exclude_labels") or []), *args.external_label]
    }
    external_uids: set[str] = set()
    external_groups: set[str] = set()
    for path in sorted(external_paths):
        for row in read_jsonl(path):
            if row.get("task") != args.task:
                continue
            uid = str(row["norm_uid"])
            if uid not in norms:
                raise KeyError(f"external label UID absent from canonical norms: {uid}")
            external_uids.add(uid)
            external_groups.add(split_group_for(norms[uid]))

    hydrated = []
    for uid in candidate_by_uid:
        norm = norms[uid]
        group = split_group_for(norm)
        if split_for(group) != "train":
            raise ValueError(f"candidate is not assigned to train: {uid}")
        hydrated.append({**norm, "split_group": group, "split": "train"})
    selected = balanced_sample(hydrated, args.total)
    selected_uids = {row["norm_uid"] for row in selected}
    selected_groups = {row["split_group"] for row in selected}
    if len(selected_groups) != len(selected):
        raise ValueError("selected pack contains repeated source groups")
    if selected_uids & external_uids or selected_groups & external_groups:
        raise ValueError("selected pack overlaps an external panel")

    output_root.mkdir(parents=True, exist_ok=True)
    items_path = output_root / "items.jsonl"
    candidates_path = output_root / "candidates.full-bank.jsonl"
    bank_output = output_root / "bank.json"
    write_jsonl(items_path, selected)
    write_jsonl(
        candidates_path,
        [candidate_by_uid[row["norm_uid"]] for row in selected],
    )
    bank_output.write_text(
        json.dumps(bank_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    chunk_paths = []
    for start in range(0, len(selected), args.chunk_size):
        path = output_root / "chunks" / f"part-{start // args.chunk_size:03d}.jsonl"
        write_jsonl(path, selected[start : start + args.chunk_size])
        chunk_paths.append(path)
    report = {
        "schema_version": "silver-match-v3-independent-teacher-pack-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "count": len(selected),
        "chunk_size": args.chunk_size,
        "chunk_count": len(chunk_paths),
        "selected_by_corpus": dict(sorted(Counter(row["corpus"] for row in selected).items())),
        "selected_source_groups": len(selected_groups),
        "external_uids": len(external_uids),
        "external_source_groups": len(external_groups),
        "uid_overlap_with_external": len(selected_uids & external_uids),
        "source_group_overlap_with_external": len(selected_groups & external_groups),
        "train_split_count": sum(row["split"] == "train" for row in selected),
        "bank_metric_count": len(bank_ids),
        "bank_source_sha256": bank_hash,
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "candidate_slate": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
            "candidate_meta": {"path": str(candidate_meta_path), "sha256": sha256_file(candidate_meta_path)},
            "external_labels": {str(path): sha256_file(path) for path in sorted(external_paths)},
        },
        "outputs": {
            "items": {"path": str(items_path), "sha256": sha256_file(items_path)},
            "candidates": {"path": str(candidates_path), "sha256": sha256_file(candidates_path)},
            "bank": {"path": str(bank_output), "sha256": sha256_file(bank_output)},
            "chunks": {str(path): sha256_file(path) for path in chunk_paths},
        },
    }
    report_path = output_root / "validation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
