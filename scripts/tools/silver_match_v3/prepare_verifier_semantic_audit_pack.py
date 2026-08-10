#!/usr/bin/env python3
"""Prepare a truth-hidden full-bank semantic audit for frozen verifier rows.

The original implementation was Humor-specific.  ``--task`` now makes the
same hidden-ID/full-bank rendering reusable for every task while retaining
Humor as the backwards-compatible default for already frozen commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .common import read_jsonl, sha256_file, write_jsonl
from .make_calibration import split_group_for


def stable(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\x1f{namespace}\x1f{value}".encode()).hexdigest()


def resolve(path: str, anchor: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (anchor.parent / value).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", default="humor")
    parser.add_argument("--items", required=True)
    parser.add_argument(
        "--forbidden-items",
        required=True,
        action="append",
        help=(
            "JSONL identity set that must remain disjoint from the rendered pack; "
            "repeat the flag to enforce multiple frozen exclusion sets"
        ),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--chunk-size", type=int, default=100)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    items_path = Path(args.items).resolve()
    forbidden_paths = [Path(value).resolve() for value in args.forbidden_items]
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(output_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_items = list(read_jsonl(items_path))
    item_by_uid = {str(row["norm_uid"]): row for row in source_items}
    if len(item_by_uid) != len(source_items):
        raise ValueError("duplicate audit item UID")
    forbidden = [
        row for forbidden_path in forbidden_paths for row in read_jsonl(forbidden_path)
    ]
    forbidden_uids = {str(row["norm_uid"]) for row in forbidden}
    if len(forbidden_uids) != len(forbidden) or "" in forbidden_uids:
        raise ValueError("duplicate or missing permanent-blind UID")

    norm_by_uid = {}
    for corpus, value in manifest["corpora"].items():
        if value.get("task") != args.task:
            continue
        for row in read_jsonl(resolve(value["path"], manifest_path)):
            norm_by_uid[str(row["norm_uid"])] = row
    missing = sorted(set(item_by_uid) - set(norm_by_uid))
    if missing:
        raise ValueError(f"audit items absent from manifest norms: {missing[:5]}")
    missing_forbidden = sorted(forbidden_uids - set(norm_by_uid))
    if missing_forbidden:
        raise ValueError(
            f"permanent-blind items absent from task norms: {missing_forbidden[:5]}"
        )
    item_groups = set()
    for uid, source in item_by_uid.items():
        canonical = split_group_for(norm_by_uid[uid])
        if str(source.get("source_group") or "") != canonical:
            raise ValueError(f"audit item source_group mismatch: {uid}")
        if canonical in item_groups:
            raise ValueError(f"audit pack repeats a source_group: {canonical}")
        item_groups.add(canonical)
    forbidden_groups = set()
    for source in forbidden:
        uid = str(source["norm_uid"])
        canonical = split_group_for(norm_by_uid[uid])
        if str(source.get("source_group") or "") != canonical:
            raise ValueError(f"permanent-blind source_group mismatch: {uid}")
        forbidden_groups.add(canonical)
    if set(item_by_uid) & forbidden_uids:
        raise ValueError("semantic audit exposes a permanent-blind UID")
    if item_groups & forbidden_groups:
        raise ValueError("semantic audit overlaps permanent blind by source group")
    if args.task not in manifest.get("banks", {}):
        raise ValueError(f"manifest has no bank for task {args.task}")
    bank_meta = manifest["banks"][args.task]
    bank_path = resolve(bank_meta["path"], manifest_path)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    metrics = list(bank["metrics"])
    metrics.sort(key=lambda row: stable(args.seed, "metric", str(row["metric_id"])))
    bank_hash = str(bank_meta.get("source_sha256") or bank.get("source_sha256"))

    rows = []
    for uid in sorted(item_by_uid, key=lambda value: stable(args.seed, "item", value)):
        norm = norm_by_uid[uid]
        source = item_by_uid[uid]
        rows.append(
            {
                "schema_version": "silver-match-v3-verifier-semantic-audit-item-v1",
                "norm_uid": uid,
                "task": args.task,
                "corpus": norm.get("corpus"),
                "row": norm.get("row"),
                "source_id": norm.get("source_id"),
                "source_group": source.get("source_group"),
                "norm": norm.get("norm"),
                "context": norm.get("context"),
                "source_segment": norm.get("source_segment"),
                "aspect": norm.get("aspect"),
                "polarity": norm.get("polarity"),
                "manual_decision": None,
                "manual_metric_id": None,
                "manual_confidence": None,
                "manual_reason": None,
                "auditor": None,
            }
        )
    output_root.mkdir(parents=True, exist_ok=False)
    item_output, bank_output = output_root / "items.jsonl", output_root / "bank.json"
    write_jsonl(item_output, rows)
    bank_output.write_text(json.dumps({**bank, "metrics": metrics}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    chunks = {}
    for start in range(0, len(rows), args.chunk_size):
        path = output_root / "chunks" / f"part-{start // args.chunk_size:03d}.jsonl"
        write_jsonl(path, rows[start : start + args.chunk_size])
        chunks[str(path)] = {"count": min(args.chunk_size, len(rows) - start), "sha256": sha256_file(path)}
    report = {
        "schema_version": "silver-match-v3-verifier-semantic-audit-pack-v1",
        "task": args.task,
        "count": len(rows),
        "bank_metric_count": len(metrics),
        "bank_source_sha256": bank_hash,
        "truth_hidden": True,
        "adjudicator_outputs_read": False,
        "label_pass_outputs_read": False,
        "permanent_blind_rows_excluded": len(forbidden_uids),
        "permanent_blind_source_groups_excluded": len(forbidden_groups),
        "input_hashes": {
            "manifest": sha256_file(manifest_path),
            "items": sha256_file(items_path),
            "forbidden_items": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in forbidden_paths
            ],
            "bank": sha256_file(bank_path),
        },
        "outputs": {
            "items": {"path": str(item_output), "sha256": sha256_file(item_output)},
            "bank": {"path": str(bank_output), "sha256": sha256_file(bank_output)},
            "chunks": chunks,
        },
    }
    validation = output_root / "validation.json"
    validation.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "validation_sha256": sha256_file(validation)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
