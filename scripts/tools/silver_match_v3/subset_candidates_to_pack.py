#!/usr/bin/env python3
"""Project a frozen candidate artifact onto an immutable item pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--candidate-meta")
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-k", type=int, default=50)
    args = parser.parse_args()
    if args.output_k < 1:
        parser.error("--output-k must be positive")

    candidates_path = Path(args.candidates).resolve()
    candidate_meta_path = (
        Path(args.candidate_meta).resolve()
        if args.candidate_meta
        else candidates_path.with_suffix(candidates_path.suffix + ".meta.json")
    )
    pack_root = Path(args.pack_root).resolve()
    output_path = Path(args.output).resolve()
    report_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if output_path.exists() or report_path.exists():
        raise FileExistsError(f"refusing to overwrite candidate subset: {output_path}")

    validation_path = pack_root / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    items_path = pack_root / "items.jsonl"
    if sha256_file(items_path) != validation["outputs"]["items"]["sha256"]:
        raise ValueError("pack items hash mismatch")
    items = list(read_jsonl(items_path))
    item_uids = [str(row["norm_uid"]) for row in items]
    if len(item_uids) != len(set(item_uids)):
        raise ValueError("pack contains duplicate UIDs")
    wanted = set(item_uids)
    task = str(validation["task"])
    bank_hash = str(validation["bank_source_sha256"])

    candidate_sha256 = sha256_file(candidates_path)
    candidate_meta: dict[str, Any] | None = None
    if candidate_meta_path.exists():
        candidate_meta = json.loads(candidate_meta_path.read_text(encoding="utf-8"))
        output_meta = candidate_meta.get("output")
        recorded = str(
            candidate_meta.get("output_sha256")
            or (output_meta.get("sha256") if isinstance(output_meta, dict) else None)
            or candidate_meta.get("sha256")
            or ""
        )
        if recorded and recorded != candidate_sha256:
            raise ValueError("source candidate metadata hash mismatch")

    selected: dict[str, dict[str, Any]] = {}
    duplicate_wanted: set[str] = set()
    for row in read_jsonl(candidates_path):
        uid = str(row.get("norm_uid") or "")
        if uid not in wanted:
            continue
        if uid in selected:
            duplicate_wanted.add(uid)
            continue
        if row.get("task") != task or row.get("bank_source_sha256") != bank_hash:
            raise ValueError(f"candidate task/bank mismatch: {uid}")
        values = list(row.get("candidates") or [])
        if len(values) < args.output_k:
            raise ValueError(
                f"candidate row shorter than requested K={args.output_k}: {uid}/{len(values)}"
            )
        metric_ids = [str(value["metric_id"]) for value in values]
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError(f"candidate row has duplicate metric IDs: {uid}")
        top = [{**value, "rank": rank} for rank, value in enumerate(values[: args.output_k], 1)]
        selected[uid] = {**row, "candidates": top}
    if duplicate_wanted:
        raise ValueError(f"source candidates duplicate requested UIDs: {sorted(duplicate_wanted)[:3]}")
    if set(selected) != wanted:
        missing = sorted(wanted - set(selected))
        raise ValueError(f"source candidates do not cover pack: {missing[:3]}")

    rows = [selected[uid] for uid in item_uids]
    write_jsonl(output_path, rows)
    report = {
        "schema_version": "silver-match-v3-pack-candidate-subset-v1",
        "task": task,
        "count": len(rows),
        "output_k": args.output_k,
        "bank_source_sha256": bank_hash,
        "inputs": {
            "candidates": {
                "path": str(candidates_path),
                "sha256": candidate_sha256,
            },
            "candidate_meta": (
                {
                    "path": str(candidate_meta_path),
                    "sha256": sha256_file(candidate_meta_path),
                }
                if candidate_meta_path.exists()
                else None
            ),
            "pack_validation": {
                "path": str(validation_path),
                "sha256": sha256_file(validation_path),
            },
        },
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
