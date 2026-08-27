#!/usr/bin/env python3
"""Reorder a truth-hidden anchor pack for an independent second full-bank pass."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .common import read_jsonl, sha256_file, write_jsonl


def _key(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{namespace}\0{value}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--chunk-size", type=int, default=25)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")

    source = Path(args.pack_root).resolve()
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite anchor relabel pack: {output}")
    validation_path = source / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    items_path, bank_path = source / "items.jsonl", source / "bank.json"
    if sha256_file(items_path) != validation["outputs"]["items"]:
        raise ValueError("anchor items hash mismatch")
    if sha256_file(bank_path) != validation["outputs"]["bank"]:
        raise ValueError("anchor bank hash mismatch")
    items = list(read_jsonl(items_path))
    if len({str(row["norm_uid"]) for row in items}) != len(items):
        raise ValueError("anchor items contain duplicate UIDs")
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if bank.get("task") != validation.get("task") or bank.get(
        "source_sha256"
    ) != validation.get("bank_source_sha256"):
        raise ValueError("anchor bank identity mismatch")
    if len({str(row["metric_id"]) for row in bank["metrics"]}) != len(bank["metrics"]):
        raise ValueError("anchor bank contains duplicate IDs")

    items.sort(key=lambda row: (_key(args.seed, "item", str(row["norm_uid"])), row["norm_uid"]))
    metrics = list(bank["metrics"])
    metrics.sort(
        key=lambda row: (_key(args.seed, "metric", str(row["metric_id"])), row["metric_id"])
    )
    output.mkdir(parents=True, exist_ok=True)
    output_items, output_bank = output / "items.jsonl", output / "bank.json"
    write_jsonl(output_items, items)
    output_bank.write_text(
        json.dumps({**bank, "metrics": metrics}, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    chunks = []
    for start in range(0, len(items), args.chunk_size):
        path = output / "chunks" / f"part-{start // args.chunk_size:03d}.jsonl"
        write_jsonl(path, items[start : start + args.chunk_size])
        chunks.append(path)
    report = {
        **validation,
        "schema_version": "silver-match-v3-labeler-anchor-relabel-pack-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "chunk_size": args.chunk_size,
        "chunk_counts": {
            path.stem: sum(1 for _ in read_jsonl(path)) for path in chunks
        },
        "source_pack": {
            "path": str(source),
            "validation_sha256": sha256_file(validation_path),
        },
        "truth_hidden": True,
        "bank_order_reshuffled": True,
        "item_order_reshuffled": True,
        "outputs": {
            "items": sha256_file(output_items),
            "bank": sha256_file(output_bank),
            "chunks": {str(path): sha256_file(path) for path in chunks},
        },
    }
    (output / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
