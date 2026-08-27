#!/usr/bin/env python3
"""Permute every item and bank card in an immutable independent teacher pack."""

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
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=25)
    args = parser.parse_args()
    source, output = Path(args.pack_root).resolve(), Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite permuted pack: {output}")
    validation_path = source / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    items_path, bank_path = source / "items.jsonl", source / "bank.json"
    if sha256_file(items_path) != validation["outputs"]["items"]["sha256"]:
        raise ValueError("source items hash mismatch")
    if sha256_file(bank_path) != validation["outputs"]["bank"]["sha256"]:
        raise ValueError("source bank hash mismatch")
    items = list(read_jsonl(items_path))
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if bank.get("task") != validation.get("task") or bank.get(
        "source_sha256"
    ) != validation.get("bank_source_sha256"):
        raise ValueError("source bank identity mismatch")
    items.sort(
        key=lambda row: (_key(args.seed, "item", str(row["norm_uid"])), row["norm_uid"])
    )
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
        "schema_version": "silver-match-v3-permuted-independent-teacher-pack-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "chunk_size": args.chunk_size,
        "chunk_count": len(chunks),
        "source_pack": {
            "path": str(source),
            "validation_sha256": sha256_file(validation_path),
        },
        "bank_order_reshuffled": True,
        "item_order_reshuffled": True,
        "outputs": {
            "items": {"path": str(output_items), "sha256": sha256_file(output_items)},
            "bank": {"path": str(output_bank), "sha256": sha256_file(output_bank)},
            "chunks": {str(path): sha256_file(path) for path in chunks},
        },
    }
    (output / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"task": report["task"], "count": len(items), "seed": args.seed}))


if __name__ == "__main__":
    main()
