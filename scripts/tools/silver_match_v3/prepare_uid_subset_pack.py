#!/usr/bin/env python3
"""Build a truth-hidden, reordered full-bank pack for an explicit UID subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .common import read_jsonl, sha256_file, write_jsonl


def order(seed: int, namespace: str, value: str) -> tuple[str, str]:
    return hashlib.sha256(f"{seed}\0{namespace}\0{value}".encode()).hexdigest(), value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--uids", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=25)
    args = parser.parse_args()
    source, uid_path, output = (
        Path(args.pack_root).resolve(),
        Path(args.uids).resolve(),
        Path(args.output_root).resolve(),
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    validation_path = source / "validation.json"
    validation = json.loads(validation_path.read_text())
    items_path, bank_path = source / "items.jsonl", source / "bank.json"
    if sha256_file(items_path) != validation["outputs"]["items"]["sha256"]:
        raise ValueError("source items hash mismatch")
    if sha256_file(bank_path) != validation["outputs"]["bank"]["sha256"]:
        raise ValueError("source bank hash mismatch")
    items = list(read_jsonl(items_path))
    by_uid = {str(row["norm_uid"]): row for row in items}
    wanted_rows = list(read_jsonl(uid_path))
    wanted = [str(row["norm_uid"]) for row in wanted_rows]
    if len(wanted) != len(set(wanted)) or not set(wanted).issubset(by_uid):
        raise ValueError("UID subset is duplicate or outside source pack")
    selected = [by_uid[uid] for uid in wanted]
    selected.sort(key=lambda row: order(args.seed, "item", str(row["norm_uid"])))
    bank = json.loads(bank_path.read_text())
    metrics = sorted(
        bank["metrics"], key=lambda row: order(args.seed, "metric", str(row["metric_id"]))
    )
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "items.jsonl", selected)
    (output / "bank.json").write_text(
        json.dumps({**bank, "metrics": metrics}, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    )
    chunks = []
    for start in range(0, len(selected), args.chunk_size):
        path = output / "chunks" / f"part-{start // args.chunk_size:03d}.jsonl"
        write_jsonl(path, selected[start : start + args.chunk_size])
        chunks.append(path)
    report = {
        "schema_version": "silver-match-v3-truth-hidden-uid-subset-pack-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": validation["task"],
        "count": len(selected),
        "chunk_size": args.chunk_size,
        "chunk_count": len(chunks),
        "bank_source_sha256": validation["bank_source_sha256"],
        "truth_hidden": True,
        "prior_decisions_metric_ids_and_proposals_hidden": True,
        "permanent_blind_rows_in_source": 0,
        "seed": args.seed,
        "inputs": {
            "source_pack_validation": {
                "path": str(validation_path), "sha256": sha256_file(validation_path)
            },
            "uid_reference": {"path": str(uid_path), "sha256": sha256_file(uid_path)},
        },
        "outputs": {
            "items": {"path": str(output / "items.jsonl"), "sha256": sha256_file(output / "items.jsonl")},
            "bank": {"path": str(output / "bank.json"), "sha256": sha256_file(output / "bank.json")},
            "chunks": {str(path): sha256_file(path) for path in chunks},
        },
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
