#!/usr/bin/env python3
"""Re-slate first-pass MATCH claims for a truth-blind full-bank relabel."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


def _key(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{namespace}\0{value}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--first-labels", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument(
        "--confidence", action="append", choices=["high", "medium", "low"], default=[]
    )
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")

    pack_root = Path(args.pack_root).resolve()
    labels_path = Path(args.first_labels).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite relabel pack: {output_root}")

    pack_report_path = pack_root / "validation.json"
    pack_report = json.loads(pack_report_path.read_text(encoding="utf-8"))
    items_path, bank_path = pack_root / "items.jsonl", pack_root / "bank.json"
    if sha256_file(items_path) != pack_report["outputs"]["items"]["sha256"]:
        raise ValueError("source pack items hash mismatch")
    if sha256_file(bank_path) != pack_report["outputs"]["bank"]["sha256"]:
        raise ValueError("source pack bank hash mismatch")

    task = str(pack_report["task"])
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    bank_hash = str(pack_report["bank_source_sha256"])
    if bank.get("task") != task or bank.get("source_sha256") != bank_hash:
        raise ValueError("source pack bank identity mismatch")
    metric_ids = [str(row["metric_id"]) for row in bank["metrics"]]
    if len(metric_ids) != len(set(metric_ids)):
        raise ValueError("source bank has duplicate metric IDs")

    items = list(read_jsonl(items_path))
    item_by_uid = {str(row["norm_uid"]): row for row in items}
    if len(item_by_uid) != len(items):
        raise ValueError("source pack has duplicate item UIDs")
    labels = list(read_jsonl(labels_path))
    if len({str(row["norm_uid"]) for row in labels}) != len(labels):
        raise ValueError("first labels have duplicate UIDs")
    allowed_confidence = set(args.confidence or ["high", "medium"])
    selected_labels = []
    for row in labels:
        uid = str(row["norm_uid"])
        if uid not in item_by_uid:
            raise KeyError(f"first label absent from source pack: {uid}")
        if row.get("task") != task or row.get("current_bank_source_sha256") != bank_hash:
            raise ValueError(f"first label task/bank mismatch: {uid}")
        if row.get("decision") != "MATCH" or row.get("confidence") not in allowed_confidence:
            continue
        if str(row.get("metric_id")) not in metric_ids:
            raise ValueError(f"first MATCH ID absent from bank: {uid}")
        selected_labels.append(row)
    if not selected_labels:
        raise ValueError("no eligible first-pass MATCH labels")

    selected = [item_by_uid[str(row["norm_uid"])] for row in selected_labels]
    selected.sort(key=lambda row: (_key(args.seed, "item", str(row["norm_uid"])), row["norm_uid"]))
    shuffled_metrics = list(bank["metrics"])
    shuffled_metrics.sort(
        key=lambda row: (_key(args.seed, "metric", str(row["metric_id"])), row["metric_id"])
    )
    shuffled_bank = {**bank, "metrics": shuffled_metrics}

    output_root.mkdir(parents=True, exist_ok=True)
    output_items = output_root / "items.jsonl"
    output_bank = output_root / "bank.json"
    write_jsonl(output_items, selected)
    output_bank.write_text(
        json.dumps(shuffled_bank, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    chunks = []
    for start in range(0, len(selected), args.chunk_size):
        path = output_root / "chunks" / f"part-{start // args.chunk_size:03d}.jsonl"
        write_jsonl(path, selected[start : start + args.chunk_size])
        chunks.append(path)

    report: dict[str, Any] = {
        "schema_version": "silver-match-v3-independent-relabel-pack-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "count": len(selected),
        "chunk_size": args.chunk_size,
        "chunk_count": len(chunks),
        "selected_by_corpus": dict(
            sorted(Counter(str(row["corpus"]) for row in selected).items())
        ),
        "selected_source_groups": len({row["split_group"] for row in selected}),
        "train_split_count": sum(row.get("split") == "train" for row in selected),
        "bank_metric_count": len(metric_ids),
        "bank_source_sha256": bank_hash,
        "seed": args.seed,
        "selection": {
            "decision": "MATCH",
            "confidence": sorted(allowed_confidence),
            "first_metric_ids_hidden": True,
            "bank_order_reshuffled": True,
        },
        "inputs": {
            "source_pack_validation": {
                "path": str(pack_report_path),
                "sha256": sha256_file(pack_report_path),
            },
            "first_labels": {"path": str(labels_path), "sha256": sha256_file(labels_path)},
        },
        "outputs": {
            "items": {"path": str(output_items), "sha256": sha256_file(output_items)},
            "bank": {"path": str(output_bank), "sha256": sha256_file(output_bank)},
            "chunks": {str(path): sha256_file(path) for path in chunks},
        },
    }
    report_path = output_root / "validation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
