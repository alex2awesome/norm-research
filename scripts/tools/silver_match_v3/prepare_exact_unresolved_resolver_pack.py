#!/usr/bin/env python3
"""Build a truth-hidden next-round pack from exact-consensus unresolved UIDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


def _order(seed: int, namespace: str, value: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}\0{namespace}\0{value}".encode()).hexdigest()
    return digest, value


def _index(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    output = {str(row.get("norm_uid") or ""): row for row in rows}
    if "" in output or len(output) != len(rows):
        raise ValueError(f"{name} contains missing or duplicate norm_uid values")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--unresolved", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=25)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")

    source = Path(args.pack_root).resolve()
    unresolved_path = Path(args.unresolved).resolve()
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite resolver pack: {output}")

    validation_path = source / "validation.json"
    items_path, bank_path = source / "items.jsonl", source / "bank.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if sha256_file(items_path) != validation["outputs"]["items"]["sha256"]:
        raise ValueError("source items hash mismatch")
    if sha256_file(bank_path) != validation["outputs"]["bank"]["sha256"]:
        raise ValueError("source bank hash mismatch")
    task = str(validation["task"])
    items = list(read_jsonl(items_path))
    item_by_uid = _index(items, "source pack")
    unresolved = list(read_jsonl(unresolved_path))
    unresolved_by_uid = _index(unresolved, "unresolved input")
    if not unresolved:
        raise ValueError("unresolved selection is empty")
    extra = set(unresolved_by_uid) - set(item_by_uid)
    if extra:
        raise ValueError(f"unresolved UIDs outside source pack: {sorted(extra)[:3]}")
    for uid, row in unresolved_by_uid.items():
        if str(row.get("task")) != task or not row.get("unresolved_reason"):
            raise ValueError(f"invalid unresolved provenance: {uid}")

    selected = [item_by_uid[uid] for uid in unresolved_by_uid]
    selected.sort(key=lambda row: _order(args.seed, "item", str(row["norm_uid"])))
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    metrics = list(bank["metrics"])
    metrics.sort(key=lambda row: _order(args.seed, "metric", str(row["metric_id"])))

    output.mkdir(parents=True, exist_ok=True)
    output_items, output_bank = output / "items.jsonl", output / "bank.json"
    write_jsonl(output_items, selected)
    output_bank.write_text(
        json.dumps({**bank, "metrics": metrics}, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    chunks = []
    for start in range(0, len(selected), args.chunk_size):
        path = output / "chunks" / f"part-{start // args.chunk_size:03d}.jsonl"
        write_jsonl(path, selected[start : start + args.chunk_size])
        chunks.append(path)

    report = {
        "schema_version": "silver-match-v3-exact-unresolved-resolver-pack-v1",
        "status": "FROZEN_TRUTH_HIDDEN_BEFORE_LABELING",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "source_count": len(items),
        "count": len(selected),
        "chunk_size": args.chunk_size,
        "chunk_count": len(chunks),
        "bank_metric_count": len(metrics),
        "bank_source_sha256": validation["bank_source_sha256"],
        "seed": args.seed,
        "truth_hidden": True,
        "prior_decisions_and_metric_ids_hidden": True,
        "selection_rule": "all_and_only_current_exact_consensus_unresolved_uids",
        "inputs": {
            "source_pack_validation": {
                "path": str(validation_path),
                "sha256": sha256_file(validation_path),
            },
            "unresolved": {
                "path": str(unresolved_path),
                "sha256": sha256_file(unresolved_path),
            },
        },
        "outputs": {
            "items": {"path": str(output_items), "sha256": sha256_file(output_items)},
            "bank": {"path": str(output_bank), "sha256": sha256_file(output_bank)},
            "chunks": {str(path): sha256_file(path) for path in chunks},
        },
    }
    (output / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
