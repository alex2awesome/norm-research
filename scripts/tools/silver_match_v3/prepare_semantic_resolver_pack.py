#!/usr/bin/env python3
"""Build a truth-hidden resolver pack from confidence and corroboration rules."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


def _index(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    output = {str(row["norm_uid"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate norm_uid values: {path}")
    return output


def _decision_key(row: dict[str, Any]) -> tuple[str, str | None]:
    decision = str(row.get("decision") or "")
    metric_id = str(row["metric_id"]) if decision == "MATCH" else None
    return decision, metric_id


def _order(seed: int, namespace: str, value: str) -> tuple[str, str]:
    return hashlib.sha256(f"{seed}\0{namespace}\0{value}".encode()).hexdigest(), value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--semantic-labels", required=True)
    parser.add_argument("--strict-key", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument(
        "--selection-mode",
        choices=("confidence_and_corroboration", "exact_disagreements_only"),
        default="confidence_and_corroboration",
        help=(
            "Use the legacy confidence/corroboration union, or construct a third "
            "truth-hidden pass from only exact decision-and-leaf disagreements "
            "between two complete independent label passes."
        ),
    )
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")

    source = Path(args.pack_root).resolve()
    semantic_path = Path(args.semantic_labels).resolve()
    key_path = Path(args.strict_key).resolve()
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite resolver pack: {output}")
    validation_path, items_path, bank_path = (
        source / "validation.json",
        source / "items.jsonl",
        source / "bank.json",
    )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if sha256_file(items_path) != validation["outputs"]["items"]["sha256"]:
        raise ValueError("source items hash mismatch")
    if sha256_file(bank_path) != validation["outputs"]["bank"]["sha256"]:
        raise ValueError("source bank hash mismatch")
    items = list(read_jsonl(items_path))
    item_by_uid = {str(row["norm_uid"]): row for row in items}
    if len(item_by_uid) != len(items):
        raise ValueError("source pack has duplicate UIDs")
    semantic, strict = _index(semantic_path), _index(key_path)
    if set(semantic) != set(item_by_uid):
        raise ValueError("semantic labels must cover the source pack exactly")
    if not set(strict).issubset(item_by_uid):
        raise ValueError("strict key contains UIDs outside the source pack")
    task = str(validation["task"])
    bank_hash = str(validation["bank_source_sha256"])
    for name, rows in (("semantic", semantic), ("strict", strict)):
        for uid, row in rows.items():
            if row.get("task") != task or row.get("current_bank_source_sha256") != bank_hash:
                raise ValueError(f"{name} task/bank mismatch: {uid}")

    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for item in items:
        uid = str(item["norm_uid"])
        semantic_row = semantic[uid]
        key_row = strict.get(uid)
        low_confidence = str(semantic_row.get("confidence")) != "high"
        mismatch = key_row is not None and _decision_key(key_row) != _decision_key(
            semantic_row
        )
        match_lacks_corroboration = semantic_row.get("decision") == "MATCH" and (
            key_row is None or mismatch
        )
        if low_confidence:
            counts["medium_or_low_any_decision"] += 1
        if mismatch:
            counts["strict_key_exact_mismatch"] += 1
        if match_lacks_corroboration:
            counts["semantic_match_lacks_exact_strict_corroboration"] += 1
        selected_by_policy = (
            mismatch
            if args.selection_mode == "exact_disagreements_only"
            else low_confidence or mismatch or match_lacks_corroboration
        )
        if selected_by_policy:
            selected.append(item)
    if not selected:
        raise ValueError("resolver selection is empty")

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
        "schema_version": "silver-match-v3-semantic-resolver-pack-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "source_count": len(items),
        "count": len(selected),
        "chunk_size": args.chunk_size,
        "chunk_count": len(chunks),
        "bank_metric_count": len(metrics),
        "bank_source_sha256": bank_hash,
        "seed": args.seed,
        "truth_hidden": True,
        "prior_decisions_and_metric_ids_hidden": True,
        "permanent_blind_rows_in_source": 0,
        "selection_rule": {
            "mode": args.selection_mode,
            "all_semantic_medium_or_low_any_decision": (
                args.selection_mode == "confidence_and_corroboration"
            ),
            "all_exact_strict_key_mismatches": True,
            "all_semantic_matches_without_exact_strict_corroboration": (
                args.selection_mode == "confidence_and_corroboration"
            ),
            "counts_before_union_deduplication": dict(sorted(counts.items())),
        },
        "inputs": {
            "source_pack_validation": {
                "path": str(validation_path),
                "sha256": sha256_file(validation_path),
            },
            "semantic_labels": {
                "path": str(semantic_path),
                "sha256": sha256_file(semantic_path),
            },
            "strict_key": {"path": str(key_path), "sha256": sha256_file(key_path)},
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
