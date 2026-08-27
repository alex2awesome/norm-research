#!/usr/bin/env python3
"""Materialize every bank leaf for each item in a truth-hidden label pack.

This is an inference-only bridge for independent full-bank model passes.  It
binds the immutable pack, exposes no prior decisions/proposals, and preserves
the pack's deterministic metric order so alternate order runs can be audited.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


def materialize(pack_root: Path, output: Path, report: Path) -> dict[str, Any]:
    pack_root = pack_root.resolve()
    output = output.resolve()
    report = report.resolve()
    if output.exists() or report.exists():
        raise FileExistsError("refusing to overwrite full-bank blind candidates")
    validation_path = pack_root / "validation.json"
    items_path = pack_root / "items.jsonl"
    bank_path = pack_root / "bank.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("truth_hidden") is not True:
        raise ValueError("source pack is not truth-hidden")
    if sha256_file(items_path) != validation["outputs"]["items"]["sha256"]:
        raise ValueError("source item hash mismatch")
    if sha256_file(bank_path) != validation["outputs"]["bank"]["sha256"]:
        raise ValueError("source bank hash mismatch")
    items = list(read_jsonl(items_path))
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    task = str(validation["task"])
    bank_sha = str(validation["bank_source_sha256"])
    if bank.get("task") != task or bank.get("source_sha256") != bank_sha:
        raise ValueError("source pack task/bank provenance mismatch")
    metrics = list(bank.get("metrics") or [])
    metric_ids = [str(row["metric_id"]) for row in metrics]
    if not metric_ids or len(metric_ids) != len(set(metric_ids)):
        raise ValueError("bank contains no metrics or duplicate IDs")
    uids = [str(row.get("norm_uid") or "") for row in items]
    if "" in uids or len(uids) != len(set(uids)):
        raise ValueError("items contain missing or duplicate UIDs")
    foreign = [row.get("norm_uid") for row in items if row.get("task") != task]
    if foreign:
        raise ValueError(f"pack contains foreign task rows: {foreign[:3]}")

    candidates = [
        {
            "metric_id": metric_id,
            "rank": rank,
            "score": None,
            "candidate_source": "truth_hidden_full_bank",
        }
        for rank, metric_id in enumerate(metric_ids, 1)
    ]
    rows = [
        {
            "schema_version": row.get("schema_version") or "silver-match-v3.0",
            "norm_uid": row["norm_uid"],
            "corpus": row["corpus"],
            "task": task,
            "row": row["row"],
            "bank_source_sha256": bank_sha,
            "candidates": candidates,
            "candidate_depth": len(candidates),
            "truth_hidden": True,
            "prior_predictions_hidden": True,
        }
        for row in items
    ]
    write_jsonl(output, rows)
    result = {
        "schema_version": "silver-match-v3-full-bank-blind-candidates-freeze-v1",
        "status": "FROZEN_BEFORE_INFERENCE",
        "task": task,
        "count": len(rows),
        "unique_uids": len(uids),
        "candidate_depth": len(metric_ids),
        "bank_source_sha256": bank_sha,
        "truth_hidden": True,
        "prior_decisions_metric_ids_predictions_and_proposals_read": False,
        "inputs": {
            "pack_validation": {
                "path": str(validation_path),
                "sha256": sha256_file(validation_path),
            },
            "items": {"path": str(items_path), "sha256": sha256_file(items_path)},
            "bank": {"path": str(bank_path), "sha256": sha256_file(bank_path)},
        },
        "output": {"path": str(output), "sha256": sha256_file(output)},
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**result, "report": {"path": str(report), "sha256": sha256_file(report)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(Path(args.pack_root), Path(args.output), Path(args.report)),
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
