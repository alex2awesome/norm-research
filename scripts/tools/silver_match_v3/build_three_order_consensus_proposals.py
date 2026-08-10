#!/usr/bin/env python3
"""Keep exact MATCH proposals stable across original, hashed, and reverse order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


ORDERS = ("original", "hashed", "reverse")


def _index(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    indexed = {str(row.get("norm_uid") or ""): row for row in rows}
    if not rows or "" in indexed or len(indexed) != len(rows):
        raise ValueError(f"empty, missing, or duplicate norm_uid: {path}")
    return indexed


def build(paths: dict[str, Path], task: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values = {order: _index(paths[order]) for order in ORDERS}
    uids = set(values[ORDERS[0]])
    if any(set(values[order]) != uids for order in ORDERS[1:]):
        raise ValueError("three order proposal inputs have different coverage")
    prompt_hashes = {
        str(row.get("prompt_sha256") or "")
        for order in ORDERS
        for row in values[order].values()
    }
    bank_hashes = {
        str(row.get("candidate_bank_source_sha256") or row.get("bank_source_sha256") or "")
        for order in ORDERS
        for row in values[order].values()
    }
    models = {
        str(row.get("model") or "")
        for order in ORDERS
        for row in values[order].values()
    }
    if len(prompt_hashes) != 1 or "" in prompt_hashes:
        raise ValueError("three order outputs do not share one prompt hash")
    if len(bank_hashes) != 1 or "" in bank_hashes:
        raise ValueError("three order outputs do not share one bank hash")
    if len(models) != 1 or "" in models:
        raise ValueError("three order outputs do not share one model")

    selected: list[dict[str, Any]] = []
    exact_agreement = decision_agreement = 0
    for uid in sorted(uids):
        rows = [values[order][uid] for order in ORDERS]
        if any(row.get("task") != task or row.get("order_mode") != order for order, row in zip(ORDERS, rows, strict=True)):
            raise ValueError(f"task/order mismatch: {uid}")
        decisions = {str(row.get("decision") or "") for row in rows}
        keys = {(str(row.get("decision") or ""), row.get("metric_id")) for row in rows}
        decision_agreement += len(decisions) == 1
        exact_agreement += len(keys) == 1
        if len(keys) == 1 and rows[0].get("decision") == "MATCH":
            selected.append(
                {
                    **rows[0],
                    "consensus_order_modes": list(ORDERS),
                    "consensus_metric_id": rows[0]["metric_id"],
                    "order_confidences": {
                        order: row.get("confidence")
                        for order, row in zip(ORDERS, rows, strict=True)
                    },
                    "order_reasons": {
                        order: row.get("reason")
                        for order, row in zip(ORDERS, rows, strict=True)
                    },
                    "order_output_sha256": {
                        order: sha256_file(paths[order]) for order in ORDERS
                    },
                    "label_source": "three_order_exact_match_proposal",
                }
            )
    report = {
        "schema_version": "silver-match-v3-three-order-consensus-proposals-v1",
        "task": task,
        "input_count": len(uids),
        "decision_agreement_count": decision_agreement,
        "exact_decision_and_id_agreement_count": exact_agreement,
        "consensus_match_count": len(selected),
        "prompt_sha256": next(iter(prompt_hashes)),
        "bank_source_sha256": next(iter(bank_hashes)),
        "model": next(iter(models)),
        "inputs": {
            order: {"path": str(paths[order]), "sha256": sha256_file(paths[order])}
            for order in ORDERS
        },
    }
    return selected, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for order in ORDERS:
        parser.add_argument(f"--{order}", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-freeze", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {order: Path(getattr(args, order)).resolve() for order in ORDERS}
    output = Path(args.output).resolve()
    report_path = output.with_suffix(output.suffix + ".report.json")
    freeze_path = Path(args.output_freeze).resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError(output)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") not in {
        "FROZEN_COMPLETE_BEFORE_TRUTH_JOIN",
        "FROZEN_COMPLETE_BEFORE_VERIFIER_AUTHORING",
    }:
        raise ValueError("outputs were not frozen before consensus construction")
    frozen_paths = {
        order: Path(freeze["outputs"][order]["predictions"]["path"]).resolve()
        for order in ORDERS
    }
    if frozen_paths != paths or any(
        sha256_file(paths[order]) != freeze["outputs"][order]["predictions"]["sha256"]
        for order in ORDERS
    ):
        raise ValueError("consensus inputs differ from frozen outputs")
    selected, report = build(paths, args.task)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, selected)
    report.update(
        {
            "output_freeze": {"path": str(freeze_path), "sha256": sha256_file(freeze_path)},
            "output": {"path": str(output), "sha256": sha256_file(output)},
        }
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "report_sha256": sha256_file(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
