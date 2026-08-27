#!/usr/bin/env python3
"""Join frozen optimize-only proposals to truth for verifier prompt authorship.

The packet is training/GEPA evidence only.  It cannot contain verifier-dev or
blind-audit identities, and every proposal is assigned an exact binary target:
CONFIRM_MATCH only when both the gold decision and bank leaf agree; otherwise
REJECT.  Fresh verifier-dev truth is neither an input nor an output.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


def _index(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    values = {str(row.get("norm_uid") or ""): row for row in rows}
    if not rows or "" in values or len(values) != len(rows):
        raise ValueError(f"empty, missing, or duplicate norm_uid values: {path}")
    return values


def build_packet(
    *,
    task: str,
    truth_path: Path,
    items_path: Path,
    bank_path: Path,
    proposals_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    paths = {
        "truth": truth_path.resolve(),
        "items": items_path.resolve(),
        "bank": bank_path.resolve(),
        "proposals": proposals_path.resolve(),
    }
    truth = _index(paths["truth"])
    items = _index(paths["items"])
    proposals = _index(paths["proposals"])
    if not set(proposals) <= set(truth) & set(items):
        raise ValueError("proposal UIDs are not covered by optimize truth/items")
    for uid, row in truth.items():
        if (
            row.get("task") != task
            or row.get("gepa_role") != "optimize"
            or (row.get("predeclared_split") or row.get("split")) != "train"
            or row.get("prompt_gradient_eligible") is not True
        ):
            raise ValueError(f"truth is not optimize-only gradient evidence: {uid}")

    bank = json.loads(paths["bank"].read_text(encoding="utf-8"))
    metrics = {str(row.get("metric_id") or ""): row for row in bank.get("metrics") or []}
    if (
        bank.get("task") != task
        or not metrics
        or "" in metrics
        or len(metrics) != len(bank.get("metrics") or [])
    ):
        raise ValueError("invalid task-local metric bank")

    rows: list[dict[str, Any]] = []
    targets: Counter[str] = Counter()
    proposal_metrics: Counter[str] = Counter()
    for uid in sorted(proposals):
        proposal, gold, item = proposals[uid], truth[uid], items[uid]
        proposal_id = str(proposal.get("metric_id") or "")
        if proposal.get("task") != task or proposal.get("decision") != "MATCH":
            raise ValueError(f"proposal is not a task-local MATCH: {uid}")
        if proposal_id not in metrics:
            raise ValueError(f"proposal leaf is absent from current bank: {uid}")
        gold_id = str(gold.get("metric_id") or "")
        correct = gold.get("decision") == "MATCH" and gold_id == proposal_id
        target = "CONFIRM_MATCH" if correct else "REJECT"
        card_ids = {proposal_id}
        if gold.get("decision") == "MATCH":
            if gold_id not in metrics:
                raise ValueError(f"gold leaf is absent from current bank: {uid}")
            card_ids.add(gold_id)
        source_group = str(gold.get("source_group") or item.get("source_group") or "")
        if not source_group:
            raise ValueError(f"missing optimize source group: {uid}")
        targets[target] += 1
        proposal_metrics[proposal_id] += 1
        rows.append(
            {
                "schema_version": "silver-match-v3-verifier-author-example-v1",
                "norm_uid": uid,
                "task": task,
                "corpus": item.get("corpus"),
                "source_group": source_group,
                "gepa_role": "optimize",
                "predeclared_split": "train",
                "norm": item.get("norm"),
                "context": item.get("context"),
                "proposal": {
                    key: proposal.get(key)
                    for key in ("decision", "metric_id", "confidence", "reason", "model", "prompt_sha256")
                },
                "gold": {
                    key: gold.get(key)
                    for key in ("decision", "metric_id", "confidence", "agreement_sources")
                },
                "target": target,
                "metric_cards": {
                    metric_id: metrics[metric_id] for metric_id in sorted(card_ids)
                },
                "use_contract": {
                    "verifier_prompt_authorship_or_gepa_optimize_only": True,
                    "verifier_selection": False,
                    "final_blind_audit": False,
                    "retriever_training": False,
                    "mi_or_outcome_estimation": False,
                },
            }
        )
    groups = [str(row["source_group"]) for row in rows]
    if len(groups) != len(set(groups)):
        raise ValueError("optimize proposal packet repeats a source group")
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    examples_path = output_root / "examples.jsonl"
    write_jsonl(examples_path, rows)
    report = {
        "schema_version": "silver-match-v3-verifier-author-training-packet-v1",
        "status": "FROZEN_OPTIMIZE_ONLY_AUTHORSHIP_EVIDENCE",
        "task": task,
        "count": len(rows),
        "source_groups": len(groups),
        "target_counts": dict(sorted(targets.items())),
        "proposal_metric_counts": dict(sorted(proposal_metrics.items())),
        "fresh_verifier_dev_truth_read": False,
        "blind_audit_truth_read": False,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "outputs": {
            "examples": {
                "path": str(examples_path),
                "sha256": sha256_file(examples_path),
            }
        },
    }
    report_path = output_root / "REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {**report, "report_sha256": sha256_file(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--items", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    report = build_packet(
        task=args.task,
        truth_path=Path(args.truth),
        items_path=Path(args.items),
        bank_path=Path(args.bank),
        proposals_path=Path(args.proposals),
        output_root=Path(args.output_root),
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
