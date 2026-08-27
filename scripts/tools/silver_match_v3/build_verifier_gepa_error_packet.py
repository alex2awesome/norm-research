#!/usr/bin/env python3
"""Build a verifier false-retain/false-reject packet from optimize truth only."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


def _index(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    out = {str(row.get("norm_uid") or ""): row for row in rows}
    if not rows or "" in out or len(out) != len(rows):
        raise ValueError(f"empty, missing, or duplicate norm_uid values: {path}")
    return out


def _confirm(row: dict[str, Any], proposal: str) -> bool:
    return row.get("decision") == "CONFIRM_MATCH" and str(row.get("metric_id")) == proposal


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in ("decision", "metric_id", "confidence", "reason", "order_mode", "prompt_sha256")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--items", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--verifier", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    truth_path, items_path, bank_path, proposal_path = map(
        lambda value: Path(value).resolve(),
        (args.truth, args.items, args.bank, args.proposals),
    )
    truth, items, proposals = map(_index, (truth_path, items_path, proposal_path))
    if any(
        row.get("task") != args.task
        or (row.get("predeclared_split") or row.get("split")) != "train"
        or row.get("gepa_role") != "optimize"
        or row.get("prompt_gradient_eligible") is not True
        for row in truth.values()
    ):
        raise ValueError("truth is not wholly authoritative-train optimize evidence")
    if not set(proposals) <= set(items):
        raise ValueError("proposal universe is not covered by source items")
    eligible_uids = set(proposals) & set(truth)
    if not eligible_uids:
        raise ValueError("no proposal rows intersect authoritative-train optimize truth")
    verifiers: dict[str, dict[str, dict[str, Any]]] = {}
    verifier_paths: dict[str, Path] = {}
    for spec in args.verifier:
        if "=" not in spec:
            raise ValueError("--verifier must be NAME=PATH")
        name, raw = spec.split("=", 1)
        if not name or name in verifiers:
            raise ValueError(f"invalid/duplicate verifier name: {name!r}")
        path = Path(raw).resolve()
        values = _index(path)
        if set(values) != set(proposals):
            raise ValueError(f"verifier/proposal UID mismatch: {name}")
        verifiers[name] = values
        verifier_paths[name] = path
    if len(verifiers) < 2:
        raise ValueError("at least two verifier orders are required")

    bank_payload = json.loads(bank_path.read_text(encoding="utf-8"))
    metrics = {str(row["metric_id"]): row for row in bank_payload["metrics"]}
    rows: list[dict[str, Any]] = []
    outcome_counts: Counter[str] = Counter()
    truth_counts: Counter[str] = Counter()
    leaf_confusions: Counter[tuple[str, str]] = Counter()
    for uid in sorted(eligible_uids):
        gold, proposal = truth[uid], proposals[uid]
        proposal_id = str(proposal.get("metric_id") or "")
        if proposal.get("decision") != "MATCH" or proposal_id not in metrics:
            raise ValueError(f"proposal is not a current-bank MATCH: {uid}")
        target_confirm = gold.get("decision") == "MATCH" and str(gold.get("metric_id")) == proposal_id
        confirms = {name: _confirm(values[uid], proposal_id) for name, values in verifiers.items()}
        retained = all(confirms.values())
        outcome = (
            "true_retain" if target_confirm and retained else
            "false_reject" if target_confirm else
            "false_retain" if retained else
            "true_reject"
        )
        outcome_counts[outcome] += 1
        truth_counts[str(gold.get("decision"))] += 1
        gold_id = str(gold.get("metric_id") or "")
        if gold.get("decision") == "MATCH" and gold_id != proposal_id:
            leaf_confusions[(gold_id, proposal_id)] += 1
        card_ids = {proposal_id}
        if gold_id:
            card_ids.add(gold_id)
        rows.append(
            {
                "norm_uid": uid,
                "task": args.task,
                "source_group": items[uid].get("source_group"),
                "norm": items[uid].get("norm"),
                "context": items[uid].get("context"),
                "outcome": outcome,
                "target": "CONFIRM_MATCH" if target_confirm else "REJECT",
                "gold": {
                    "decision": gold.get("decision"),
                    "metric_id": gold.get("metric_id"),
                    "agreement_sources": gold.get("agreement_sources"),
                    "source_predictions": gold.get("source_predictions"),
                },
                "proposal": _compact(proposal),
                "verifier_orders": {
                    name: {**_compact(values[uid]), "confirms_exact_proposal": confirms[name]}
                    for name, values in verifiers.items()
                },
                "metric_cards": {metric_id: metrics[metric_id] for metric_id in sorted(card_ids)},
                "gepa_error_axis": (
                    f"proposal_should_abstain:{gold.get('decision')}"
                    if gold.get("decision") != "MATCH"
                    else f"wrong_leaf:{proposal_id}->{gold_id}"
                    if not target_confirm
                    else "over_abstention_on_exact_leaf"
                    if outcome == "false_reject"
                    else "correct_exact_leaf"
                ),
            }
        )

    output = Path(args.output).resolve()
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    if output.exists() or meta_path.exists():
        raise FileExistsError(output)
    write_jsonl(output, rows)
    errors = [row for row in rows if row["outcome"] in {"false_retain", "false_reject"}]
    error_path = output.with_name(output.stem + ".errors.jsonl")
    write_jsonl(error_path, errors)
    report = {
        "schema_version": "silver-match-v3-verifier-gepa-error-packet-v1",
        "task": args.task,
        "panel_role": "consumed_selection_authoritative_upstream_train_optimize_only",
        "proposal_count": len(rows),
        "full_consumed_proposal_count": len(proposals),
        "proposals_excluded_as_upstream_nontrain": len(proposals) - len(rows),
        "outcomes": dict(sorted(outcome_counts.items())),
        "truth_decisions": dict(sorted(truth_counts.items())),
        "all_orders_must_confirm_exact_proposal": list(verifiers),
        "leaf_confusions": [
            {"gold_metric_id": gold, "proposal_metric_id": proposal, "count": count}
            for (gold, proposal), count in sorted(leaf_confusions.items(), key=lambda value: (-value[1], value[0]))
        ],
        "inputs": {
            "truth": {"path": str(truth_path), "sha256": sha256_file(truth_path)},
            "items": {"path": str(items_path), "sha256": sha256_file(items_path)},
            "bank": {"path": str(bank_path), "sha256": sha256_file(bank_path)},
            "proposals": {"path": str(proposal_path), "sha256": sha256_file(proposal_path)},
            "verifiers": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in verifier_paths.items()
            },
        },
        "outputs": {
            "all": {"path": str(output), "sha256": sha256_file(output)},
            "errors": {"path": str(error_path), "sha256": sha256_file(error_path)},
        },
    }
    meta_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "report_sha256": sha256_file(meta_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
