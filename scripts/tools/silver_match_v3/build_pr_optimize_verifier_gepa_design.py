#!/usr/bin/env python3
"""Freeze PR verifier-GEPA inputs from optimize truth and aggregate dev failures.

The output deliberately contains no consumed verifier-dev identity, text, metric
leaf, or row-level error.  Detailed examples come only from the authoritative
optimize panel.  The seed proposal covers every optimize row so a later builder
can deterministically create a balanced exact-match verifier slate.
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
    out = {str(row.get("norm_uid") or ""): row for row in rows}
    if not rows or "" in out or len(out) != len(rows):
        raise ValueError(f"empty, missing, or duplicate norm_uid: {path}")
    return out


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _choose_seed(
    uid: str,
    candidates: dict[str, Any],
    orders: dict[str, dict[str, dict[str, Any]]],
) -> tuple[str, str]:
    candidate_ids = [str(row["metric_id"]) for row in candidates.get("candidates") or []]
    if not candidate_ids:
        raise ValueError(f"empty candidate slate: {uid}")
    votes: Counter[str] = Counter()
    first_order: dict[str, int] = {}
    for position, name in enumerate(("original", "hashed", "reverse")):
        row = orders[name][uid]
        metric_id = str(row.get("metric_id") or "")
        if row.get("decision") == "MATCH" and metric_id in candidate_ids:
            votes[metric_id] += 1
            first_order.setdefault(metric_id, position)
    if votes:
        chosen = min(votes, key=lambda value: (-votes[value], first_order[value], value))
        return chosen, "r4_match_vote"
    return candidate_ids[0], "retriever_top1_when_r4_typed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--items", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--r4-original", required=True)
    parser.add_argument("--r4-hashed", required=True)
    parser.add_argument("--r4-reverse", required=True)
    parser.add_argument("--gemma-dev-score", required=True)
    parser.add_argument("--gemma-rejection-freeze", required=True)
    parser.add_argument("--codex-dev-score", required=True)
    parser.add_argument("--codex-rejection-freeze", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    paths = {
        name: Path(getattr(args, name)).resolve()
        for name in (
            "truth",
            "items",
            "bank",
            "candidates",
            "r4_original",
            "r4_hashed",
            "r4_reverse",
            "gemma_dev_score",
            "gemma_rejection_freeze",
            "codex_dev_score",
            "codex_rejection_freeze",
        )
    }
    output = Path(args.output_root).resolve()
    if output.exists():
        raise FileExistsError(output)

    truth = _index(paths["truth"])
    items = _index(paths["items"])
    candidates = _index(paths["candidates"])
    orders = {
        name: _index(paths[f"r4_{name}"])
        for name in ("original", "hashed", "reverse")
    }
    uids = set(truth)
    if set(items) != uids or set(candidates) != uids or any(set(rows) != uids for rows in orders.values()):
        raise ValueError("optimize inputs do not have exact UID coverage")
    groups = [str(row.get("source_group") or "") for row in truth.values()]
    if any(
        row.get("task") != "press-releases"
        or row.get("gepa_role") != "optimize"
        or (row.get("predeclared_split") or row.get("split")) != "train"
        or row.get("prompt_gradient_eligible") is not True
        for row in truth.values()
    ) or "" in groups or len(groups) != len(set(groups)):
        raise ValueError("truth is not source-disjoint authoritative optimize evidence")

    bank = _load(paths["bank"])
    bank_ids = {str(row.get("metric_id") or "") for row in bank.get("metrics") or []}
    if bank.get("task") != "press-releases" or "" in bank_ids or not bank_ids:
        raise ValueError("invalid PR metric bank")

    source_counts: Counter[str] = Counter()
    primary: list[dict[str, Any]] = []
    for uid in sorted(uids):
        metric_id, source = _choose_seed(uid, candidates[uid], orders)
        if metric_id not in bank_ids:
            raise ValueError(f"seed metric absent from bank: {uid}/{metric_id}")
        source_counts[source] += 1
        base = orders["original"][uid]
        primary.append(
            {
                **base,
                "decision": "MATCH",
                "metric_id": metric_id,
                "seed_proposal_source": source,
                "reason": "Frozen optimize-only seed proposal; target is joined only inside verifier GEPA.",
            }
        )

    gemma_score = _load(paths["gemma_dev_score"])
    codex_score = _load(paths["codex_dev_score"])
    gemma_rejection = _load(paths["gemma_rejection_freeze"])
    codex_rejection = _load(paths["codex_rejection_freeze"])
    if (
        gemma_rejection.get("status")
        != "REJECTED_APPEND_ONLY_NO_PROMOTION_OR_RETUNING_ON_CONSUMED_DEV"
        or codex_rejection.get("status")
        != "REJECTED_APPEND_ONLY_NO_PROMOTION_OR_RETUNING_ON_CONSUMED_DEV"
        or gemma_score.get("all_gates_pass") is not False
        or codex_score.get("all_gates_pass") is not False
    ):
        raise ValueError("consumed dev attempts are not append-only rejected")

    output.mkdir(parents=True)
    primary_path = output / "seed_primary.all_optimize.jsonl"
    write_jsonl(primary_path, primary)
    taxonomy = {
        "schema_version": "silver-match-v3-pr-verifier-aggregate-error-taxonomy-v1",
        "status": "FROZEN_IDENTITY_FREE_AGGREGATES_FOR_OPTIMIZE_ONLY_PROMPT_AUTHORING",
        "task": "press-releases",
        "contracts": {
            "consumed_dev_norm_uids_included": False,
            "consumed_dev_source_groups_included": False,
            "consumed_dev_text_included": False,
            "consumed_dev_metric_ids_included": False,
            "consumed_dev_row_level_reasons_included": False,
            "detailed_examples_may_come_only_from_optimize_truth": True,
            "may_not_retune_on_consumed_dev": True,
        },
        "aggregate_attempts": {
            "gemma_accepted_v4": {
                "n": int(gemma_score["n"]),
                "retained": int(gemma_score["retained"]),
                "retained_true": int(gemma_score["retained_true"]),
                "false_retained": int(gemma_score["false_retained"]),
                "retained_precision": float(gemma_score["retained_precision"]),
                "retained_precision_wilson_95": gemma_score["retained_precision_wilson_95"],
                "order_exact_agreement": float(
                    (gemma_score.get("order_stability") or {})[
                        "exact_decision_and_id_agreement"
                    ]
                ),
            },
            "proposal_hidden_codex": {
                "n": int(codex_score["n"]),
                "retained": int(codex_score["retained"]),
                "retained_true": int(codex_score["retained_true"]),
                "false_retained": int(codex_score["false_retained"]),
                "retained_precision": float(codex_score["retained_precision"]),
                "retained_precision_wilson_95": codex_score["retained_precision_wilson_95"],
                "retained_recall_of_correct_proposals": float(
                    codex_score["retained_recall_of_correct_proposals"]
                ),
            },
        },
        "prompt_axes": [
            {
                "axis": "explicit_normative_predicate_before_leaf_selection",
                "instruction": "Reject topical facts, status, examples, or outcomes unless the text explicitly evaluates or reveals a communicative norm.",
            },
            {
                "axis": "exact_leaf_entailment",
                "instruction": "Confirm only when the evidence entails the proposal leaf itself, not merely its family or a nearby sibling.",
            },
            {
                "axis": "contrastive_disconfirmation",
                "instruction": "Name the strongest competing supplied leaf and reject when it is equally or better supported.",
            },
            {
                "axis": "order_robustness",
                "instruction": "Do not use candidate rank or position as evidence; decisions must survive candidate permutation.",
            },
            {
                "axis": "confidence_is_not_a_gate_substitute",
                "instruction": "Use high confidence only after both explicit-criterion and exact-leaf tests pass; otherwise return a typed nonmatch.",
            },
        ],
    }
    taxonomy_path = output / "aggregate_error_taxonomy.json"
    taxonomy_path.write_text(
        json.dumps(taxonomy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": "silver-match-v3-pr-optimize-verifier-gepa-design-v1",
        "status": "FROZEN_OPTIMIZE_ONLY_BEFORE_THIRD_DESIGN_PROMPT_AUTHORING",
        "task": "press-releases",
        "optimize_count": len(uids),
        "optimize_source_group_count": len(set(groups)),
        "seed_proposal_source_counts": dict(sorted(source_counts.items())),
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "outputs": {
            "seed_primary": {
                "path": str(primary_path),
                "sha256": sha256_file(primary_path),
            },
            "aggregate_error_taxonomy": {
                "path": str(taxonomy_path),
                "sha256": sha256_file(taxonomy_path),
            },
        },
    }
    report_path = output / "FREEZE.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({**report, "freeze_sha256": sha256_file(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
