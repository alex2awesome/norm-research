#!/usr/bin/env python3
"""Build the Humor verifier r3 error packet from the frozen optimize split only.

This task-specific audit intentionally has no arguments for select or blind inputs.
The eligible set is optimize74 intersected with resolved truth v2, less the declared
r3 exclusions.  It reports r2 false-retains and a fixed, balanced sample of clear
correct proposals that r2 rejected in both candidate orders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


FALSE_REJECT_SAMPLE = (
    "d19ab316a9035a91d3e8e9e4701cee3d7f2f2fb445ad11fb5864d40a4dca216a",
    "9e6ca0b3d60b3e37475587d395fb90df10fddcfcf0d0d881d11b54ae7cacb4f7",
    "3ce4daca6125942fe524ca019d85b4159fa17df37a77e06ad7c2fe9d39d7b509",
    "9ce35b780b0f5b35bb3e2abfff8f8edd3a1728363f276cc099855e0266a55161",
    "f814deb83cb3f430e7244cc362a3d94d7b0a12f651105fe21758454c6ff6f00d",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def index(rows: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    output = {str(row["norm_uid"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate norm_uid in {path}")
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_prediction(row: dict[str, Any]) -> dict[str, Any]:
    return {
        field: row.get(field)
        for field in ("decision", "metric_id", "confidence", "reason", "order_mode")
    }


def is_confirm(row: dict[str, Any], proposal_metric_id: str) -> bool:
    return (
        row.get("decision") == "CONFIRM_MATCH"
        and str(row.get("metric_id")) == proposal_metric_id
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.input_root).resolve()
    paths = {
        "optimize_truth": root / "optimize74.truth.jsonl",
        "optimize_primary": root / "optimize74.primary.jsonl",
        "optimize_candidates": root / "optimize74.candidates.jsonl",
        "optimize_items": root / "optimize74.items.jsonl",
        "r2_original": root / "r2.optimize74.original.jsonl",
        "r2_hashed": root / "r2.optimize74.hashed.jsonl",
        "resolved_truth_v2": root / "resolved_truth_v2.jsonl",
        "r3_exclusions": root / "r3.false_retain.exclusions.jsonl",
        "eligible_tiebreak": root / "optimize73_eligible.tiebreak.validated.jsonl",
    }
    if any("select" in str(path).lower() or "blind" in str(path).lower() for path in paths.values()):
        raise ValueError("select/blind paths are forbidden in the optimize-only report")
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    raw = {name: read_jsonl(path) for name, path in paths.items()}
    optimize_truth = index(raw["optimize_truth"], paths["optimize_truth"])
    if len(optimize_truth) != 74:
        raise ValueError(f"expected frozen optimize74, found {len(optimize_truth)}")
    optimize_uids = set(optimize_truth)
    indexed = {
        name: index(raw[name], paths[name])
        for name in (
            "optimize_primary",
            "optimize_candidates",
            "optimize_items",
            "r2_original",
            "r2_hashed",
            "resolved_truth_v2",
            "eligible_tiebreak",
        )
    }
    for name in (
        "optimize_primary",
        "optimize_candidates",
        "optimize_items",
        "r2_original",
        "r2_hashed",
    ):
        if set(indexed[name]) != optimize_uids:
            raise ValueError(f"{name} does not exactly cover optimize74")

    excluded = {str(row["norm_uid"]) for row in raw["r3_exclusions"]}
    eligible_uids = sorted(optimize_uids & set(indexed["resolved_truth_v2"]) - excluded)
    if len(eligible_uids) != 73:
        raise ValueError(f"expected 73 revised eligible rows, found {len(eligible_uids)}")
    if not set(indexed["eligible_tiebreak"]) <= set(eligible_uids):
        raise ValueError("tie-break file contains an ineligible UID")

    bank_path = Path(args.bank).resolve()
    bank_payload = json.loads(bank_path.read_text())
    metrics = {str(row["metric_id"]): row for row in bank_payload["metrics"]}

    rows: list[dict[str, Any]] = []
    for uid in eligible_uids:
        gold = indexed["resolved_truth_v2"][uid]
        primary = indexed["optimize_primary"][uid]
        proposal = str(primary["metric_id"])
        target_confirm = gold.get("decision") == "MATCH" and str(gold.get("metric_id")) == proposal
        original = indexed["r2_original"][uid]
        hashed = indexed["r2_hashed"][uid]
        original_confirm = is_confirm(original, proposal)
        hashed_confirm = is_confirm(hashed, proposal)
        rows.append(
            {
                "norm_uid": uid,
                "target": "CONFIRM_MATCH" if target_confirm else "REJECT",
                "gold": gold,
                "primary": primary,
                "item": indexed["optimize_items"][uid],
                "original": original,
                "hashed": hashed,
                "original_confirm": original_confirm,
                "hashed_confirm": hashed_confirm,
            }
        )

    def score(order: str) -> dict[str, Any]:
        confirm_field = f"{order}_confirm"
        counts = Counter()
        for row in rows:
            target_confirm = row["target"] == "CONFIRM_MATCH"
            predicted_confirm = bool(row[confirm_field])
            if target_confirm and predicted_confirm:
                counts["true_accept"] += 1
            elif target_confirm:
                counts["false_reject"] += 1
            elif predicted_confirm:
                counts["false_retain"] += 1
            else:
                counts["true_reject"] += 1
        accepted = counts["true_accept"] + counts["false_retain"]
        positives = counts["true_accept"] + counts["false_reject"]
        return {
            **{name: counts[name] for name in ("true_accept", "false_reject", "true_reject", "false_retain")},
            "accuracy": (counts["true_accept"] + counts["true_reject"]) / len(rows),
            "accept_precision": counts["true_accept"] / accepted if accepted else None,
            "accept_recall": counts["true_accept"] / positives if positives else None,
        }

    both_counts = Counter()
    for row in rows:
        target_confirm = row["target"] == "CONFIRM_MATCH"
        predicted_confirm = row["original_confirm"] and row["hashed_confirm"]
        if target_confirm and predicted_confirm:
            both_counts["true_accept"] += 1
        elif target_confirm:
            both_counts["false_reject"] += 1
        elif predicted_confirm:
            both_counts["false_retain"] += 1
        else:
            both_counts["true_reject"] += 1
    both_accepted = both_counts["true_accept"] + both_counts["false_retain"]
    target_positive = sum(row["target"] == "CONFIRM_MATCH" for row in rows)
    both_score = {
        **{name: both_counts[name] for name in ("true_accept", "false_reject", "true_reject", "false_retain")},
        "accuracy": (both_counts["true_accept"] + both_counts["true_reject"]) / len(rows),
        "accept_precision": both_counts["true_accept"] / both_accepted if both_accepted else None,
        "accept_recall": both_counts["true_accept"] / target_positive if target_positive else None,
    }

    def audit_row(row: dict[str, Any]) -> dict[str, Any]:
        gold = row["gold"]
        primary = row["primary"]
        ids = {
            str(value)
            for value in (
                gold.get("metric_id"),
                primary.get("metric_id"),
                row["original"].get("metric_id"),
                row["hashed"].get("metric_id"),
            )
            if value is not None
        }
        item = row["item"]
        return {
            "norm_uid": row["norm_uid"],
            "source_group": item.get("source_group"),
            "norm": item.get("norm"),
            "context": item.get("context"),
            "target": row["target"],
            "gold": {
                "decision": gold.get("decision"),
                "metric_id": gold.get("metric_id"),
                "agreement_sources": gold.get("agreement_sources"),
                "source_predictions": gold.get("source_predictions"),
            },
            "primary": compact_prediction(primary),
            "r2_original": compact_prediction(row["original"]),
            "r2_hashed": compact_prediction(row["hashed"]),
            "independent_tiebreak": compact_prediction(indexed["eligible_tiebreak"][row["norm_uid"]])
            if row["norm_uid"] in indexed["eligible_tiebreak"]
            else None,
            "metric_cards": {
                metric_id: {
                    key: metrics[metric_id].get(key)
                    for key in ("metric_id", "name", "description", "examples")
                }
                for metric_id in sorted(ids)
            },
        }

    bilateral_false_retains = [
        row
        for row in rows
        if row["target"] == "REJECT" and row["original_confirm"] and row["hashed_confirm"]
    ]
    any_order_false_retains = [
        row
        for row in rows
        if row["target"] == "REJECT" and (row["original_confirm"] or row["hashed_confirm"])
    ]
    sample_index = {row["norm_uid"]: row for row in rows}
    sample = [sample_index[uid] for uid in FALSE_REJECT_SAMPLE]
    for row in sample:
        if row["target"] != "CONFIRM_MATCH" or row["original_confirm"] or row["hashed_confirm"]:
            raise ValueError(f"false-reject sample invariant failed for {row['norm_uid']}")

    report = {
        "schema_version": "silver-match-v3-humor-verifier-gepa-r3-optimize-audit-v1",
        "scope": {
            "split": "source-group-frozen optimize74 only",
            "optimize_count": len(optimize_uids),
            "resolved_truth_v2_intersection_count": len(optimize_uids & set(indexed["resolved_truth_v2"])),
            "excluded_count": len(optimize_uids & excluded),
            "eligible_count": len(rows),
            "selection_and_permanent_blind_content_inspected": False,
            "rule": "eligible = optimize74 intersect resolved_truth_v2 minus r3 exclusions",
        },
        "provenance": {
            "input_sha256": {name: sha256(path) for name, path in paths.items()},
            "bank_sha256": sha256(bank_path),
            "bank_source_sha256": bank_payload.get("source_sha256"),
            "full_independent_tiebreak_labels_sha256_reference": "f397040a3fc7da3b07e1cd1404763775c60571c1991e3ce24f77821f167747a6",
        },
        "gold_distribution": dict(sorted(Counter(row["gold"]["decision"] for row in rows).items())),
        "target_distribution": dict(sorted(Counter(row["target"] for row in rows).items())),
        "r2_scores": {
            "definition": "CONFIRM_MATCH is positive only when metric_id equals the frozen proposal; all other decisions reject.",
            "original_order": score("original"),
            "hashed_order": score("hashed"),
            "both_orders_must_confirm": both_score,
        },
        "independently_confirmed_false_retain_boundary_coverage": {
            "known_confirmed_set_size": 5,
            "eligible_optimize_rows": len(indexed["eligible_tiebreak"]),
            "outside_optimize_or_removed_by_v2_exclusions": 5 - len(indexed["eligible_tiebreak"]),
            "policy": "Only eligible optimize rows are included; no excluded/non-optimize contents are used for r3 gradients.",
        },
        "bilateral_false_retains": [audit_row(row) for row in bilateral_false_retains],
        "all_at_least_one_order_false_retains": [audit_row(row) for row in any_order_false_retains],
        "matched_correct_proposal_false_reject_sample": {
            "population_bilateral_false_reject_count": both_counts["false_reject"],
            "sample_count": len(sample),
            "selection": "Five high-specificity leaf boundaries spanning rate, pauses, ownership, de-escalation, and phonetic word choice.",
            "rows": [audit_row(row) for row in sample],
        },
        "r3_error_taxonomy": [
            {
                "name": "production_object_without_comic_device_function",
                "evidence_uid": "c3c72299baa16e302e58cd82f688a9c4695ae6183cdf557bd3d705a4333613c9",
                "error": "Technical audio-mix defect was falsely retained as a43 instead of a244.",
            },
            {
                "name": "relief_condition_mistaken_for_play_contract",
                "evidence_uid": "f9423b3c20ae288284d2225162daeec03e34f8ba22eccfbc64cd6d84cd91c704",
                "error": "Post-trauma emotional permission/relief was falsely retained as a4 instead of a61.",
            },
            {
                "name": "over_abstention_despite_leaf_owned_object",
                "evidence_uids": list(FALSE_REJECT_SAMPLE),
                "error": "Broad overlapping siblings displaced or tied exact named objects: laugh rate, pauses, recycling, non-escalatory heckler handling, and phonetic diction.",
            },
        ],
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "sha256": sha256(output), "eligible_count": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
