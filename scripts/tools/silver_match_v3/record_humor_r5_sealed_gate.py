#!/usr/bin/env python3
"""Record the permanent sealed Humor R5 gate and diagnostic decomposition."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def rate(num: int, den: int) -> float | None:
    return num / den if den else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--adjudicator-original", required=True)
    parser.add_argument("--adjudicator-hashed", required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--verifier-original", required=True)
    parser.add_argument("--verifier-hashed", required=True)
    parser.add_argument("--verifier-reverse", required=True)
    parser.add_argument("--two-order-score", required=True)
    parser.add_argument("--three-order-score", required=True)
    parser.add_argument("--adjudicator-score", required=True)
    parser.add_argument("--selection-freeze", required=True)
    parser.add_argument("--output-freeze", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = {key: Path(value).resolve() for key, value in vars(args).items() if key != "output"}
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)

    truth_rows = list(read_jsonl(paths["truth"]))
    truth = {str(row["norm_uid"]): row for row in truth_rows}
    candidates = {str(row["norm_uid"]): row for row in read_jsonl(paths["candidates"])}
    adj = {
        name: {str(row["norm_uid"]): row for row in read_jsonl(paths[f"adjudicator_{name}"])}
        for name in ("original", "hashed")
    }
    primary = {str(row["norm_uid"]): row for row in read_jsonl(paths["primary"])}
    verifier = {
        name: {str(row["norm_uid"]): row for row in read_jsonl(paths[f"verifier_{name}"])}
        for name in ("original", "hashed", "reverse")
    }
    uids = set(truth)
    if (
        len(truth_rows) != len(uids)
        or set(candidates) != uids
        or any(set(value) != uids for value in adj.values())
        or any(set(value) != set(primary) for value in verifier.values())
    ):
        raise ValueError("sealed diagnostic inputs lack exact expected coverage")

    selection = json.loads(paths["selection_freeze"].read_text(encoding="utf-8"))
    output_freeze = json.loads(paths["output_freeze"].read_text(encoding="utf-8"))
    if (
        output_freeze.get("status") != "FROZEN_COMPLETE_BEFORE_TRUTH_JOIN"
        or output_freeze.get("sealed_truth_sha256_not_opened") != sha256_file(paths["truth"])
        or output_freeze.get("permanent_blind_consumed") is not False
    ):
        raise ValueError("predictions were not frozen against this sealed truth")
    gate = selection.get("selection_gate") or {}
    required_gate = {
        "minimum_retained_precision": 0.9,
        "minimum_retained_support": 20,
        "minimum_wilson_95_lower": 0.8,
        "thresholds_lowered": False,
    }
    if gate != required_gate:
        raise ValueError("unexpected R5 selection gate")
    two = json.loads(paths["two_order_score"].read_text(encoding="utf-8"))
    three = json.loads(paths["three_order_score"].read_text(encoding="utf-8"))
    adj_score = json.loads(paths["adjudicator_score"].read_text(encoding="utf-8"))

    def gate_record(policy: dict[str, Any]) -> dict[str, Any]:
        checks = {
            "retained_support": int(policy["retained"]) >= int(gate["minimum_retained_support"]),
            "retained_precision": float(policy["retained_precision"] or 0)
            >= float(gate["minimum_retained_precision"]),
            "wilson_95_lower": float((policy["retained_precision_wilson_95"] or [0])[0])
            >= float(gate["minimum_wilson_95_lower"]),
        }
        return {"policy": policy, "checks": checks, "eligible": all(checks.values())}

    two_gate = gate_record(two["policies"]["high_only"])
    three_gate = gate_record(three["policy"])
    if two_gate["eligible"] or three_gate["eligible"]:
        raise ValueError("negative-gate recorder called on an eligible policy")

    truth_decisions = Counter(str(row["decision"]) for row in truth_rows)
    proposal_truth = Counter()
    verifier_order_decisions = {
        name: Counter(str(row["decision"]) for row in rows.values())
        for name, rows in verifier.items()
    }
    terminal = Counter()
    retained_by_truth = Counter()
    confusion = Counter()
    for uid, gold in truth.items():
        if uid not in primary:
            left, right = adj["original"][uid], adj["hashed"][uid]
            if (
                left.get("decision") == right.get("decision")
                and left.get("metric_id") == right.get("metric_id")
                and left.get("decision") != "MATCH"
            ):
                terminal[f"ABSTAIN:{left['decision']}"] += 1
            else:
                terminal["ABSTAIN:ADJUDICATOR_REJECT_OR_DISAGREE"] += 1
            continue
        proposed = str(primary[uid]["metric_id"])
        correct = gold.get("decision") == "MATCH" and str(gold.get("metric_id")) == proposed
        proposal_truth["correct"] += int(correct)
        proposal_truth["incorrect"] += int(not correct)
        rows = [verifier[name][uid] for name in ("original", "hashed", "reverse")]
        keep = all(
            row.get("decision") == "CONFIRM_MATCH"
            and str(row.get("metric_id")) == proposed
            and row.get("confidence") == "high"
            and not row.get("parse_error")
            for row in rows
        )
        confusion[f"proposal_{'correct' if correct else 'wrong'}__{'retained' if keep else 'rejected'}"] += 1
        if keep:
            terminal["MATCH"] += 1
            retained_by_truth[str(gold["decision"])] += 1
        else:
            exact_outcomes = {(row.get("decision"), row.get("metric_id")) for row in rows}
            if len(exact_outcomes) == 1 and rows[0].get("decision") != "CONFIRM_MATCH":
                terminal[f"ABSTAIN:{rows[0]['decision']}"] += 1
            elif all(row.get("decision") == "CONFIRM_MATCH" for row in rows):
                terminal["ABSTAIN:LOW_CONFIDENCE"] += 1
            else:
                terminal["ABSTAIN:VERIFIER_REJECT_OR_DISAGREE"] += 1

    match_truth = [row for row in truth_rows if row.get("decision") == "MATCH"]
    by_metric: dict[str, dict[str, int]] = defaultdict(lambda: {"truth_match_count": 0, "gold_in_k50": 0})
    gold_in_k50_uids: set[str] = set()
    for row in match_truth:
        uid = str(row["norm_uid"])
        metric = str(row["metric_id"])
        candidate_ids = {str(value["metric_id"]) for value in candidates[uid].get("candidates") or []}
        present = metric in candidate_ids
        by_metric[metric]["truth_match_count"] += 1
        by_metric[metric]["gold_in_k50"] += int(present)
        if present:
            gold_in_k50_uids.add(uid)

    conditional: dict[str, Any] = {}
    for name, rows in adj.items():
        correct = sum(
            rows[uid].get("decision") == "MATCH"
            and str(rows[uid].get("metric_id")) == str(truth[uid].get("metric_id"))
            for uid in gold_in_k50_uids
        )
        conditional[name] = {
            "gold_in_k50_count": len(gold_in_k50_uids),
            "exact_correct": correct,
            "exact_accuracy_given_gold_in_k50": rate(correct, len(gold_in_k50_uids)),
        }
    strict_correct = sum(
        uid in primary and str(primary[uid].get("metric_id")) == str(truth[uid].get("metric_id"))
        for uid in gold_in_k50_uids
    )
    strict_proposed = sum(uid in primary for uid in gold_in_k50_uids)
    by_metric_report = {
        metric: {
            **counts,
            "recall_at_50": rate(counts["gold_in_k50"], counts["truth_match_count"]),
        }
        for metric, counts in sorted(by_metric.items())
    }

    result = {
        "schema_version": "silver-match-v3-humor-r5-sealed-negative-gate-v1",
        "status": "PERMANENTLY_INELIGIBLE_SELECT_GATE_FAILED",
        "task": "humor",
        "role": "sealed_select_diagnostic_only",
        "predeclared_gate": gate,
        "gate_results": {"two_order_high_only": two_gate, "three_order_exact_high": three_gate},
        "truth_distribution": {"count": len(truth_rows), "decisions": dict(sorted(truth_decisions.items()))},
        "adjudicator": {
            "strict_consensus": adj_score["metrics"]["strict_consensus"],
            "proposal_truth": dict(sorted(proposal_truth.items())),
            "conditional_given_gold_in_k50": {
                **conditional,
                "strict_two_order_consensus": {
                    "gold_in_k50_count": len(gold_in_k50_uids),
                    "proposed": strict_proposed,
                    "proposal_rate": rate(strict_proposed, len(gold_in_k50_uids)),
                    "exact_correct": strict_correct,
                    "exact_accuracy_given_gold_in_k50": rate(strict_correct, len(gold_in_k50_uids)),
                },
            },
        },
        "retrieval": {
            "candidate_k": 50,
            "truth_match_count": len(match_truth),
            "gold_in_k50": len(gold_in_k50_uids),
            "recall_at_50": rate(len(gold_in_k50_uids), len(match_truth)),
            "gold_missing_k50": len(match_truth) - len(gold_in_k50_uids),
            "by_metric": by_metric_report,
        },
        "verifier": {
            "per_order_decisions": {
                name: dict(sorted(counts.items())) for name, counts in verifier_order_decisions.items()
            },
            "three_order_confusion": dict(sorted(confusion.items())),
            "retained_by_truth_decision": dict(sorted(retained_by_truth.items())),
        },
        "diagnostic_terminal_outputs_not_releasable": {
            "counts": dict(sorted(terminal.items())),
            "rates": {key: value / len(truth_rows) for key, value in sorted(terminal.items())},
            "match_count": terminal["MATCH"],
            "match_rate": terminal["MATCH"] / len(truth_rows),
            "abstain_count": len(truth_rows) - terminal["MATCH"],
            "abstain_rate": (len(truth_rows) - terminal["MATCH"]) / len(truth_rows),
            "explicit_noise_count": terminal["ABSTAIN:NOISE"],
            "explicit_noise_rate": terminal["ABSTAIN:NOISE"] / len(truth_rows),
        },
        "diagnosis": {
            "retriever_retraining_needed": len(gold_in_k50_uids) < len(match_truth),
            "adjudicator_or_cross_encoder_retraining_needed": strict_correct < len(gold_in_k50_uids),
            "verifier_alone_is_sufficient": False,
        },
        "permanent_constraints": {
            "promote_r5": False,
            "deploy_r5_locally": False,
            "run_r5_cross_runtime_equivalence_for_deployment": False,
            "iterate_prompt_or_threshold_on_select": False,
            "consume_permanent_blind": False,
            "allowed_next_path": "new leakage-clean learned model with sealed select used only as dev reporting",
        },
        "inputs": {key: artifact(path) for key, path in sorted(paths.items())},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**result, "output": str(output), "sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
