#!/usr/bin/env python3
"""Select the frozen Humor adjudicator/verifier stack from fresh human truth.

The complete Gemma cross-product must be sealed before this script can read
truth. Adjudicator variants are selected first from strict two-order proposal
precision. Only then are the predeclared verifier variants compared for the
chosen proposal family. No prompt, threshold, or candidate set is mutated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file
from .run_humor_fresh_select_gpu_queue import _cell_complete, validate_queue
from .score_verifier_calibration import safe_rate, wilson_interval


POLICY_SCHEMA = "silver-match-v3-humor-fresh-release-v2-prelabel-policy-v1"
TRUTH_SCHEMA = "silver-match-v3-exact-multi-pass-truth-report-v1"


def _index(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    indexed = {str(row.get("norm_uid") or ""): row for row in rows}
    if "" in indexed or len(indexed) != len(rows):
        raise ValueError(f"missing or duplicate norm_uid values: {path}")
    return indexed


def _f_beta(precision: float | None, recall: float | None, beta: float = 0.5) -> float:
    if precision is None or recall is None or precision + recall == 0:
        return 0.0
    beta2 = beta * beta
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def _lower(interval: list[float] | None) -> float:
    return -1.0 if interval is None else float(interval[0])


def _load_truth(
    truth_path: Path,
    report_path: Path,
    *,
    expected_panel_role: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    truth = _index(truth_path)
    output = (report.get("outputs") or {}).get("resolved") or {}
    if (
        report.get("schema_version") != TRUTH_SCHEMA
        or report.get("task") != "humor"
        or report.get("gepa_role") != "select"
        or report.get("gepa_panel_role") != expected_panel_role
        or output.get("sha256") != sha256_file(truth_path)
        or int(report.get("resolved_count", -1)) != len(truth)
        or int(report.get("source_count", -1))
        != int(report.get("resolved_count", -2)) + int(report.get("unresolved_count", -3))
    ):
        raise ValueError(f"truth report is not the frozen {expected_panel_role} release")
    for uid, row in truth.items():
        if (
            row.get("task") != "humor"
            or row.get("gepa_role") != "select"
            or row.get("gepa_panel_role") != expected_panel_role
            or row.get("prompt_selection_eligible") is not True
            or row.get("prompt_gradient_eligible") is not False
            or row.get("training_eligible") is not False
        ):
            raise ValueError(f"truth row violates selection-only role: {uid}")
    return truth, report


def _load_strict_consensus(
    original_path: Path,
    hashed_path: Path,
    consensus_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    original, hashed, consensus = (
        _index(path) for path in (original_path, hashed_path, consensus_path)
    )
    if set(original) != set(hashed):
        raise ValueError("adjudicator order outputs have different UID coverage")
    expected = {
        uid: row
        for uid, row in original.items()
        if row.get("decision") == hashed[uid].get("decision") == "MATCH"
        and row.get("metric_id") == hashed[uid].get("metric_id")
    }
    if set(consensus) != set(expected) or any(
        consensus[uid].get("metric_id") != expected[uid].get("metric_id")
        for uid in expected
    ):
        raise ValueError("consensus proposals are not the exact two-order MATCH set")
    return consensus, {
        "input_count": len(original),
        "consensus_match_count": len(consensus),
        "order_exact_agreement_count": sum(
            (original[uid].get("decision"), original[uid].get("metric_id"))
            == (hashed[uid].get("decision"), hashed[uid].get("metric_id"))
            for uid in original
        ),
        "inputs": {
            "original": {"path": str(original_path), "sha256": sha256_file(original_path)},
            "hashed": {"path": str(hashed_path), "sha256": sha256_file(hashed_path)},
            "consensus": {"path": str(consensus_path), "sha256": sha256_file(consensus_path)},
        },
    }


def score_adjudicator(
    truth: dict[str, dict[str, Any]],
    proposals: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    truth_matches = sum(row.get("decision") == "MATCH" for row in truth.values())
    scored_uids = sorted(set(truth) & set(proposals))
    correct = sum(
        truth[uid].get("decision") == "MATCH"
        and str(truth[uid].get("metric_id")) == str(proposals[uid].get("metric_id"))
        for uid in scored_uids
    )
    precision = safe_rate(correct, len(scored_uids))
    recall = safe_rate(correct, truth_matches)
    interval = wilson_interval(correct, len(scored_uids))
    return {
        "resolved_truth_count": len(truth),
        "resolved_truth_match_count": truth_matches,
        "proposal_count_all_panel_rows": len(proposals),
        "proposal_count_with_resolved_truth": len(scored_uids),
        "proposal_count_without_resolved_truth": len(proposals) - len(scored_uids),
        "correct_exact_proposal_count": correct,
        "exact_proposal_precision": precision,
        "exact_proposal_precision_wilson_95": interval,
        "exact_proposal_precision_wilson_95_lower": _lower(interval),
        "exact_proposal_recall": recall,
        "exact_f_beta_0_5": _f_beta(precision, recall),
    }


def _select_adjudicator(variants: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        variants,
        key=lambda row: (
            -float(row["score"]["exact_proposal_precision_wilson_95_lower"]),
            -float(row["score"]["exact_f_beta_0_5"]),
            -float(row["score"]["exact_proposal_precision"] or -1),
            -float(row["score"]["exact_proposal_recall"] or -1),
            str(row["name"]),
        ),
    )[0]


def score_verifier(
    truth: dict[str, dict[str, Any]],
    proposals: dict[str, dict[str, Any]],
    orders: dict[str, dict[str, dict[str, Any]]],
    *,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    proposal_uids = set(proposals)
    if any(set(rows) != proposal_uids for rows in orders.values()):
        raise ValueError("verifier orders do not exactly cover the frozen proposals")
    scored_uids = sorted(set(truth) & proposal_uids)
    truth_matches = sum(row.get("decision") == "MATCH" for row in truth.values())
    correct_proposals = {
        uid
        for uid in scored_uids
        if truth[uid].get("decision") == "MATCH"
        and str(truth[uid].get("metric_id")) == str(proposals[uid].get("metric_id"))
    }
    retained_uids = []
    for uid in scored_uids:
        proposed = str(proposals[uid].get("metric_id"))
        values = [orders[name][uid] for name in ("original", "hashed", "reverse")]
        if all(
            row.get("decision") == "CONFIRM_MATCH"
            and str(row.get("metric_id")) == proposed
            and row.get("confidence") == "high"
            and not row.get("parse_error")
            for row in values
        ):
            retained_uids.append(uid)
    retained_true = sum(uid in correct_proposals for uid in retained_uids)
    false_retained = len(retained_uids) - retained_true
    wrong_proposals = len(scored_uids) - len(correct_proposals)
    precision = safe_rate(retained_true, len(retained_uids))
    recall_correct = safe_rate(retained_true, len(correct_proposals))
    recall_truth = safe_rate(retained_true, truth_matches)
    interval = wilson_interval(retained_true, len(retained_uids))
    lower = _lower(interval)
    eligible = (
        len(retained_uids) >= int(thresholds["minimum_retained"])
        and precision is not None
        and precision >= float(thresholds["minimum_retained_exact_precision"])
        and lower
        >= float(thresholds["minimum_retained_exact_precision_wilson_95_lower"])
    )
    return {
        "resolved_truth_count": len(truth),
        "resolved_truth_match_count": truth_matches,
        "proposal_count_all_panel_rows": len(proposals),
        "proposal_count_with_resolved_truth": len(scored_uids),
        "proposal_count_without_resolved_truth": len(proposals) - len(scored_uids),
        "correct_exact_proposal_count": len(correct_proposals),
        "proposal_exact_accuracy_on_resolved_intersection": safe_rate(
            len(correct_proposals), len(scored_uids)
        ),
        "retained_count": len(retained_uids),
        "retained_true_count": retained_true,
        "false_retained_count": false_retained,
        "retained_exact_precision": precision,
        "retained_exact_precision_wilson_95": interval,
        "retained_exact_precision_wilson_95_lower": lower,
        "retained_exact_recall_of_correct_proposals": recall_correct,
        "retained_exact_recall_of_truth_matches": recall_truth,
        "wrong_proposal_rejection_rate": safe_rate(
            wrong_proposals - false_retained, wrong_proposals
        ),
        "exact_f_beta_0_5": _f_beta(precision, recall_truth),
        "eligible": eligible,
        "thresholds": thresholds,
    }


def _select_verifier(variants: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        variants,
        key=lambda row: (
            -int(bool(row["score"]["eligible"])),
            -float(row["score"]["retained_exact_precision_wilson_95_lower"]),
            -float(row["score"]["exact_f_beta_0_5"]),
            -float(row["score"]["retained_exact_precision"] or -1),
            -float(row["score"]["retained_exact_recall_of_truth_matches"] or -1),
            str(row["name"]),
        ),
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--complete-marker", required=True)
    parser.add_argument("--adjudicator-truth", required=True)
    parser.add_argument("--adjudicator-truth-report", required=True)
    parser.add_argument("--verifier-truth", required=True)
    parser.add_argument("--verifier-truth-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = {
        name: Path(getattr(args, name.replace("-", "_"))).resolve()
        for name in (
            "policy",
            "queue",
            "complete-marker",
            "adjudicator-truth",
            "adjudicator-truth-report",
            "verifier-truth",
            "verifier-truth-report",
        )
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)

    policy = json.loads(paths["policy"].read_text(encoding="utf-8"))
    if (
        policy.get("schema_version") != POLICY_SCHEMA
        or policy.get("status") != "FROZEN_BEFORE_INDEPENDENT_LABELS_OR_MODEL_PREDICTIONS"
        or policy.get("task") != "humor"
        or policy.get("blind_status") != "SEALED_UNCONSUMED"
        or (policy.get("selection_contract") or {}).get(
            "prompt_or_threshold_iteration_after_truth_join_allowed"
        )
        is not False
    ):
        raise ValueError("invalid or unfrozen Humor prelabel policy")

    queue = json.loads(paths["queue"].read_text(encoding="utf-8"))
    validate_queue(queue)
    marker = json.loads(paths["complete-marker"].read_text(encoding="utf-8"))
    queue_sha = sha256_file(paths["queue"])
    if (
        marker.get("schema_version")
        != "silver-match-v3-humor-queue-run-complete-v1"
        or marker.get("queue_sha256") != queue_sha
        or marker.get("all_cells_exact_complete") is not True
        or not all(
            _cell_complete(cell)
            for stage in queue["stages"]
            for cell in stage["cells"]
        )
    ):
        raise ValueError("Gemma queue is not exact-complete before truth selection")

    adjudicator_truth, adjudicator_truth_report = _load_truth(
        paths["adjudicator-truth"],
        paths["adjudicator-truth-report"],
        expected_panel_role="adjudicator_dev",
    )
    verifier_truth, verifier_truth_report = _load_truth(
        paths["verifier-truth"],
        paths["verifier-truth-report"],
        expected_panel_role="verifier_dev",
    )
    outputs = queue["outputs"]
    adjudicator_names = [str(row["name"]) for row in policy["adjudicator_variants"]]
    adjudicator_variants = []
    verifier_panel_proposals: dict[str, dict[str, dict[str, Any]]] = {}
    for name in adjudicator_names:
        panel_paths = {
            order: Path(outputs[f"adjudicator_dev.adjudicator.{name}.{order}"])
            for order in ("original", "hashed", "consensus")
        }
        proposals, proposal_meta = _load_strict_consensus(
            panel_paths["original"], panel_paths["hashed"], panel_paths["consensus"]
        )
        adjudicator_variants.append(
            {"name": name, "score": score_adjudicator(adjudicator_truth, proposals), **proposal_meta}
        )
        verifier_paths = {
            order: Path(outputs[f"verifier_dev.adjudicator.{name}.{order}"])
            for order in ("original", "hashed", "consensus")
        }
        verifier_panel_proposals[name], _ = _load_strict_consensus(
            verifier_paths["original"],
            verifier_paths["hashed"],
            verifier_paths["consensus"],
        )
    chosen_adjudicator = _select_adjudicator(adjudicator_variants)
    chosen_name = str(chosen_adjudicator["name"])

    thresholds = dict((policy["selection_contract"] or {})["final_release_gate"])
    verifier_names = [str(row["name"]) for row in policy["verifier_variants"]]
    verifier_variants = []
    for name in verifier_names:
        order_paths = {
            order: Path(outputs[f"verifier_dev.verifier.{chosen_name}.{name}.{order}"])
            for order in ("original", "hashed", "reverse")
        }
        order_rows = {order: _index(path) for order, path in order_paths.items()}
        verifier_variants.append(
            {
                "name": name,
                "adjudicator": chosen_name,
                "score": score_verifier(
                    verifier_truth,
                    verifier_panel_proposals[chosen_name],
                    order_rows,
                    thresholds=thresholds,
                ),
                "inputs": {
                    order: {"path": str(path), "sha256": sha256_file(path)}
                    for order, path in order_paths.items()
                },
            }
        )
    chosen_verifier = _select_verifier(verifier_variants)
    promoted = bool(chosen_verifier["score"]["eligible"])
    report = {
        "schema_version": "silver-match-v3-humor-fresh-release-selection-v1",
        "status": (
            "PROMOTED_STRICT_AUTOMATIC_MATCH_PATH"
            if promoted
            else "NO_AUTOMATIC_MATCH_PATH"
        ),
        "task": "humor",
        "selection_data": "fresh_independent_selection_truth_only",
        "adjudicator": {"chosen": chosen_adjudicator, "variants": adjudicator_variants},
        "verifier": {"chosen": chosen_verifier, "variants": verifier_variants},
        "production_contract": policy["production_contract"],
        "blind_status": "SEALED_UNCONSUMED",
        "permanent_blind_consumed": False,
        "prompt_or_threshold_iteration_after_truth_join": False,
        "inputs": {
            "policy": {"path": str(paths["policy"]), "sha256": sha256_file(paths["policy"])},
            "queue": {"path": str(paths["queue"]), "sha256": queue_sha},
            "complete_marker": {
                "path": str(paths["complete-marker"]),
                "sha256": sha256_file(paths["complete-marker"]),
            },
            "adjudicator_truth": {
                "path": str(paths["adjudicator-truth"]),
                "sha256": sha256_file(paths["adjudicator-truth"]),
                "report": adjudicator_truth_report,
            },
            "verifier_truth": {
                "path": str(paths["verifier-truth"]),
                "sha256": sha256_file(paths["verifier-truth"]),
                "report": verifier_truth_report,
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "chosen_adjudicator": chosen_name,
                "chosen_verifier": chosen_verifier["name"],
                "output": str(output),
                "output_sha256": sha256_file(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
