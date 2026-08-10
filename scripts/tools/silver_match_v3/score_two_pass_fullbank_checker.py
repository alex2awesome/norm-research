#!/usr/bin/env python3
"""Score a frozen two-pass, proposal-hidden full-bank verifier on optimize truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file
from .score_verifier_calibration import safe_rate, wilson_interval


POLICY_NAME = "original_and_hashed_exact_high"


def _index(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    indexed = {str(row.get("norm_uid") or ""): row for row in rows}
    if not rows or "" in indexed or len(indexed) != len(rows):
        raise ValueError(f"missing or duplicate norm_uid: {path}")
    return indexed


def score_rows(
    labels_a: dict[str, dict[str, Any]],
    labels_b: dict[str, dict[str, Any]],
    primary: dict[str, dict[str, Any]],
    truth: dict[str, dict[str, Any]],
    *,
    minimum_retained: int,
    minimum_point_precision: float,
    minimum_wilson_lower: float,
) -> dict[str, Any]:
    uids = set(primary)
    if set(labels_a) != uids or set(labels_b) != uids or set(truth) != uids:
        raise ValueError("two passes, proposals, and truth lack identical UID coverage")
    retained_uids: list[str] = []
    retained_true = 0
    exact_agreement_uids: list[str] = []
    correct_proposals = 0
    for uid in sorted(uids):
        proposal, gold = primary[uid], truth[uid]
        proposal_id = str(proposal.get("metric_id") or "")
        if proposal.get("decision") != "MATCH" or not proposal_id:
            raise ValueError(f"screened proposal is not an exact MATCH: {uid}")
        proposal_true = (
            gold.get("decision") == "MATCH"
            and str(gold.get("metric_id") or "") == proposal_id
        )
        correct_proposals += int(proposal_true)
        left, right = labels_a[uid], labels_b[uid]
        exact_agreement = (
            left.get("decision") == right.get("decision")
            and (
                str(left.get("metric_id") or "")
                if left.get("decision") == "MATCH"
                else None
            )
            == (
                str(right.get("metric_id") or "")
                if right.get("decision") == "MATCH"
                else None
            )
        )
        if exact_agreement:
            exact_agreement_uids.append(uid)
        keep = (
            left.get("decision") == "MATCH"
            and right.get("decision") == "MATCH"
            and str(left.get("metric_id") or "") == proposal_id
            and str(right.get("metric_id") or "") == proposal_id
            and left.get("confidence") == "high"
            and right.get("confidence") == "high"
        )
        if keep:
            retained_uids.append(uid)
            retained_true += int(proposal_true)
    retained = len(retained_uids)
    false_retained = retained - retained_true
    precision = safe_rate(retained_true, retained)
    interval = wilson_interval(retained_true, retained)
    lower = interval[0] if interval else None
    gates = {
        "minimum_retained": retained >= minimum_retained,
        "minimum_point_precision": (
            precision is not None and precision >= minimum_point_precision
        ),
        "minimum_wilson_95_lower": (
            lower is not None and lower >= minimum_wilson_lower
        ),
    }
    return {
        "n": len(uids),
        "correct_proposals": correct_proposals,
        "wrong_proposals": len(uids) - correct_proposals,
        "retained": retained,
        "retained_true": retained_true,
        "false_retained": false_retained,
        "retained_precision": precision,
        "retained_precision_wilson_95": interval,
        "retained_recall_of_correct_proposals": safe_rate(
            retained_true, correct_proposals
        ),
        "exact_two_pass_agreement": len(exact_agreement_uids),
        "exact_two_pass_agreement_rate": safe_rate(
            len(exact_agreement_uids), len(uids)
        ),
        "retained_uids": retained_uids,
        "gate_results": gates,
        "all_gates_pass": all(gates.values()),
    }


def _load_validation(path: Path, labels: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version")
        != "silver-match-v3-independent-label-validation-v1"
        or payload.get("complete") is not True
        or (payload.get("output") or {}).get("sha256") != sha256_file(labels)
    ):
        raise ValueError(f"invalid independent label validation: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--screen-freeze", required=True)
    parser.add_argument("--postlabel-audit", required=True)
    parser.add_argument("--labels-a", required=True)
    parser.add_argument("--labels-validation-a", required=True)
    parser.add_argument("--labels-b", required=True)
    parser.add_argument("--labels-validation-b", required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--truth-meta", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        name: Path(getattr(args, name)).resolve()
        for name in (
            "plan",
            "policy",
            "screen_freeze",
            "postlabel_audit",
            "labels_a",
            "labels_validation_a",
            "labels_b",
            "labels_validation_b",
            "primary",
            "truth",
            "truth_meta",
        )
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    policy = json.loads(paths["policy"].read_text(encoding="utf-8"))
    screen = json.loads(paths["screen_freeze"].read_text(encoding="utf-8"))
    audit = json.loads(paths["postlabel_audit"].read_text(encoding="utf-8"))
    truth_meta = json.loads(paths["truth_meta"].read_text(encoding="utf-8"))
    task = str(plan.get("task") or "")
    if (
        plan.get("status") != "FROZEN_BEFORE_EITHER_INDEPENDENT_LABEL_PASS"
        or not task
        or policy.get("status")
        != "POLICY_FROZEN_BEFORE_NEW_CONTENT_TASK_PREDICTIONS_OR_LABELS"
        or task not in (policy.get("scope") or [])
        or (plan.get("inputs", {}).get("external_policy") or {}).get("sha256")
        != sha256_file(paths["policy"])
        or screen.get("status")
        != "FROZEN_TRUTH_BLIND_SCREEN_REQUIRES_STRONG_VERIFIER"
        or screen.get("task") != task
        or (screen.get("output") or {}).get("sha256")
        != sha256_file(paths["primary"])
        or audit.get("status")
        != "AUDITED_POSTLABEL_FROM_PRELABEL_FREEZES_AND_ISOLATED_TRANSCRIPTS"
        or audit.get("task") != task
        or audit.get("candidate_proposals_exposed_to_either_pass") is not False
        or audit.get("prior_truth_or_predictions_exposed_to_either_pass") is not False
        or audit.get("pass_predictions_mutually_visible") is not False
        or truth_meta.get("schema_version")
        != "silver-match-v3-jsonl-reference-subset-v1"
        or (truth_meta.get("inputs", {}).get("reference") or {}).get("sha256")
        != sha256_file(paths["primary"])
        or (truth_meta.get("output") or {}).get("sha256")
        != sha256_file(paths["truth"])
    ):
        raise ValueError("frozen plan, policy, screen, audit, or truth binding is invalid")
    validations = {
        "A": _load_validation(paths["labels_validation_a"], paths["labels_a"]),
        "B": _load_validation(paths["labels_validation_b"], paths["labels_b"]),
    }
    transcript_audits = audit.get("transcript_isolation_audits") or {}
    plan_staged = {
        "A": plan.get("inputs", {}).get("staged_pass_a") or {},
        "B": plan.get("inputs", {}).get("staged_pass_b") or {},
    }
    for name in ("A", "B"):
        validation = validations[name]
        if (
            validation.get("task") != task
            or validation.get("count") != int(plan.get("row_count", -1))
            or (validation.get("transcript_audit") or {}).get("sha256")
            != (transcript_audits.get(name) or {}).get("sha256")
            or (validation.get("pack_validation") or {}).get("sha256")
            != (audit.get("passes", {}).get(name) or {}).get("validation_sha256")
            or (validation.get("pack_validation") or {}).get("sha256")
            != (plan_staged[name].get("validation") or {}).get("sha256")
        ):
            raise ValueError(f"pass {name} validation is not bound to the frozen run")
    verifier_policy = policy.get("verifier_policy") or {}
    if POLICY_NAME not in (verifier_policy.get("eligible_policies") or []):
        raise ValueError(f"policy does not authorize {POLICY_NAME}")
    result = score_rows(
        _index(paths["labels_a"]),
        _index(paths["labels_b"]),
        _index(paths["primary"]),
        _index(paths["truth"]),
        minimum_retained=int(verifier_policy["minimum_retained"]),
        minimum_point_precision=float(verifier_policy["minimum_point_precision"]),
        minimum_wilson_lower=float(verifier_policy["minimum_wilson_95_lower"]),
    )
    if result["n"] != int(plan.get("row_count", -1)):
        raise ValueError("scored universe differs from frozen plan")
    report = {
        "schema_version": "silver-match-v3-two-pass-fullbank-checker-score-v1",
        "status": (
            "PASS_ELIGIBLE_FOR_NEW_FRESH_SOURCE_DISJOINT_SELECT"
            if result["all_gates_pass"]
            else "REJECTED_OPTIMIZE_GATE_DO_NOT_PROMOTE"
        ),
        "task": task,
        "role": "optimize_only_verifier_design",
        "eligible_policy": POLICY_NAME,
        **result,
        "gate_thresholds": {
            "minimum_retained": int(verifier_policy["minimum_retained"]),
            "minimum_point_precision": float(
                verifier_policy["minimum_point_precision"]
            ),
            "minimum_wilson_95_lower": float(
                verifier_policy["minimum_wilson_95_lower"]
            ),
        },
        "next_action": (
            "freeze and score a new source-disjoint select panel; no production claim yet"
            if result["all_gates_pass"]
            else "do not lower thresholds; revise on optimize rows or use another verifier"
        ),
        "contracts": {
            "optimize_rows_only": True,
            "proposal_hidden_from_both_full_bank_passes": True,
            "both_passes_must_exactly_confirm_proposal_at_high_confidence": True,
            "successful_optimize_gate_is_not_final_blind_evidence": True,
            "thresholds_unchanged_after_labels": True,
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**report, "output_sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
