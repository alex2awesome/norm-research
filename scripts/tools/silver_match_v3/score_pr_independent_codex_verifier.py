#!/usr/bin/env python3
"""Score the predeclared proposal-hidden Codex verifier exactly once."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file
from .score_verifier_calibration import safe_rate, wilson_interval


def _index(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    values = {str(row.get("norm_uid") or ""): row for row in rows}
    if not rows or "" in values or len(values) != len(rows):
        raise ValueError(f"invalid UID coverage: {path}")
    return values


def _observed_tokens(log_root: Path) -> int | None:
    values = []
    pattern = re.compile(r"^tokens used\s*\n\s*([0-9,]+)\s*$", re.MULTILINE)
    for path in sorted(log_root.glob("part-*.log")):
        matches = pattern.findall(path.read_text(encoding="utf-8", errors="replace"))
        if matches:
            values.append(int(matches[-1].replace(",", "")))
    return sum(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--pair-freeze", required=True)
    parser.add_argument("--transcript-audit", required=True)
    parser.add_argument("--labels-validation", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        name: Path(getattr(args, name)).resolve()
        for name in (
            "plan",
            "pair_freeze",
            "transcript_audit",
            "labels_validation",
            "labels",
            "primary",
            "truth",
            "targets",
            "log_root",
        )
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    pairs = json.loads(paths["pair_freeze"].read_text(encoding="utf-8"))
    transcript = json.loads(paths["transcript_audit"].read_text(encoding="utf-8"))
    validation = json.loads(paths["labels_validation"].read_text(encoding="utf-8"))
    if (
        plan.get("status") != "FROZEN_BEFORE_ANY_FALLBACK_CODEX_LABEL"
        or plan.get("task") != "press-releases"
        or plan.get("contracts", {}).get("proposal_not_in_label_workspace") is not True
        or plan.get("contracts", {}).get("truth_or_target_not_in_label_workspace")
        is not True
        or plan.get("contracts", {}).get("score_exactly_once_without_tuning")
        is not True
        or pairs.get("status") != "FROZEN_BALANCED_BEFORE_VERIFIER_INFERENCE"
        or transcript.get("status") != "PASS"
        or transcript.get("complete") is not True
        or validation.get("complete") is not True
        or (validation.get("transcript_audit") or {}).get("sha256")
        != sha256_file(paths["transcript_audit"])
        or (validation.get("output") or {}).get("sha256")
        != sha256_file(paths["labels"])
    ):
        raise ValueError("fallback plan/transcript/label validation is incomplete")
    pair_outputs = pairs.get("outputs") or {}
    for name in ("primary", "truth", "targets"):
        if pair_outputs.get(name, {}).get("sha256") != sha256_file(paths[name]):
            raise ValueError(f"scoring input differs from frozen pair universe: {name}")

    labels = _index(paths["labels"])
    primary = _index(paths["primary"])
    truth = _index(paths["truth"])
    targets = _index(paths["targets"])
    uids = set(targets)
    if (
        len(uids) != int(plan.get("frontier_count", -1))
        or set(labels) != uids
        or set(primary) != uids
        or set(truth) != uids
    ):
        raise ValueError("Codex verifier inputs lack exact frozen frontier coverage")
    accepted_confidences = set(plan["keep_rule"]["accepted_confidences"])
    retained_uids: list[str] = []
    retained_true = independent_exact_truth = 0
    for uid in sorted(uids):
        label, proposal, gold, target = (
            labels[uid],
            primary[uid],
            truth[uid],
            targets[uid],
        )
        proposal_id = str(proposal.get("metric_id") or "")
        expected_target = (
            "CONFIRM_MATCH"
            if gold.get("decision") == "MATCH"
            and str(gold.get("metric_id") or "") == proposal_id
            else "REJECT"
        )
        if (
            target.get("target") != expected_target
            or target.get("proposal_metric_id") != proposal_id
            or label.get("label_source") != "independent_codex_full_bank"
        ):
            raise ValueError(f"truth/target/independent-label drift: {uid}")
        keep = (
            label.get("decision") == "MATCH"
            and str(label.get("metric_id") or "") == proposal_id
            and label.get("confidence") in accepted_confidences
        )
        exact_truth = (
            label.get("decision") == gold.get("decision")
            and (
                str(label.get("metric_id") or "")
                if label.get("decision") == "MATCH"
                else None
            )
            == (
                str(gold.get("metric_id") or "")
                if gold.get("decision") == "MATCH"
                else None
            )
        )
        independent_exact_truth += exact_truth
        if keep:
            retained_uids.append(uid)
            retained_true += expected_target == "CONFIRM_MATCH"
    retained = len(retained_uids)
    false_retained = retained - retained_true
    positives = sum(row["target"] == "CONFIRM_MATCH" for row in targets.values())
    negatives = len(targets) - positives
    precision = safe_rate(retained_true, retained)
    precision_wilson = wilson_interval(retained_true, retained)
    wilson_lower = precision_wilson[0] if precision_wilson else None
    gates = plan["gate_rule"]
    results = {
        "minimum_retained_proposals": retained
        >= int(gates["minimum_retained_proposals"]),
        "minimum_exact_precision": precision is not None
        and precision >= float(gates["minimum_exact_precision"]),
        "minimum_wilson_lower_95": wilson_lower is not None
        and wilson_lower >= float(gates["minimum_wilson_lower_95"]),
    }
    passed = all(results.values())
    report = {
        "schema_version": "silver-match-v3-pr-independent-codex-verifier-score-v1",
        "status": (
            "PASS_ELIGIBLE_FOR_NEW_INDEPENDENT_BLIND_AUDIT"
            if passed
            else "REJECTED_FALLBACK_DEV_GATE_ABSTAIN"
        ),
        "task": "press-releases",
        "role": "verifier_dev_codex_fallback",
        "policy": plan["keep_rule"],
        "n": len(uids),
        "target_counts": {"CONFIRM_MATCH": positives, "REJECT": negatives},
        "retained": retained,
        "retained_true": retained_true,
        "false_retained": false_retained,
        "retained_precision": precision,
        "retained_precision_wilson_95": precision_wilson,
        "retained_recall_of_correct_proposals": safe_rate(retained_true, positives),
        "wrong_proposal_rejection_rate": safe_rate(negatives - false_retained, negatives),
        "wrong_proposal_rejection_wilson_95": wilson_interval(
            negatives - false_retained, negatives
        ),
        "independent_full_bank_exact_truth_accuracy": safe_rate(
            independent_exact_truth, len(uids)
        ),
        "gate_thresholds": gates,
        "gate_results": results,
        "all_gates_pass": passed,
        "retained_uids": retained_uids,
        "cost": {
            **plan["cost_estimate"],
            "observed_tokens_from_transcripts": _observed_tokens(paths["log_root"]),
        },
        "next_action": (
            "new_independent_blind_audit_required_before_production"
            if passed
            else "do_not_promote; abstain or obtain a new source-disjoint verifier design"
        ),
        "contracts": {
            "scored_once": True,
            "no_post_label_threshold_or_prompt_choice": True,
            "gemma_v4_remains_ineligible_advisory_only": True,
            "successful_dev_gate_is_not_final_blind_evidence": True,
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
            if path.is_file()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({**report, "output_sha256": sha256_file(output)}, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
