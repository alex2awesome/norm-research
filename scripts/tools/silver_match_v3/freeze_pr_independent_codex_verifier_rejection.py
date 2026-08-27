#!/usr/bin/env python3
"""Seal the failed proposal-hidden Codex verifier result append-only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import sha256_file


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--transcript-audit", required=True)
    parser.add_argument("--labels-validation", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--score", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        name: Path(getattr(args, name)).resolve()
        for name in (
            "plan",
            "transcript_audit",
            "labels_validation",
            "labels",
            "score",
        )
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)

    plan = _load(paths["plan"])
    audit = _load(paths["transcript_audit"])
    validation = _load(paths["labels_validation"])
    score = _load(paths["score"])
    if (
        plan.get("status") != "FROZEN_BEFORE_ANY_FALLBACK_CODEX_LABEL"
        or audit.get("status") != "PASS"
        or audit.get("complete") is not True
        or audit.get("violations") != []
        or validation.get("complete") is not True
        or int(validation.get("count", -1)) != int(plan.get("frontier_count", -2))
        or (validation.get("transcript_audit") or {}).get("sha256")
        != sha256_file(paths["transcript_audit"])
        or (validation.get("output") or {}).get("sha256")
        != sha256_file(paths["labels"])
        or score.get("status") != "REJECTED_FALLBACK_DEV_GATE_ABSTAIN"
        or score.get("all_gates_pass") is not False
        or int(score.get("n", -1)) != int(plan.get("frontier_count", -2))
        or (score.get("inputs") or {}).get("plan", {}).get("sha256")
        != sha256_file(paths["plan"])
        or (score.get("inputs") or {}).get("transcript_audit", {}).get("sha256")
        != sha256_file(paths["transcript_audit"])
        or (score.get("inputs") or {}).get("labels", {}).get("sha256")
        != sha256_file(paths["labels"])
        or any(
            bool((score.get("gate_results") or {}).get(name))
            for name in ("minimum_exact_precision", "minimum_wilson_lower_95")
        )
    ):
        raise ValueError("proposal-hidden Codex rejection chain is incomplete or inconsistent")

    payload = {
        "schema_version": "silver-match-v3-pr-independent-codex-verifier-rejection-freeze-v1",
        "status": "REJECTED_APPEND_ONLY_NO_PROMOTION_OR_RETUNING_ON_CONSUMED_DEV",
        "task": "press-releases",
        "variant": "proposal_hidden_independent_full_bank_codex",
        "consumed_verifier_dev_count": int(score["n"]),
        "failed_gates": [
            name
            for name, passed in (score.get("gate_results") or {}).items()
            if not passed
        ],
        "observed": {
            "retained": int(score["retained"]),
            "retained_true": int(score["retained_true"]),
            "false_retained": int(score["false_retained"]),
            "retained_precision": float(score["retained_precision"]),
            "retained_precision_wilson_95": score[
                "retained_precision_wilson_95"
            ],
            "retained_recall_of_correct_proposals": float(
                score["retained_recall_of_correct_proposals"]
            ),
        },
        "disposition": {
            "eligible_for_production": False,
            "eligible_for_final_blind_audit": False,
            "advisory_only": True,
            "may_edit_rule_or_threshold_on_consumed_dev": False,
            "consumed_identities_and_source_groups_must_be_excluded": True,
            "allowed_next_design": "optimize-truth-only GEPA followed by a newly drawn source-group-disjoint verifier test",
            "fallback_if_next_design_fails": "MATCH abstain-only; preserve typed nonmatches",
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({**payload, "output_sha256": sha256_file(output)}, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
