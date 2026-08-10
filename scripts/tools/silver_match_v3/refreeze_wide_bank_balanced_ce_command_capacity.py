#!/usr/bin/env python3
"""Bind queue negative-count arguments to an increased balanced-CE capacity."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from .common import sha256_file
from .freeze_metric_balanced_ce_policy import _artifact


PROFILE = "silver-match-v4-balanced-wide-bank-capacity-v3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-capacity-policy", required=True)
    parser.add_argument("--failed-pair-audit", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source_path = Path(args.source_capacity_policy).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    eligibility_source = source_path.with_suffix(".ELIGIBILITY.json")
    eligibility = json.loads(eligibility_source.read_text(encoding="utf-8"))
    if eligibility.get("policy_sha256") != sha256_file(source_path):
        raise ValueError("source capacity eligibility is stale")
    if (
        source.get("development_evidence_status")
        != "INDEPENDENT_UNOBSERVED_BEFORE_BALANCED_TRAINING"
        or source.get("blind_status") != "SEALED_UNCONSUMED"
    ):
        raise ValueError("source policy no longer has independent DEV and sealed blind")

    failed_path = Path(args.failed_pair_audit).resolve()
    failed = json.loads(failed_path.read_text(encoding="utf-8"))
    stderr = failed.get("stderr", "")
    if (
        failed.get("status") != "AUDIT_PROCESS_FAILURE"
        or int(failed.get("returncode", 0)) == 0
        or "negative count differs from balanced policy" not in stderr
    ):
        raise ValueError("input is not a pre-training queue/policy count mismatch")

    policy = copy.deepcopy(source)
    balanced = policy["balanced_training"]
    match_total = int(balanced["hard_negatives_per_match"]) + int(
        balanced["global_balanced_negatives_per_match"]
    )
    abstain_total = int(balanced["hard_negatives_per_abstain"]) + int(
        balanced["global_balanced_negatives_per_abstain"]
    )
    fixed = policy["fixed_training"]
    if (
        int(fixed["negatives_per_positive"]) == match_total
        and int(fixed["negatives_per_abstain"]) == abstain_total
    ):
        raise ValueError("source fixed training already matches balanced capacity")
    fixed.update(
        {
            "negatives_per_positive": match_total,
            "negatives_per_abstain": abstain_total,
        }
    )
    balanced["profile"] = PROFILE

    task = policy["scope"][0]
    freezer = Path(__file__).resolve()
    repo_root = Path(args.repo_root).resolve()
    now = datetime.now(timezone.utc).isoformat()
    policy.update(
        {
            "policy_revision": f"{task}-metric-balanced-ce-v4-wide-bank-capacity-v3",
            "balanced_profile": PROFILE,
            "status": "REFROZEN_AFTER_PRETRAIN_QUEUE_POLICY_COUNT_MISMATCH_BEFORE_TASK_DEV_SCORING",
            "frozen_at": now,
            "source_capacity_policy": _artifact(source_path),
            "source_capacity_policy_eligibility": _artifact(eligibility_source),
            "failed_pair_exposure_audit": _artifact(failed_path),
        }
    )
    policy["implementation"].update(
        {
            "refreeze_wide_bank_balanced_ce_command_capacity_path": str(
                freezer.relative_to(repo_root)
            ),
            "refreeze_wide_bank_balanced_ce_command_capacity_sha256": sha256_file(
                freezer
            ),
        }
    )

    output = Path(args.output).resolve()
    eligibility_output = output.with_suffix(".ELIGIBILITY.json")
    if output.exists() or eligibility_output.exists():
        raise FileExistsError(output if output.exists() else eligibility_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    new_eligibility = {
        **eligibility,
        "policy_path": str(output),
        "policy_sha256": sha256_file(output),
        "balanced_profile": PROFILE,
        "development_evidence_status": policy["development_evidence_status"],
        "refrozen_at": now,
    }
    eligibility_output.write_text(
        json.dumps(new_eligibility, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "task": task,
                "policy": _artifact(output),
                "eligibility": _artifact(eligibility_output),
                "balanced_profile": PROFILE,
                "negatives_per_positive": match_total,
                "negatives_per_abstain": abstain_total,
                "development_evidence_status": policy["development_evidence_status"],
                "blind_status": policy["blind_status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
