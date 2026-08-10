#!/usr/bin/env python3
"""Freeze balanced CE after exposure failure but before task DEV scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import sha256_file
from . import freeze_metric_balanced_ce_policy as dense
PREFLIGHT_PROFILE = "silver-match-v4-balanced-preflight-independent-dev-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-policy", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--failure-report", action="append", required=True)
    parser.add_argument("--pointwise-exposure-audit", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    eligibility_path = output.with_suffix(".ELIGIBILITY.json")
    if output.exists() or eligibility_path.exists():
        raise FileExistsError(output if output.exists() else eligibility_path)
    exposure_path = Path(args.pointwise_exposure_audit).resolve()
    exposure = json.loads(exposure_path.read_text(encoding="utf-8"))
    if (
        exposure.get("status") != "FAIL_POINTWISE_EXPOSURE_GATE"
        or exposure.get("task") != args.task
        or exposure.get("model_initialized") is not False
        or exposure.get("gpu_consumed") is not False
    ):
        raise ValueError("task exposure audit is not a clean pre-training failure")
    policy, eligibility = dense.freeze(args)
    config = policy["balanced_training"]
    config.update(
        {
            "profile": PREFLIGHT_PROFILE,
            "max_unique_positive_uids_per_metric": 64,
            "hard_negatives_per_match": 2,
            "global_balanced_negatives_per_match": 6,
            "hard_negatives_per_abstain": 2,
            "global_balanced_negatives_per_abstain": 6,
            "min_negative_exposure_per_bank_metric": 16,
            "target_negative_to_positive_pair_ratio": 2.0,
            "minimum_negative_to_positive_pair_ratio": 2.0,
            "maximum_positive_pair_fraction_per_metric": 0.3333333333333333,
        }
    )
    freezer = Path(__file__).resolve()
    policy.update(
        {
            "balanced_profile": PREFLIGHT_PROFILE,
            "policy_revision": f"{args.task}-metric-balanced-ce-v4-preflight",
            "status": "FROZEN_AFTER_POINTWISE_EXPOSURE_FAILURE_BEFORE_ANY_TASK_DEV_SCORING",
            "development_evidence_status": "INDEPENDENT_UNOBSERVED_BEFORE_BALANCED_TRAINING",
            "pointwise_exposure_failure": dense._artifact(exposure_path),
        }
    )
    policy["role_contract"]["dev"] = (
        "independent frozen development panel unobserved before balanced training"
    )
    policy["implementation"].update(
        {
            "freeze_preflight_metric_balanced_ce_policy_path": str(
                freezer.relative_to(Path(args.repo_root).resolve())
            ),
            "freeze_preflight_metric_balanced_ce_policy_sha256": sha256_file(
                freezer
            ),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    eligibility.update(
        {
            "policy_sha256": sha256_file(output),
            "policy_path": str(output),
            "balanced_profile": PREFLIGHT_PROFILE,
            "development_evidence_status": policy[
                "development_evidence_status"
            ],
        }
    )
    eligibility_path.write_text(
        json.dumps(eligibility, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "policy": dense._artifact(output),
                "eligibility": dense._artifact(eligibility_path),
                "task": args.task,
                "balanced_profile": PREFLIGHT_PROFILE,
                "development_evidence_status": policy[
                    "development_evidence_status"
                ],
                "blind_status": "SEALED_UNCONSUMED",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
