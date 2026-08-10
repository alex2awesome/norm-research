#!/usr/bin/env python3
"""Freeze the predeclared sparse-teacher profile of balanced CE v4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import sha256_file
from . import freeze_metric_balanced_ce_policy as dense


SPARSE_PROFILE = "silver-match-v4-balanced-sparse-teacher-profile-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-policy", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--failure-report", action="append", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    eligibility_path = output.with_suffix(".ELIGIBILITY.json")
    if output.exists() or eligibility_path.exists():
        raise FileExistsError(output if output.exists() else eligibility_path)
    policy, eligibility = dense.freeze(args)
    config = policy["balanced_training"]
    config.update(
        {
            "profile": SPARSE_PROFILE,
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
    policy["balanced_profile"] = SPARSE_PROFILE
    policy["policy_revision"] = f"{args.task}-metric-balanced-ce-v4-sparse"
    policy["implementation"].update(
        {
            "freeze_sparse_metric_balanced_ce_policy_path": str(
                freezer.relative_to(Path(args.repo_root).resolve())
            ),
            "freeze_sparse_metric_balanced_ce_policy_sha256": sha256_file(
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
            "balanced_profile": SPARSE_PROFILE,
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
                "balanced_profile": SPARSE_PROFILE,
                "blind_status": "SEALED_UNCONSUMED",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
