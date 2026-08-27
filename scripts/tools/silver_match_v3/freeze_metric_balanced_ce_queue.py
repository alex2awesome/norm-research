#!/usr/bin/env python3
"""Freeze a v3-compatible queue that invokes the balanced v4 CE trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import freeze_cross_encoder_queue as base
from .common import sha256_file
from .train_cross_encoder_balanced import BALANCED_OBJECTIVE_REVISION


BALANCED_MODULE = "scripts.tools.silver_match_v3.train_cross_encoder_balanced"
BALANCED_RELATIVE_PATH = Path(
    "scripts/tools/silver_match_v3/train_cross_encoder_balanced.py"
)


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    policy_path = Path(args.policy).resolve()
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("balanced_objective_revision") != BALANCED_OBJECTIVE_REVISION:
        raise ValueError("queue policy is not metric-balanced v4")
    result = base.freeze(args)
    repo_root = Path(args.repo_root).resolve()
    balanced_path = repo_root / BALANCED_RELATIVE_PATH
    implementation = policy.get("implementation") or {}
    if (
        implementation.get("balanced_train_cross_encoder_path")
        != str(BALANCED_RELATIVE_PATH)
        or implementation.get("balanced_train_cross_encoder_sha256")
        != sha256_file(balanced_path)
    ):
        raise ValueError("balanced trainer implementation differs from policy")
    for entry in result["commands"]:
        command = entry["command"]
        if command[2:4] != ["-m", "scripts.tools.silver_match_v3.train_cross_encoder"]:
            raise ValueError("unexpected base CE command")
        command[3] = BALANCED_MODULE
    result["implementation"] = base._artifact(balanced_path)
    result["implementation_audit"] = {
        **result["implementation_audit"],
        "balanced_objective_revision": BALANCED_OBJECTIVE_REVISION,
        "balanced_trainer": base._artifact(balanced_path),
        "underlying_v3_trainer": base._artifact(
            repo_root / "scripts/tools/silver_match_v3/train_cross_encoder.py"
        ),
    }
    result["balanced_training"] = policy["balanced_training"]
    result["balanced_exposure_gate_timing"] = "before_model_initialization"
    result["blind_status"] = policy["blind_status"]
    result["development_evidence_status"] = policy[
        "development_evidence_status"
    ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-teachers", action="append", required=True)
    parser.add_argument("--dev-teachers", action="append", required=True)
    parser.add_argument("--candidates", action="append", required=True)
    parser.add_argument("--extra-binding", action="append")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--python", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    result = freeze(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256_file(output),
                **result,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
