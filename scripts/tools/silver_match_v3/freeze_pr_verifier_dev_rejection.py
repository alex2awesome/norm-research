#!/usr/bin/env python3
"""Seal the failed accepted-v4 verifier-dev result append-only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--pair-freeze", required=True)
    parser.add_argument("--scoring-policy", required=True)
    parser.add_argument("--output-freeze", required=True)
    parser.add_argument("--gate-score", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        name: Path(getattr(args, name)).resolve()
        for name in (
            "plan",
            "pair_freeze",
            "scoring_policy",
            "output_freeze",
            "gate_score",
        )
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    plan, pairs, scoring, frozen, score = [
        json.loads(paths[name].read_text(encoding="utf-8"))
        for name in (
            "plan",
            "pair_freeze",
            "scoring_policy",
            "output_freeze",
            "gate_score",
        )
    ]
    if (
        plan.get("status")
        != "FROZEN_BEFORE_FIRST_FRESH_DEV_VERIFIER_INFERENCE"
        or pairs.get("status") != "FROZEN_BALANCED_BEFORE_VERIFIER_INFERENCE"
        or scoring.get("status")
        != "FROZEN_BEFORE_FIRST_FRESH_DEV_VERIFIER_OUTPUT"
        or frozen.get("status") != "FROZEN_COMPLETE_BEFORE_FRESH_DEV_SCORING"
        or score.get("status") != "REJECTED_FRESH_DEV_GATE"
        or score.get("all_gates_pass") is not False
        or any((score.get("gate_results") or {}).values())
        or int(score.get("n", -1)) != int(pairs.get("selected_count", -2))
        or (frozen.get("plan") or {}).get("sha256") != sha256_file(paths["plan"])
        or (scoring.get("inputs") or {}).get("plan", {}).get("sha256")
        != sha256_file(paths["plan"])
        or (score.get("inputs") or {}).get("output_freeze", {}).get("sha256")
        != sha256_file(paths["output_freeze"])
        or (score.get("inputs") or {}).get("scoring_policy", {}).get("sha256")
        != sha256_file(paths["scoring_policy"])
    ):
        raise ValueError("accepted-v4 rejection chain is incomplete or inconsistent")
    payload = {
        "schema_version": "silver-match-v3-pr-verifier-dev-rejection-freeze-v1",
        "status": "REJECTED_APPEND_ONLY_NO_PROMOTION_OR_RETUNING_ON_CONSUMED_DEV",
        "task": "press-releases",
        "variant": "accepted_v4",
        "consumed_verifier_dev_count": int(score["n"]),
        "failed_gates": [
            name for name, passed in score["gate_results"].items() if not passed
        ],
        "disposition": {
            "eligible_for_production": False,
            "eligible_for_final_blind_audit": False,
            "advisory_only": True,
            "may_edit_or_select_prompt_on_consumed_dev": False,
            "allowed_fallback": "proposal-hidden independent full-bank Codex verification or abstain",
            "future_gemma_work": "GEPA on optimize truth only, followed by newly sampled source-disjoint verifier test",
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
