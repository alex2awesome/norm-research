#!/usr/bin/env python3
"""Append-only execution correction adding the globally predeclared reverse order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-plan", required=True)
    parser.add_argument("--global-policy", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prior_path = Path(args.prior_plan).resolve()
    policy_path = Path(args.global_policy).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError("refusing to overwrite PR R4 execution supersession")
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    required_orders = (policy.get("adjudicator_policy") or {}).get("select_orders")
    if (
        prior.get("schema_version") != "silver-match-v3-pr-r4-select-openrouter-plan-v1"
        or prior.get("status") != "FROZEN_BEFORE_SELECT_INFERENCE"
        or prior.get("orders") != ["original", "hashed"]
        or required_orders != ["original", "hashed", "reverse"]
        or policy.get("status")
        != "POLICY_FROZEN_BEFORE_NEW_CONTENT_TASK_PREDICTIONS_OR_LABELS"
        or "press-releases" not in policy.get("scope", [])
    ):
        raise ValueError("prior plan/global policy do not authorize this narrow correction")
    outputs = dict(prior["outputs"])
    outputs["reverse"] = str(Path(outputs["original"]).with_name("reverse.jsonl"))
    plan = {
        **prior,
        "schema_version": "silver-match-v3-pr-r4-select-openrouter-plan-v2",
        "status": "FROZEN_EXECUTION_SUPERSESSION_BEFORE_ANY_SELECT_SCORE",
        "orders": required_orders,
        "max_total_api_requests": 750,
        "outputs": outputs,
        "supersession": {
            "relationship": "execution-only completeness correction",
            "reason": "restore reverse order predeclared by the earlier global content GEPA policy",
            "prior_plan": {"path": str(prior_path), "sha256": sha256_file(prior_path)},
            "global_policy": {"path": str(policy_path), "sha256": sha256_file(policy_path)},
            "prompt_changed": False,
            "model_changed": False,
            "rendering_changed": False,
            "candidate_pack_changed": False,
            "request_cap_per_order_changed": False,
            "select_score_opened_before_supersession": False,
            "new_variant_added": False,
        },
        "contracts": {
            **prior["contracts"],
            "all_global_select_orders_restored": True,
            "score_requires_original_hashed_reverse_frozen": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**plan, "plan_sha256": sha256_file(output_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
