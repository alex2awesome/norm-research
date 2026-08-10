import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.freeze_explicit_role_verifier_selection import (
    build_freeze,
)


def _artifacts(tmp_path: Path) -> tuple[Path, Path, list[Path]]:
    outputs = [tmp_path / "truth.jsonl", tmp_path / "original.jsonl"]
    plan_path = tmp_path / "COMMAND_PLAN.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-explicit-role-task-local-gepa-plan-v1",
                "status": "FROZEN_BEFORE_TASK_LOCAL_GEPA_INFERENCE",
                "task": "code-review",
                "thresholds": {
                    "minimum_point_precision": 0.9,
                    "minimum_wilson_95_lower": 0.8,
                    "minimum_retained": 30,
                    "selection_rule": "Wilson, support, point",
                },
                "adjudicator_variants": [
                    {"name": "r0", "combined_prompt_sha256": "a" * 64}
                ],
                "verifier_variants": [
                    {"name": "v0", "combined_prompt_sha256": "b" * 64}
                ],
                "commands": [
                    {
                        "stage": "verifier_subset_truth",
                        "adjudicator_variant": "r0",
                        "verifier_variant": "v0",
                        "role": "prompt_dev",
                        "command": {
                            "module": "unused",
                            "argv": ["--output", str(outputs[0])],
                        },
                    },
                    {
                        "stage": "verifier",
                        "adjudicator_variant": "r0",
                        "verifier_variant": "v0",
                        "role": "prompt_dev",
                        "order": "original",
                        "direct_batch_command": {
                            "module": "unused",
                            "argv": ["--output", str(outputs[1])],
                        },
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    role_freeze = tmp_path / "FREEZE.json"
    role_freeze.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-explicit-role-task-local-gepa-freeze-v1",
                "status": "FROZEN_BEFORE_TASK_LOCAL_GEPA_INFERENCE",
                "task": "code-review",
                "command_plan": {"sha256": sha256_file(plan_path)},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return plan_path, role_freeze, outputs


def test_freezes_established_exact_high_policy_universe(tmp_path: Path) -> None:
    plan, role_freeze, _ = _artifacts(tmp_path)

    result = build_freeze(plan, role_freeze)

    assert result["status"] == "FROZEN_BEFORE_ANY_VERIFIER_INFERENCE_OR_SCORE"
    assert [row["name"] for row in result["candidate_policies"]] == [
        "two_order_exact_high",
        "all_three_order_exact_high",
    ]
    assert result["diagnostic_policies"][0]["production_eligible"] is False
    assert result["preinference_audit"]["checked_output_path_count"] == 2


def test_refuses_freeze_after_any_verifier_output_exists(tmp_path: Path) -> None:
    plan, role_freeze, outputs = _artifacts(tmp_path)
    outputs[0].write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="before any verifier output"):
        build_freeze(plan, role_freeze)
