import json
from pathlib import Path

from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.score_verifier_calibration import wilson_interval
from scripts.tools.silver_match_v3.select_explicit_role_verifier import build_selection


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _policy(*, correct: int, retained: int, retained_true: int, n: int = 50) -> dict:
    return {
        "n": n,
        "proposal_correct": correct,
        "retained": retained,
        "retained_true": retained_true,
        "retained_precision": retained_true / retained if retained else None,
        "retained_precision_wilson_95": wilson_interval(retained_true, retained),
        "retained_recall_of_correct_proposals": retained_true / correct,
        "wrong_proposal_rejection_rate": 1.0,
    }


def test_selects_predeclared_exact_high_policy_by_frozen_ranking(tmp_path: Path) -> None:
    prompt = "b" * 64
    commands = []
    for role, panel, split in (
        ("optimize", "prompt_train", "optimize"),
        ("select", "prompt_dev", "dev"),
    ):
        inputs = {}
        for key in ("truth", "primary", "original", "hashed", "reverse"):
            path = tmp_path / role / f"{key}.jsonl"
            rows = [
                {
                    "norm_uid": f"u{i}",
                    "prompt_sha256": prompt,
                    "parse_error": None,
                }
                for i in range(50)
            ]
            _write_jsonl(path, rows)
            inputs[key] = path
        two_path = tmp_path / f"v0.{role}.two.json"
        two_path.write_text(
            json.dumps(
                {
                    "schema_version": "silver-match-v3-two-order-verifier-score-v1",
                    "selection_split": split,
                    "explicit_role": role,
                    "policies": {
                        "high_only": _policy(
                            correct=48, retained=30, retained_true=29
                        ),
                        "medium_or_high": _policy(
                            correct=48, retained=40, retained_true=36
                        ),
                    },
                    "input_hashes": {
                        key: sha256_file(inputs[key])
                        for key in ("truth", "primary", "original", "hashed")
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        three_path = tmp_path / f"v0.{role}.three.json"
        three_path.write_text(
            json.dumps(
                {
                    "schema_version": "silver-match-v3-three-order-verifier-score-v1",
                    "selection_split": split,
                    "explicit_role": role,
                    "policy": _policy(correct=48, retained=47, retained_true=45),
                    "input_hashes": {
                        key: sha256_file(inputs[key])
                        for key in ("truth", "primary", "original", "hashed", "reverse")
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        commands.extend(
            [
                {
                    "stage": "verifier_score_two_order",
                    "conditional_on_selected_adjudicator_variant": "r0",
                    "verifier_variant": "v0",
                    "role": panel,
                    "command": {
                        "module": "unused",
                        "argv": [
                            "--truth", str(inputs["truth"]),
                            "--primary", str(inputs["primary"]),
                            "--original", str(inputs["original"]),
                            "--hashed", str(inputs["hashed"]),
                            "--output", str(two_path),
                        ],
                    },
                },
                {
                    "stage": "verifier_score_three_order",
                    "conditional_on_selected_adjudicator_variant": "r0",
                    "verifier_variant": "v0",
                    "role": panel,
                    "command": {
                        "module": "unused",
                        "argv": [
                            "--truth", str(inputs["truth"]),
                            "--primary", str(inputs["primary"]),
                            "--original", str(inputs["original"]),
                            "--hashed", str(inputs["hashed"]),
                            "--reverse", str(inputs["reverse"]),
                            "--output", str(three_path),
                        ],
                    },
                },
            ]
        )
    plan_path = tmp_path / "COMMAND_PLAN.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-explicit-role-task-local-gepa-plan-v1",
                "status": "FROZEN_BEFORE_TASK_LOCAL_GEPA_INFERENCE",
                "task": "code-review",
                "thresholds": {
                    "minimum_point_precision": 0.9,
                    "minimum_retained": 30,
                    "minimum_wilson_95_lower": 0.8,
                },
                "adjudicator_variants": [
                    {"name": "r0", "combined_prompt_sha256": "a" * 64}
                ],
                "verifier_variants": [
                    {"name": "v0", "combined_prompt_sha256": prompt}
                ],
                "commands": commands,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    plan_sha = sha256_file(plan_path)
    role_freeze = tmp_path / "FREEZE.json"
    role_freeze.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-explicit-role-task-local-gepa-freeze-v1",
                "task": "code-review",
                "command_plan": {"sha256": plan_sha},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    role_sha = sha256_file(role_freeze)
    adj = tmp_path / "adjudicator.json"
    adj.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-explicit-role-adjudicator-selection-v1",
                "task": "code-review",
                "status": "selected",
                "chosen": {"name": "r0"},
                "inputs": {"command_plan": {"sha256": plan_sha}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    policy = tmp_path / "VERIFIER_SELECTION_FREEZE.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-explicit-role-verifier-selection-freeze-v1",
                "status": "FROZEN_BEFORE_ANY_VERIFIER_INFERENCE_OR_SCORE",
                "task": "code-review",
                "thresholds": {
                    "minimum_point_precision": 0.9,
                    "minimum_retained": 30,
                    "minimum_wilson_95_lower": 0.8,
                },
                "selection_rule": "Wilson, support, point",
                "candidate_policies": [
                    {
                        "name": "two_order_exact_high",
                        "score_stage": "verifier_score_two_order",
                        "score_field": "policies.high_only",
                        "production_eligible": True,
                    },
                    {
                        "name": "all_three_order_exact_high",
                        "score_stage": "verifier_score_three_order",
                        "score_field": "policy",
                        "production_eligible": True,
                    },
                ],
                "preinference_audit": {"all_verifier_outputs_absent": True},
                "inputs": {
                    "command_plan": {"sha256": plan_sha},
                    "role_freeze": {"sha256": role_sha},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = build_selection(plan_path, role_freeze, adj, policy)

    assert result["status"] == "selected"
    assert result["chosen"]["verifier_variant"] == "v0"
    assert result["chosen"]["policy"] == "all_three_order_exact_high"
    assert result["diagnostics"][0]["production_eligible"] is False

    failed_adj = json.loads(adj.read_text(encoding="utf-8"))
    failed_adj["status"] = "failed_closed_no_eligible_adjudicator"
    failed_adj["chosen"] = None
    adj.write_text(json.dumps(failed_adj) + "\n", encoding="utf-8")
    failed = build_selection(plan_path, role_freeze, adj, policy)
    assert failed["status"] == "not_run_no_eligible_adjudicator"
    assert failed["chosen"] is None
    assert failed["candidates"] == []
