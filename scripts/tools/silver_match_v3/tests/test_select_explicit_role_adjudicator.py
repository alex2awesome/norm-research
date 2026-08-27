import json
from pathlib import Path

from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.select_explicit_role_adjudicator import (
    build_selection,
)


def _score(path: Path, *, role: str, prompt: str, correct: int, support: int) -> None:
    input_path = path.with_suffix(".input.jsonl")
    input_path.write_text("{}\n", encoding="utf-8")
    panel = "prompt_train" if role == "optimize" else "prompt_dev"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-two-order-gepa-score-v1",
                "selection_universe": "predeclared_train_only",
                "explicit_role": role,
                "panel_role": panel,
                "prompt_sha256": prompt,
                "metrics": {
                    "strict_consensus": {
                        "confirmed_match_count": support,
                        "correct_exact_id_count": correct,
                        "exact_id_precision": correct / support,
                    }
                },
                "inputs": {
                    "truth": {
                        "path": str(input_path.resolve()),
                        "sha256": sha256_file(input_path),
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_selects_by_wilson_then_support_then_point(tmp_path: Path) -> None:
    variants = [("a", "a" * 64, 29, 30), ("b", "b" * 64, 45, 47)]
    commands = []
    for name, prompt, correct, support in variants:
        for role, panel in (("optimize", "prompt_train"), ("select", "prompt_dev")):
            score = tmp_path / f"{name}.{role}.json"
            _score(score, role=role, prompt=prompt, correct=correct, support=support)
            commands.append(
                {
                    "stage": "adjudicator_score",
                    "variant": name,
                    "role": panel,
                    "command": {
                        "module": "unused",
                        "argv": ["--output", str(score)],
                    },
                }
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
                    "minimum_wilson_95_lower": 0.8,
                    "minimum_retained": 30,
                    "selection_rule": "Wilson, support, point",
                },
                "adjudicator_variants": [
                    {"name": name, "combined_prompt_sha256": prompt}
                    for name, prompt, _, _ in variants
                ],
                "commands": commands,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    freeze_path = tmp_path / "FREEZE.json"
    freeze_path.write_text(
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

    result = build_selection(plan_path, freeze_path)

    assert result["status"] == "selected"
    assert result["chosen"]["name"] == "b"
    assert all(row["eligible"] for row in result["variants"])


def test_fails_closed_on_exact_selection_key_tie(tmp_path: Path) -> None:
    prompt = "c" * 64
    commands = []
    declared = []
    for name in ("a", "b"):
        declared.append({"name": name, "combined_prompt_sha256": prompt})
        for role, panel in (("optimize", "prompt_train"), ("select", "prompt_dev")):
            score = tmp_path / f"{name}.{role}.json"
            _score(score, role=role, prompt=prompt, correct=29, support=30)
            commands.append(
                {
                    "stage": "adjudicator_score",
                    "variant": name,
                    "role": panel,
                    "command": {"module": "unused", "argv": ["--output", str(score)]},
                }
            )
    plan_path = tmp_path / "COMMAND_PLAN.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-explicit-role-task-local-gepa-plan-v1",
                "status": "FROZEN_BEFORE_TASK_LOCAL_GEPA_INFERENCE",
                "task": "math-stackexchange",
                "thresholds": {
                    "minimum_point_precision": 0.9,
                    "minimum_wilson_95_lower": 0.8,
                    "minimum_retained": 30,
                    "selection_rule": "Wilson, support, point",
                },
                "adjudicator_variants": declared,
                "commands": commands,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    freeze_path = tmp_path / "FREEZE.json"
    freeze_path.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-explicit-role-task-local-gepa-freeze-v1",
                "status": "FROZEN_BEFORE_TASK_LOCAL_GEPA_INFERENCE",
                "task": "math-stackexchange",
                "command_plan": {"sha256": sha256_file(plan_path)},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = build_selection(plan_path, freeze_path)

    assert result["status"] == "failed_closed_exact_selection_key_tie"
    assert result["chosen"] is None
