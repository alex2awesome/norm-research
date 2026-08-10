import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.seal_explicit_role_verifier_production_policy import (
    seal_policy,
)


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n")


def _fixture(tmp_path: Path):
    prompt = tmp_path / "verify.txt"
    prompt.write_text("verify")
    prompt_sha = sha256_file(prompt)
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text("{}\n")
    primary = tmp_path / "primary.jsonl"
    primary.write_text("{}\n")
    model = "/models/gemma"
    commands = []
    for order in ("original", "hashed", "reverse"):
        output = tmp_path / f"{order}.jsonl"
        output.write_text("{}\n")
        meta = {
            "model": model,
            "max_alternatives": 49,
            "batch_size": 256,
            "max_model_len": 8192,
            "max_tokens": 180,
            "gpu_memory_utilization": 0.9,
            "enforce_eager": False,
            "seed": 29,
            "context_chars": 1400,
            "description_chars": 520,
            "example_chars": 180,
            "max_examples": 2,
            "order_mode": order,
            "output_sha256": sha256_file(output),
            "prompt_sha256": prompt_sha,
            "input_candidates_sha256": sha256_file(candidates),
            "primary_sha256": sha256_file(primary),
            "invalid_count": 0,
            "prompt_component_sha256": {str(prompt.resolve()): prompt_sha},
        }
        _write(output.with_suffix(".jsonl.meta.json"), meta)
        commands.append(
            {
                "stage": "verifier",
                "role": "prompt_dev",
                "adjudicator_variant": "r0",
                "verifier_variant": "v0",
                "order": order,
                "direct_batch_command": {
                    "module": "scripts.tools.silver_match_v3.verify_gemma",
                    "argv": [
                        "--output",
                        str(output),
                        "--order-mode",
                        order,
                        "--model",
                        model,
                        "--max-alternatives",
                        "49",
                    ],
                },
            }
        )
    plan = tmp_path / "COMMAND_PLAN.json"
    _write(
        plan,
        {
            "schema_version": "silver-match-v3-explicit-role-task-local-gepa-plan-v1",
            "status": "FROZEN_BEFORE_TASK_LOCAL_GEPA_INFERENCE",
            "task": "t",
            "thresholds": {
                "minimum_point_precision": 0.9,
                "minimum_retained": 30,
                "minimum_wilson_95_lower": 0.8,
            },
            "verifier_variants": [
                {
                    "name": "v0",
                    "combined_prompt_sha256": prompt_sha,
                    "components": [
                        {"path": str(prompt.resolve()), "sha256": prompt_sha}
                    ],
                }
            ],
            "commands": commands,
        },
    )
    plan_sha = sha256_file(plan)
    role_freeze = tmp_path / "FREEZE.json"
    _write(
        role_freeze,
        {
            "schema_version": "silver-match-v3-explicit-role-task-local-gepa-freeze-v1",
            "task": "t",
            "command_plan": {"sha256": plan_sha},
            "test_or_blind_audit_consumed": False,
            "production_consumed": False,
        },
    )
    adjudicator = tmp_path / "adjudicator.json"
    _write(
        adjudicator,
        {
            "schema_version": "silver-match-v3-explicit-role-adjudicator-selection-v1",
            "task": "t",
            "status": "selected",
            "selection_role": "prompt_dev",
            "test_or_blind_audit_consumed": False,
            "production_consumed": False,
            "outcomes_or_mi_used": False,
            "chosen": {"name": "r0"},
            "inputs": {"command_plan": {"sha256": plan_sha}},
        },
    )
    policy_freeze = tmp_path / "VERIFIER_SELECTION_FREEZE.json"
    _write(
        policy_freeze,
        {
            "schema_version": "silver-match-v3-explicit-role-verifier-selection-freeze-v1",
            "status": "FROZEN_BEFORE_ANY_VERIFIER_INFERENCE_OR_SCORE",
            "task": "t",
            "test_or_blind_audit_consumed": False,
            "production_consumed": False,
            "outcomes_or_mi_used": False,
            "inputs": {"command_plan": {"sha256": plan_sha}},
        },
    )
    optimize = tmp_path / "optimize-score.json"
    select = tmp_path / "select-score.json"
    _write(optimize, {"score": 1})
    _write(select, {"score": 1})
    selection = tmp_path / "verifier.json"
    _write(
        selection,
        {
            "schema_version": "silver-match-v3-explicit-role-verifier-selection-v1",
            "task": "t",
            "status": "selected",
            "selection_role": "prompt_dev",
            "test_or_blind_audit_consumed": False,
            "production_consumed": False,
            "outcomes_or_mi_used": False,
            "thresholds": {
                "minimum_point_precision": 0.9,
                "minimum_retained": 30,
                "minimum_wilson_95_lower": 0.8,
            },
            "chosen": {
                "eligible": True,
                "policy": "all_three_order_exact_high",
                "adjudicator_variant": "r0",
                "verifier_variant": "v0",
                "verifier_prompt_sha256": prompt_sha,
                "select_metrics": {
                    "retained": 40,
                    "retained_precision": 0.9,
                    "retained_precision_wilson_95": [0.81, 0.97],
                },
                "optimize_score": {
                    "path": str(optimize),
                    "sha256": sha256_file(optimize),
                },
                "select_score": {
                    "path": str(select),
                    "sha256": sha256_file(select),
                },
            },
            "inputs": {
                "plan": {"path": str(plan), "sha256": plan_sha},
                "role_freeze": {
                    "path": str(role_freeze),
                    "sha256": sha256_file(role_freeze),
                },
                "adjudicator_selection": {
                    "path": str(adjudicator),
                    "sha256": sha256_file(adjudicator),
                },
                "policy_freeze": {
                    "path": str(policy_freeze),
                    "sha256": sha256_file(policy_freeze),
                },
            },
        },
    )
    return plan, role_freeze, adjudicator, selection


def test_seals_selected_three_order_policy_and_runtime(tmp_path: Path) -> None:
    plan, role_freeze, adjudicator, selection = _fixture(tmp_path)
    policy = seal_policy(
        plan_path=plan,
        role_freeze_path=role_freeze,
        adjudicator_selection_path=adjudicator,
        verifier_selection_path=selection,
    )
    assert policy["dev_gate"]["cleared"] is True
    assert policy["order_policy"]["orders"] == ["original", "hashed", "reverse"]
    assert set(policy["selected_prompt_dev_runs"]) == {
        "original",
        "hashed",
        "reverse",
    }


def test_rejects_runtime_drift_in_one_selected_order(tmp_path: Path) -> None:
    plan, role_freeze, adjudicator, selection = _fixture(tmp_path)
    payload = json.loads(plan.read_text())
    reverse = next(
        cell
        for cell in payload["commands"]
        if cell.get("stage") == "verifier" and cell.get("order") == "reverse"
    )
    output = Path(reverse["direct_batch_command"]["argv"][1])
    meta_path = output.with_suffix(".jsonl.meta.json")
    meta = json.loads(meta_path.read_text())
    meta["seed"] = 31
    _write(meta_path, meta)
    with pytest.raises(ValueError, match="reverse verifier dev runtime differs"):
        seal_policy(
            plan_path=plan,
            role_freeze_path=role_freeze,
            adjudicator_selection_path=adjudicator,
            verifier_selection_path=selection,
        )
