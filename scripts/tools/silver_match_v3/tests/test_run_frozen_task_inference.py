import json

import pytest

from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.run_frozen_task_inference import build_command


def _artifact(path):
    return {"path": str(path), "sha256": sha256_file(path)}


def _fixture(tmp_path):
    repo = tmp_path / "repo"
    implementation_dir = repo / "scripts" / "tools" / "silver_match_v3"
    prompt_dir = implementation_dir / "prompts"
    prompt_dir.mkdir(parents=True)
    adjudicator_impl = implementation_dir / "adjudicate_gemma.py"
    verifier_impl = implementation_dir / "verify_gemma.py"
    adjudicator_impl.write_text("# adjudicator\n", encoding="utf-8")
    verifier_impl.write_text("# verifier\n", encoding="utf-8")
    adjudicator_prompt = prompt_dir / "adjudicate.txt"
    verifier_prompt = prompt_dir / "verify.txt"
    addon = prompt_dir / "addon.txt"
    for path, value in (
        (adjudicator_prompt, "adjudicate"),
        (verifier_prompt, "verify"),
        (addon, "addon"),
    ):
        path.write_text(value, encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    candidates = tmp_path / "candidates.jsonl"
    candidates_meta = tmp_path / "candidates.jsonl.meta.json"
    selection = tmp_path / "selection.json"
    verifier_selection = tmp_path / "verifier-selection.json"
    policy = tmp_path / "policy.json"
    primary = tmp_path / "primary.jsonl"
    for path in (
        manifest,
        candidates,
        candidates_meta,
        selection,
        verifier_selection,
        policy,
        primary,
    ):
        path.write_text("{}\n", encoding="utf-8")
    plan = {
        "schema_version": "silver-match-v3-task-production-plan-v1",
        "status": "FROZEN_READY_FOR_UNLABELED_PRODUCTION",
        "manifest": _artifact(manifest),
        "candidate_union": _artifact(candidates),
        "candidate_union_meta": _artifact(candidates_meta),
        "adjudicator": {
            "implementation": _artifact(adjudicator_impl),
            "selection": _artifact(selection),
            "prompt": str(adjudicator_prompt),
            "prompt_addons": [str(addon)],
            "prompt_components": {
                str(adjudicator_prompt): {"sha256": sha256_file(adjudicator_prompt)},
                str(addon): {"sha256": sha256_file(addon)},
            },
            "model": "/model/gemma",
            "candidate_depth": 50,
            "prompt_rendering": {
                "context_chars": 1200,
                "description_chars": 260,
                "example_chars": 80,
                "max_examples": 0,
            },
            "production_sampling": {
                "max_model_len": 8192,
                "max_tokens": 160,
                "seed": 17,
            },
        },
        "verifier": {
            "implementation": _artifact(verifier_impl),
            "selection": _artifact(verifier_selection),
            "production_policy": _artifact(policy),
            "prompt": str(verifier_prompt),
            "prompt_addons": [str(addon)],
            "prompt_components": {
                str(verifier_prompt): {"sha256": sha256_file(verifier_prompt)},
                str(addon): {"sha256": sha256_file(addon)},
            },
            "rendering": {
                "model": "/model/gemma",
                "max_alternatives": 49,
                "context_chars": 1200,
                "description_chars": 260,
                "example_chars": 180,
                "max_examples": 0,
                "max_model_len": 8192,
                "max_tokens": 180,
                "seed": 29,
            },
        },
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return plan_path, primary, adjudicator_prompt


def test_adjudicator_command_comes_entirely_from_frozen_plan(tmp_path):
    plan, _, _ = _fixture(tmp_path)
    command, repo, _ = build_command(
        plan_path=plan,
        stage="adjudicator",
        order="hashed",
        output_path=tmp_path / "output.jsonl",
        shard_id=2,
        num_shards=4,
        batch_size=128,
        gpu_memory_utilization=.88,
    )
    assert repo == tmp_path / "repo"
    assert "scripts.tools.silver_match_v3.adjudicate_gemma" in command
    assert command[command.index("--max-candidates") + 1] == "50"
    assert command[command.index("--model") + 1] == "/model/gemma"
    assert command[command.index("--shard-id") + 1] == "2"
    assert command[command.index("--order-mode") + 1] == "hashed"


def test_verifier_command_requires_primary_and_uses_frozen_rendering(tmp_path):
    plan, primary, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="requires an existing"):
        build_command(
            plan_path=plan,
            stage="verifier",
            order="original",
            output_path=tmp_path / "output.jsonl",
            shard_id=0,
            num_shards=1,
            batch_size=128,
            gpu_memory_utilization=.88,
        )
    command, _, _ = build_command(
        plan_path=plan,
        stage="verifier",
        order="original",
        output_path=tmp_path / "output.jsonl",
        primary_path=primary,
        shard_id=0,
        num_shards=1,
        batch_size=128,
        gpu_memory_utilization=.88,
    )
    assert "scripts.tools.silver_match_v3.verify_gemma" in command
    assert command[command.index("--max-alternatives") + 1] == "49"
    assert command[command.index("--seed") + 1] == "29"


def test_command_rejects_changed_frozen_prompt(tmp_path):
    plan, _, prompt = _fixture(tmp_path)
    prompt.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen artifact changed"):
        build_command(
            plan_path=plan,
            stage="adjudicator",
            order="original",
            output_path=tmp_path / "output.jsonl",
            shard_id=0,
            num_shards=1,
            batch_size=128,
            gpu_memory_utilization=.88,
        )
