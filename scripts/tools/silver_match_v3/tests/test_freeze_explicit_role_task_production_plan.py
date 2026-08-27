import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.freeze_explicit_role_task_production_plan import (
    freeze_plan,
)


def _dump(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload) + "\n")
    return path


def _artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _fixture(tmp_path: Path):
    repo = tmp_path / "repo"
    implementation = repo / "scripts" / "tools" / "silver_match_v3"
    implementation.mkdir(parents=True)
    for name in (
        "adjudicate_gemma.py",
        "verify_gemma.py",
        "combine_ordered_verifications.py",
        "finalize_adjudications.py",
        "run_task_production.py",
    ):
        (implementation / name).write_text(f"# {name}\n")
    adj_prompt = repo / "adj.txt"
    verify_prompt = repo / "verify.txt"
    adj_prompt.write_text("adjudicate")
    verify_prompt.write_text("verify")
    adj_prompt_sha = sha256_file(adj_prompt)
    verify_prompt_sha = sha256_file(verify_prompt)

    source_a = tmp_path / "a.jsonl"
    source_b = tmp_path / "b.jsonl"
    source_a.write_text('{"norm_uid":"a"}\n')
    source_b.write_text('{"norm_uid":"b"}\n')
    candidates = tmp_path / "all.jsonl"
    candidates.write_text(source_a.read_text() + source_b.read_text())
    manifest = _dump(
        tmp_path / "manifest.json",
        {
            "banks": {"task": {"source_sha256": "b" * 64}},
            "corpora": {
                "a": {"task": "task", "count": 1},
                "b": {"task": "task", "count": 1},
            },
        },
    )
    audits = []
    for corpus, source in (("a", source_a), ("b", source_b)):
        audits.append(
            _dump(
                tmp_path / f"{corpus}.audit.json",
                {
                    "complete": True,
                    "task": "task",
                    "corpus": corpus,
                    "manifest_sha256": sha256_file(manifest),
                    "bank_source_sha256": "b" * 64,
                    "observed_count": 1,
                    "candidate_inputs": {str(source): {"sha256": sha256_file(source)}},
                },
            )
        )
    _dump(
        candidates.with_suffix(".jsonl.meta.json"),
        {
            "count": 2,
            "sha256": sha256_file(candidates),
            "inputs": {
                str(source_a): {"sha256": sha256_file(source_a)},
                str(source_b): {"sha256": sha256_file(source_b)},
            },
        },
    )
    retriever = _dump(
        tmp_path / "retriever.json",
        {"task": "task", "selection_split": "dev"},
    )

    commands = []
    for order in ("original", "hashed"):
        output = tmp_path / f"adj.{order}.jsonl"
        output.write_text("{}\n")
        _dump(
            output.with_suffix(".jsonl.meta.json"),
            {
                "order_mode": order,
                "output_sha256": sha256_file(output),
                "prompt_sha256": adj_prompt_sha,
                "max_candidates": 50,
                "prompt_component_sha256": {
                    str(adj_prompt.resolve()): adj_prompt_sha
                },
                "invalid_count": 0,
                "model": "/models/gemma",
                "prompt_rendering": {
                    "context_chars": 1400,
                    "description_chars": 520,
                    "example_chars": 180,
                    "max_examples": 2,
                },
            },
        )
        commands.append(
            {
                "stage": "adjudicator",
                "role": "prompt_dev",
                "variant": "r0",
                "order": order,
                "direct_batch_command": {
                    "module": "scripts.tools.silver_match_v3.adjudicate_gemma",
                    "argv": [
                        "--output",
                        str(output),
                        "--order-mode",
                        order,
                        "--model",
                        "/models/gemma",
                        "--max-candidates",
                        "50",
                        "--batch-size",
                        "256",
                        "--gpu-memory-utilization",
                        "0.9",
                    ],
                },
            }
        )
    explicit = _dump(
        tmp_path / "COMMAND_PLAN.json",
        {
            "schema_version": "silver-match-v3-explicit-role-task-local-gepa-plan-v1",
            "status": "FROZEN_BEFORE_TASK_LOCAL_GEPA_INFERENCE",
            "task": "task",
            "candidate_bank_source_sha256": "b" * 64,
            "candidate_k": 50,
            "inputs": {"manifest": _artifact(manifest)},
            "scientific_scope": {
                "test_or_blind_audit_consumed": False,
                "production_consumed": False,
                "outcomes_or_mi_used": False,
            },
            "adjudicator_variants": [
                {
                    "name": "r0",
                    "combined_prompt_sha256": adj_prompt_sha,
                    "components": [_artifact(adj_prompt)],
                }
            ],
            "verifier_variants": [
                {
                    "name": "v0",
                    "combined_prompt_sha256": verify_prompt_sha,
                    "components": [_artifact(verify_prompt)],
                }
            ],
            "commands": commands,
        },
    )
    explicit_sha = sha256_file(explicit)
    role_freeze = _dump(
        tmp_path / "FREEZE.json",
        {
            "schema_version": "silver-match-v3-explicit-role-task-local-gepa-freeze-v1",
            "task": "task",
            "command_plan": {"sha256": explicit_sha},
            "test_or_blind_audit_consumed": False,
            "production_consumed": False,
        },
    )
    adjudicator = _dump(
        tmp_path / "adjudicator.json",
        {
            "schema_version": "silver-match-v3-explicit-role-adjudicator-selection-v1",
            "task": "task",
            "status": "selected",
            "selection_role": "prompt_dev",
            "test_or_blind_audit_consumed": False,
            "production_consumed": False,
            "outcomes_or_mi_used": False,
            "chosen": {"name": "r0", "prompt_sha256": adj_prompt_sha},
            "inputs": {"command_plan": {"sha256": explicit_sha}},
        },
    )
    verifier = _dump(
        tmp_path / "verifier.json",
        {
            "schema_version": "silver-match-v3-explicit-role-verifier-selection-v1",
            "task": "task",
            "status": "selected",
            "selection_role": "prompt_dev",
            "test_or_blind_audit_consumed": False,
            "production_consumed": False,
            "outcomes_or_mi_used": False,
            "chosen": {
                "policy": "all_three_order_exact_high",
                "verifier_variant": "v0",
                "verifier_prompt_sha256": verify_prompt_sha,
            },
            "inputs": {"plan": {"sha256": explicit_sha}},
        },
    )
    verifier_dev_runs = {}
    rendering = {
        "model": "/models/gemma",
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
    }
    for order in ("original", "hashed", "reverse"):
        output = tmp_path / f"verify.{order}.jsonl"
        output.write_text("{}\n")
        meta = _dump(output.with_suffix(".jsonl.meta.json"), {"order_mode": order})
        verifier_dev_runs[order] = {
            "output": _artifact(output),
            "meta": _artifact(meta),
        }
    policy = _dump(
        tmp_path / "policy.json",
        {
            "schema_version": "silver-match-v3-verifier-production-policy-v2",
            "task": "task",
            "selection_role": "prompt_dev",
            "test_or_blind_audit_consumed": False,
            "production_consumed": False,
            "outcomes_or_mi_used": False,
            "may_run_on_production_unlabeled_norms": True,
            "selected_policy": "all_three_order_exact_high",
            "prompt": {
                "rendered_prompt_sha256": verify_prompt_sha,
                "components": [_artifact(verify_prompt)],
            },
            "rendering": rendering,
            "order_policy": {
                "orders": ["original", "hashed", "reverse"],
                "acceptance_mode": "all_orders_exact_high_same_id_no_parse_error",
                "retain_only_if": "every frozen order confirms exact/high",
            },
            "dev_gate": {"cleared": True},
            "selected_prompt_dev_runs": verifier_dev_runs,
            "inputs": {
                "selection": _artifact(verifier),
                "command_plan": _artifact(explicit),
            },
        },
    )
    return {
        "manifest_path": manifest,
        "task": "task",
        "candidate_path": candidates,
        "candidate_audit_paths": audits,
        "retriever_selection_path": retriever,
        "explicit_plan_path": explicit,
        "role_freeze_path": role_freeze,
        "adjudicator_selection_path": adjudicator,
        "verifier_selection_path": verifier,
        "verifier_policy_path": policy,
        "repo_root": repo,
    }


def test_freezes_exact_selected_three_order_topology(tmp_path: Path) -> None:
    payload = freeze_plan(**_fixture(tmp_path))
    assert payload["status"] == "FROZEN_READY_FOR_UNLABELED_PRODUCTION"
    assert payload["expected_count"] == 2
    assert payload["corpora"] == ["a", "b"]
    assert payload["adjudicator"]["orders"] == ["original", "hashed"]
    assert payload["verifier"]["orders"] == ["original", "hashed", "reverse"]
    assert payload["verifier"]["acceptance_mode"] == (
        "all_orders_exact_high_same_id_no_parse_error"
    )


def test_rejects_incomplete_all_corpus_candidate_audits(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    kwargs["candidate_audit_paths"] = kwargs["candidate_audit_paths"][:1]
    with pytest.raises(ValueError, match="candidate audits are incomplete"):
        freeze_plan(**kwargs)
