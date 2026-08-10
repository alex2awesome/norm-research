import json
from pathlib import Path

import pytest

import scripts.tools.silver_match_v3.combine_ordered_verifications as combine_module
from scripts.tools.silver_match_v3.combine_ordered_verifications import combine
from scripts.tools.silver_match_v3.common import read_jsonl, sha256_file


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _fixture(tmp_path: Path):
    adj_prompt = "a" * 64
    verifier_prompt = "v" * 64
    bank = "b" * 64
    primary = tmp_path / "primary.jsonl"
    _write_jsonl(
        primary,
        [
            {
                "norm_uid": "u",
                "corpus": "c",
                "task": "t",
                "row": 0,
                "decision": "MATCH",
                "metric_id": "m1",
                "prompt_sha256": adj_prompt,
                "candidate_bank_source_sha256": bank,
            }
        ],
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-explicit-role-verifier-selection-v1",
                "task": "t",
                "status": "selected",
                "selection_role": "prompt_dev",
                "test_or_blind_audit_consumed": False,
                "production_consumed": False,
                "outcomes_or_mi_used": False,
                "chosen": {
                    "eligible": True,
                    "verifier_prompt_sha256": verifier_prompt,
                },
            }
        )
    )
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "task": "t",
                "inputs": {"selection": {"sha256": sha256_file(selection)}},
                "may_run_on_production_unlabeled_norms": True,
                "dev_gate": {"cleared": True},
                "prompt": {"rendered_prompt_sha256": verifier_prompt},
                "order_policy": {
                    "orders": ["original", "hashed", "reverse"],
                    "acceptance_mode": "all_orders_exact_high_same_id_no_parse_error",
                },
            }
        )
    )
    paths = {}
    for order in ("original", "hashed", "reverse"):
        path = tmp_path / f"{order}.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "norm_uid": "u",
                    "order_mode": order,
                    "decision": "CONFIRM_MATCH",
                    "metric_id": "m1",
                    "confidence": "high",
                    "parse_error": None,
                    "primary_metric_id": "m1",
                    "primary_prompt_sha256": adj_prompt,
                    "prompt_sha256": verifier_prompt,
                    "candidate_bank_source_sha256": bank,
                    "model": "/gemma",
                    "alternative_ids": ["m2", "m3"],
                }
            ],
        )
        paths[order] = path
    return primary, selection, policy, paths


def test_three_order_policy_requires_and_accepts_all_three(tmp_path: Path) -> None:
    primary, selection, policy, paths = _fixture(tmp_path)
    output = tmp_path / "combined.jsonl"
    report = combine(
        primary_path=primary,
        verification_paths=paths,
        selection_path=selection,
        policy_path=policy,
        output_path=output,
    )
    row = list(read_jsonl(output))[0]
    assert report["complete"] is True
    assert row["decision"] == "CONFIRM_MATCH"
    assert row["verification_orders"] == ["original", "hashed", "reverse"]
    assert row["strict_all_order_acceptance"] is True


def test_three_order_policy_rejects_missing_reverse(tmp_path: Path) -> None:
    primary, selection, policy, paths = _fixture(tmp_path)
    paths.pop("reverse")
    with pytest.raises(ValueError, match="exact frozen order list"):
        combine(
            primary_path=primary,
            verification_paths=paths,
            selection_path=selection,
            policy_path=policy,
            output_path=tmp_path / "combined.jsonl",
        )


def test_one_order_rejection_drops_match(tmp_path: Path) -> None:
    primary, selection, policy, paths = _fixture(tmp_path)
    reverse = list(read_jsonl(paths["reverse"]))[0]
    reverse["decision"] = "REJECT_MATCH"
    reverse["metric_id"] = None
    _write_jsonl(paths["reverse"], [reverse])
    output = tmp_path / "combined.jsonl"
    combine(
        primary_path=primary,
        verification_paths=paths,
        selection_path=selection,
        policy_path=policy,
        output_path=output,
    )
    row = list(read_jsonl(output))[0]
    assert row["decision"] == "REJECT_MATCH"
    assert row["strict_all_order_acceptance"] is False


def test_rescue_runtime_binding_is_hash_linked_and_exact(tmp_path: Path) -> None:
    primary, selection, policy, paths = _fixture(tmp_path)
    candidates = tmp_path / "finalists.jsonl"
    candidates.write_text("{}\n")
    component = tmp_path / "verify-prompt.txt"
    component.write_text("verify")
    component_sha = sha256_file(component)
    verify_implementation = tmp_path / "verify_gemma.py"
    verify_implementation.write_text("# verify\n")
    rendering = {
        "model": "/gemma",
        "max_alternatives": 2,
        "batch_size": 64,
        "max_model_len": 8192,
        "max_tokens": 180,
        "gpu_memory_utilization": 0.8,
        "enforce_eager": False,
        "seed": 29,
        "context_chars": 1400,
        "description_chars": 520,
        "example_chars": 180,
        "max_examples": 2,
    }
    for order, path in paths.items():
        (path.with_suffix(".jsonl.meta.json")).write_text(
            json.dumps(
                {
                    **rendering,
                    "input_candidates": str(candidates),
                    "input_candidates_sha256": sha256_file(candidates),
                    "primary_sha256": sha256_file(primary),
                    "prompt_sha256": "v" * 64,
                    "order_mode": order,
                    "output_sha256": sha256_file(path),
                    "invalid_count": 0,
                    "prompt_component_sha256": {
                        str(component): component_sha
                    },
                }
            )
        )
    rescue_plan = tmp_path / "rescue-plan.json"
    rescue_plan.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-task-rescue-plan-v3",
                "status": "FROZEN_READY_FOR_REPEATED_FULL_BANK_RESCUE",
                "task": "t",
                "rescue_policy": {"max_finalists": 3},
                "verifier": {
                    "selection": {
                        "path": str(selection),
                        "sha256": sha256_file(selection),
                    },
                    "production_policy": {
                        "path": str(policy),
                        "sha256": sha256_file(policy),
                    },
                    "orders": ["original", "hashed", "reverse"],
                    "rendering": rendering,
                    "prompt_components": {
                        str(component): {"sha256": component_sha}
                    },
                },
                "implementations": {
                    "combine_ordered_verifications.py": {
                        "path": str(Path(combine_module.__file__).resolve()),
                        "sha256": sha256_file(
                            Path(combine_module.__file__).resolve()
                        ),
                    },
                    "verify_gemma.py": {
                        "path": str(verify_implementation),
                        "sha256": sha256_file(verify_implementation),
                    },
                },
            }
        )
    )
    output = tmp_path / "rescue-combined.jsonl"
    report = combine(
        primary_path=primary,
        verification_paths=paths,
        selection_path=selection,
        policy_path=policy,
        rescue_plan_path=rescue_plan,
        output_path=output,
    )
    row = list(read_jsonl(output))[0]
    assert report["rescue_plan"]["sha256"] == sha256_file(rescue_plan)
    assert row["rescue_plan_sha256"] == sha256_file(rescue_plan)
