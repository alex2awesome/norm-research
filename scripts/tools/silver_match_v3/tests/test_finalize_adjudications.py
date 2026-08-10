import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.tools.silver_match_v3.finalize_adjudications import (
    final_match_decision,
    run,
    selected_prompt_sha,
)
from scripts.tools.silver_match_v3.common import read_jsonl, sha256_file


def test_verified_exact_match_requires_both_checks():
    primary = {"decision": "MATCH", "metric_id": "a1"}
    order = {"decision": "MATCH", "metric_id": "a1"}
    verification = {
        "decision": "CONFIRM_MATCH",
        "metric_id": "a1",
        "confidence": "medium",
    }
    assert final_match_decision(primary, order, verification) == (
        "MATCH", "a1", "verified_exact_match"
    )


def test_order_disagreement_abstains():
    result = final_match_decision(
        {"decision": "MATCH", "metric_id": "a1"},
        {"decision": "MATCH", "metric_id": "a2"},
        {"decision": "CONFIRM_MATCH", "metric_id": "a1", "confidence": "high"},
    )
    assert result == ("UNSTABLE_MATCH", None, "order_check_disagreed")


def test_low_confidence_verifier_abstains():
    result = final_match_decision(
        {"decision": "MATCH", "metric_id": "a1"},
        {"decision": "MATCH", "metric_id": "a1"},
        {"decision": "CONFIRM_MATCH", "metric_id": "a1", "confidence": "low"},
    )
    assert result == ("UNSTABLE_MATCH", None, "contrastive_verifier_low_confidence")


def test_primary_typed_abstention_is_retained():
    assert final_match_decision(
        {"decision": "NO_CANDIDATE_FITS", "metric_id": None}, None, None
    ) == ("NO_CANDIDATE_FITS", None, "primary_typed_abstention")


def test_selection_prompt_must_be_task_matched_and_dev_selected(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps(
            {
                "task": "t",
                "selection_split": "external_dev_only",
                "chosen": {"prompt_sha256": "a" * 64},
            }
        )
    )
    prompt_sha, payload = selected_prompt_sha(path, "t", "adjudicator")
    assert prompt_sha == "a" * 64
    assert payload["task"] == "t"
    with pytest.raises(ValueError, match="task mismatch"):
        selected_prompt_sha(path, "other", "adjudicator")


def test_selection_prompt_rejects_test_selected_artifact(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps(
            {
                "task": "t",
                "selection_split": "test",
                "chosen": {"prompt_sha256": "a" * 64},
            }
        )
    )
    with pytest.raises(ValueError, match="not made on dev"):
        selected_prompt_sha(path, "t", "verifier")


def test_explicit_role_selection_uses_role_specific_prompt_hash(tmp_path):
    verifier = tmp_path / "verifier-explicit.json"
    verifier.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-explicit-role-verifier-selection-v1",
                "task": "t",
                "status": "selected",
                "selection_role": "prompt_dev",
                "test_or_blind_audit_consumed": False,
                "production_consumed": False,
                "outcomes_or_mi_used": False,
                "chosen": {"verifier_prompt_sha256": "v" * 64},
            }
        )
    )
    prompt, _ = selected_prompt_sha(verifier, "t", "verifier")
    assert prompt == "v" * 64

    payload = json.loads(verifier.read_text())
    payload["test_or_blind_audit_consumed"] = True
    verifier.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="not cleanly made"):
        selected_prompt_sha(verifier, "t", "verifier")


def test_strict_production_requires_frozen_two_order_verifier_policy(tmp_path):
    bank = tmp_path / "bank.json"
    metric_ids = [f"m{i}" for i in range(50)]
    bank.write_text(
        json.dumps(
            {
                "metrics": [
                    {"metric_id": metric_id, "name": metric_id, "description": metric_id}
                    for metric_id in metric_ids
                ]
            }
        )
    )
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"norm_uid": "u", "row": 0}) + "\n")
    bank_sha = "b" * 64
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3.0",
                "banks": {"t": {"path": str(bank), "source_sha256": bank_sha}},
                "corpora": {
                    "c": {"path": str(corpus), "task": "t", "count": 1}
                },
            }
        )
    )
    adjudicator_prompt = "a" * 64
    verifier_prompt = "c" * 64
    primary = tmp_path / "primary.jsonl"
    primary.write_text(
        json.dumps(
            {
                "norm_uid": "u",
                "corpus": "c",
                "task": "t",
                "row": 0,
                "decision": "MATCH",
                "metric_id": "m0",
                "confidence": "high",
                "reason": "best",
                "candidate_ids": metric_ids,
                "candidate_bank_source_sha256": bank_sha,
                "prompt_sha256": adjudicator_prompt,
                "model": "/gemma/snapshot",
                "order_mode": "original",
                "parse_error": None,
            }
        )
        + "\n"
    )
    order = tmp_path / "order.jsonl"
    order.write_text(
        json.dumps(
            {
                "norm_uid": "u",
                "decision": "MATCH",
                "metric_id": "m0",
                "confidence": "high",
                "reason": "best",
                "candidate_ids": list(reversed(metric_ids)),
                "candidate_bank_source_sha256": bank_sha,
                "prompt_sha256": adjudicator_prompt,
                "model": "/gemma/snapshot",
                "order_mode": "hashed",
            }
        )
        + "\n"
    )
    adjudicator_selection = tmp_path / "adj-selection.json"
    adjudicator_selection.write_text(
        json.dumps(
            {
                "task": "t",
                "selection_split": "dev",
                "candidate_depth": 50,
                "chosen": {"prompt_sha256": adjudicator_prompt},
            }
        )
    )
    verifier_selection = tmp_path / "verifier-selection.json"
    verifier_selection.write_text(
        json.dumps(
            {
                "task": "t",
                "selection_split": "external_dev_only",
                "chosen": {"prompt_sha256": verifier_prompt},
            }
        )
    )
    verifier_policy = tmp_path / "verifier-policy.json"
    verifier_policy.write_text(
        json.dumps(
            {
                "task": "t",
                "inputs": {
                    "selection": {"sha256": sha256_file(verifier_selection)}
                },
                    "may_run_on_production_unlabeled_norms": True,
                    "dev_gate": {"cleared": True},
                    "order_policy": {"orders": ["original", "hashed"]},
            }
        )
    )
    production_plan = tmp_path / "production-plan.json"
    production_plan.write_text(
        json.dumps(
            {
                "status": "FROZEN_READY_FOR_UNLABELED_PRODUCTION",
                "task": "t",
                "manifest": {"sha256": sha256_file(manifest)},
                "bank_source_sha256": bank_sha,
                "adjudicator": {
                    "selection": {"sha256": sha256_file(adjudicator_selection)}
                },
                    "verifier": {
                        "selection": {"sha256": sha256_file(verifier_selection)},
                        "production_policy": {"sha256": sha256_file(verifier_policy)},
                        "orders": ["original", "hashed"],
                    },
            }
        )
    )
    verification = tmp_path / "verification.jsonl"
    verification.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-two-order-production-verification-v1",
                "norm_uid": "u",
                "decision": "CONFIRM_MATCH",
                "metric_id": "m0",
                "confidence": "high",
                "reason": "both_orders_high_confidence_confirm_same_id",
                "primary_metric_id": "m0",
                "primary_prompt_sha256": adjudicator_prompt,
                "candidate_bank_source_sha256": bank_sha,
                "prompt_sha256": verifier_prompt,
                "model": "/gemma/snapshot",
                "verification_orders": ["original", "hashed"],
                "strict_two_order_acceptance": True,
                "verifier_selection_sha256": sha256_file(verifier_selection),
                "verifier_policy_sha256": sha256_file(verifier_policy),
                "production_plan_sha256": sha256_file(production_plan),
            }
        )
        + "\n"
    )
    output = tmp_path / "final.jsonl"
    args = SimpleNamespace(
        manifest=str(manifest),
        corpus="c",
        primary=[str(primary)],
        order_check=[str(order)],
        verification=[str(verification)],
        adjudicator_selection=str(adjudicator_selection),
        verifier_selection=str(verifier_selection),
        verifier_policy=str(verifier_policy),
        production_plan=str(production_plan),
        strict_production=True,
        output=str(output),
    )
    report = run(args)
    assert report["complete"] is True
    assert list(read_jsonl(output))[0]["decision"] == "MATCH"

    policy_payload = json.loads(verifier_policy.read_text())
    policy_payload["order_policy"]["orders"] = ["original", "hashed", "reverse"]
    verifier_policy.write_text(json.dumps(policy_payload))
    plan_payload = json.loads(production_plan.read_text())
    plan_payload["verifier"]["orders"] = ["original", "hashed", "reverse"]
    plan_payload["verifier"]["production_policy"]["sha256"] = sha256_file(
        verifier_policy
    )
    production_plan.write_text(json.dumps(plan_payload))
    verification_payload = json.loads(verification.read_text().strip())
    verification_payload.update(
        {
            "schema_version": "silver-match-v3-multi-order-production-verification-v1",
            "verification_orders": ["original", "hashed", "reverse"],
            "strict_two_order_acceptance": None,
            "strict_all_order_acceptance": True,
            "accepted_by_order": {
                "original": True,
                "hashed": True,
                "reverse": True,
            },
            "verifier_policy_sha256": sha256_file(verifier_policy),
            "production_plan_sha256": sha256_file(production_plan),
        }
    )
    verification.write_text(json.dumps(verification_payload) + "\n")
    args.output = str(tmp_path / "multi-order-final.jsonl")
    report = run(args)
    assert report["complete"] is True
    assert list(read_jsonl(Path(args.output)))[0]["decision"] == "MATCH"

    args.output = str(tmp_path / "should-not-exist.jsonl")
    args.verifier_policy = None
    with pytest.raises(ValueError, match="requires the production plan"):
        run(args)
