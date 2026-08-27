import json

import pytest

from scripts.tools.silver_match_v3.combine_two_order_verifications import combine
from scripts.tools.silver_match_v3.common import read_jsonl, sha256_file


def _rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _fixture(tmp_path):
    primary = _rows(
        tmp_path / "primary.jsonl",
        [
            {
                "norm_uid": "keep",
                "corpus": "c",
                "task": "t",
                "row": 0,
                "decision": "MATCH",
                "metric_id": "m1",
                "prompt_sha256": "a" * 64,
                "candidate_bank_source_sha256": "b" * 64,
            },
            {
                "norm_uid": "drop",
                "corpus": "c",
                "task": "t",
                "row": 1,
                "decision": "MATCH",
                "metric_id": "m2",
                "prompt_sha256": "a" * 64,
                "candidate_bank_source_sha256": "b" * 64,
            },
            {"norm_uid": "not-proposed", "decision": "NOISE"},
        ],
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "selection_split": "external_dev_only",
                "calibration_power_status": "supported",
                "chosen": {
                    "statistically_supported": True,
                    "prompt_sha256": "c" * 64,
                },
            }
        )
    )
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "inputs": {"selection": {"sha256": sha256_file(selection)}},
                "may_run_on_production_unlabeled_norms": True,
                "dev_gate": {"cleared": True},
                "prompt": {"rendered_prompt_sha256": "c" * 64},
                "order_policy": {
                    "retain_only_if": (
                        "both orders return CONFIRM_MATCH for the identical proposed "
                        "metric_id with high confidence"
                    )
                },
            }
        )
    )

    def verification(uid, metric, order, confidence):
        return {
            "norm_uid": uid,
            "primary_metric_id": metric,
            "primary_prompt_sha256": "a" * 64,
            "candidate_bank_source_sha256": "b" * 64,
            "prompt_sha256": "c" * 64,
            "model": "/model/snapshot",
            "order_mode": order,
            "decision": "CONFIRM_MATCH",
            "metric_id": metric,
            "confidence": confidence,
            "alternative_ids": ["other"],
            "parse_error": None,
        }

    original = _rows(
        tmp_path / "original.jsonl",
        [
            verification("keep", "m1", "original", "high"),
            verification("drop", "m2", "original", "high"),
        ],
    )
    hashed = _rows(
        tmp_path / "hashed.jsonl",
        [
            verification("keep", "m1", "hashed", "high"),
            verification("drop", "m2", "hashed", "medium"),
        ],
    )
    return primary, original, hashed, selection, policy


def test_combine_enforces_both_orders_high_confidence(tmp_path):
    primary, original, hashed, selection, policy = _fixture(tmp_path)
    output = tmp_path / "combined.jsonl"
    report = combine(
        primary_path=primary,
        original_path=original,
        hashed_path=hashed,
        selection_path=selection,
        policy_path=policy,
        output_path=output,
    )
    rows = {row["norm_uid"]: row for row in read_jsonl(output)}
    assert report["counts"]["accepted"] == 1
    assert rows["keep"]["decision"] == "CONFIRM_MATCH"
    assert rows["keep"]["strict_two_order_acceptance"] is True
    assert rows["drop"]["decision"] == "REJECT_MATCH"
    assert rows["drop"]["metric_id"] is None


def test_combine_rejects_incomplete_order_coverage(tmp_path):
    primary, original, hashed, selection, policy = _fixture(tmp_path)
    hashed.write_text(hashed.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="coverage mismatch"):
        combine(
            primary_path=primary,
            original_path=original,
            hashed_path=hashed,
            selection_path=selection,
            policy_path=policy,
            output_path=tmp_path / "combined.jsonl",
        )


def test_verifier_parse_failure_becomes_rejection_not_forced_match(tmp_path):
    primary, original, hashed, selection, policy = _fixture(tmp_path)
    rows = [json.loads(line) for line in hashed.read_text().splitlines()]
    rows[0].update(
        {
            "decision": "INVALID_OUTPUT",
            "metric_id": None,
            "confidence": "low",
            "parse_error": "no_json",
        }
    )
    _rows(hashed, rows)
    output = tmp_path / "combined.jsonl"
    combine(
        primary_path=primary,
        original_path=original,
        hashed_path=hashed,
        selection_path=selection,
        policy_path=policy,
        output_path=output,
    )
    combined = {row["norm_uid"]: row for row in read_jsonl(output)}
    assert combined["keep"]["decision"] == "REJECT_MATCH"
    assert combined["keep"]["strict_two_order_acceptance"] is False
