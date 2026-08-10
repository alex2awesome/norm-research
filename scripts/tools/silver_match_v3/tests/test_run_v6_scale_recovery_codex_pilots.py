from __future__ import annotations

import copy

import pytest

from scripts.tools.silver_match_v3.run_v6_scale_recovery_codex_pilots import (
    _prompt,
    validate_payload,
)


def _exact(uid: str) -> dict[str, object]:
    return {
        "norm_uid": uid,
        "decision": "EXACT",
        "primary_metric_id": "a0",
        "pair_labels": [
            {"metric_id": "a0", "relation": "EXACT"},
            {"metric_id": "a1", "relation": "FAMILY"},
            {"metric_id": "a2", "relation": "REJECT"},
            {"metric_id": "a3", "relation": "REJECT"},
        ],
        "confidence": "high",
        "reason": "The criterion directly matches a0 and defeats its nearest sibling.",
    }


def test_valid_payload_requires_exact_uid_coverage_and_semantic_contract() -> None:
    uids = ["0" * 64, "1" * 64]
    payload = {
        "task": "peer-review",
        "chunk_id": "part-0000",
        "labels": [_exact(uid) for uid in reversed(uids)],
    }
    summary = validate_payload(
        payload,
        task="peer-review",
        chunk_id="part-0000",
        expected_uids=uids,
        metric_ids={"a0", "a1", "a2", "a3"},
    )
    assert summary == {
        "row_count": 2,
        "decision_counts": {"EXACT": 2},
        "pair_relation_counts": {"EXACT": 2, "FAMILY": 2, "REJECT": 4},
    }


@pytest.mark.parametrize("fault", ["missing", "duplicate", "unknown_metric", "bad_envelope"])
def test_payload_validation_fails_closed(fault: str) -> None:
    uids = ["0" * 64, "1" * 64]
    payload = {
        "task": "peer-review",
        "chunk_id": "part-0000",
        "labels": [_exact(uid) for uid in uids],
    }
    if fault == "missing":
        payload["labels"].pop()
    elif fault == "duplicate":
        payload["labels"][1]["norm_uid"] = uids[0]
    elif fault == "unknown_metric":
        payload["labels"][0]["pair_labels"][2]["metric_id"] = "a99"
    else:
        payload["unexpected"] = True
    with pytest.raises(ValueError):
        validate_payload(
            copy.deepcopy(payload),
            task="peer-review",
            chunk_id="part-0000",
            expected_uids=uids,
            metric_ids={"a0", "a1", "a2", "a3"},
        )


def test_prompt_requires_complete_bank_and_forbids_external_discovery() -> None:
    prompt = _prompt("peer-review", "part-0000", 20)
    assert "complete current peer-review metric bank" in prompt
    assert "Consider every bank metric" in prompt
    assert "Do not inspect any path outside" in prompt
    assert "PRIVATE_SELECTION_LEDGER" not in prompt
    assert "part-0000.jsonl" in prompt

