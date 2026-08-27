import pytest

from scripts.tools.silver_match_v3.validate_manual_labels import validate_labels


ITEM = {
    "schema_version": "3",
    "norm_uid": "u",
    "corpus": "c",
    "task": "t",
    "row": 1,
    "split_group": "g",
    "split": "test",
}


def test_manual_match_is_bank_constrained_and_gets_rank():
    rows = validate_labels(
        [
            {
                "norm_uid": "u",
                "decision": "MATCH",
                "metric_id": "a1",
                "confidence": "high",
                "reason": "directly stated",
            }
        ],
        {"u": ITEM},
        {"t": {"a1"}},
        annotator="agent",
        candidate_ranks={"u": {"a1": 7}},
    )
    assert rows[0]["retrieved_rank"] == 7


def test_manual_abstention_cannot_carry_metric():
    with pytest.raises(ValueError, match="abstention"):
        validate_labels(
            [
                {
                    "norm_uid": "u",
                    "decision": "NO_CANDIDATE_FITS",
                    "metric_id": "a1",
                    "confidence": "high",
                    "reason": "gap",
                }
            ],
            {"u": ITEM},
            {"t": {"a1"}},
            annotator="agent",
        )
