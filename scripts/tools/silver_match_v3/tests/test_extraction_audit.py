import pytest

from scripts.tools.silver_match_v3.sample_extraction_audit import (
    audit_rows,
    deterministic_sample,
)


def test_audit_rows_align_and_stratify():
    deploy = [
        {"unit_id": "zero", "signals": []},
        {
            "unit_id": "one",
            "signals": [
                {"signal_text": "explain the rule", "passage_text": "please explain"},
                {"signal_text": "mere fact", "passage_text": "a fact"},
            ],
        },
    ]
    score = [
        {
            "unit_id": "one",
            "scored": [
                {"signal_text": "explain the rule", "faithful": 1, "valid": 1},
                {"signal_text": "mere fact", "faithful": 1, "valid": 0},
            ],
        }
    ]
    rows = audit_rows(deploy, score, {"one": "please explain this; a fact follows"})
    sample = deterministic_sample(rows, 1, 1)
    assert len(sample) == 2
    assert {row["judge_accepted"] for row in sample} == {True, False}
    assert "please explain" in rows[0]["source_context"]


def test_audit_rows_reject_partial_score():
    with pytest.raises(ValueError, match="incomplete positional pair"):
        audit_rows([{"unit_id": "one", "signals": [{"signal_text": "x"}]}], [], {})
