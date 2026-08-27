import pytest

from scripts.tools.silver_match_v3.combine_cross_encoder_proposals import combine_rows


def row(metric, decision="PROVISIONAL_MATCH"):
    return {
        "norm_uid": "u1",
        "task": "t",
        "corpus": "c",
        "row": 0,
        "bank_source_sha256": "bank",
        "candidates": [{"metric_id": "a1"}, {"metric_id": "a2"}],
        "ce_proposal": {"decision": decision, "metric_id": metric},
    }


def test_requires_two_exact_gated_metric_votes():
    result = combine_rows([row("a1"), row("a1")])
    assert result["ce_consensus"] == {
        "decision": "PROVISIONAL_MATCH",
        "metric_id": "a1",
        "agreement_count": 2,
        "variant_proposals": [
            {"decision": "PROVISIONAL_MATCH", "metric_id": "a1"},
            {"decision": "PROVISIONAL_MATCH", "metric_id": "a1"},
        ],
    }


def test_disagreement_or_single_gate_abstains():
    assert combine_rows([row("a1"), row("a2")])["ce_consensus"]["decision"] == "PROVISIONAL_ABSTAIN"
    assert (
        combine_rows([row("a1"), row(None, "PROVISIONAL_ABSTAIN")])["ce_consensus"]["decision"]
        == "PROVISIONAL_ABSTAIN"
    )


def test_rejects_different_candidate_universes():
    second = row("a1")
    second["candidates"] = [{"metric_id": "a1"}, {"metric_id": "a3"}]
    with pytest.raises(ValueError, match="different candidate universes"):
        combine_rows([row("a1"), second])
