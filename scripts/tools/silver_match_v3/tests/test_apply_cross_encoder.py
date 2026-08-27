import pytest

from scripts.tools.silver_match_v3.apply_cross_encoder import rerank_candidates


def test_rerank_candidates_emits_only_provisional_match():
    rows, proposal = rerank_candidates(
        [{"metric_id": "a1", "rank": 1}, {"metric_id": "a2", "rank": 2}],
        [0.2, 0.9],
        score_threshold=0.8,
        margin_threshold=0.2,
        output_k=2,
    )
    assert [row["metric_id"] for row in rows] == ["a2", "a1"]
    assert [row["ce_rank"] for row in rows] == [1, 2]
    assert proposal["decision"] == "PROVISIONAL_MATCH"
    assert proposal["metric_id"] == "a2"


def test_rerank_candidates_abstains_on_small_margin():
    _, proposal = rerank_candidates(
        [{"metric_id": "a1"}, {"metric_id": "a2"}],
        [0.91, 0.90],
        score_threshold=0.8,
        margin_threshold=0.05,
        output_k=2,
    )
    assert proposal["decision"] == "PROVISIONAL_ABSTAIN"
    assert proposal["metric_id"] is None
    assert proposal["top_metric_id"] == "a1"


def test_rerank_candidates_uses_deterministic_original_order_for_ties():
    rows, _ = rerank_candidates(
        [{"metric_id": "a2"}, {"metric_id": "a1"}],
        [0.5, 0.5],
        score_threshold=1.0,
        margin_threshold=0.0,
        output_k=2,
    )
    assert [row["metric_id"] for row in rows] == ["a2", "a1"]


def test_rerank_candidates_rejects_invalid_slate():
    with pytest.raises(ValueError, match="empty candidate"):
        rerank_candidates([], [], score_threshold=0.5, margin_threshold=0.1, output_k=2)
    with pytest.raises(ValueError, match="length mismatch"):
        rerank_candidates(
            [{"metric_id": "a1"}],
            [],
            score_threshold=0.5,
            margin_threshold=0.1,
            output_k=2,
        )
