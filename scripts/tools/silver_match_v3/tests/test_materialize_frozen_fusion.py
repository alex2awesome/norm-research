import pytest

from scripts.tools.silver_match_v3.materialize_frozen_fusion import rerank_candidates


def row(metric_id, index, dense, char):
    return {
        "metric_id": metric_id,
        "metric_index": index,
        "dense_rank": dense,
        "dense_statement_rank": dense,
        "word_rank": dense,
        "word_statement_rank": dense,
        "char_rank": char,
        "char_statement_rank": char,
    }


def test_rerank_uses_frozen_weights_and_metric_index_ties():
    weights = {
        "dense_rank": 2 / 3,
        "dense_statement_rank": 0,
        "word_rank": 0,
        "word_statement_rank": 0,
        "char_rank": 1 / 3,
        "char_statement_rank": 0,
    }
    values = [row("m2", 2, 2, 2), row("m1", 1, 1, 8), row("m0", 0, 2, 2)]
    ranked = rerank_candidates(values, weights, 60, 2)
    assert [value["metric_id"] for value in ranked] == ["m0", "m2"]
    assert [value["rank"] for value in ranked] == [1, 2]


def test_positive_weight_missing_rank_is_rejected():
    weights = {
        "dense_rank": 1,
        "dense_statement_rank": 0,
        "word_rank": 0,
        "word_statement_rank": 0,
        "char_rank": 0,
        "char_statement_rank": 0,
    }
    value = row("m0", 0, 1, 1)
    del value["dense_rank"]
    with pytest.raises(ValueError, match="missing dense_rank"):
        rerank_candidates([value], weights, 60, 1)
