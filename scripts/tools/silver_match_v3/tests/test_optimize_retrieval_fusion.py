import numpy as np

from scripts.tools.silver_match_v3.optimize_retrieval_fusion import (
    component_weights,
    score_components,
    select_weights,
    tensorize,
)


def test_component_weights_are_normalized():
    weights = component_weights(0.75, 2.0, 1.0, 0.5)
    assert np.isclose(weights.sum(), 1.0)
    assert weights[0] > weights[1]


def test_dev_selects_evidence_view_without_using_test():
    # Two metrics, six component ranks. Evidence dense ranks are correct on
    # dev; statement dense ranks are wrong. Test is deliberately the reverse
    # and therefore must not control selection.
    tensor = np.asarray(
        [
            [[1, 2, 1, 2, 1, 2], [2, 1, 2, 1, 2, 1]],
            [[2, 1, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2]],
        ],
        dtype=float,
    )
    best, _ = select_weights(
        tensor,
        np.asarray([0, 0]),
        ["a", "a"],
        ["dev", "test"],
        rank_constant=60,
        evidence_grid=(0.0, 1.0),
        modality_grid=(0.0, 1.0),
    )
    assert best["evidence_share"] == 1.0
    scores = score_components(
        tensor[:1],
        np.asarray([best["component_weights"][key] for key in (
            "dense_rank", "dense_statement_rank", "word_rank",
            "word_statement_rank", "char_rank", "char_statement_rank",
        )]),
        60,
    )
    assert int(np.argmax(scores[0])) == 0


def test_selector_can_freeze_production_depth_objective():
    tensor = np.asarray(
        [
            [[1, 2, 3, 1, 2, 3], [2, 1, 1, 2, 1, 1]],
            [[2, 1, 2, 2, 1, 2], [1, 2, 1, 1, 2, 1]],
        ],
        dtype=float,
    )
    best, trials = select_weights(
        tensor,
        np.asarray([0, 1]),
        ["a", "b"],
        ["dev", "dev"],
        rank_constant=60,
        evidence_grid=(0.5,),
        modality_grid=(0.0, 1.0),
        primary_k=50,
    )
    assert best in trials
    assert best["dev"]["recall_at_50"] == 1.0


def test_tensorize_uses_bank_metric_index_not_lexicographic_id():
    labels = [{"norm_uid": "u", "metric_id": "a2"}]
    candidates = {
        "u": {
            "candidates": [
                {"metric_id": "a10", "metric_index": 10, **{key: 2 for key in (
                    "dense_rank", "dense_statement_rank", "word_rank",
                    "word_statement_rank", "char_rank", "char_statement_rank")}},
                {"metric_id": "a2", "metric_index": 2, **{key: 1 for key in (
                    "dense_rank", "dense_statement_rank", "word_rank",
                    "word_statement_rank", "char_rank", "char_statement_rank")}},
            ]
        }
    }
    metric_ids, _, gold = tensorize(labels, candidates)
    assert metric_ids == ["a2", "a10"]
    assert gold.tolist() == [0]
