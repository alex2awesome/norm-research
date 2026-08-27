import numpy as np

from scripts.tools.silver_match_v3.retrieve import (
    reciprocal_rank_union,
    stable_shard,
    top_indices,
    weighted_reciprocal_rank_union,
    uses_nemotron_query_format,
)


def test_nemotron_query_format_auto_detects_base_and_adapter():
    assert uses_nemotron_query_format("auto", "nvidia/llama-embed-nemotron-8b", None)
    assert uses_nemotron_query_format("auto", "some/base", "/tmp/adapter")
    assert not uses_nemotron_query_format("auto", "BAAI/bge-large-en-v1.5", None)
    assert uses_nemotron_query_format("nemotron", "anything", None)


def test_rrf_rewards_cross_lane_agreement():
    ranked = reciprocal_rank_union(([0, 1, 2], [2, 3, 0], [2, 4, 1]))
    assert ranked[0][0] == 2


def test_top_indices_tie_breaks_by_index():
    scores = np.array([[0.5, 0.8, 0.8, 0.2]])
    assert top_indices(scores, 3).tolist() == [[1, 2, 0]]


def test_stable_shard_is_deterministic():
    uid = "a" * 64
    assert stable_shard(uid, 7) == stable_shard(uid, 7)


def test_weighted_rrf_can_prefer_one_lane_and_reject_negative_weights():
    ranked = weighted_reciprocal_rank_union([([0, 1], 0.1), ([1, 0], 1.0)])
    assert ranked[0][0] == 1

    import pytest

    with pytest.raises(ValueError):
        weighted_reciprocal_rank_union([([0], -1.0)])
