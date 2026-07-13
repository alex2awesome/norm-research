import json

import pytest

from methods.metric_implementer.experiments.complete_r3_census import (
    build_complete_r3_partition,
    materialize_complete_r3_partition,
)


def _parent(index):
    return {
        "parent_name": f"R2 metric {index}",
        "parent_description": f"Description for R2 metric {index}.",
        "children": [{"rubrics": [{"key": f"k{index}", "name": f"leaf {index}"}]}],
    }


def _inputs():
    r2 = {
        "task": "demo",
        "bucket": "general",
        "parented_trees": [_parent(index) for index in range(4)],
    }
    r3 = {
        "task": "demo",
        "bucket": "general",
        "round": 3,
        "model": "test-model",
        "n_r2_clusters_in": 4,
        "merged_groups": [{
            "merged_name": "Merged zero and two",
            "merged_description": "A real merge.",
            "source_r2_cluster_ids": [0, 2],
            "source_r2_cluster_names": ["R2 metric 0", "R2 metric 2"],
            "total_leaf_rubrics": 2,
            "all_leaves": [{"key": "k0"}, {"key": "k2"}],
        }],
        "grandparents": [{"grandparent_name": "Auxiliary parent"}],
    }
    return r2, r3


def test_complete_partition_preserves_merges_and_appends_singletons(tmp_path):
    r2, r3 = _inputs()
    result = build_complete_r3_partition(
        r2, r3, r2_input_sha256="a" * 64, r3_expanded_sha256="b" * 64)

    assert result["n_merged_groups"] == 3
    assert result["n_multi_r2_merges"] == 1
    assert result["n_singleton_carry_forwards"] == 2
    assert [row["source_r2_cluster_ids"] for row in result["merged_groups"]] == [
        [0, 2], [1], [3]
    ]
    assert [row["r3_membership_type"] for row in result["merged_groups"]] == [
        "multi_r2_merge", "singleton_carry_forward", "singleton_carry_forward"
    ]
    assert result["merged_groups"][0]["source_r3_merged_group_index"] == 0
    assert result["grandparents_role"].startswith("higher_order_auxiliary")

    r2_path = tmp_path / "r2.json"
    r3_path = tmp_path / "r3.json"
    out_path = tmp_path / "complete.json"
    r2_path.write_text(json.dumps(r2))
    r3_path.write_text(json.dumps(r3))
    first = materialize_complete_r3_partition(r2_path, r3_path, out_path)
    first_bytes = out_path.read_bytes()
    second = materialize_complete_r3_partition(r2_path, r3_path, out_path)
    assert first == second
    assert out_path.read_bytes() == first_bytes


def test_complete_partition_rejects_overlapping_merges():
    r2, r3 = _inputs()
    r3["merged_groups"].append({
        "merged_name": "Overlapping merge",
        "merged_description": "Invalid.",
        "source_r2_cluster_ids": [1, 2],
    })
    with pytest.raises(ValueError, match="overlap"):
        build_complete_r3_partition(
            r2, r3, r2_input_sha256="a" * 64, r3_expanded_sha256="b" * 64)


def test_complete_partition_rejects_missing_source_rows():
    r2, r3 = _inputs()
    r2["parented_trees"].pop()
    with pytest.raises(ValueError, match="declares 4 R2 inputs"):
        build_complete_r3_partition(
            r2, r3, r2_input_sha256="a" * 64, r3_expanded_sha256="b" * 64)
