import json
from pathlib import Path

from scripts.tools.silver_match_v3.build_relations import (
    Thresholds,
    build_relations,
)


def _write_json(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _hierarchies(root: Path, task: str = "peer-review"):
    groups = [
        {"merged_name": f"m{i}", "merged_description": f"d{i}", "all_leaves": []}
        for i in range(5)
    ]
    for metric, keys in enumerate((("k0a", "k0b"), ("k1a", "k1b"), ("k2",), ("k3",), ("k4",))):
        groups[metric]["all_leaves"] = [{"key": key, "name": key} for key in keys]
    _write_json(
        root / f"{task}_general_r2_expanded.json",
        {"merged_groups": groups},
    )
    _write_json(
        root / f"{task}_general_r3_expanded.json",
        {
            "merged_groups": [
                {
                    "merged_name": "supported",
                    "merged_description": "same",
                    "source_r2_cluster_ids": [0, 1],
                },
                {
                    "merged_name": "unsupported transitive group",
                    "merged_description": "same",
                    "source_r2_cluster_ids": [2, 3, 4],
                },
            ],
            "grandparents": [
                {
                    "grandparent_name": "broad family",
                    "grandparent_description": "related but distinct",
                    "children": [{"r2_cluster_id": 1}, {"r2_cluster_id": 2}],
                }
            ],
        },
    )


def test_equivalence_requires_pairwise_support_and_family_is_broader(tmp_path):
    _hierarchies(tmp_path)
    labels = tmp_path / "pairs.jsonl"
    _write_jsonl(
        labels,
        [
            {"task": "peer-review", "key_a": "k0a", "key_b": "k1a", "score": 2},
            {"task": "peer-review", "key_a": "k0a", "key_b": "k1b", "score": 2},
            {"task": "peer-review", "key_a": "k0b", "key_b": "k1a", "score": 2},
            # Only one of the three pairs in [2,3,4] is supported.  The whole
            # group must not receive transitive equivalence credit.
            {"task": "peer-review", "key_a": "k2", "key_b": "k3", "score": 2},
            {"task": "peer-review", "key_a": "k2", "key_b": "k3", "score": 2},
            {"task": "peer-review", "key_a": "k2", "key_b": "k3", "score": 2},
        ],
    )
    result = build_relations(
        tmp_path,
        labels,
        ["peer-review"],
        Thresholds(min_same_pairs=3, min_same_rate=0.8, max_unrelated_pairs=0),
    )
    task = result["tasks"]["peer-review"]
    assert [group["metric_ids"] for group in task["equivalence_groups"]] == [["a0", "a1"]]
    assert task["metric_relations"]["a0"]["equivalent_metric_ids"] == ["a0", "a1"]
    assert task["metric_relations"]["a2"]["equivalent_metric_ids"] == ["a2"]
    assert "a2" in task["metric_relations"]["a1"]["family_metric_ids"]
    assert "a1" not in task["metric_relations"]["a2"]["equivalent_metric_ids"]


def test_unrelated_label_vetoes_equivalence(tmp_path):
    _hierarchies(tmp_path)
    labels = tmp_path / "pairs.jsonl"
    _write_jsonl(
        labels,
        [
            {"task": "peer-review", "key_a": "k0a", "key_b": "k1a", "score": 2},
            {"task": "peer-review", "key_a": "k0a", "key_b": "k1b", "score": 2},
            {"task": "peer-review", "key_a": "k0b", "key_b": "k1a", "score": 2},
            {"task": "peer-review", "key_a": "k0b", "key_b": "k1b", "score": 0},
        ],
    )
    result = build_relations(tmp_path, labels, ["peer-review"], Thresholds())
    task = result["tasks"]["peer-review"]
    assert task["equivalence_groups"] == []
    audit = task["r3_merge_audit"][0]["pair_evidence"][0]
    assert audit["unrelated"] == 1
    assert "unrelated_contradiction" in audit["reasons"]


def test_duplicate_raw_pair_disagreement_uses_conservative_minimum(tmp_path):
    _hierarchies(tmp_path)
    labels = tmp_path / "pairs.jsonl"
    rows = [
        {"task": "peer-review", "key_a": "k0a", "key_b": "k1a", "score": 2},
        {"task": "peer-review", "key_a": "k0a", "key_b": "k1a", "score": 0},
        {"task": "peer-review", "key_a": "k0a", "key_b": "k1b", "score": 2},
        {"task": "peer-review", "key_a": "k0b", "key_b": "k1a", "score": 2},
        {"task": "peer-review", "key_a": "k0b", "key_b": "k1b", "score": 2},
    ]
    _write_jsonl(labels, rows)
    result = build_relations(tmp_path, labels, ["peer-review"], Thresholds())
    evidence = result["tasks"]["peer-review"]["r3_merge_audit"][0]["pair_evidence"][0]
    assert evidence["conflicting_raw_pairs"] == 1
    assert evidence["unrelated"] == 1
    assert not evidence["qualified"]
