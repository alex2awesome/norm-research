import json
from pathlib import Path

from methods.metric_implementer.experiments import mine_clusters


def test_mining_inputs_are_repo_anchored_outside_caller_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    repo_root = Path(mine_clusters.__file__).resolve().parents[3]
    assert Path(mine_clusters._HIER_DIR) == repo_root / "outputs" / "hierarchy"
    assert Path(mine_clusters._STRUCT_DIR) == repo_root / "outputs" / "analyses" / "structural_metrics"
    assert Path(mine_clusters._CANON) == repo_root / "outputs" / "analyses" / "canon_all_real_forms.jsonl"


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def test_complete_upper_inventory_appends_grandparents_without_reindexing_merged(
        monkeypatch, tmp_path):
    monkeypatch.setattr(mine_clusters, "_HIER_DIR", str(tmp_path))
    _write(tmp_path / "demo_general_r2_expanded.json", {
        "merged_groups": [
            {"merged_name": "m0", "merged_description": "first", "all_leaves": [
                {"name": "leaf zero", "key": "k0"}]},
            {"merged_name": "m1", "merged_description": "second", "all_leaves": [
                {"name": "leaf one", "key": "k1"}]},
        ],
        "grandparents": [{
            "grandparent_name": "g0",
            "grandparent_description": "umbrella",
            "total_leaf_rubrics": 4,
            "children": [{"name": "child", "description": "child definition", "n_leaves": 4}],
        }],
    })

    groups = mine_clusters.hierarchy_groups("demo", "general", "R2")
    assert [row["group_idx"] for row in groups] == [0, 1, 2]
    assert [row["source_kind"] for row in groups] == [
        "merged_group", "merged_group", "grandparent"]
    assert len({row["node_id"] for row in groups}) == 3
    # Historical accessors remain merged-only so existing sweep/checkpoint indices do not grow.
    assert [row["merged_name"] for row in mine_clusters.r2_groups(
        "demo", "general")] == ["m0", "m1"]


def test_refined_r1_inventory_flattens_both_materializations(monkeypatch, tmp_path):
    monkeypatch.setattr(mine_clusters, "_HIER_DIR", str(tmp_path))
    _write(tmp_path / "demo_general_r1_refined.json", {
        "parented_trees": [{
            "parent_name": "parent",
            "parent_description": "parent description",
            "total_leaf_rubrics": 2,
            "children": [{"rubrics": [
                {"name": "a", "key": "ka"}, {"name": "b", "key": "kb"}]}],
        }],
        "merged_trees": [{
            "merged_name": "merged",
            "merged_description": "merged description",
            "total_rubric_count": 1,
            "all_rubrics": [{"name": "c", "key": "kc"}],
        }],
    })

    groups = mine_clusters.hierarchy_groups("demo", "general", "R1")
    assert [row["merged_name"] for row in groups] == ["parent", "merged"]
    assert [[leaf["name"] for leaf in row["all_leaves"]] for row in groups] == [
        ["a", "b"], ["c"]]
    assert all(row["task"] == "demo" and row["level"] == "R1" for row in groups)


def test_node_identity_is_level_and_bucket_safe(monkeypatch, tmp_path):
    monkeypatch.setattr(mine_clusters, "_HIER_DIR", str(tmp_path))
    for bucket in ("general", "specific"):
        for level in ("r2", "r3"):
            _write(tmp_path / f"demo_{bucket}_{level}_expanded.json", {
                "merged_groups": [{
                    "merged_name": "same name",
                    "merged_description": "same description",
                    "all_leaves": [{"name": "leaf"}],
                }],
                "grandparents": [],
            })
    ids = {
        mine_clusters.hierarchy_groups("demo", bucket, level)[0]["node_id"]
        for bucket in ("general", "specific") for level in ("R2", "R3")
    }
    assert len(ids) == 4


def test_native_action_nodes_expose_overlap_dependency_blocks(monkeypatch, tmp_path):
    monkeypatch.setattr(mine_clusters, "_HIER_DIR", str(tmp_path))
    _write(tmp_path / "demo_general_r2_expanded.json", {
        "merged_groups": [{
            "merged_name": "canonical construct",
            "merged_description": "A sufficiently long canonical construct description for test",
            "source_r2_cluster_ids": [0, 1],
            "all_leaves": [{"name": "leaf a", "key": "ka"}],
        }, {
            "merged_name": "independent construct",
            "merged_description": "A sufficiently long independent construct description for test",
            "source_r2_cluster_ids": [4, 5],
            "all_leaves": [{"name": "leaf b", "key": "kb"}],
        }],
        "grandparents": [{
            "grandparent_name": "shared umbrella",
            "grandparent_description": "A sufficiently long shared umbrella description for test",
            "children": [
                {"r2_cluster_id": 1, "name": "one"},
                {"r2_cluster_id": 2, "name": "two"},
            ],
        }],
    })

    merged, independent, umbrella = mine_clusters.hierarchy_groups(
        "demo", "general", "R2")
    assert merged["dependency_component_id"] == umbrella["dependency_component_id"]
    assert merged["dependency_component_size"] == umbrella["dependency_component_size"] == 2
    assert merged["dependency_degree"] == umbrella["dependency_degree"] == 1
    assert independent["dependency_component_size"] == 1
    assert independent["dependency_degree"] == 0
    assert merged["source_assignment_multiplicity_max"] == 2
    assert merged["immediate_source_ids"] == [0, 1]
    assert "_leaf_support_ids" not in merged


def test_terminal_frontier_absorbs_laminar_actions_and_carries_uncovered_inputs(
        monkeypatch, tmp_path):
    monkeypatch.setattr(mine_clusters, "_HIER_DIR", str(tmp_path))
    inputs = [{
        "parent_name": f"input {index}",
        "parent_description": f"A long source description for recorded input number {index}",
        "children": [{
            "cluster_id": index,
            "medoid_name": f"child {index}",
            "rubrics": [{"name": f"leaf {index}", "key": f"k{index}"}],
        }],
    } for index in range(8)]
    _write(tmp_path / "demo_general_r1_refined.json", {
        "parented_trees": inputs,
        "merged_trees": [],
    })
    _write(tmp_path / "demo_general_r2_expanded.json", {
        "task": "demo",
        "bucket": "general",
        "n_r2_clusters_in": 8,
        "n_merged_groups": 2,
        "n_grandparents": 2,
        "merged_groups": [{
            "merged_name": "absorbed merge",
            "merged_description": "A long description for the absorbed merge action record",
            "source_r2_cluster_ids": [0, 1],
            "source_r2_cluster_names": ["input 0", "input 1"],
            "all_leaves": [{"name": "leaf 0", "key": "k0"}],
        }, {
            "merged_name": "retained merge",
            "merged_description": "A long description for the retained merge action record",
            "source_r2_cluster_ids": [3, 4],
            "source_r2_cluster_names": ["input 3", "input 4"],
            "all_leaves": [{"name": "leaf 3", "key": "k3"}],
        }],
        "grandparents": [{
            "grandparent_name": "retained parent",
            "grandparent_description": "A long description for the retained parent action record",
            "children": [
                {"r2_cluster_id": 0, "name": "input 0"},
                {"r2_cluster_id": 1, "name": "input 1"},
                {"r2_cluster_id": 2, "name": "input 2"},
            ],
        }, {
            "grandparent_name": "other parent",
            "grandparent_description": "A long description for another retained parent action",
            "children": [
                {"r2_cluster_id": 5, "name": "input 5"},
                {"r2_cluster_id": 6, "name": "input 6"},
            ],
        }],
    })

    rows = mine_clusters.hierarchy_terminal_frontier("demo", "general", "R2")
    audit = mine_clusters.hierarchy_terminal_frontier_audit("demo", "general", "R2")
    assert [row["frontier_role"] for row in rows] == [
        "retained_parent", "retained_parent", "retained_merge",
        "carried_uncovered_input",
    ]
    assert rows[0]["absorbed_descendant_action_ids"] == ["merged_group:0"]
    assert rows[-1]["merged_name"] == "input 7"
    assert [row["frontier_source_ids"] for row in rows] == [
        [0, 1, 2], [5, 6], [3, 4], [7],
    ]
    assert all(row["dependency_component_size"] == 1 for row in rows)
    assert audit["exact_once_input_coverage"] is True
    assert audit["n_frontier_nodes"] == 4
    assert audit["n_carried_inputs"] == 1
