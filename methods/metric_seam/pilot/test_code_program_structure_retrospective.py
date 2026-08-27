from __future__ import annotations

import json
from pathlib import Path

import pytest

from methods.metric_seam.pilot.code_program_structure_retrospective import (
    DEPTH_RESULT,
    TASK_DIR,
    StructureError,
    _call_graph_descriptors,
    _local_call_graph,
    _strongly_connected_components,
    evaluate,
    render_report,
    source_descriptors,
)


def test_scc_condensation_handles_recursion_and_depth() -> None:
    graph = {
        "a": {"b"},
        "b": {"a", "c"},
        "c": {"d"},
        "d": set(),
        "self": {"self"},
    }
    components = _strongly_connected_components(graph)
    assert ("a", "b") in components
    assert ("self",) in components
    result = _call_graph_descriptors(graph)
    assert result["recursive_components"] == 2
    assert result["condensed_longest_path_edges"] == 2


def test_source_descriptors_measure_control_and_local_call_structure() -> None:
    source = '''
import ast

def helper(x):
    return x + 1

def score(x):
    if x:
        for value in x:
            helper(value)
    return 0
'''
    result = source_descriptors(source)
    assert result["function_defs"] == 2
    assert result["control_nodes"] == 2
    assert result["max_control_nesting"] == 2
    assert result["local_call_graph_edges"] == 1
    assert result["condensed_longest_path_edges"] == 1
    assert result["imported_top_level_modules"] == ["ast"]


def test_scope_qualified_call_graph_does_not_collapse_duplicate_nested_names() -> None:
    tree = __import__("ast").parse(
        """
def first():
    def walk():
        return 1
    return walk()

def second():
    def walk():
        return 2
    return walk()
"""
    )
    graph = _local_call_graph(tree)
    assert set(graph) == {"first", "first.walk", "second", "second.walk"}
    assert graph["first"] == {"first.walk"}
    assert graph["second"] == {"second.walk"}


def test_actual_full_family_structure_panel_is_complete() -> None:
    result = evaluate()
    assert result["scope"] == {
        "active_criteria": 18,
        "deep_programs": 18,
        "train_selected_shallow_programs": 15,
        "runtime_channel_both_arms": "code",
        "selection": "retrospective_full_family",
    }
    assert len(result["programs"]) == 33
    assert len(result["pairs"]) == 15
    assert sum(
        row["reconstruction_delta_spearman"] is not None for row in result["pairs"]
    ) == 13
    assert result["paired_summary"]["ast_nodes"]["pair_n"] == 15
    sensitivity = result["association_sensitivity"]
    assert sensitivity["all_defined"]["n"] == 13
    assert sensitivity["minimum_common_n_20"]["n"] == 8
    assert sensitivity["comparison_support_eligible"]["n"] == 4
    all_defined = sensitivity["all_defined"]["metrics"]["ast_nodes"]
    eligible = sensitivity["comparison_support_eligible"]["metrics"]["ast_nodes"]
    assert all_defined["spearman_structure_delta_vs_reconstruction_delta"] < 0
    assert eligible["spearman_structure_delta_vs_reconstruction_delta"] > 0
    assert result["upstream_reconstruction_family"][
        "multiplicity_controlled_improvements"
    ] == 0
    assert "0 BH-FDR improvements" in render_report(result)


def test_manifest_depth_cross_binding_fails_before_source_measurement(
    tmp_path: Path,
) -> None:
    manifest = json.loads(
        (TASK_DIR / "code_scores_cpu_manifest.json").read_text(encoding="utf-8")
    )
    manifest["output_sha256"] = "0" * 64
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "code_scores_cpu_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(StructureError, match="not bound to the depth result"):
        evaluate(task_dir=task_dir, depth_result=DEPTH_RESULT)
