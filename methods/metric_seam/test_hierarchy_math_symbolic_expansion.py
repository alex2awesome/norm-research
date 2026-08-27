from __future__ import annotations

import json
from pathlib import Path

import pytest

from methods.metric_seam.adjudicate_math_symbolic_capability_expansion import (
    SymbolicExpansionAdjudicationError,
    build_symbolic_expansion_adjudication,
)
from methods.metric_seam.hierarchy_math_symbolic_capability_mapper import (
    RELATION_ID,
    SymbolicCapabilityMapError,
    build_symbolic_capability_map,
    inspect_symbolic_capability,
)
from methods.metric_seam.hierarchy_math_symbolic_expansion_prevalence import (
    EXPANSION_KEY,
    SymbolicExpansionPrevalenceError,
    build_symbolic_expansion_prevalence,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"
PANEL = BASE / "panel_v3.json"
CANONICAL = BASE / "math_stackexchange_construct_fidelity_merged_v1.json"
V1 = ROOT / "methods/metric_seam/hybrids/ops_symbolic_steps_v1.py"
V2 = ROOT / "methods/metric_seam/hybrids/ops_symbolic_steps_v2.py"
V1_TEST = ROOT / "methods/metric_seam/hybrids/test_ops_symbolic_steps_v1.py"
V2_TEST = ROOT / "methods/metric_seam/hybrids/test_ops_symbolic_steps_v2.py"
MAP_ARTIFACT = BASE / "math_stackexchange_symbolic_capability_source_map_v1.json"
ADJUDICATION_ARTIFACT = (
    BASE / "math_stackexchange_symbolic_capability_construct_fidelity_v1.json"
)
PREVALENCE_ARTIFACT = (
    BASE / "math_stackexchange_symbolic_capability_expansion_prevalence_v1.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _capability() -> dict:
    return inspect_symbolic_capability(
        V1.read_text(encoding="utf-8"),
        V2.read_text(encoding="utf-8"),
        v1_path=str(V1),
        v2_path=str(V2),
        v1_test_path=str(V1_TEST),
        v2_test_path=str(V2_TEST),
        v1_tests_present=V1_TEST.is_file(),
        v2_tests_present=V2_TEST.is_file(),
    )


def _map() -> dict:
    return build_symbolic_capability_map(_load(PANEL), _capability())


def _adjudication() -> dict:
    return build_symbolic_expansion_adjudication(
        _load(PANEL), _map(), _load(CANONICAL)
    )


def test_static_capability_inspection_prefers_v2_and_retains_v1() -> None:
    capability = _capability()

    assert capability["relation_id"] == RELATION_ID
    assert capability["selection_provenance"].startswith("manually_designed_pipeline_seed")
    assert capability["v2_explicitly_imports_v1"] is True
    assert capability["frozen_matched_relation_depth"] == 3
    receipts = capability["isolation_and_test_receipts"]
    assert receipts["isolation_wrapper_present_by_static_source"] is True
    assert receipts["isolation_wrapper_executed_for_this_sensitivity"] is False
    assert receipts["isolation_does_not_increment_matched_relation_depth"] is True
    assert receipts["v1_tests_present"] is True
    assert receipts["v2_tests_present"] is True
    assert receipts["capability_tests_executed_for_this_sensitivity"] is False


def test_static_capability_inspection_rejects_relation_or_provenance_drift() -> None:
    v1 = V1.read_text(encoding="utf-8")
    v2 = V2.read_text(encoding="utf-8")
    with pytest.raises(SymbolicCapabilityMapError, match="relation identity drifted"):
        inspect_symbolic_capability(
            v1,
            v2.replace(RELATION_ID, "generic_proof_correctness"),
        )
    with pytest.raises(SymbolicCapabilityMapError, match="v1 provenance"):
        inspect_symbolic_capability(
            v1,
            v2.replace(
                "from .ops_symbolic_steps_v1 import (",
                "from .unrelated_module import (",
            ),
        )


def test_source_only_mapper_retrieves_broad_candidates_without_fidelity_credit() -> None:
    result = _map()

    assert result["summary"] == {
        "n_cells": 90,
        "n_retrieved_candidates": 15,
        "retrieved_by_level": {"R1": 4, "R2": 6, "R3": 5},
        "n_construct_fidelity_decisions": 0,
    }
    assert result["programs_imported_or_executed"] is False
    assert result["items_or_articles_loaded"] is False
    assert result["certificate_counts_loaded"] is False
    assert result["prompt_outputs_loaded"] is False
    assert result["reference_values_loaded"] is False
    assert result["outcome_labels_loaded"] is False
    assert result["correlations_or_reconstruction_loaded"] is False
    assert result["models_apis_or_gpus_used"] is False
    assert all(
        row["candidate_capability_id"] is None
        for row in result["rows"]
        if not row["retrieved_candidate"]
    )


def test_source_only_mapper_is_text_deterministic_and_rejects_panel_drift() -> None:
    first = _map()
    second = _map()
    assert first == second

    panel = _load(PANEL)
    removed = next(
        cell for cell in panel["cells"] if cell["task"] == "math-stackexchange"
    )
    panel["cells"].remove(removed)
    with pytest.raises(SymbolicCapabilityMapError, match="30 R1/R2/R3"):
        build_symbolic_capability_map(panel, _capability())


def test_independent_adjudication_is_narrow_and_additive() -> None:
    result = _adjudication()

    assert result["summary"] == {
        "n_cells": 90,
        "n_retrieved_candidates": 15,
        "n_relation_local_static_matches": 7,
        "n_retrieved_relation_mismatches": 8,
        "n_newly_covered_cells": 5,
        "n_existing_cells_adding_formal_symbolic_relation": 2,
        "canonical_relation_local_cells_unchanged": 33,
        "additive_sensitivity_union_cells": 38,
        "n_whole_construct_exact": 0,
        "accepted_by_level": {"R1": 1, "R2": 3, "R3": 3},
        "newly_covered_by_level": {"R2": 2, "R3": 3},
    }
    assert result["canonical_artifact_modified"] is False
    assert result["programs_or_items_executed"] is False
    assert result["models_apis_or_gpus_used"] is False
    assert all(not row["whole_construct_exact"] for row in result["rows"])


def test_every_retrieved_row_has_a_five_dimension_decision() -> None:
    result = _adjudication()
    retrieved = [row for row in result["rows"] if row["retrieved_candidate"]]

    assert len(retrieved) == 15
    assert all(
        set(row["dimension_audit"])
        == {"object", "relation", "polarity", "applicability", "aggregation"}
        for row in retrieved
    )
    accepted = [row for row in retrieved if row["symbolic_relation_local_static_fidelity"]]
    assert all(row["matched_relation_depth"] == 3 for row in accepted)
    assert all(
        row["dimension_audit"]["aggregation"]["status"] == "mismatch_disclosed"
        for row in accepted
    )
    assert all(row["verdict"] != "exact" for row in accepted)


def test_new_and_existing_formal_symbolic_cells_are_explicit() -> None:
    result = _adjudication()
    new = result["newly_covered_cells"]
    existing = result["existing_cells_adding_formal_symbolic_relation"]

    assert [(row["level"], row["metric_name"]) for row in new] == [
        ("R2", "Logical completeness (no gaps)"),
        ("R2", "Stepwise logical validity and explicit justification"),
        ("R3", "Logical rigor, validity, and completeness"),
        ("R3", "Logical correctness and completeness (no gaps)"),
        ("R3", "Stepwise logical validity and explicit justification"),
    ]
    assert [(row["level"], row["metric_name"]) for row in existing] == [
        ("R1", "Formal correctness and checkability"),
        ("R2", "Logical correctness, rigor, and completeness"),
    ]


def test_adjudication_rejects_retrieval_or_canonical_boundary_drift() -> None:
    retrieval = _map()
    candidate = next(row for row in retrieval["rows"] if row["retrieved_candidate"])
    candidate["retrieved_candidate"] = False
    candidate["candidate_capability_id"] = None
    with pytest.raises(SymbolicExpansionAdjudicationError, match="retrieval set differs"):
        build_symbolic_expansion_adjudication(
            _load(PANEL), retrieval, _load(CANONICAL)
        )

    canonical = _load(CANONICAL)
    canonical["program_outputs_loaded"] = True
    with pytest.raises(SymbolicExpansionAdjudicationError, match="forbidden boundary"):
        build_symbolic_expansion_adjudication(_load(PANEL), _map(), canonical)


def test_prevalence_reports_balanced_and_weighted_sensitivity() -> None:
    result = build_symbolic_expansion_prevalence(_load(PANEL), _adjudication())
    pooled = result["pooled_eligible_action_nodes"]
    balanced = pooled["balanced_panel"]
    weighted = pooled[EXPANSION_KEY]

    assert result["sampling_frame"]["n_eligible_action_node_records"] == 1185
    assert balanced["canonical_relation_local_unchanged"]["rate"] == 0.366667
    assert balanced["formal_symbolic_relation_local"]["rate"] == 0.077778
    assert balanced["newly_covered_by_formal_symbolic_relation"]["rate"] == 0.055556
    assert balanced["existing_cell_adds_formal_symbolic_relation"]["rate"] == 0.022222
    assert balanced["additive_sensitivity_union_relation_local"]["rate"] == 0.422222
    assert balanced["whole_construct_exact"]["rate"] == 0.0
    assert weighted["canonical_relation_local_unchanged"]["rate"] == 0.361266
    assert weighted["formal_symbolic_relation_local"]["rate"] == 0.056315
    assert weighted["newly_covered_by_formal_symbolic_relation"]["rate"] == 0.014965
    assert weighted["existing_cell_adds_formal_symbolic_relation"]["rate"] == 0.04135
    assert weighted["additive_sensitivity_union_relation_local"]["rate"] == 0.376231
    assert weighted["whole_construct_exact"]["rate"] == 0.0


def test_prevalence_preserves_depth_and_runtime_separation() -> None:
    result = build_symbolic_expansion_prevalence(_load(PANEL), _adjudication())
    receipt = result["relation_depth_receipt"]

    assert receipt["depth"] == 3
    assert receipt["formal_symbolic_matched_cells"] == 7
    assert receipt["newly_covered_at_depth3"] == 5
    assert receipt["already_covered_adding_depth3_relation"] == 2
    assert receipt["isolation_or_test_execution_adds_depth"] is False
    runtime = receipt["capability_runtime_receipt"]
    assert runtime["isolation_wrapper_executed_for_this_sensitivity"] is False
    assert runtime["capability_tests_executed_for_this_sensitivity"] is False
    assert result["program_or_item_execution_emitted"] is False
    assert result["prompt_reference_outcome_or_reconstruction_stages_emitted"] is False
    assert result["canonical_artifact_modified"] is False


def test_prevalence_by_level_is_descriptive_not_a_trend() -> None:
    result = build_symbolic_expansion_prevalence(_load(PANEL), _adjudication())
    by_level = result["by_level"]

    assert {
        level: int(
            by_level[level]["balanced_panel"]["formal_symbolic_relation_local"][
                "expanded_positive_nodes"
            ]
        )
        for level in ("R1", "R2", "R3")
    } == {"R1": 1, "R2": 3, "R3": 3}
    assert any("hierarchy" not in limit.casefold() for limit in result["claim_limits"])
    assert any("codability" in limit.casefold() for limit in result["claim_limits"])


def test_prevalence_rejects_missing_dimension_or_weight_drift() -> None:
    adjudication = _adjudication()
    accepted = next(
        row
        for row in adjudication["rows"]
        if row["symbolic_relation_local_static_fidelity"]
    )
    accepted["dimension_audit"].pop("aggregation")
    with pytest.raises(SymbolicExpansionPrevalenceError, match="five-dimension"):
        build_symbolic_expansion_prevalence(_load(PANEL), adjudication)

    panel = _load(PANEL)
    math_cell = next(cell for cell in panel["cells"] if cell["task"] == "math-stackexchange")
    math_cell["design_weight"] = float(math_cell["design_weight"]) + 1
    with pytest.raises(SymbolicExpansionPrevalenceError, match="design weight drift"):
        build_symbolic_expansion_prevalence(panel, _adjudication())


def test_generated_artifacts_match_the_instrument() -> None:
    if not (MAP_ARTIFACT.exists() and ADJUDICATION_ARTIFACT.exists() and PREVALENCE_ARTIFACT.exists()):
        pytest.skip("generated additive artifacts have not been materialized yet")
    assert _load(MAP_ARTIFACT)["summary"] == _map()["summary"]
    assert _load(ADJUDICATION_ARTIFACT)["summary"] == _adjudication()["summary"]
    generated = _load(PREVALENCE_ARTIFACT)
    recomputed = build_symbolic_expansion_prevalence(_load(PANEL), _adjudication())
    assert generated["pooled_eligible_action_nodes"] == recomputed[
        "pooled_eligible_action_nodes"
    ]
