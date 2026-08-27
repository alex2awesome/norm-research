from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from methods.metric_seam.hierarchy_patent_claim_structure_gate import (
    DEFAULT_EXECUTION,
    DEFAULT_FIDELITY,
    DEFAULT_OUTPUT,
    EXECUTION_BASENAME,
    FIDELITY_BASENAME,
    R3_CATEGORY,
    SCHEMA,
    SELECTED_IDS,
    STATIC_ONLY_IDS,
    STATUS,
    PatentTrainGateError,
    _write_new,
    build_patent_train_gate,
)


ROOT = Path(__file__).resolve().parents[2]


def _load(relative: Path):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _inputs():
    return _load(DEFAULT_EXECUTION), _load(DEFAULT_FIDELITY)


def _build(execution=None, fidelity=None):
    real_execution, real_fidelity = _inputs()
    return build_patent_train_gate(
        real_execution if execution is None else execution,
        real_fidelity if fidelity is None else fidelity,
        execution_source=DEFAULT_EXECUTION,
        fidelity_source=DEFAULT_FIDELITY,
    )


def test_real_gate_freezes_exactly_five_operational_and_three_static_cells():
    gate = _build()
    assert gate["schema"] == SCHEMA
    assert gate["status"] == STATUS
    assert gate["summary"] == {
        "n_compiler_train_rows": 150,
        "n_conservative_static_fidelity_cells": 8,
        "n_selected_operational_cells": 5,
        "n_static_only_constant_cells": 3,
        "selected_cells_by_level": {"R2": 1, "R3": 4},
        "selected_cells_by_maximum_depth": {"1": 4, "2": 1},
        "n_whole_construct_cells": 0,
        "prompt_scored_cells": 0,
        "reconstruction_evaluable_cells": 0,
        "isomorphism_evaluable_cells": 0,
    }
    assert tuple(row["cell_id"] for row in gate["selected_operational_cells"]) == SELECTED_IDS
    assert tuple(row["cell_id"] for row in gate["static_only_cells"]) == STATIC_ONLY_IDS
    assert all(
        row["selected_for_heldout_pre_reference_execution"]
        for row in gate["selected_operational_cells"]
    )
    assert not any(
        row["selected_for_heldout_pre_reference_execution"]
        for row in gate["static_only_cells"]
    )


def test_below_cap_profiles_freeze_nonconstancy_and_section_exclusion():
    profiles = _build()["below_cap_relation_profiles"]
    assert profiles["application_section_presence"] == {
        "n_rows": 31,
        "n_measured": 31,
        "n_abstained": 0,
        "minimum": 1.0,
        "maximum": 1.0,
        "n_unique_values": 1,
        "nonconstant": False,
        "n_positive": 31,
        "n_zero": 0,
    }
    assert profiles["claim_dependency_well_formedness"]["nonconstant"] is True
    assert profiles["claim_set_layering"]["nonconstant"] is True
    assert profiles["functional_limitation_incidence"]["nonconstant"] is True
    assert profiles["abstract_word_count"]["nonconstant"] is True
    # The category scalar is summarized for audit, but the output contract below demotes it.
    assert profiles["statutory_category_surface_coverage"]["nonconstant"] is True


def test_cap_contact_evidence_is_finite_and_category_scalar_is_demoted():
    gate = _build()
    assert gate["cap_policy"]["n_cap_contact_rows"] == 119
    assert gate["cap_policy"]["n_below_cap_rows"] == 31
    evidence = gate["finite_evidence_profiles"]
    assert evidence["dependency_certificates"]["cap_contact_rows"] == {
        "n_rows": 119,
        "n_rows_with_certificate": 114,
        "n_certificates": 957,
        "certificate_kind_counts": {
            "counter_witness": 2,
            "positive_witness": 955,
        },
    }
    assert evidence["positive_local_layering_witnesses"]["cap_contact_rows"] == {
        "n_rows": 119,
        "n_positive_local_witnesses": 110,
    }
    assert evidence["category_surface_span_certificates"]["cap_contact_rows"][
        "n_certificates"
    ] == 150
    category = next(
        row for row in gate["selected_operational_cells"] if row["cell_id"] == R3_CATEGORY
    )
    relation = category["relations"][0]
    assert relation["output_mode"] == "positive_category_surface_span_certificates_only"
    assert "demoted" in relation["scalar_policy"]
    assert relation["absence_or_whole_source_inference_permitted"] is False
    assert all("absence" not in output for output in relation["allowed_output"])


def test_channel_flags_and_output_modes_preserve_claim_limits():
    gate = _build()
    assert gate["channel_boundaries"] == {
        "input_fields": ["item_key", "ctext"],
        "reference_or_prompt_values_loaded": False,
        "outcomes_loaded": False,
        "prior_art_or_examiner_evidence_loaded": False,
        "external_supervision_loaded": False,
        "heldout_items_or_outputs_loaded": False,
        "models_or_apis_called": False,
        "whole_patent_score_emitted": False,
    }
    modes = {
        relation["output_mode"]
        for cell in gate["selected_operational_cells"]
        for relation in cell["relations"]
    }
    assert modes == {
        "finite_dependency_edge_and_local_counter_witnesses",
        "positive_local_root_plus_dependent_edge_witness",
        "positive_category_surface_span_certificates_only",
        "positive_marker_certificates_plus_below_cap_presented_text_incidence",
        "exact_named_presented_abstract_word_count",
    }
    assert all(
        relation["absence_or_whole_source_inference_permitted"] is False
        for cell in gate["selected_operational_cells"]
        for relation in cell["relations"]
    )


def test_wrong_versions_sources_and_forbidden_channels_fail_closed():
    execution, fidelity = copy.deepcopy(_inputs())
    execution["program_schema"] = "metric-seam.patent-claim-structure.v12"
    with pytest.raises(PatentTrainGateError, match="runner, program, or phase"):
        _build(execution=execution)

    execution, fidelity = copy.deepcopy(_inputs())
    execution["design"]["outcome_or_reference_values_loaded"] = True
    with pytest.raises(PatentTrainGateError, match="outcome_or_reference"):
        _build(execution=execution)

    real_execution, real_fidelity = _inputs()
    with pytest.raises(PatentTrainGateError, match="v14"):
        build_patent_train_gate(
            real_execution,
            real_fidelity,
            execution_source=Path(EXECUTION_BASENAME.replace("v14", "v13")),
            fidelity_source=Path(FIDELITY_BASENAME),
        )


def test_summary_certificate_and_construct_map_drift_fail_closed():
    execution, fidelity = copy.deepcopy(_inputs())
    execution["summary"]["items_at_declared_character_cap"] = 118
    with pytest.raises(PatentTrainGateError, match="summary does not replay"):
        _build(execution=execution)

    execution, fidelity = copy.deepcopy(_inputs())
    certificate = next(
        certificate
        for row in execution["rows"]
        for certificate in row["result"]["certificates"]
        if certificate["relation"] == "statutory_category_surface_coverage"
    )
    certificate["span"] = [4, 4]
    with pytest.raises(PatentTrainGateError, match="category surface/span"):
        _build(execution=execution)

    execution, fidelity = copy.deepcopy(_inputs())
    selected = next(row for row in fidelity["rows"] if row["cell_id"] == R3_CATEGORY)
    selected["verdict"] = "sensitivity_near_miss_not_accepted"
    with pytest.raises(PatentTrainGateError, match="eight-cell map drifted"):
        _build(fidelity=fidelity)


def test_frozen_artifact_replays_and_writer_refuses_overwrite(tmp_path: Path):
    assert _load(DEFAULT_OUTPUT) == _build()
    path = tmp_path / "gate.json"
    _write_new(path, {"schema": SCHEMA})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _write_new(path, {"schema": SCHEMA})
