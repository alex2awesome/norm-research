from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from methods.metric_seam.hierarchy_patent_claim_structure_gate import (
    DEFAULT_EXECUTION as DEFAULT_TRAIN_EXECUTION,
    DEFAULT_FIDELITY,
    DEFAULT_OUTPUT as DEFAULT_TRAIN_GATE,
    R3_ARCHITECTURE,
    R3_CATEGORY,
    SELECTED_IDS,
    STATIC_ONLY_IDS,
)
from methods.metric_seam.hierarchy_patent_claim_structure_operational_summary import (
    DEFAULT_HELDOUT_EXECUTION,
    DEFAULT_OUTPUT,
    SCHEMA,
    STATUS,
    PatentOperationalSummaryError,
    _write_new,
    build_operational_summary,
)


ROOT = Path(__file__).resolve().parents[2]


def _load(relative: Path):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _inputs():
    return (
        _load(DEFAULT_TRAIN_EXECUTION),
        _load(DEFAULT_FIDELITY),
        _load(DEFAULT_TRAIN_GATE),
        _load(DEFAULT_HELDOUT_EXECUTION),
    )


def _build(train=None, fidelity=None, gate=None, heldout=None):
    real = _inputs()
    return build_operational_summary(
        real[0] if train is None else train,
        real[1] if fidelity is None else fidelity,
        real[2] if gate is None else gate,
        real[3] if heldout is None else heldout,
        train_gate_source=DEFAULT_TRAIN_GATE,
        heldout_execution_source=DEFAULT_HELDOUT_EXECUTION,
    )


def test_real_summary_credits_exactly_five_heldout_relation_measurements():
    summary = _build()
    assert summary["schema"] == SCHEMA
    assert summary["status"] == STATUS
    assert summary["stage_summary"] == {
        "n_static_relation_local_cells": 8,
        "n_train_operational_cells": 5,
        "n_heldout_relation_measurable_cells": 5,
        "n_static_only_constant_cells": 3,
        "n_prompt_articulability_measured_cells": 0,
        "n_reference_reconstruction_measured_cells": 0,
        "n_prompt_code_isomorphism_evaluable_cells": 0,
        "n_whole_criterion_codability_established_cells": 0,
        "heldout_relation_measurable_by_level": {"R2": 1, "R3": 4},
        "heldout_relation_measurable_by_maximum_depth": {"1": 4, "2": 1},
    }
    assert tuple(row["cell_id"] for row in summary["heldout_operational_cells"]) == (
        SELECTED_IDS
    )
    assert all(row["heldout_relation_measurable"] for row in summary["heldout_operational_cells"])


def test_heldout_receipt_distinguishes_cap_contact_from_cap_measurement():
    receipt = _build()["heldout_receipt"]
    assert receipt == {
        "n_rows": 150,
        "n_cap_contact_rows": 123,
        "n_cap_contact_rows_with_claim_measurement": 122,
        "n_below_cap_rows": 27,
        "status_counts": {
            "measured": 27,
            "measured_with_possible_truncation": 122,
            "relation_abstained": 1,
        },
        "failure_types": {},
    }


def test_cap_evidence_uses_certificates_and_local_witnesses_only():
    summary = _build()
    cells = {row["cell_id"]: row for row in summary["heldout_operational_cells"]}
    architecture = cells[R3_ARCHITECTURE]
    dependency, layering = architecture["relations"]
    assert dependency["evidence"]["cap_contact_rows"] == {
        "n_rows": 123,
        "n_rows_with_positive_or_local_counter_certificate": 121,
        "n_rows_without_certificate_treated_as_abstention_not_absence": 2,
        "n_certificates": 1116,
        "n_distinct_certificate_payloads": 168,
        "certificate_kind_counts": {
            "counter_witness": 1,
            "positive_witness": 1115,
        },
    }
    assert layering["evidence"]["cap_contact_rows"] == {
        "n_rows": 123,
        "n_positive_local_root_plus_edge_witnesses": 119,
        "n_rows_without_positive_witness_treated_as_abstention_not_source_absence": 4,
    }
    assert summary["cap_policy"]["category_coverage_scalar_operationally_used"] is False
    assert summary["cap_policy"]["whole_source_completeness_or_compliance_permitted"] is False


def test_category_functional_and_abstract_outputs_have_nonconstant_support():
    summary = _build()
    cells = {row["cell_id"]: row for row in summary["heldout_operational_cells"]}
    category = cells[R3_CATEGORY]["relations"][0]
    evidence = category["evidence"]
    assert evidence["all_presented_rows"]["n_certificates"] == 168
    assert evidence["all_presented_rows"][
        "n_rows_without_certificate_treated_as_abstention_not_absence"
    ] == 32
    assert evidence["certificate_diversity"] == {
        "n_distinct_surfaces": 9,
        "n_distinct_categories": 4,
    }
    assert evidence["coverage_scalar_operationally_used"] is False

    functional_cells = [
        row for row in summary["heldout_operational_cells"]
        if row["ordered_relation_ids"] == ["functional_limitation_incidence"]
    ]
    assert len(functional_cells) == 2
    for cell in functional_cells:
        functional = cell["relations"][0]["evidence"]
        assert functional["all_presented_rows"]["n_certificates"] == 206
        assert functional["cap_contact_rows"]["n_certificates"] == 191
        assert functional["below_cap_relation_profile"]["nonconstant"] is True

    abstract = next(
        row for row in summary["heldout_operational_cells"]
        if row["ordered_relation_ids"] == ["abstract_word_count"]
    )["relations"][0]
    assert abstract["evidence"]["all_presented_rows_relation_profile"][
        "n_unique_values"
    ] == 85


def test_three_section_cells_remain_constant_static_only():
    static = _build()["static_only_cells"]
    assert tuple(row["cell_id"] for row in static) == STATIC_ONLY_IDS
    assert all(row["heldout_relation_measurable"] is False for row in static)
    assert all(row["heldout_relation_profile"]["n_measured"] == 150 for row in static)
    assert all(row["heldout_relation_profile"]["nonconstant"] is False for row in static)


def test_gate_tampering_and_forbidden_heldout_inputs_fail_closed():
    train, fidelity, gate, heldout = copy.deepcopy(_inputs())
    gate["selected_operational_cells"][0]["ordered_relation_ids"] = []
    with pytest.raises(PatentOperationalSummaryError, match="does not replay"):
        _build(train=train, fidelity=fidelity, gate=gate, heldout=heldout)

    train, fidelity, gate, heldout = copy.deepcopy(_inputs())
    heldout["design"]["outcome_or_reference_values_loaded"] = True
    with pytest.raises(PatentOperationalSummaryError, match="outcome_or_reference"):
        _build(train=train, fidelity=fidelity, gate=gate, heldout=heldout)

    train, fidelity, gate, heldout = copy.deepcopy(_inputs())
    heldout["summary"]["items_at_declared_character_cap"] = 122
    with pytest.raises(PatentOperationalSummaryError, match="summary does not replay"):
        _build(train=train, fidelity=fidelity, gate=gate, heldout=heldout)


def test_invalid_category_span_fails_before_operational_credit():
    train, fidelity, gate, heldout = copy.deepcopy(_inputs())
    certificate = next(
        certificate
        for row in heldout["rows"]
        for certificate in row["result"]["certificates"]
        if certificate["relation"] == "statutory_category_surface_coverage"
    )
    certificate["span"] = [7, 7]
    with pytest.raises(PatentOperationalSummaryError, match="category surface/span"):
        _build(train=train, fidelity=fidelity, gate=gate, heldout=heldout)


def test_frozen_artifact_replays_and_writer_refuses_overwrite(tmp_path: Path):
    assert _load(DEFAULT_OUTPUT) == _build()
    path = tmp_path / "summary.json"
    _write_new(path, {"schema": SCHEMA})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _write_new(path, {"schema": SCHEMA})
