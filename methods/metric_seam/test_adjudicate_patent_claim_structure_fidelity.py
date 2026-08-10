from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from methods.metric_seam.adjudicate_patent_claim_structure_fidelity import (
    build_audit,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "metric_seam_pilot" / "hierarchy_r123"
PANEL = OUT / "panel_v3.json"
TRAIN = OUT / "patents_claim_structure_compiler_train_v14.json"
HISTORICAL = OUT / "patents_construct_fidelity_v1.json"
ARTIFACT = OUT / "patents_claim_structure_construct_fidelity_v1.json"


EXPECTED = {
    "TB::patents::specific::R1::merged_tree::151::a6737bddab8d451d7ae9": {
        "application_section_presence"
    },
    "TB::patents::specific::R2::grandparent::10::41a099074657b4acc7f5": {
        "functional_limitation_incidence"
    },
    "TB::patents::specific::R2::merged_group::40::bb89d6d56dcc9ea9c238": {
        "application_section_presence"
    },
    "TB::patents::specific::R3::grandparent::0::ed76386d4408681be502": {
        "functional_limitation_incidence"
    },
    "TB::patents::specific::R3::merged_group::12::4a62e79af29087e6ff96": {
        "application_section_presence"
    },
    "TB::patents::specific::R3::merged_group::3::6d907639386384acc1da": {
        "abstract_word_count"
    },
    "TB::patents::specific::R3::grandparent::14::b26fd00c6c47f2854678": {
        "claim_dependency_well_formedness",
        "claim_set_layering",
    },
    "TB::patents::specific::R3::merged_group::7::ac30b4e148a5c6a11ec7": {
        "statutory_category_surface_coverage"
    },
}

SENSITIVITY = {
    "TB::patents::specific::R1::parented_tree::252::f491e1d963d7235b9f55",
    "TB::patents::specific::R1::merged_tree::254::1e6c67e300daccfa0331",
    "TB::patents::specific::R2::grandparent::20::d5241fc9bf0f24e2d9fc",
    "TB::patents::specific::R3::merged_group::2::dc0365c77ceff8c35701",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build() -> dict:
    return build_audit(_load(PANEL), _load(TRAIN), _load(HISTORICAL))


def test_exact_conservative_eight_and_relation_ids() -> None:
    audit = _build()
    assert len(audit["rows"]) == 90
    assert len({row["cell_id"] for row in audit["rows"]}) == 90
    accepted = {
        row["cell_id"]: {
            relation["relation_id"] for relation in row["matched_relations"]
        }
        for row in audit["rows"]
        if row["verdict"] == "partial_relation_local"
    }
    assert accepted == EXPECTED
    assert audit["summary"]["verdict_counts"] == {
        "no_faithful_relation": 78,
        "partial_relation_local": 8,
        "sensitivity_near_miss_not_accepted": 4,
    }
    assert audit["summary"]["by_level"] == {
        "R1": {
            "n_cells": 30,
            "n_partial_relation_local": 1,
            "n_sensitivity_near_miss": 2,
        },
        "R2": {
            "n_cells": 30,
            "n_partial_relation_local": 2,
            "n_sensitivity_near_miss": 1,
        },
        "R3": {
            "n_cells": 30,
            "n_partial_relation_local": 5,
            "n_sensitivity_near_miss": 1,
        },
    }
    assert audit["summary"]["maximum_matching_relation_depth_counts"] == {
        "1": 7,
        "2": 1,
    }


def test_near_misses_are_visible_but_not_credited() -> None:
    audit = _build()
    near = {
        row["cell_id"]: row
        for row in audit["rows"]
        if row["verdict"] == "sensitivity_near_miss_not_accepted"
    }
    assert set(near) == SENSITIVITY
    assert all(not row["matched_relations"] for row in near.values())
    assert all(row["rejection_or_demotion_reason"] for row in near.values())
    assert all(
        row["sensitivity_near_miss"]["train_operational_applicability"]
        for row in near.values()
    )


def test_train_applicability_does_not_promote_constant_or_absence_channels() -> None:
    audit = _build()
    relations = [
        relation
        for row in audit["rows"]
        if row["verdict"] == "partial_relation_local"
        for relation in row["matched_relations"]
    ]
    sections = [
        row for row in relations if row["relation_id"] == "application_section_presence"
    ]
    assert len(sections) == 3
    assert all(
        row["train_operational_applicability"]["classification"]
        == "measured_but_constant_non_operational"
        for row in sections
    )
    assert all(
        row["train_operational_applicability"][
            "absence_or_whole_source_inference_permitted"
        ]
        is False
        for row in relations
    )
    assert "antecedent_reference_surface_coverage" not in {
        row["relation_id"] for row in relations
    }
    category = next(
        row
        for row in relations
        if row["relation_id"] == "statutory_category_surface_coverage"
    )
    assert category["certificate_policy"].startswith("positive surface-and-span")


def test_historical_union_is_disjoint_and_provenance_separated() -> None:
    union = _build()["summary"]["additive_union_with_historical"]
    assert union["n_current_partial_cells"] == 8
    assert union["n_historical_partial_cells"] == 6
    assert union["n_overlapping_cells"] == 0
    assert union["n_additive_union_cells"] == 14
    assert union["by_level"] == {
        "R1": {"n_current": 1, "n_historical": 2, "n_union": 3},
        "R2": {"n_current": 2, "n_historical": 1, "n_union": 3},
        "R3": {"n_current": 5, "n_historical": 3, "n_union": 8},
    }
    assert union["maximum_matching_relation_depth_counts"] == {
        "1": 8,
        "2": 1,
        "3": 5,
    }
    assert "oracle-conditioned" in union["provenance_warning"]


def test_weighted_numbers_are_explicitly_non_certifying() -> None:
    weighted = _build()["summary"][
        "posthoc_design_weighted_conditional_sensitivity"
    ]
    assert weighted["not_a_codability_or_prevalence_certification"] is True
    assert weighted["conservative_eight"]["weighted_fraction"] == pytest.approx(
        0.04273879142300195
    )
    assert weighted["broader_twelve_including_near_misses"][
        "weighted_fraction"
    ] == pytest.approx(0.09917153996101363)
    assert weighted["additive_union_fourteen"]["weighted_fraction"] == pytest.approx(
        0.09449317738791424
    )


def test_every_rejected_row_has_a_specific_reason() -> None:
    audit = _build()
    rejected = [
        row for row in audit["rows"] if row["verdict"] != "partial_relation_local"
    ]
    assert len(rejected) == 82
    assert all(row["rejection_or_demotion_reason"] for row in rejected)
    assert all(row["exact_whole_construct_fidelity"] is False for row in audit["rows"])


def test_train_contract_fails_closed_if_reference_or_outcome_was_loaded() -> None:
    train = deepcopy(_load(TRAIN))
    train["design"]["outcome_or_reference_values_loaded"] = True
    with pytest.raises(ValueError, match="blind"):
        build_audit(_load(PANEL), train, _load(HISTORICAL))


def test_checked_in_artifact_equals_fresh_build() -> None:
    assert _load(ARTIFACT) == _build()
