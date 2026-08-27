from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from methods.metric_seam.adjudicate_patent_claim_graph_additive_v1 import build_audit


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "metric_seam_pilot" / "hierarchy_r123"
PANEL = OUT / "panel_v3.json"
CANONICAL = OUT / "patents_claim_structure_construct_fidelity_v1.json"
HISTORICAL = OUT / "patents_construct_fidelity_v1.json"
ARTIFACT = OUT / "patents_claim_graph_additive_construct_fidelity_v1.json"

EXPECTED = {
    ("R1", 17): {"numeric_constraint_definition_graph"},
    ("R1", 26): {"formula_variable_definition_alignment"},
    ("R2", 2): {"two_part_or_jepson_structure"},
    ("R2", 3): {
        "bounded_antecedent_term_reference_graph",
        "formula_variable_definition_alignment",
    },
    ("R2", 20): {"markush_closed_group_structure"},
    ("R3", 11): {"claim_status_and_local_listing_witnesses"},
    ("R3", 12): {
        "bounded_antecedent_term_reference_graph",
        "numeric_constraint_definition_graph",
        "formula_variable_definition_alignment",
    },
    ("R3", 17): {"numeric_constraint_definition_graph"},
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build() -> dict:
    return build_audit(_load(PANEL), _load(CANONICAL), _load(HISTORICAL))


def test_exact_additive_cell_envelope_and_depths() -> None:
    audit = _build()
    assert len(audit["rows"]) == 90
    assert len({row["cell_id"] for row in audit["rows"]}) == 90
    accepted = {
        (row["level"], row["selection_rank"]): {
            relation["relation_id"] for relation in row["matched_relations"]
        }
        for row in audit["rows"]
        if row["verdict"] == "partial_relation_local"
    }
    assert accepted == EXPECTED
    assert audit["summary"]["verdict_counts"] == {
        "bounded_non_discovery": 82,
        "partial_relation_local": 8,
    }
    assert audit["summary"]["by_level"] == {
        "R1": {
            "n_cells": 30,
            "n_partial_relation_local": 2,
            "n_bounded_non_discovery": 28,
        },
        "R2": {
            "n_cells": 30,
            "n_partial_relation_local": 3,
            "n_bounded_non_discovery": 27,
        },
        "R3": {
            "n_cells": 30,
            "n_partial_relation_local": 3,
            "n_bounded_non_discovery": 27,
        },
    }
    assert audit["summary"]["maximum_matching_relation_depth_counts"] == {
        "2": 3,
        "3": 5,
    }
    assert audit["summary"]["n_relation_mappings"] == 11


def test_every_mapping_is_narrow_excluded_and_certificate_scoped() -> None:
    mappings = [
        relation
        for row in _build()["rows"]
        for relation in row["matched_relations"]
    ]
    assert len(mappings) == 11
    assert all(row["requested_subrelation"] for row in mappings)
    assert all(row["partial_scope"] for row in mappings)
    assert all(row["exclusions"] for row in mappings)
    assert all(row["certificate_policy"] for row in mappings)
    assert {row["channel"] for row in mappings} == {"code"}
    assert {row["depth"] for row in mappings} == {2, 3}


def test_nonselected_cells_are_explicit_bounded_non_discovery() -> None:
    rejected = [
        row for row in _build()["rows"] if row["verdict"] == "bounded_non_discovery"
    ]
    assert len(rejected) == 82
    assert all(row["bounded_non_discovery_reason"] for row in rejected)
    assert all(row["matched_relations"] == [] for row in rejected)
    assert all(row["exact_whole_construct_fidelity"] is False for row in rejected)
    assert "not evidence of tacitness" in _build()["audit_design"][
        "negative_result_policy"
    ]


def test_provenance_lanes_are_disjoint_and_not_codability() -> None:
    union = _build()["summary"]["provenance_separate_descriptive_union"]
    assert union == {
        "canonical_pure_code_cells": 8,
        "historical_oracle_hybrid_cells": 6,
        "additive_claim_graph_cells": 8,
        "additive_overlap_with_canonical": 0,
        "additive_overlap_with_historical": 0,
        "three_lane_union_cells": 22,
        "interpretation": (
            "descriptive coverage union only; provenance and channels remain separate "
            "and this is not a codability, reconstruction, or isomorphism estimate"
        ),
    }
    limits = _build()["claim_limits"]
    assert not limits["codability_claim_permitted"]
    assert not limits["prompt_articulability_measured"]
    assert not limits["reference_reconstruction_measured"]
    assert not limits["isomorphism_measured"]


def test_panel_and_provenance_artifacts_fail_closed() -> None:
    panel = deepcopy(_load(PANEL))
    panel["schema"] = "wrong"
    with pytest.raises(ValueError, match="panel schema"):
        build_audit(panel, _load(CANONICAL), _load(HISTORICAL))

    canonical = deepcopy(_load(CANONICAL))
    canonical["task"] = "wrong"
    with pytest.raises(ValueError, match="canonical"):
        build_audit(_load(PANEL), canonical, _load(HISTORICAL))


def test_checked_in_artifact_equals_fresh_source_only_audit() -> None:
    assert _load(ARTIFACT) == _build()
