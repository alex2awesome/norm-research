from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from methods.metric_seam.adjudicate_math_construct_fidelity_r3 import (
    OUTPUT,
    ROOT,
    SCHEMA,
    SOURCE_MAP,
    build_artifact,
)


def _seed_map():
    return json.loads(SOURCE_MAP.read_text(encoding="utf-8"))


def test_r3_audit_closes_all_cells_without_whole_construct_claims():
    artifact = build_artifact(_seed_map())
    assert artifact["schema"] == SCHEMA
    assert artifact["status"] == "complete_static_code_only_adjudication_pre_execution"
    assert artifact["levels"] == ["R3"]
    assert artifact["n_rows"] == 30
    assert artifact["counts"]["R3"] == {
        "n_cells": 30,
        "n_retrieved_candidates": 19,
        "verdicts": {
            "mismatch": 4,
            "no_candidate_bounded_non_discovery": 11,
            "partial": 15,
        },
        "eligible_for_relation_local_execution": 15,
        "eligible_fraction_of_cells": 0.5,
        "eligible_fraction_of_retrieved_candidates": 0.789474,
        "audited_depths": {"1": 9, "2": 10, "null": 11},
        "eligible_audited_depths": {"1": 5, "2": 10},
    }
    assert len({row["cell_id"] for row in artifact["rows"]}) == 30
    assert all(row["level"] == "R3" for row in artifact["rows"])
    assert not any(row["verdict"] == "exact" for row in artifact["rows"])


def test_candidate_identity_provenance_and_llm_exclusion_are_mechanical():
    artifact = build_artifact(_seed_map())
    for row in artifact["rows"]:
        candidate = row["candidate"]
        if candidate is None:
            assert row["verdict"] == "no_candidate_bounded_non_discovery"
            assert row["audited_depth"] is None
            assert row["implemented_relations"] == []
            continue
        source = ROOT / candidate["source_path"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == candidate[
            "program_sha256"
        ]
        assert "manual historical hybrid" in candidate[
            "historical_hybrid_provenance"
        ]
        relation_text = " ".join(row["implemented_relations"])
        assert all(
            field not in relation_text
            for field in candidate[
                "llm_fields_excluded_from_implemented_relations"
            ]
        )
        assert row["verdict"] in {"partial", "mismatch"}
        assert row["audited_depth"] in {1, 2}


def test_functional_audit_records_applicability_polarity_and_aggregation():
    artifact = build_artifact(_seed_map())
    for row in artifact["rows"]:
        functional = row["polarity_aggregation_applicability_caveats"]
        assert isinstance(functional, list) and len(functional) >= 5
        assert functional[0].startswith("Applicability: ")
        assert functional[1].startswith("Polarity: ")
        assert functional[2].startswith("Aggregation: ")
        assert all(isinstance(value, str) and value.strip() for value in functional)
        assert isinstance(row["residual_construct"], str) and row[
            "residual_construct"
        ].strip()
        assert row["justification"]


def test_known_gated_proxies_are_mismatches_and_structural_relations_stay_partial():
    artifact = build_artifact(_seed_map())
    by_id = {row["cell_id"]: row for row in artifact["rows"]}
    a42_rows = [
        row for row in artifact["rows"]
        if row["candidate"] and row["candidate"]["aspect_id"] == "a42"
    ]
    assert len(a42_rows) == 3
    assert {row["verdict"] for row in a42_rows} == {"mismatch"}
    a108 = next(
        row for row in artifact["rows"]
        if row["candidate"] and row["candidate"]["aspect_id"] == "a108"
    )
    assert a108["verdict"] == "mismatch"

    notation = by_id[
        "TB::math-stackexchange::general::R3::merged_group::3::e7911c7b707a53bacba4"
    ]
    typesetting = by_id[
        "TB::math-stackexchange::general::R3::grandparent::16::570eed33fe5f1ce2a120"
    ]
    modular = by_id[
        "TB::math-stackexchange::general::R3::merged_group::14::da47b04ffaa9bf294ae9"
    ]
    assert (notation["verdict"], notation["audited_depth"]) == ("partial", 2)
    assert (typesetting["verdict"], typesetting["audited_depth"]) == ("partial", 2)
    assert (modular["verdict"], modular["audited_depth"]) == ("partial", 2)


def test_unknown_outcome_fields_cannot_change_static_adjudication():
    source = _seed_map()
    baseline = build_artifact(source)
    poisoned = copy.deepcopy(source)
    for index, row in enumerate(poisoned["rows"]):
        row["reference_judgment"] = index / 100
        row["outcome_label"] = "high" if index % 2 else "low"
        row["program_output"] = {"score": 0.999}
        row["correlation"] = 0.999
        row["reconstruction"] = {"isomorphic": True}
    assert build_artifact(poisoned) == baseline


def test_candidate_identity_drift_fails_closed():
    source = _seed_map()
    r3_selected = next(
        row for row in source["rows"]
        if row["level"] == "R3" and row["selected_seed"] is not None
    )
    r3_selected["selected_seed"]["aspect_id"] = "a999"
    with pytest.raises(ValueError, match="candidate identity drift"):
        build_artifact(source)


def test_generated_artifact_matches_builder_when_present():
    if not OUTPUT.exists():
        pytest.skip("generated artifact not present")
    stored = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert stored == build_artifact(_seed_map())
