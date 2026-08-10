from pathlib import Path

import pytest

from methods.metric_seam.cross_audit_patent_claim_graph_additive_v1 import (
    QUARANTINED_RELATIONS,
    adversarial_probes,
    build_cross_audit,
)


@pytest.fixture(scope="module")
def audit() -> dict:
    return build_cross_audit(Path(__file__).resolve().parents[2])


def test_execution_receipts_and_pipeline_replay_exact(audit: dict) -> None:
    assert audit["summary"]["all_execution_source_hashes_match"] is True
    assert audit["summary"]["all_pipeline_replays_exact"] is True
    assert all(audit["replay"].values())
    assert all(
        check["matches"]
        for phase in audit["bound_source_hash_checks"].values()
        for check in phase.values()
    )


def test_all_eleven_mappings_receive_six_component_adjudication(audit: dict) -> None:
    rows = audit["mapping_adjudications"]
    assert len(rows) == 11
    assert len({row["cell_id"] for row in rows}) == 8
    for row in rows:
        assert row["construct_mapping_disposition"] == "retain_relation_local_mapping"
        for component in ("object", "relation", "polarity", "applicability", "aggregation", "depth"):
            assert row[component]


def test_numeric_and_formula_are_quarantined_without_erasing_mapping(audit: dict) -> None:
    quarantined = [
        row
        for row in audit["mapping_adjudications"]
        if not row["counts_as_current_executable_coverage"]
    ]
    assert {row["relation_id"] for row in quarantined} == QUARANTINED_RELATIONS
    assert len(quarantined) == 6
    assert audit["summary"]["n_current_executable_cells_after_cross_audit"] == 5
    assert audit["summary"]["n_current_executable_mappings_after_cross_audit"] == 5
    assert audit["summary"]["current_executable_cells_by_level"] == {"R2": 3, "R3": 2}
    assert audit["summary"]["current_executable_cell_depth_counts"] == {"2": 3, "3": 2}


def test_observed_links_replay_but_adversarial_program_class_fails(audit: dict) -> None:
    train = audit["receipt_validation"]["train"]
    heldout = audit["receipt_validation"]["heldout"]
    assert train["term"]["n_edges_replayed"] == 7642
    assert heldout["term"]["n_edges_replayed"] == 7725
    assert train["term"]["n_candidate_scope_violations"] == 0
    assert heldout["term"]["n_candidate_scope_violations"] == 0
    assert train["numeric"]["n_links_replayed"] == 4
    assert heldout["numeric"]["n_links_replayed"] == 1
    assert train["numeric"]["n_observed_scope_violations"] == 2
    assert heldout["numeric"]["n_observed_scope_violations"] == 0
    assert train["formula"]["n_links_replayed"] == 1
    assert train["formula"]["n_observed_link_violations"] == 0
    assert train["formula"]["n_ghost_definition_nodes"] == 3


def test_markush_truncation_filter_is_exact(audit: dict) -> None:
    train = audit["receipt_validation"]["train"]["markush"]
    heldout = audit["receipt_validation"]["heldout"]["markush"]
    assert train["n_original_certificates"] == 37
    assert train["n_retained_certificates"] == 37
    assert train["n_excluded_truncated_certificates"] == 0
    assert heldout["n_original_certificates"] == 25
    assert heldout["n_retained_certificates"] == 23
    assert heldout["n_excluded_truncated_certificates"] == 2
    assert heldout["n_items_with_excluded_certificates"] == 1
    assert {row["item_key"] for row in heldout["excluded"]} == {"heldout_0036"}


def test_disjoint_union_recomputed_with_trusted_current_correction(audit: dict) -> None:
    union = audit["descriptive_union_check"]
    assert union["canonical_cells"] == 8
    assert union["historical_cells"] == 6
    assert union["original_additive_cells"] == 8
    assert union["canonical_historical_overlap"] == 0
    assert union["canonical_additive_overlap"] == 0
    assert union["historical_additive_overlap"] == 0
    assert union["original_three_lane_union"] == 22
    assert union["trusted_current_additive_cells"] == 5
    assert union["trusted_three_lane_union"] == 19


def test_all_five_adversarial_defects_reproduce() -> None:
    probes = adversarial_probes()
    assert len(probes) == 5
    assert all(row["defect_reproduced"] for row in probes.values())


def test_provenance_gap_is_narrowly_stated(audit: dict) -> None:
    provenance = audit["provenance_correction"]
    assert provenance["execution_receipts_hash_bound"] is True
    assert provenance["construct_audit_hash_bound_by_freeze"] is False
    assert provenance["freeze_hash_bound_by_operational_summary"] is False
