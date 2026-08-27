"""Regression tests for the additive a104 V4 channel-taxonomy correction."""
from __future__ import annotations

import json

from methods.metric_seam.pilot.correct_code_review_a104_channel_taxonomy_v4 import (
    EXPECTED_HISTORICAL_SHA256,
    PROVENANCE_KEYS_PRESERVED,
    V2_JSON,
    V2_REPORT,
    V3_JSON,
    V3_REPORT,
    V4_JSON,
    V4_NOTICE,
    V4_REPORT,
    _numeric_leaves,
    _sha,
    build_v4,
    expected_artifacts,
)


def test_historical_v2_v3_artifacts_remain_frozen() -> None:
    assert _sha(V2_JSON) == EXPECTED_HISTORICAL_SHA256[V2_JSON]
    assert _sha(V2_REPORT) == EXPECTED_HISTORICAL_SHA256[V2_REPORT]
    assert _sha(V3_JSON) == EXPECTED_HISTORICAL_SHA256[V3_JSON]
    assert _sha(V3_REPORT) == EXPECTED_HISTORICAL_SHA256[V3_REPORT]


def test_v4_preserves_every_number_and_provenance_receipt() -> None:
    v3 = json.loads(V3_JSON.read_text())
    v4 = build_v4()
    assert _numeric_leaves(v4) == _numeric_leaves(v3)
    for key in PROVENANCE_KEYS_PRESERVED:
        assert v4[key] == v3[key]


def test_v4_classifies_prompt_generated_scorers_by_runtime_channel() -> None:
    v4 = build_v4()
    taxonomy = v4["channel_taxonomy_v4"]
    shallow = taxonomy["program_poles"]["prompt_generated_shallow_code"]
    assert shallow["authoring_origin"] == "Claude-generated"
    assert shallow["runtime_channel"] == "code"
    assert set(shallow["source_receipts"]) == {
        "a104_v0_keyword",
        "a104_v1_structure",
        "a104_v2_holistic",
    }
    assert all(
        receipt["runtime_channel"] == "code"
        for receipt in shallow["source_receipts"].values()
    )
    assert taxonomy["comparison_type"] == "within_code_channel_program_depth_comparison"
    assert not taxonomy["direct_prompt_articulability_candidate_present"]
    assert taxonomy["code_verifiability_candidates_present"]


def test_v4_separates_reconstruction_from_isomorphism_and_channel_axes() -> None:
    v4 = build_v4()
    measurement = v4["measurement_taxonomy_v4"]
    assert measurement["reported_rho_quantity"] == (
        "reconstruction_agreement_with_frozen_llm_judgment"
    )
    assert not measurement["rho_is_channel_capability_measure"]
    assert not measurement["rho_alone_certifies_isomorphism"]
    assert measurement["isomorphism_required_fidelities"] == [
        "construct",
        "input_representation",
        "program_channel",
        "reference_instrument",
    ]
    assert not measurement["external_anchor_used"]
    forbidden = v4["channel_taxonomy_v4"]["not_permitted"]
    assert "code-over-prompt conclusion" in forbidden
    assert "isomorphism claim from correlation alone" in forbidden


def test_v4_keeps_legacy_quantitative_paths_but_corrects_claim_text() -> None:
    v3 = json.loads(V3_JSON.read_text())
    v4 = build_v4()
    assert v4["heldout_rhos_common_intersection"] == (
        v3["heldout_rhos_common_intersection"]
    )
    assert "prompt_compiled_baseline" in v4["heldout_rhos_common_intersection"]
    assert "code-versus-code" in v4["preexisting_deep_coded_checker"]["interpretation"]
    assert "not code-over-prompt" in (
        v4["preexisting_deep_coded_checker"]["interpretation"]
    )


def test_checked_in_v4_artifacts_are_canonical() -> None:
    expected = expected_artifacts()
    assert set(expected) == {V4_JSON, V4_REPORT, V4_NOTICE}
    for path, payload in expected.items():
        assert path.read_bytes() == payload


def test_v4_report_and_notice_state_the_narrow_claim() -> None:
    report = V4_REPORT.read_text()
    notice = V4_NOTICE.read_text()
    assert "within-code-channel program-depth comparison" in report
    assert "not a direct\nprompt-articulability versus code-verifiability comparison" in report
    assert "must\nnot be quoted as code-over-prompt" in notice
    assert "construct + input + program + reference fidelity" in notice
