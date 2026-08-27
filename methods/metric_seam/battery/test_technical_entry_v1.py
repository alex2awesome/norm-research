"""Tests for the additive real-artifact technical-entry preflight."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from technical_entry_v1 import (  # noqa: E402
    BindingError,
    DEPTH_SCHEMA_VERSION,
    PreflightConfig,
    SchemaError,
    _canonical_sha256,
    normalize_adversary,
    normalize_technical_entry,
)


CANDIDATE_SHA = "a" * 64
CRITERION = "math__a1"
RELATION = "worked_relation"


def base_artifacts():
    candidate = {
        "schema": "metric-seam.blind-reconstruction.sealed-candidate-scores.v2",
        "task": "math",
        "aspect_id": "a1",
        "candidate_sha256": CANDIDATE_SHA,
        "score_map": {
            "h1": 0.1,
            "h2": 0.2,
            "h3": 0.3,
            "h4": 0.4,
            "h5": None,
        },
    }
    reference = {
        "schema": "metric-seam.blind-reconstruction.sealed-llm-reference.v2",
        "task": "math",
        "aspect_id": "a1",
        "score_map": {"h1": 0.1, "h2": 0.2, "h3": 0.3},
    }
    manifest = {
        "schema": "metric-seam.blind-reconstruction.sealed-evaluation-manifest.v2",
        "task": "math",
        "aspect_id": "a1",
        "partition": {"heldout_count": 5},
        "artifacts": {"candidate_frozen.py": {"sha256": CANDIDATE_SHA}},
        "inputs": {},
    }
    contract_check = {
        "contract_sha256": "b" * 64,
        "code_gate": {"status": "PASS", "passed": True, "n_declared": 2},
        "hybrid_gate": {"status": "FAIL", "passed": False, "n_declared": 4},
        "discrimination_gate": {"status": "PASS", "passed": True, "n_items": 40},
    }
    adversary = {
        "verdict": "ACCEPT",
        "candidate_sha256": CANDIDATE_SHA,
        "criterion_id": CRITERION,
        "relation_id": RELATION,
    }
    config = PreflightConfig(
        candidate_coverage_min=0.80,
        reference_availability_min=0.50,
        common_given_reference_min=1.0,
        minimum_common_pairs=3,
        absolute_rho_min=0.30,
    )
    return manifest, candidate, reference, contract_check, adversary, config


def normalize_fixture(*, channel="code", **overrides):
    manifest, candidate, reference, contract_check, adversary, config = base_artifacts()
    arguments = {
        "criterion_id": CRITERION,
        "relation_id": RELATION,
        "candidate_channel": channel,
        "sealed_manifest": manifest,
        "candidate_scores": candidate,
        "frozen_llm_reference": reference,
        "contract_check": contract_check,
        "adversary": adversary,
        "config": config,
    }
    arguments.update(overrides)
    return normalize_technical_entry(**arguments)


def test_full_universe_preserves_null_and_separates_coverage_planes():
    receipt = normalize_fixture()
    coverage = receipt["coverage"]
    assert coverage["heldout_n"] == 5
    assert coverage["candidate_enumerated_n"] == 5
    assert coverage["candidate_finite_n"] == 4
    assert coverage["candidate_null_n"] == 1
    assert coverage["candidate_fraction"] == pytest.approx(0.8)
    assert coverage["reference_available_n"] == 3
    assert coverage["reference_fraction"] == pytest.approx(0.6)
    assert coverage["common_n"] == 3
    assert coverage["common_given_reference"] == 1.0
    assert coverage["status"] == "pass"
    assert receipt["construct_fidelity"]["status"] == "pass"
    assert receipt["reference_reconstruction"]["rho_candidate"] == pytest.approx(1.0)
    assert receipt["reference_reconstruction"]["absolute_floor_met"] is True
    assert receipt["inferential_preflight"]["eligible"] is True
    # A valid preflight remains explicitly non-confirmatory.
    assert receipt["claim_permissions"]["may_claim_confirmatory_batch"] is False


def test_candidate_score_artifact_must_enumerate_nulls_for_full_heldout():
    manifest, candidate, reference, contract_check, adversary, config = base_artifacts()
    candidate["score_map"].pop("h5")
    with pytest.raises(BindingError, match="full held-out universe"):
        normalize_technical_entry(
            criterion_id=CRITERION,
            relation_id=RELATION,
            candidate_channel="code",
            sealed_manifest=manifest,
            candidate_scores=candidate,
            frozen_llm_reference=reference,
            contract_check=contract_check,
            adversary=adversary,
            config=config,
        )


def test_contract_gate_routing_is_explicit_and_channel_faithful():
    code = normalize_fixture(channel="code")
    hybrid = normalize_fixture(channel="hybrid")
    assert code["construct_fidelity"]["contract"]["selected_gate"] == "code_gate"
    assert code["construct_fidelity"]["contract"]["status"] == "pass"
    assert hybrid["construct_fidelity"]["contract"]["selected_gate"] == "hybrid_gate"
    assert hybrid["construct_fidelity"]["contract"]["status"] == "fail"
    assert hybrid["construct_fidelity"]["status"] == "fail"


@pytest.mark.parametrize(
    ("payload", "expected_status", "schema_family"),
    [
        ({"decision": "ACCEPT"}, "pass", "decision"),
        ({"decision": "REJECT"}, "fail", "decision"),
        ({"suite_pass": True, "freeze_verified": True}, "pass", "suite_pass"),
        ({"suite_pass": False, "freeze_verified": True}, "fail", "suite_pass"),
        ({"verdict": "PASS"}, "pass", "verdict"),
        ({"verdict": "ACCEPT", "integrity_ok": False}, "fail", "verdict"),
    ],
)
def test_real_adversary_schema_families_normalize(payload, expected_status, schema_family):
    result = normalize_adversary(
        payload,
        candidate_sha256=CANDIDATE_SHA,
        criterion_id=CRITERION,
        relation_id=RELATION,
        universe_sha256="c" * 64,
    )
    assert result["status"] == expected_status
    assert result["schema_family"] == schema_family


def test_candidate_criterion_relation_and_universe_bindings_fail_closed():
    with pytest.raises(BindingError, match="candidate mismatch"):
        normalize_fixture(expected_candidate_sha256="f" * 64)

    manifest, candidate, reference, contract_check, adversary, config = base_artifacts()
    reference["criterion_id"] = "math__wrong"
    with pytest.raises(BindingError, match="criterion_id mismatch"):
        normalize_technical_entry(
            criterion_id=CRITERION,
            relation_id=RELATION,
            candidate_channel="code",
            sealed_manifest=manifest,
            candidate_scores=candidate,
            frozen_llm_reference=reference,
            contract_check=contract_check,
            adversary=adversary,
            config=config,
        )

    manifest, candidate, reference, contract_check, adversary, config = base_artifacts()
    adversary["candidate_sha256"] = "d" * 64
    with pytest.raises(BindingError, match="different candidate"):
        normalize_technical_entry(
            criterion_id=CRITERION,
            relation_id=RELATION,
            candidate_channel="code",
            sealed_manifest=manifest,
            candidate_scores=candidate,
            frozen_llm_reference=reference,
            contract_check=contract_check,
            adversary=adversary,
            config=config,
        )

    manifest, candidate, reference, contract_check, adversary, config = base_artifacts()
    universe_sha = _canonical_sha256(sorted(candidate["score_map"]))
    adversary["universe_sha256"] = "e" * 64
    assert universe_sha != adversary["universe_sha256"]
    with pytest.raises(BindingError, match="universe mismatch"):
        normalize_technical_entry(
            criterion_id=CRITERION,
            relation_id=RELATION,
            candidate_channel="code",
            sealed_manifest=manifest,
            candidate_scores=candidate,
            frozen_llm_reference=reference,
            contract_check=contract_check,
            adversary=adversary,
            config=config,
        )


def test_certificate_counts_and_relation_depth_are_independent_planes():
    _, candidate, _, _, _, _ = base_artifacts()
    universe_sha = _canonical_sha256(sorted(candidate["score_map"]))
    certificate = {
        "schema": "metric-seam.certificate-summary.v1",
        "criterion_id": CRITERION,
        "relation_id": RELATION,
        "candidate_sha256": CANDIDATE_SHA,
        "universe_sha256": universe_sha,
        "counts": {
            "verified_positive": 1,
            "verified_absence": 1,
            "abstain": 2,
            "error": 1,
        },
    }
    depth = {
        "scale": DEPTH_SCHEMA_VERSION,
        "criterion_id": CRITERION,
        "relation_id": RELATION,
        "candidate_sha256": CANDIDATE_SHA,
        "universe_sha256": universe_sha,
        "nodes": [
            {
                "node_id": "parse",
                "implementation": "code",
                "relation_depth": 1,
                "contributes_to_output": True,
            },
            {
                "node_id": "solver",
                "implementation": "code",
                "relation_depth": 3,
                "contributes_to_output": True,
            },
            {
                "node_id": "aggregate",
                "implementation": "aggregation",
                "relation_depth": None,
                "contributes_to_output": True,
            },
        ],
        "static_max_relation_depth": 3,
        "longest_path_edges": 2,
        "dynamic_contributing_depth_histogram": {"1": 2, "3": 3},
    }
    receipt = normalize_fixture(certificate_summary=certificate, depth_profile=depth)
    cert = receipt["certificate_plane"]
    assert cert["verified_positive_n"] == 1
    assert cert["verified_absence_n"] == 1
    assert cert["abstain_n"] == 2
    assert cert["error_n"] == 1
    assert cert["unclassified_n"] == 0
    assert cert["certificate_coverage"] == pytest.approx(0.4)
    profile = receipt["program_depth"]
    assert profile["static_max_relation_depth"] == 3
    assert profile["dynamic_contributing_depth_histogram"] == {
        "0": 0,
        "1": 2,
        "2": 0,
        "3": 3,
        "4": 0,
    }
    # Depth is not promoted into construct fidelity.
    assert "depth" not in receipt["construct_fidelity"]


def test_relation_depth_rejects_out_of_scale_and_incomplete_dynamic_accounting():
    _, candidate, _, _, _, _ = base_artifacts()
    universe_sha = _canonical_sha256(sorted(candidate["score_map"]))
    depth = {
        "scale": DEPTH_SCHEMA_VERSION,
        "candidate_sha256": CANDIDATE_SHA,
        "universe_sha256": universe_sha,
        "nodes": [
            {
                "node_id": "bad",
                "implementation": "code",
                "relation_depth": 5,
                "contributes_to_output": True,
            }
        ],
    }
    with pytest.raises(SchemaError, match=r"relation_depth in \[0, 4\]"):
        normalize_fixture(depth_profile=depth)

    depth["nodes"][0]["relation_depth"] = 4
    depth["dynamic_contributing_depth_histogram"] = {"4": 4}
    with pytest.raises(SchemaError, match="every held-out item"):
        normalize_fixture(depth_profile=depth)


def test_missing_discrimination_is_unavailable_not_synthetic_failure():
    manifest, candidate, reference, contract_check, adversary, config = base_artifacts()
    contract_check["discrimination_gate"] = {
        "status": "NOT_RUN",
        "passed": None,
        "n_items": 0,
    }
    receipt = normalize_technical_entry(
        criterion_id=CRITERION,
        relation_id=RELATION,
        candidate_channel="code",
        sealed_manifest=manifest,
        candidate_scores=candidate,
        frozen_llm_reference=reference,
        contract_check=contract_check,
        adversary=adversary,
        config=config,
    )
    assert receipt["construct_fidelity"]["contract"]["discrimination_status"] == "unavailable"
    assert receipt["construct_fidelity"]["status"] == "unavailable"
    assert receipt["inferential_preflight"]["eligible"] is False


def test_real_a144_sealed_artifacts_preflight_as_proxy_mismatch():
    run = ROOT / "outputs/metric_seam_pilot/reconstruction_v2/blind_math_a144_001"
    sealed = run / "sealed_eval_002"
    receipt = normalize_technical_entry(
        criterion_id="math__a144",
        relation_id="explicit_witness_verification_and_scope",
        candidate_channel="code",
        sealed_manifest=sealed / "sealed_manifest.json",
        candidate_scores=sealed / "candidate_scores.json",
        frozen_llm_reference=sealed / "llm_reference_scores.json",
        metrics=sealed / "metrics.json",
        contract=(
            ROOT
            / "outputs/metric_seam_pilot/battery/effort_ladder/contracts_v3/math__a144.json"
        ),
        adversary=run / "adversary_001/RESULTS.json",
        expected_candidate_sha256=(
            "1eb9f07166da53365b901b0bad223991b71421afd07c4c9aa918816795c33ecd"
        ),
    )
    coverage = receipt["coverage"]
    assert coverage["heldout_n"] == 100
    assert coverage["candidate_finite_n"] == 100
    assert coverage["reference_available_n"] == 52
    assert coverage["common_n"] == 52
    assert coverage["candidate_fraction"] == 1.0
    assert coverage["reference_fraction"] == pytest.approx(0.52)
    assert receipt["reference_reconstruction"]["rho_candidate"] == pytest.approx(
        0.06598665482101687
    )
    assert receipt["reference_reconstruction"]["absolute_floor_met"] is False
    construct = receipt["construct_fidelity"]
    assert construct["adversary"]["schema_family"] == "decision"
    assert construct["adversary"]["status"] == "fail"
    assert construct["adversary"]["summary"]["ordering_passes"] == 14
    assert construct["status"] == "fail"
    assert receipt["inferential_preflight"]["eligible"] is False
    assert receipt["certificate_plane"]["status"] == "unavailable"
    assert receipt["program_depth"]["status"] == "unavailable"
    assert receipt["claim_permissions"]["may_claim_isomorphism"] is False
