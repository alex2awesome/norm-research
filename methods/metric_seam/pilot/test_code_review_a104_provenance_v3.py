"""Regression checks for the additive a104 V3 provenance correction."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "outputs/metric_seam_pilot/tasks/code_review"


def test_v3_changes_provenance_not_numbers() -> None:
    v2 = json.loads((TASK / "a104_cpu_sealed_eval_v2.json").read_text())
    v3 = json.loads((TASK / "a104_cpu_sealed_eval_v3.json").read_text())
    for key in (
        "baseline_selection", "common_heldout_n", "gate_floor",
        "preexisting_deep_coded_checker", "split", "structural_profile_coverage",
    ):
        assert v3[key] == v2[key]
    assert v3["h0_freeze_checks"] == v2["blind_freeze_checks"]
    assert v3["retrospective_relation_h0"]["P_gate"] == v2["blind_relation_h0"]["P_gate"]
    assert v3["retrospective_relation_h0"]["P_beats_baseline"] == (
        v2["blind_relation_h0"]["P_beats_baseline"]
    )
    assert v3["retrospective_relation_h0"]["delta_vs_prompt_baseline"] == (
        v2["blind_relation_h0"]["delta_vs_prompt_baseline"]
    )
    assert v3["heldout_rhos_common_intersection"]["retrospective_relation_h0"] == (
        v2["heldout_rhos_common_intersection"]["blind_relation_h0"]
    )
    assert "blind_relation_h0" not in v3


def test_v3_does_not_claim_mechanical_authoring_blindness() -> None:
    v3 = json.loads((TASK / "a104_cpu_sealed_eval_v3.json").read_text())
    assert v3["h0_discovery_provenance"] == "manual_mock_retrospective_seed"
    assert v3["h0_authoring_certification"] == "not_mechanically_blind"
    assert v3["execution_blindness"]["classification"] == (
        "label_unreferenced_not_label_inaccessible"
    )
    assert v3["execution_blindness"]["serialized_input_contains_merge_judgement"]
    assert not v3["execution_blindness"]["merge_judgement_referenced_by_scorer"]


def test_v3_discloses_corpus_and_model_lineage() -> None:
    v3 = json.loads((TASK / "a104_cpu_sealed_eval_v3.json").read_text())
    data = v3["data_provenance"]
    model = v3["model_provenance"]
    assert "pr_test_execution" in data["source_diff_directory"]
    assert not data["uses_legacy_f2p_mock_program_or_output"]
    assert not data["uses_prior_test_execution_telemetry_or_test_outcome"]
    assert not model["model_or_gpu_inference_in_this_cpu_run"]
    assert "model-produced" in model["reference_judgement"]
    assert "Claude-produced" in model["prompt_compiled_baselines"]


def test_v3_records_profile_counts_and_identity_limitation() -> None:
    v3 = json.loads((TASK / "a104_cpu_sealed_eval_v3.json").read_text())
    audit = v3["structural_profile_audit"]
    assert audit["n_items"] == 250
    assert audit["n_ctext_head_tail_truncated"] == 168
    assert audit["n_items_with_ast_edge"] == 18
    assert audit["n_ast_edges_raw"] == 44
    assert audit["n_ast_edges_unique_qualified_triples"] == 41
    assert audit["n_items_with_assertion"] == 33
    assert audit["n_assertions"] == 114
    assert "scope-aware" in audit["qualified_name_collision_limitation"]


def test_v3_has_a_narrow_correction_time_comparison_receipt() -> None:
    v3 = json.loads((TASK / "a104_cpu_sealed_eval_v3.json").read_text())
    receipt = v3["comparison_inputs_verified_at_correction"]
    assert len(receipt["code_scores_sha256"]) == 64
    assert set(receipt["sources"]) == {
        "a104_v0_keyword", "a104_v1_structure", "a104_v2_holistic",
        "a104_coded_checker",
    }
    assert "V2 did not itself" in receipt["verification_time_semantics"]
