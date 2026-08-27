from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from methods.metric_seam.hierarchy_code_review_registry import RegistryError, build_registry
from methods.metric_seam.hierarchy_batch import build_readiness


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"


def _load(name: str):
    return json.loads((BASE / name).read_text(encoding="utf-8"))


def _inputs():
    return (
        _load("code_review_construct_fidelity_v2.json"),
        _load("code_review_train_execution_v2.json"),
        _load("code_review_train_gate_v1.json"),
        _load("code_review_heldout_readiness_v1.json"),
        _load("code_review_reconstruction_prompt_manifest_v2.json"),
    )


def _source_paths():
    return {
        "construct_fidelity": str(BASE / "code_review_construct_fidelity_v2.json"),
        "train_execution": str(BASE / "code_review_train_execution_v2.json"),
        "train_gate": str(BASE / "code_review_train_gate_v1.json"),
        "heldout_readiness": str(BASE / "code_review_heldout_readiness_v1.json"),
        "prompt_manifest": str(BASE / "code_review_reconstruction_prompt_manifest_v2.json"),
    }


def test_real_registry_keeps_code_prompt_and_isomorphism_gates_separate():
    registry = build_registry(*_inputs())
    assert registry["summary"]["n_cells"] == 90
    assert registry["summary"]["n_relation_local_static_fidelity"] == 56
    assert registry["summary"]["n_train_operational_relation_mappings"] == 30
    assert registry["summary"]["n_heldout_confirmatory_reconstruction_ready"] == 21
    assert registry["summary"]["n_prompt_references_compiled_unscored"] == 21
    assert registry["summary"]["n_prompt_channels_compiled"] == 3
    assert registry["summary"]["n_unique_prompt_program_vectors"] == 12
    assert registry["summary"]["n_prompt_jobs_compiled_unscored"] == 15750
    assert registry["summary"]["n_whole_construct_code_fidelity"] == 0
    assert registry["summary"]["n_prompt_references_scored"] == 0
    assert registry["summary"]["n_isomorphism_adjudications"] == 0


def test_legacy_prompt_manifest_remains_readable_but_is_not_the_current_registry_input():
    inputs = list(_inputs())
    inputs[-1] = _load("code_review_reconstruction_prompt_manifest_v1.json")
    registry = build_registry(*inputs)
    assert registry["summary"]["n_prompt_channels_compiled"] == 2
    assert registry["summary"]["n_unique_prompt_program_vectors"] is None


def test_prompt_manifest_cannot_expand_after_code_coverage_gate():
    inputs = list(_inputs())
    prompt = copy.deepcopy(inputs[-1])
    prompt["cell_ids"].append("invented")
    inputs[-1] = prompt
    with pytest.raises(RegistryError, match="prompt cells"):
        build_registry(*inputs)


def test_current_prompt_manifest_must_keep_panel_binding_and_partial_scope():
    inputs = list(_inputs())
    prompt = copy.deepcopy(inputs[-1])
    prompt["panel_content_sha256"] = "wrong"
    inputs[-1] = prompt
    with pytest.raises(RegistryError, match="different panels"):
        build_registry(*inputs)

    inputs = list(_inputs())
    prompt = copy.deepcopy(inputs[-1])
    prompt["scope_statements"]["selected_construct_fidelity_verdict_counts"] = {
        "exact": 21
    }
    inputs[-1] = prompt
    with pytest.raises(RegistryError, match="partial-only"):
        build_registry(*inputs)


def test_registry_advances_code_arm_without_claiming_prompt_or_isomorphism_completion():
    registry = build_registry(*_inputs(), sources=_source_paths())
    readiness = build_readiness(
        _load("panel_v3.json"),
        prompt_bank=_load("prompt_arm_bank_v3.json"),
        program_registry=registry["registry"],
    )
    code_review = readiness["progress_matrix"]["code-review"]
    assert sum(level["completed_decomposition"] for level in code_review.values()) == 56
    assert sum(level["operational_relation_local_witness"] for level in code_review.values()) == 30
    assert sum(level["heldout_confirmatory_reconstruction_ready"] for level in code_review.values()) == 21
    assert sum(level["prompt_reference_compiled_unscored"] for level in code_review.values()) == 21
    assert sum(level["prompt_reference_scored"] for level in code_review.values()) == 0
    assert sum(level["isomorphism_complete"] for level in code_review.values()) == 0
    assert not any(row["completed_deep_metric_seam_run"] for row in readiness["rows"])
