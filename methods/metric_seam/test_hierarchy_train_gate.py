from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from methods.metric_seam.hierarchy_code_runner import apply_program_selection, build_execution_plan
from methods.metric_seam.hierarchy_train_gate import TrainGateError, build_train_gate


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"


def _load(name: str):
    return json.loads((BASE / name).read_text(encoding="utf-8"))


def test_real_train_replay_freezes_minimal_operational_gate():
    execution = _load("code_review_train_execution_v2.json")
    audit = _load("code_review_construct_fidelity_v1.json")
    gate = build_train_gate(execution, audit)
    assert gate["reference_values_used"] is False
    assert gate["outcome_labels_used"] is False
    assert gate["heldout_items_or_outputs_used"] is False
    assert gate["summary"]["n_selected_programs"] == 16
    assert gate["summary"]["n_selected_relation_mappings"] == 30
    assert gate["summary"]["selected_relation_mappings_by_level"] == {
        "R1": 10, "R2": 7, "R3": 13
    }
    selected = apply_program_selection(build_execution_plan(audit), gate)
    assert len(selected) == 16


def test_gate_rejects_reference_or_outcome_bearing_execution():
    execution = _load("code_review_train_execution_v2.json")
    audit = _load("code_review_construct_fidelity_v1.json")
    for field in ("reference_fields_passed_to_worker", "outcome_fields_passed_to_worker"):
        poisoned = copy.deepcopy(execution)
        poisoned[field] = True
        with pytest.raises(TrainGateError):
            build_train_gate(poisoned, audit)


def test_gate_program_identity_must_match_static_audit():
    execution = _load("code_review_train_execution_v2.json")
    audit = _load("code_review_construct_fidelity_v1.json")
    poisoned = copy.deepcopy(execution)
    poisoned["programs"][0]["aspect_id"] = "a999"
    with pytest.raises(TrainGateError, match="identities drifted"):
        build_train_gate(poisoned, audit)
