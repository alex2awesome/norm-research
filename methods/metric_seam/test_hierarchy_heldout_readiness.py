from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from methods.metric_seam.hierarchy_heldout_readiness import (
    HeldoutReadinessError,
    build_heldout_readiness,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"


def _load(name: str):
    return json.loads((BASE / name).read_text(encoding="utf-8"))


def test_real_pre_reference_replay_yields_21_confirmatory_relation_mappings():
    result = build_heldout_readiness(
        _load("code_review_heldout_execution_v1.json"),
        _load("code_review_train_gate_v1.json"),
    )
    assert result["reference_values_used"] is False
    assert result["prompt_outputs_used"] is False
    assert result["summary"]["n_confirmatory_programs"] == 12
    assert result["summary"]["n_confirmatory_relation_mappings"] == 21
    assert result["summary"]["confirmatory_relation_mappings_by_level"] == {
        "R1": 8, "R2": 5, "R3": 8
    }
    assert result["summary"]["confirmatory_relation_mappings_by_depth"] == {
        "1": 14, "2": 7
    }


def test_readiness_rejects_reference_bearing_or_unselected_programs():
    execution = _load("code_review_heldout_execution_v1.json")
    gate = _load("code_review_train_gate_v1.json")
    poisoned = copy.deepcopy(execution)
    poisoned["reference_fields_passed_to_worker"] = True
    with pytest.raises(HeldoutReadinessError, match="reference"):
        build_heldout_readiness(poisoned, gate)

    poisoned = copy.deepcopy(execution)
    poisoned["programs"].pop()
    with pytest.raises(HeldoutReadinessError, match="frozen training selection"):
        build_heldout_readiness(poisoned, gate)
