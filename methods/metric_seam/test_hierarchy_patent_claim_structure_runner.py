from __future__ import annotations

import pytest

from methods.metric_seam.hierarchy_patent_claim_structure_runner import (
    PatentExecutionError,
    execute_split,
    validate_manifest,
)


ITEMS = [
    {
        "item_key": "train_0001",
        "ctext": "ABSTRACT:\nA valve.\nCLAIMS:\n1. A system comprising a valve.",
    },
    {
        "item_key": "train_0002",
        "ctext": (
            "ABSTRACT:\nA sensor.\nCLAIMS:\n"
            "1. A system comprising a sensor.\n"
            "2. The system of claim 1, wherein the sensor is configured to sample at 10 Hz."
        ),
    },
]


def test_executes_text_only_train_split_and_summarizes_relation_coverage() -> None:
    result = execute_split(ITEMS, phase="compiler_train")
    assert result["summary"]["status_counts"] == {"measured": 2}
    assert result["summary"]["failure_types"] == {}
    dependency = result["summary"]["relation_measurement"][
        "claim_dependency_well_formedness"
    ]
    assert dependency == {
        "n_measured": 1,
        "n_abstained": 1,
        "minimum": 1.0,
        "maximum": 1.0,
        "nonconstant": False,
    }
    assert result["design"]["outcome_or_reference_values_loaded"] is False


def test_rejects_extra_fields_and_wrong_split_aliases() -> None:
    with pytest.raises(PatentExecutionError, match="exactly"):
        execute_split(
            [{"item_key": "train_0001", "ctext": "x", "judgement": 1}],
            phase="compiler_train",
        )
    with pytest.raises(PatentExecutionError, match="split key"):
        execute_split(
            [{"item_key": "heldout_0001", "ctext": "x"}],
            phase="compiler_train",
        )


def test_heldout_phase_accepts_only_heldout_opaque_aliases() -> None:
    heldout = [{"item_key": "heldout_0001", "ctext": ITEMS[0]["ctext"]}]
    result = execute_split(heldout, phase="heldout_pre_reference")
    assert result["phase"] == "heldout_pre_reference"
    assert result["rows"][0]["item_key"] == "heldout_0001"


def test_manifest_binds_same_ctext_representation_and_marks_cap() -> None:
    manifest = {
        "schema": "metric-seam.hierarchy-shared-items.v1",
        "task": "patents",
        "representation": {
            "field": "ctext",
            "same_bytes_required_for_prompt_and_code": True,
            "max_chars": len(ITEMS[0]["ctext"]),
        },
        "selection": {"train_n": 1, "heldout_n": 1},
        "policy": {"outcome_columns_emitted": False, "external_supervision_used": False},
    }
    max_chars = validate_manifest(manifest, ITEMS[:1], phase="compiler_train")
    result = execute_split(
        ITEMS[:1], phase="compiler_train", representation_max_chars=max_chars
    )
    assert result["summary"]["items_at_declared_character_cap"] == 1
    assert result["summary"]["status_counts"] == {
        "measured_with_possible_truncation": 1
    }
    assert result["rows"][0]["relation_applicability"] == {
        "finite_witnesses_replayable_on_presented_bytes": True,
        "absence_or_whole_claim_set_inference_permitted": False,
        "train_gate_scope": "finite_witnesses_only",
    }
    assert result["design"]["absence_certificate_permitted"] is False
    assert result["design"]["finite_local_counter_witness_permitted"] is True


def test_manifest_rejects_wrong_task_or_supervised_policy() -> None:
    manifest = {
        "schema": "metric-seam.hierarchy-shared-items.v1",
        "task": "peer-review",
        "representation": {
            "field": "ctext",
            "same_bytes_required_for_prompt_and_code": True,
            "max_chars": 4000,
        },
        "selection": {"train_n": 2},
        "policy": {"outcome_columns_emitted": False, "external_supervision_used": False},
    }
    with pytest.raises(PatentExecutionError, match="not for patents"):
        validate_manifest(manifest, ITEMS, phase="compiler_train")
    manifest["task"] = "patents"
    manifest["policy"]["external_supervision_used"] = True
    with pytest.raises(PatentExecutionError, match="outcome-blind"):
        validate_manifest(manifest, ITEMS, phase="compiler_train")
