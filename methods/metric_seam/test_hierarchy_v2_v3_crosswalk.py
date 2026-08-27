from __future__ import annotations

import hashlib
import json

import pytest

from methods.metric_seam.hierarchy_v2_v3_crosswalk import (
    BASE,
    DEFAULT_OUTPUT,
    CrosswalkError,
    _assert_projection,
    _canonical_bytes,
    compile_from_paths,
)


@pytest.fixture(scope="module")
def compiled_crosswalk():
    return compile_from_paths()


def test_canonical_crosswalk_proves_scientific_content_and_prompt_identity(
    compiled_crosswalk,
):
    panel = compiled_crosswalk["panel_result"]
    bank = compiled_crosswalk["prompt_bank_result"]
    frame = compiled_crosswalk["inference_frame_change"]

    assert panel["n_cells"] == 990
    assert panel["n_task_level_strata"] == 33
    assert all(row["n_cells"] == 30 for row in panel["cells_per_task_level"])
    assert panel["cell_order_identical"] is True
    assert panel["node_order_identical"] is True
    assert panel["all_v2_cell_fields_identical"] is True
    assert panel["construct_scientific_content_unchanged"] is True

    assert bank["n_cells"] == 990
    assert bank["n_semantic_prompt_arms"] == 28_335
    assert bank["cell_order_identical"] is True
    assert bank["semantic_arm_order_identical"] is True
    assert bank["semantic_arm_objects_identical"] is True

    assert frame["v3_generation_label"] == "legacy-expanded-source-action-node-dag-v1"
    assert frame["v3_is_partition"] is False
    assert frame["primary_frame"] == "overlapping native action-node DAG"
    assert compiled_crosswalk["scientific_disposition"] == {
        "metric_selection_rerun_required": False,
        "prompt_generation_rerun_required": False,
        "model_scoring_rerun_required": False,
        "claim": (
            "V3 preserves all 990 selected metric objects and all 28,335 semantic prompt "
            "arms. It changes the analysis metadata and inference frame, not the construct "
            "or prompt content."
        ),
    }


def test_checked_in_crosswalk_is_exact_deterministic_regeneration(compiled_crosswalk):
    checked_in = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert checked_in == compiled_crosswalk
    declared = checked_in["crosswalk_content_sha256"]
    core = {
        key: value
        for key, value in checked_in.items()
        if key != "crosswalk_content_sha256"
    }
    assert hashlib.sha256(_canonical_bytes(core)).hexdigest() == declared
    assert not (BASE / "CURRENT.json").resolve() == DEFAULT_OUTPUT.resolve()


def test_projection_guard_rejects_construct_or_prompt_mutation():
    old = [{"id": "TB::one", "construct": "same", "arms": [{"id": "name"}]}]
    changed_construct = [{
        "id": "TB::one",
        "construct": "different",
        "arms": [{"id": "name"}],
        "dependency_component_id": "d1",
    }]
    with pytest.raises(CrosswalkError, match="scientific/content fields changed"):
        _assert_projection(
            old,
            changed_construct,
            added_fields=("dependency_component_id",),
            identity_field="id",
            label="synthetic",
        )

    changed_prompt = [{
        "id": "TB::one",
        "construct": "same",
        "arms": [{"id": "renamed"}],
        "dependency_component_id": "d1",
    }]
    with pytest.raises(CrosswalkError, match="scientific/content fields changed"):
        _assert_projection(
            old,
            changed_prompt,
            added_fields=("dependency_component_id",),
            identity_field="id",
            label="synthetic",
        )


def test_projection_guard_rejects_unexpected_metadata_field():
    old = [{"id": "TB::one", "construct": "same"}]
    new = [{
        "id": "TB::one",
        "construct": "same",
        "dependency_component_id": "d1",
        "undeclared": "not allowed",
    }]
    with pytest.raises(CrosswalkError, match="unexpected added or missing fields"):
        _assert_projection(
            old,
            new,
            added_fields=("dependency_component_id",),
            identity_field="id",
            label="synthetic",
        )
