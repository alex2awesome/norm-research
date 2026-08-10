from __future__ import annotations

import copy
import gzip
import itertools
import json
from pathlib import Path

import pytest

from methods.metric_seam.hierarchy_math_lclamp_runner import (
    CANONICAL_ITEMS_ROOT,
    load_bound_items,
)
from methods.metric_seam.hierarchy_math_prompt_batch import (
    FIXED_HELDOUT_BANK_ARMS,
    FORM_IDS,
    IMPLEMENTATION_ARM_ID,
    MathPromptBatchError,
    _bank_fingerprint,
    _write_jobs,
    compile_prompt_batch,
    validate_prompt_response,
)
from methods.metric_seam.hierarchy_prompt_batch import PromptBatchError


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"
AUDIT_PATH = BASE / "math_stackexchange_construct_fidelity_merged_v1.json"
BANK_PATH = BASE / "prompt_arm_bank_v3.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def real_inputs():
    return {"audit": _load(AUDIT_PATH), "bank": _load(BANK_PATH)}


@pytest.fixture(scope="module")
def train_batch(real_inputs):
    items, path = load_bound_items(CANONICAL_ITEMS_ROOT, "compiler_train")
    return compile_prompt_batch(
        real_inputs["audit"],
        real_inputs["bank"],
        items,
        phase="compiler_train",
        audit_source=str(AUDIT_PATH),
        bank_source=str(BANK_PATH),
        items_source=str(path),
    )


@pytest.fixture(scope="module")
def heldout_batch(real_inputs):
    items, path = load_bound_items(CANONICAL_ITEMS_ROOT, "heldout_pre_reference")
    return compile_prompt_batch(
        real_inputs["audit"],
        real_inputs["bank"],
        items,
        phase="heldout_pre_reference",
        audit_source=str(AUDIT_PATH),
        bank_source=str(BANK_PATH),
        items_source=str(path),
    )


def test_real_train_batch_covers_full_bank_for_final_33_without_scoring(train_batch):
    manifest = train_batch.manifest
    summary = manifest["summary"]
    assert manifest["status"] == "compiled_unscored"
    assert manifest["phase"] == "compiler_train"
    assert summary == {
        "n_cells": 33,
        "n_cells_by_level": {"R1": 12, "R2": 6, "R3": 15},
        "n_cells_by_audited_depth": {"1": 10, "2": 23},
        "n_unique_program_vectors": 16,
        "n_items": 150,
        "n_passes": 2,
        "n_bank_arms_across_selected_cells": 951,
        "n_source_articulation_arms_across_selected_cells": 306,
        "n_control_arms_across_selected_cells": 612,
        "n_bank_prompt_specs_in_phase": 951,
        "n_post_code_disclosure_specs_in_phase": 33,
        "n_prompt_specs_in_phase": 984,
        "n_jobs": 295200,
        "n_prompt_responses": 0,
        "n_reconstruction_estimates": 0,
        "n_isomorphism_adjudications": 0,
    }
    assert manifest["construct_scope"].startswith("33 partial")
    assert manifest["jobs_artifact"] is None
    assert set(manifest["forbidden_inputs"].values()) == {False}
    assert manifest["prompt_judgment_role"].endswith("never an external truth label")
    assert manifest["design_provenance"]["compiler_runtime_score_blind"] is True
    assert manifest["design_provenance"][
        "investigator_level_score_blindness_claimed"
    ] is False


def test_fixed_heldout_batch_is_predeclared_not_calibration_selected(heldout_batch):
    manifest = heldout_batch.manifest
    summary = manifest["summary"]
    assert manifest["phase"] == "heldout_pre_reference"
    assert manifest["batch_role"] == "fixed_predeclared_heldout_confirmation"
    assert summary["n_bank_prompt_specs_in_phase"] == 33 * 4 * 3
    assert summary["n_post_code_disclosure_specs_in_phase"] == 33
    assert summary["n_prompt_specs_in_phase"] == 429
    assert summary["n_jobs"] == 128700
    assert manifest["arm_bank_contract"]["fixed_heldout_bank_arms"] == list(
        FIXED_HELDOUT_BANK_ARMS
    )
    assert manifest["arm_bank_contract"]["heldout_fixed_forms"] == list(FORM_IDS)
    assert "not compiled here" in manifest["phase_design"][
        "future_calibrated_release"
    ]


def test_jobs_expose_only_request_to_model_and_embed_exact_shared_ctext(train_batch):
    first, second = list(itertools.islice(train_batch.iter_jobs(), 2))
    assert set(first) == {
        "request_id",
        "request",
        "executor_metadata",
        "audit_metadata",
    }
    assert set(first["request"]) == {"system", "user"}
    assert "ctext" not in first
    assert "ctext" not in first["executor_metadata"]
    assert "ctext" not in first["audit_metadata"]
    ctext = train_batch.items[0]["ctext"]
    assert first["request"]["user"].count(ctext) == 1
    assert "UNTRUSTED_MATH_DOCUMENT" in first["request"]["system"]
    assert "UNTRUSTED_MATH_DOCUMENT" in first["request"]["user"]
    assert first["audit_metadata"]["ctext_sha256"]
    assert first["audit_metadata"]["arm_role"] == "source_name_baseline"
    assert first["request"] == second["request"]
    assert first["audit_metadata"]["pass_id"] == 1
    assert second["audit_metadata"]["pass_id"] == 2
    assert first["executor_metadata"]["sampling_seed"] < 1_000_000_000
    assert second["executor_metadata"]["sampling_seed"] >= 1_000_000_000
    assert first["executor_metadata"]["stateless_separate_call"] is True
    assert first["executor_metadata"]["cache_and_context_reuse_forbidden"] is True


def test_post_code_disclosure_is_visibly_separate_from_source_articulation(train_batch):
    disclosure = next(
        job
        for job in train_batch.iter_jobs()
        if job["audit_metadata"]["arm_id"] == IMPLEMENTATION_ARM_ID
    )
    assert disclosure["audit_metadata"]["arm_role"] == (
        "post_code_relation_disclosure"
    )
    prompt = disclosure["request"]["user"]
    assert "not independent source articulation" in prompt
    assert "Audited executable subrelations" in prompt
    assert "Explicitly unimplemented residual" in prompt
    first_source = next(train_batch.iter_jobs())
    assert "POST-CODE" not in first_source["request"]["user"]
    assert first_source["audit_metadata"]["arm_provenance"] == "construct_name"


def test_fixed_heldout_first_item_has_exact_bank_form_orbits_and_disclosure(
    heldout_batch,
):
    first_cell = heldout_batch.manifest["cells"][0]["cell_id"]
    first_item = heldout_batch.items[0]["item_key"]
    observed = set()
    for job in heldout_batch.iter_jobs():
        metadata = job["audit_metadata"]
        if metadata["cell_id"] != first_cell:
            break
        if metadata["item_key"] == first_item:
            observed.add(
                (metadata["arm_id"], metadata["form_id"], metadata["pass_id"])
            )
    expected = {
        (arm_id, form_id, pass_id)
        for arm_id in FIXED_HELDOUT_BANK_ARMS
        for form_id in FORM_IDS
        for pass_id in (1, 2)
    }
    expected |= {(IMPLEMENTATION_ARM_ID, "canonical", pass_id) for pass_id in (1, 2)}
    assert observed == expected


def test_train_selection_contract_is_unsupervised_and_heldout_blind(train_batch):
    selection = train_batch.manifest["train_only_selection_preregistration"]
    assert selection["heldout_information_permitted"] is False
    assert selection["external_supervised_anchor_permitted"] is False
    assert selection["minimum_common_support"] == 30
    assert selection["matched_controls_required"] == [
        "wrong_construct",
        "inert_length",
    ]
    assert "retain one" in selection["null_cell_rule"]
    assert "absolute Spearman" in selection["source_statistic"]
    analysis = train_batch.manifest["heldout_analysis_preregistration"]
    assert "16 exact executable-vector clusters" in analysis["clustered_unit"]
    assert "cannot establish whole-construct isomorphism" in analysis["claim_limit"]
    assert analysis["primary_reconstruction"].startswith("raw signed Spearman")
    assert "cannot rescue" in analysis["isomorphism_polarity_gate"]
    assert "require sign stability" in analysis["form_robustness"]


def test_response_validator_preserves_three_states_without_score_coercion():
    assert validate_prompt_response(
        {
            "measurement_status": "not_applicable",
            "evidence": [],
            "rationale": "No observable occasion.",
        }
    )["measurement_status"] == "not_applicable"
    assert validate_prompt_response(
        {
            "measurement_status": "applicable_abstain",
            "evidence": ["Question: ..."],
            "rationale": "The scalar is not defensible.",
        }
    )["measurement_status"] == "applicable_abstain"
    assert validate_prompt_response(
        {
            "measurement_status": "scored",
            "score": 0.4,
            "evidence": ["Answer: ..."],
            "rationale": "Direct evidence.",
        }
    )["score"] == 0.4
    with pytest.raises(PromptBatchError):
        validate_prompt_response(
            {
                "measurement_status": "scored",
                "score": float("nan"),
                "evidence": [],
                "rationale": "invalid",
            }
        )


def test_official_phase_items_are_required_byte_for_byte(real_inputs):
    train, path = load_bound_items(CANONICAL_ITEMS_ROOT, "compiler_train")
    with pytest.raises(MathPromptBatchError, match="official L-clamp ctext bytes"):
        compile_prompt_batch(
            real_inputs["audit"],
            real_inputs["bank"],
            train[:-1],
            phase="compiler_train",
            items_source=str(path),
        )
    contaminated = copy.deepcopy(train)
    contaminated[0]["outcome"] = 1
    with pytest.raises(MathPromptBatchError, match="official L-clamp ctext bytes"):
        compile_prompt_batch(
            real_inputs["audit"],
            real_inputs["bank"],
            contaminated,
            phase="compiler_train",
            items_source=str(path),
        )
    with pytest.raises(MathPromptBatchError, match="items_source is not"):
        compile_prompt_batch(
            real_inputs["audit"],
            real_inputs["bank"],
            train,
            phase="compiler_train",
            items_source=str(CANONICAL_ITEMS_ROOT / "sealed_heldout.json"),
        )


def test_arm_bank_panel_and_control_tampering_fail_closed(real_inputs):
    bad_digest = dict(real_inputs["bank"])
    bad_digest["bank_content_sha256"] = "0" * 64
    with pytest.raises(MathPromptBatchError, match="content identity"):
        compile_prompt_batch(
            real_inputs["audit"],
            bad_digest,
            load_bound_items(CANONICAL_ITEMS_ROOT, "compiler_train")[0],
            phase="compiler_train",
        )

    bad_control = dict(real_inputs["bank"])
    bad_control["cells"] = list(bad_control["cells"])
    target_index = next(
        index
        for index, cell in enumerate(bad_control["cells"])
        if cell["task"] == "math-stackexchange"
    )
    cell = dict(bad_control["cells"][target_index])
    cell["arms"] = list(cell["arms"])
    arm_index = next(
        index
        for index, arm in enumerate(cell["arms"])
        if arm["id"] == "control_wrong_definition"
    )
    arm = dict(cell["arms"][arm_index])
    arm["control_for"] = "source_rules"
    cell["arms"][arm_index] = arm
    bad_control["cells"][target_index] = cell
    bad_control["bank_content_sha256"] = _bank_fingerprint(bad_control)
    with pytest.raises(MathPromptBatchError, match="missing exact wrong control"):
        compile_prompt_batch(
            real_inputs["audit"],
            bad_control,
            load_bound_items(CANONICAL_ITEMS_ROOT, "compiler_train")[0],
            phase="compiler_train",
        )


def test_final_audit_scope_and_summary_are_not_hand_editable(real_inputs):
    audit = copy.deepcopy(real_inputs["audit"])
    row = next(row for row in audit["rows"] if row["eligible_for_relation_local_execution"])
    row["verdict"] = "exact"
    row["scope"] = "whole_construct"
    with pytest.raises(MathPromptBatchError, match="summary does not match"):
        compile_prompt_batch(
            audit,
            real_inputs["bank"],
            load_bound_items(CANONICAL_ITEMS_ROOT, "compiler_train")[0],
            phase="compiler_train",
        )


def test_job_writer_is_exclusive_count_checked_and_gzip_readable(tmp_path):
    path = tmp_path / "jobs.jsonl.gz"
    rows = ({"request_id": str(index)} for index in range(3))
    assert _write_jobs(path, rows, 3) == 3
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        assert [json.loads(line)["request_id"] for line in handle] == ["0", "1", "2"]
    with pytest.raises(FileExistsError):
        _write_jobs(path, [], 0)
    mismatch = tmp_path / "mismatch.jsonl.gz"
    with pytest.raises(MathPromptBatchError, match="job count drift"):
        _write_jobs(mismatch, [{"request_id": "only"}], 2)
    assert not mismatch.exists()


def test_job_writer_removes_partial_file_on_iterator_failure_and_is_path_independent(
    tmp_path,
):
    def failing_rows():
        yield {"request_id": "written-before-failure"}
        raise RuntimeError("synthetic iterator failure")

    partial = tmp_path / "partial.jsonl.gz"
    with pytest.raises(RuntimeError, match="synthetic iterator failure"):
        _write_jobs(partial, failing_rows(), 2)
    assert not partial.exists()

    left = tmp_path / "left.jsonl.gz"
    right = tmp_path / "different-name.jsonl.gz"
    payload = [{"request_id": "stable", "request": {"user": "same bytes"}}]
    _write_jobs(left, payload, 1)
    _write_jobs(right, payload, 1)
    assert left.read_bytes() == right.read_bytes()
