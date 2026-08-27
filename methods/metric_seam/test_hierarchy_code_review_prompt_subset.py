from __future__ import annotations

import json
from pathlib import Path

import pytest

from methods.metric_seam.hierarchy_code_review_prompt_subset import (
    PromptSubsetError,
    filter_jobs,
    validate_prompt_subset,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"


def _load(name: str) -> dict:
    return json.loads((BASE / name).read_text(encoding="utf-8"))


def test_v3_prompt_manifest_has_corrected_scope_and_no_outputs() -> None:
    manifest = _load("code_review_reconstruction_prompt_manifest_v3.json")
    assert manifest["schema"] == "metric-seam.hierarchy-reconstruction-prompt-batch.v3"
    assert manifest["status"] == "compiled_unscored_static_cross_audit_filtered"
    assert manifest["n_cells"] == 18
    assert manifest["n_unique_program_vectors"] == 10
    assert manifest["n_jobs"] == manifest["expected_n_jobs"] == 13500
    assert manifest["scope_statements"]["selected_construct_fidelity_verdict_counts"] == {
        "partial": 18
    }
    assert manifest["candidate_scores_read_or_embedded"] is False
    assert manifest["prompt_outputs_used"] is False
    assert manifest["outcome_labels_used"] is False
    assert manifest["external_ground_truth_used"] is False
    assert manifest["static_cross_audit_filter"]["prompt_execution_performed"] is False


def test_v3_excludes_exact_three_cells_and_refreezes_controls() -> None:
    manifest = _load("code_review_reconstruction_prompt_manifest_v3.json")
    excluded = {
        row["cell_id"]
        for row in manifest["static_cross_audit_filter"]["excluded_mappings"]
    }
    assert excluded == {
        "TB::code-review::general::R1::merged_tree::171::33b7ed9b7e4e601644ef",
        "TB::code-review::general::R2::merged_group::131::43ed2014b9a1669be3ca",
        "TB::code-review::general::R3::grandparent::3::681c2abce3bef33e3781",
    }
    survivors = set(manifest["cell_ids"])
    controls = manifest["analysis_preregistration"]["wrong_relation_control"]
    assert len(controls["rows"]) == 18
    assert len(controls["reassignments_from_v2"]) == 4
    assert {row["cell_id"] for row in controls["rows"]} == survivors
    assert all(row["control_prompt_cell_id"] in survivors for row in controls["rows"])
    assert all(
        row["code_vector_aspect_id"] != row["control_prompt_aspect_id"]
        for row in controls["rows"]
    )


def test_v3_jobs_are_exact_surviving_v2_subsequence() -> None:
    manifest = _load("code_review_reconstruction_prompt_manifest_v3.json")
    survivors = set(manifest["cell_ids"])
    expected = filter_jobs(
        BASE / "code_review_reconstruction_prompt_jobs_v2.jsonl.gz", survivors
    )
    observed = filter_jobs(
        BASE / "code_review_reconstruction_prompt_jobs_v3.jsonl.gz", survivors
    )
    assert observed == expected == manifest["static_cross_audit_filter"]["job_filter_summary"]


def test_exact_subset_guard_rejects_retained_excluded_jobs() -> None:
    manifest = _load("code_review_reconstruction_prompt_manifest_v3.json")
    with pytest.raises(PromptSubsetError, match="out-of-scope cell retained"):
        filter_jobs(
            BASE / "code_review_reconstruction_prompt_jobs_v2.jsonl.gz",
            set(manifest["cell_ids"]),
            reject_non_survivors=True,
        )


def test_v3_prompt_artifacts_validate_against_corrected_funnel() -> None:
    validate_prompt_subset(
        _load("code_review_reconstruction_prompt_manifest_v3.json"),
        _load("code_review_reconstruction_prompt_manifest_v2.json"),
        _load("code_review_corrected_funnel_v1.json"),
        _load("code_review_construct_fidelity_v2.json"),
        BASE / "code_review_reconstruction_prompt_jobs_v2.jsonl.gz",
        BASE / "code_review_reconstruction_prompt_jobs_v3.jsonl.gz",
    )
