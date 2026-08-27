from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from methods.metric_seam.hierarchy_code_runner import CANONICAL_ITEMS_ROOT, load_bound_items
from methods.metric_seam.hierarchy_prompt_batch import (
    CHANNELS,
    RESPONSE_SCHEMA,
    PromptBatchError,
    compile_prompt_batch,
    validate_prompt_response,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/metric_seam_pilot/hierarchy_r123"


def _load(name: str):
    return json.loads((BASE / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def inputs():
    items, items_path = load_bound_items(CANONICAL_ITEMS_ROOT, "heldout_pre_reference")
    return {
        "audit": _load("code_review_construct_fidelity_v2.json"),
        "readiness": _load("code_review_heldout_readiness_v1.json"),
        "panel": _load("panel_v3.json"),
        "items": items,
        "items_path": items_path,
    }


@pytest.fixture(scope="module")
def compiled(inputs):
    return compile_prompt_batch(
        inputs["audit"], inputs["readiness"], inputs["panel"], inputs["items"]
    )


def test_real_v2_batch_is_unscored_source_separated_and_vector_clustered(compiled):
    manifest, jobs = compiled
    assert manifest["n_cells"] == 21
    assert manifest["n_unique_program_vectors"] == 12
    assert manifest["n_items_per_cell"] == 125
    assert manifest["n_channels"] == 4
    assert manifest["n_jobs"] == 21000
    assert manifest["expected_n_jobs"] == len(jobs)
    assert manifest["candidate_scores_read_or_embedded"] is False
    assert manifest["prompt_outputs_used"] is False
    assert manifest["outcome_labels_used"] is False
    assert manifest["external_ground_truth_used"] is False
    assert manifest["panel_content_sha256"] == (
        "ed61147ac70a93e88b57e7807608c31171430928cf2983036a5de3f3bfecc206"
    )
    assert set(manifest["channels"]) == set(CHANNELS)
    assert "relation_matched" not in manifest["channels"]
    assert len(manifest["source_only_subrelation_selection"]["rows"]) == 21
    assert (
        manifest["source_only_subrelation_selection"][
            "code_or_construct_fidelity_fields_used"
        ]
        is False
    )


def test_model_projection_has_only_prompts_and_ctext_is_audit_metadata(compiled):
    manifest, jobs = compiled
    assert manifest["model_input_projection_contract"]["allowed_request_keys"] == [
        "system", "user"
    ]
    first = jobs[0]
    assert set(first) == {
        "request_id", "request", "executor_metadata", "audit_metadata"
    }
    assert set(first["request"]) == {"system", "user"}
    assert "ctext" not in first
    assert "ctext" not in first["executor_metadata"]
    assert first["audit_metadata"]["ctext"] in first["request"]["user"]
    assert first["audit_metadata"]["ctext_sha256"]
    assert "UNTRUSTED_CODE_REVIEW_DIFF" in first["request"]["system"]
    assert "UNTRUSTED_CODE_REVIEW_DIFF" in first["request"]["user"]
    assert len({job["request_id"] for job in jobs}) == len(jobs)
    assert all(
        not any(key.startswith("candidate_score") for key in job["audit_metadata"])
        for job in jobs
    )


def test_channels_are_honestly_named_and_passes_are_separate_calls(compiled):
    _, jobs = compiled
    first_cell = jobs[0]["audit_metadata"]["cell_id"]
    first_item = jobs[0]["audit_metadata"]["item_key"]
    selected = {
        (job["audit_metadata"]["channel"], job["audit_metadata"]["pass_id"]): job
        for job in jobs
        if job["audit_metadata"]["cell_id"] == first_cell
        and job["audit_metadata"]["item_key"] == first_item
    }
    assert set(channel for channel, _ in selected) == set(CHANNELS)
    for channel in CHANNELS:
        pass1, pass2 = selected[channel, 1], selected[channel, 2]
        assert pass1["request"] == pass2["request"]
        assert pass1["executor_metadata"]["sampling_seed"] < 1_000_000_000
        assert 1_000_000_000 <= pass2["executor_metadata"]["sampling_seed"] < 2_000_000_000
        assert pass1["executor_metadata"]["sampling_seed"] != pass2[
            "executor_metadata"
        ]["sampling_seed"]
        assert pass1["executor_metadata"]["stateless_separate_call"] is True
        assert pass1["executor_metadata"]["cache_and_context_reuse_forbidden"] is True
    assert selected["source_only_whole_construct", 1]["request"]["user"] != selected[
        "source_only_subrelation", 1
    ]["request"]["user"]
    assert selected["source_only_subrelation", 1]["request"]["user"] != selected[
        "implementation_disclosed", 1
    ]["request"]["user"]
    assert "executable" not in selected["source_only_subrelation", 1]["request"]["user"].lower()
    assert "code program" not in selected["source_only_subrelation", 1]["request"]["user"].lower()


def test_response_contract_has_conditional_score_and_three_statuses(compiled):
    manifest, _ = compiled
    schema = manifest["model_input_projection_contract"]["response_validation"]
    assert schema == RESPONSE_SCHEMA
    assert schema["properties"]["measurement_status"]["enum"] == [
        "not_applicable", "applicable_abstain", "scored"
    ]
    assert schema["allOf"][0]["then"] == {"required": ["score"]}
    assert schema["allOf"][0]["else"] == {"not": {"required": ["score"]}}
    projection = manifest["executor_projection"]
    assert projection["measurement_status_to_applicability"] == {
        "not_applicable": False,
        "applicable_abstain": True,
        "scored": True,
    }


def test_response_validator_preserves_three_states_and_never_coerces_scores():
    assert validate_prompt_response({
        "measurement_status": "not_applicable",
        "evidence": [],
        "rationale": "No observable occasion.",
    })["measurement_status"] == "not_applicable"
    assert validate_prompt_response({
        "measurement_status": "applicable_abstain",
        "evidence": ["def changed():"],
        "rationale": "An occasion exists but a scalar is not defensible.",
    })["measurement_status"] == "applicable_abstain"
    scored = validate_prompt_response({
        "measurement_status": "scored",
        "score": 0.75,
        "evidence": ["return parsed.depth"],
        "rationale": "Direct evidence supports a scalar.",
    })
    assert scored["score"] == 0.75
    for invalid in (
        {
            "measurement_status": "not_applicable",
            "score": 0.0,
            "evidence": [],
            "rationale": "extra score",
        },
        {
            "measurement_status": "scored",
            "evidence": [],
            "rationale": "missing score",
        },
        {
            "measurement_status": "scored",
            "score": float("nan"),
            "evidence": [],
            "rationale": "nonfinite",
        },
    ):
        with pytest.raises(PromptBatchError):
            validate_prompt_response(invalid)


def test_wrong_relation_controls_differ_by_program_and_remain_within_level(compiled):
    manifest, _ = compiled
    controls = manifest["analysis_preregistration"]["wrong_relation_control"]["rows"]
    assert len(controls) == 21
    assert len({row["cell_id"] for row in controls}) == 21
    assert all(
        row["code_vector_aspect_id"] != row["control_prompt_aspect_id"]
        for row in controls
    )
    assert {row["level"] for row in controls} == {"R1", "R2", "R3"}
    clusters = manifest["clustered_inference"]["vector_clusters"]
    assert len(clusters) == 12
    assert sum(len(cluster["cell_ids"]) for cluster in clusters) == 21


def test_official_heldout_items_are_required_and_outcome_rows_fail_closed(inputs):
    with pytest.raises(PromptBatchError, match="official heldout"):
        compile_prompt_batch(
            inputs["audit"],
            inputs["readiness"],
            inputs["panel"],
            inputs["items"][:-1],
        )
    contaminated = copy.deepcopy(inputs["items"])
    contaminated[0]["outcome"] = 1
    with pytest.raises(PromptBatchError, match="official heldout"):
        compile_prompt_batch(
            inputs["audit"], inputs["readiness"], inputs["panel"], contaminated
        )
    duplicated = copy.deepcopy(inputs["items"])
    duplicated[1]["item_key"] = duplicated[0]["item_key"]
    with pytest.raises(PromptBatchError, match="official heldout"):
        compile_prompt_batch(
            inputs["audit"], inputs["readiness"], inputs["panel"], duplicated
        )


def test_readiness_must_bind_exact_audited_aspect_path_and_source(inputs):
    readiness = copy.deepcopy(inputs["readiness"])
    readiness["confirmatory_programs"][0]["source_sha256"] = "0" * 64
    with pytest.raises(PromptBatchError, match="readiness field drift"):
        compile_prompt_batch(
            inputs["audit"], readiness, inputs["panel"], inputs["items"]
        )

    readiness = copy.deepcopy(inputs["readiness"])
    readiness["confirmatory_programs"][1]["aspect_id"] = readiness[
        "confirmatory_programs"
    ][0]["aspect_id"]
    with pytest.raises(PromptBatchError, match="duplicate/invalid confirmatory"):
        compile_prompt_batch(
            inputs["audit"], readiness, inputs["panel"], inputs["items"]
        )


def test_panel_binding_and_pre_code_subrelation_fail_closed(inputs):
    panel = copy.deepcopy(inputs["panel"])
    selected_id = inputs["readiness"]["confirmatory_programs"][0]["cell_ids"][0]
    for cell in panel["cells"]:
        if cell["id"] == selected_id:
            cell["components"] = []
            break
    # Recompute only the panel's declared identity so the test reaches the source-only gate.
    core = {key: value for key, value in panel.items() if key != "panel_content_sha256"}
    import hashlib

    panel["panel_content_sha256"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    audit = copy.deepcopy(inputs["audit"])
    audit["panel_content_sha256"] = panel["panel_content_sha256"]
    with pytest.raises(PromptBatchError, match="no pre-code hierarchy component"):
        compile_prompt_batch(audit, inputs["readiness"], panel, inputs["items"])


def test_manifest_preregisters_common_support_and_no_isomorphism_claim(compiled):
    manifest, _ = compiled
    analysis = manifest["analysis_preregistration"]
    assert "n>=30" in analysis["confirmatory_support_gate"]
    assert "identical intersection" in analysis["channel_contrast_support"]
    assert "Reconstruction agreement alone is insufficient" in analysis[
        "isomorphism_adjudication"
    ]
    assert manifest["status"] == "compiled_unscored"
    assert manifest["isomorphism"].startswith("not available")
    # v3 shipped with no ceiling arm, which left every rho unanchored: a null could
    # not be separated from disclosure loss.  The ceiling arm is now compiled.
    assert manifest["omitted_channels"] == {}
    ceiling = manifest["ceiling_arm"]
    assert ceiling["channel"] == "full_executable_contract"
    assert "upper-bounds" in ceiling["reads_as"]
    assert "tacitness" in ceiling["not_a_claim_of"]
    scope = manifest["scope_statements"]
    assert scope["selected_construct_fidelity_verdict_counts"] == {"partial": 21}
    assert "cannot establish whole-construct isomorphism" in scope[
        "whole_construct_limit"
    ]
    assert "compiled post hoc" in scope["source_subrelation_timing"]
    assert "not a historically preregistered selection" in scope[
        "source_subrelation_timing"
    ]
    assert "applicability, polarity, abstention, or aggregation" in scope[
        "implementation_disclosure_limit"
    ]
    assert "reconstruction mismatch" in scope["implementation_disclosure_limit"]
