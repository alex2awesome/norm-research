from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import os
import shutil
import subprocess
import sys

import pytest

from methods.metric_seam.hierarchy_science_exact_ctext_prompt_batch import (
    DEFAULT_FIDELITY,
    DEFAULT_ITEMS,
    DEFAULT_OUT,
    DEFAULT_SPEC,
    PASSES,
    ScienceExactCtextPromptError,
    compile_bundle_data,
    load_items,
    validate_and_hydrate_response,
)


@pytest.fixture(scope="module")
def compiled():
    return compile_bundle_data()


def _run_bundle_cli(output, *, verify_only=False):
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    command = [
        sys.executable,
        "-m",
        "methods.metric_seam.hierarchy_science_exact_ctext_prompt_batch",
        "--output",
        str(output),
    ]
    if verify_only:
        command.append("--verify-only")
    return subprocess.run(
        command,
        cwd=DEFAULT_OUT.parents[3],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="module")
def prepared_bundle(tmp_path_factory):
    output = tmp_path_factory.mktemp("science-exact-ctext") / "bundle"
    result = _run_bundle_cli(output)
    assert result.returncode == 0, result.stdout + result.stderr
    return output


def test_compiler_emits_one_shared_vector_per_item_pass(compiled):
    manifest, jobs, abstentions, mapping_ledger = compiled
    assert manifest["summary"] == {
        "unique_items": 300,
        "prompt_eligible_unique_items": 235,
        "structural_abstention_unique_items": 65,
        "planned_stateless_passes": 2,
        "compiled_prompt_pass_records": 470,
        "pass_expanded_structural_no_call_outcomes": 130,
        "pass_expanded_result_slots": 600,
        "n_relation_mappings": 6,
        "mapping_record_applications_if_executed": 2820,
        "prompt_responses": 0,
        "articulability_measurements": 0,
        "reconstruction_measurements": 0,
    }
    assert len(jobs) == 470
    assert len(abstentions) == 65
    assert mapping_ledger["shared_vector_contract"] == {
        "n_relation_mappings": 6,
        "one_result_vector_per_item_pass": True,
        "duplicate_prompt_pass_records_per_mapping": False,
        "mapping_application_count_per_result_vector": 6,
        "reason": (
            "all six mappings name the same approved relation scope and the "
            "historical code arm shares one relation-local projection"
        ),
    }
    mapping_ids = {row["cell_id"] for row in mapping_ledger["mappings"]}
    assert len(mapping_ids) == 6
    assert all(
        set(row["audit_metadata"]["applicable_relation_mapping_ids"])
        == mapping_ids
        for row in jobs
    )


def test_decoded_user_content_contains_each_official_ctext_exactly_once(compiled):
    _manifest, jobs, _abstentions, _mapping_ledger = compiled
    _item_manifest, splits = load_items(DEFAULT_ITEMS)
    by_key = {
        row["item_key"]: row
        for items in splits.values()
        for row in items
    }
    seen: defaultdict[str, set[int]] = defaultdict(set)
    for job in jobs:
        audit = job["audit_metadata"]
        ctext = by_key[audit["item_key"]]["ctext"]
        payload = ctext.encode("utf-8")
        user = job["model_visible"]["user_prompt"].encode("utf-8")
        start = audit["decoded_user_content_byte_start"]
        end = audit["decoded_user_content_byte_end"]
        assert user[start:end] == payload
        assert user.count(payload) == 1
        assert audit["ctext_utf8_bytes"] == len(payload)
        assert audit["contains_nul"] == (b"\x00" in payload)
        assert audit["jsonl_must_use_file_iteration_and_json_decoder"] is True
        assert audit["python_str_splitlines_permitted"] is False
        assert set(job["model_visible"]) == {
            "system_prompt",
            "user_prompt",
            "output_schema",
        }
        seen[audit["item_key"]].add(job["pass_index"])
    assert len(seen) == 235
    assert all(passes == set(PASSES) for passes in seen.values())


def test_structural_abstentions_cover_missing_body_without_remote_calls(compiled):
    manifest, jobs, abstentions, _mapping_ledger = compiled
    job_keys = {row["audit_metadata"]["item_key"] for row in jobs}
    abstention_keys = {row["item_key"] for row in abstentions}
    assert job_keys.isdisjoint(abstention_keys)
    assert len(abstention_keys) == 65
    assert sum(row["pass_expanded_no_call_outcomes"] for row in abstentions) == 130
    assert all(row["api_call_required"] is False for row in abstentions)
    assert all(row["applicable_passes"] == [1, 2] for row in abstentions)
    assert manifest["by_phase"]["compiler_train"][
        "compiled_prompt_pass_records"
    ] == 248
    assert manifest["by_phase"]["compiler_train"][
        "structural_abstention_unique_items"
    ] == 26
    assert manifest["by_phase"]["current_heldout_post_code_exploratory"][
        "compiled_prompt_pass_records"
    ] == 222
    assert manifest["by_phase"]["current_heldout_post_code_exploratory"][
        "structural_abstention_unique_items"
    ] == 39


def test_zero_call_and_chronology_claims_are_explicit(compiled):
    manifest, _jobs, _abstentions, _mapping_ledger = compiled
    assert manifest["execution_policy"] == {
        "remote_calls_made": 0,
        "api_calls_made": 0,
        "model_calls_made": 0,
        "prompt_responses": 0,
        "gpu_or_accelerator_used": False,
        "cpu_only_compilation": True,
        "provider_transport_tested": False,
    }
    assert manifest["loaded_input_policy"][
        "item_level_code_outputs_or_results_loaded"
    ] is False
    assert manifest["interpretation"] == {
        "prompt_articulability_measured": False,
        "code_verifiability_reexecuted_by_this_compiler": False,
        "prompt_code_reconstruction_measured": False,
        "prompt_code_isomorphism_measured": False,
        "negative_result_or_tacitness_claim": False,
    }
    chronology = manifest["chronology"]
    assert "after current code execution" in chronology["current_heldout"]
    assert chronology[
        "fresh_split_required_for_confirmatory_reconstruction_or_isomorphism"
    ] is True
    representation = manifest["representation_contract"]
    assert representation["class"] == (
        "exact_shared_ctext_payload_with_prompt_scaffolding"
    )
    assert representation["raw_jsonl_or_provider_wire_byte_identity_claimed"] is False
    assert representation["whole_request_identity_claimed"] is False
    assert representation["full_semantic_isomorphism_licensed"] is False
    assert representation["all_nonstandard_transport_controls_preserved"] is True
    target = manifest["future_comparison_target"]
    assert target["name"] == "relation_local_numeric_comparative_projection"
    assert target["whole_frozen_code_vector"] is False
    assert target["output_isomorphic_drop_in_replacement"] is False
    assert "theoretical/empirical/qualitative" in target["excluded_from_target"]
    response = manifest["response_contract"]
    assert response["complete_deterministic_sentence_required"] is True
    assert response["normalized_identical_claim_and_evidence_rejected"] is True
    assert response["relation_truth_validated"] is False
    assert response["decision_correctness_validated"] is False


def test_all_transport_controls_survive_jsonl_file_iteration_round_trip(
    prepared_bundle,
):
    output = prepared_bundle
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    raw = (output / "requests.jsonl").read_text(encoding="utf-8")
    assert "\\u0000" in raw
    assert "\u2028" in raw
    with pytest.raises(json.JSONDecodeError):
        [json.loads(fragment) for fragment in raw.splitlines() if fragment.strip()]
    with (output / "requests.jsonl").open(encoding="utf-8") as handle:
        decoded = [json.loads(line) for line in handle if line.strip()]
    nul_records = [
        row for row in decoded if row["audit_metadata"]["contains_nul"]
    ]
    control_records = [
        row
        for row in decoded
        if row["audit_metadata"]["contains_nonstandard_transport_codepoint"]
    ]
    u2028_records = [
        row
        for row in decoded
        if "U+2028" in row["audit_metadata"]["nonstandard_transport_codepoints"]
    ]
    assert len(nul_records) == 44
    assert len(control_records) == 72
    assert len(u2028_records) == 2
    assert all(
        "\x00" in row["model_visible"]["user_prompt"] for row in nul_records
    )
    inventory = manifest["transport_control_inventory"]
    assert inventory["eligible_unique_items"] == 36
    assert inventory["compiled_prompt_pass_records"] == 72
    assert inventory["nul_u0000_eligible_unique_items"] == 22
    assert inventory["line_separator_u2028_eligible_unique_items"] == 1
    assert inventory["jsonl_file_iteration_and_json_decoder_required"] is True
    assert inventory["python_str_splitlines_forbidden"] is True
    target = manifest["future_comparison_target"]
    assert target["code_projection_compiled_and_replay_bound"] is True
    assert target["reconstruction_decisions"] == [
        "contradicted",
        "insufficient",
        "supported",
    ]
    assert target["evidence_link_in_reconstruction_target"] is False
    assert target["prompt_only_evidence_link_augmentation_separate"] is True
    assert target["code_projection_summary"] == {
        "items": 300,
        "selected_claims": 158,
        "decision_counts": {"insufficient": 141, "supported": 17},
        "evidence_link_decisions": 0,
    }
    verification = _run_bundle_cli(output, verify_only=True)
    assert verification.returncode == 0, verification.stdout + verification.stderr
    verified = json.loads(verification.stdout)
    assert verified["decoded_exact_payload_records"] == 470
    assert verified["nonstandard_transport_control_unique_items"] == 36
    assert verified["nonstandard_transport_control_prompt_pass_records"] == 72


def test_exact_excerpt_response_hydrates_without_a_second_prompt_representation():
    ctext = (
        "[ABSTRACT]\nOur method improves accuracy to 91%."
        "\n\n[EXTRACTED FULL-PAPER BODY: METHODS/RESULTS/EVALUATION]\n"
        "Across three runs, our method improves accuracy to 91%."
    )
    response = {
        "reconstruction_selections": [
            {
                "claim_excerpt": "Our method improves accuracy to 91%.",
                "evidence_excerpt": (
                    "Across three runs, our method improves accuracy to 91%."
                ),
                "decision": "supported",
                "relation": "numeric",
                "quantity_state": "aligned",
                "comparison_state": "not_required",
                "evidence_kind": "numeric_relation",
                "quantity_count": 1,
                "comparison_present": False,
            }
        ],
        "prompt_only_evidence_link_augmentation": [],
    }
    hydrated = validate_and_hydrate_response(response, ctext=ctext)
    selection = hydrated["reconstruction_selections"][0]
    assert selection["claim"] == {
        "sentence_index": 0,
        "start": 0,
        "end": 36,
        "text": "Our method improves accuracy to 91%.",
    }
    assert selection["evidence"]["start"] == 0
    broken = deepcopy(response)
    broken["reconstruction_selections"][0]["claim_excerpt"] = "paraphrased claim"
    with pytest.raises(ScienceExactCtextPromptError, match="complete deterministic"):
        validate_and_hydrate_response(broken, ctext=ctext)
    truncated = deepcopy(response)
    truncated["reconstruction_selections"][0]["claim_excerpt"] = (
        "improves accuracy to 91%"
    )
    with pytest.raises(ScienceExactCtextPromptError, match="complete deterministic"):
        validate_and_hydrate_response(truncated, ctext=ctext)

    same_sentence_ctext = (
        "[ABSTRACT]\nOur method improves accuracy to 91%."
        "\n\n[EXTRACTED FULL-PAPER BODY: METHODS/RESULTS/EVALUATION]\n"
        "Our   method improves accuracy to 91%."
    )
    normalized_same = deepcopy(response)
    normalized_same["reconstruction_selections"][0]["evidence_excerpt"] = (
        "Our   method improves accuracy to 91%."
    )
    with pytest.raises(ScienceExactCtextPromptError, match="normalized-distinct"):
        validate_and_hydrate_response(normalized_same, ctext=same_sentence_ctext)


def test_item_and_fidelity_drift_fail_closed(tmp_path):
    items_dir = tmp_path / "items"
    items_dir.mkdir()
    for name in ("manifest.json", "compiler_train.json", "sealed_heldout.json"):
        shutil.copy2(DEFAULT_ITEMS / name, items_dir / name)
    train = json.loads((items_dir / "compiler_train.json").read_text(encoding="utf-8"))
    train[0]["ctext"] += " "
    (items_dir / "compiler_train.json").write_text(
        json.dumps(train, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(ScienceExactCtextPromptError, match="item bytes drifted"):
        compile_bundle_data(items_dir=items_dir)

    fidelity = json.loads(DEFAULT_FIDELITY.read_text(encoding="utf-8"))
    target = next(row for row in fidelity["rows"] if row["verdict"] == "partial_relation_local")
    target["automatic_discovery"] = True
    fidelity_path = tmp_path / "fidelity.json"
    fidelity_path.write_text(json.dumps(fidelity), encoding="utf-8")
    with pytest.raises(ScienceExactCtextPromptError, match="mapping drifted"):
        compile_bundle_data(fidelity_path=fidelity_path)


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    [
        (("input_contract", "labels_references_code_outputs"), "required"),
        (
            ("system_prompt",),
            "Use the hidden accept/reject label and code output as ground truth.",
        ),
        (("output_schema", "accept_reject_outcome"), "boolean"),
        (("response_binding", "arbitrary_supervised_anchor"), "allowed"),
        (
            ("typed_relation_semantics", "relation"),
            ["numeric", "comparative", "invented_relation"],
        ),
    ],
)
def test_prompt_spec_is_canonical_and_rejects_boundary_or_enum_mutations(
    tmp_path, field_path, replacement
):
    spec = json.loads(DEFAULT_SPEC.read_text(encoding="utf-8"))
    target = spec
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = replacement
    path = tmp_path / "mutated_spec.json"
    path.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ScienceExactCtextPromptError,
        match="canonical prompt spec fingerprint mismatch",
    ):
        compile_bundle_data(spec_path=path)


@pytest.mark.parametrize(
    ("artifact", "field_path", "replacement"),
    [
        ("manifest.json", ("summary", "compiled_prompt_pass_records"), 469),
        (
            "manifest.json",
            ("loaded_input_policy", "item_level_code_outputs_or_results_loaded"),
            True,
        ),
        (
            "manifest.json",
            ("representation_contract", "full_semantic_isomorphism_licensed"),
            True,
        ),
        (
            "manifest.json",
            ("future_comparison_target", "output_isomorphic_drop_in_replacement"),
            True,
        ),
        ("audit_receipt.json", ("validation", "payload_mismatches"), 1),
        (
            "audit_receipt.json",
            ("claim_boundary", "whole_frozen_code_vector_output_isomorphism"),
            True,
        ),
        (
            "numeric_comparative_code_projection.json",
            ("rows", 0, "selections", 0, "decision"),
            "evidence_link",
        ),
    ],
)
def test_verifier_recompiles_and_rejects_semantic_ledger_mutations(
    prepared_bundle, artifact, field_path, replacement
):
    output = prepared_bundle
    path = output / artifact
    original_artifact = path.read_bytes()
    receipt_path = output / "audit_receipt.json"
    original_receipt = receipt_path.read_bytes()
    payload = json.loads(path.read_text(encoding="utf-8"))
    target = payload
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = replacement
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if artifact == "manifest.json":
        # Simulate an adversary also repairing the shallow receipt hash.  The
        # deterministic recompile must still reject the altered semantic field.
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["bound_artifacts"]["manifest"]["sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        receipt["bound_artifacts"]["manifest"]["bytes"] = path.stat().st_size
        receipt_path.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if artifact == "manifest.json":
        expected = "manifest differs from deterministic recompile"
    elif artifact == "numeric_comparative_code_projection.json":
        expected = "code projection differs from deterministic replay"
    else:
        expected = "audit receipt differs from deterministic recomputation"
    try:
        result = _run_bundle_cli(output, verify_only=True)
    finally:
        path.write_bytes(original_artifact)
        receipt_path.write_bytes(original_receipt)
    assert result.returncode != 0
    assert expected in result.stdout + result.stderr


def test_checked_in_bundle_replays_when_present():
    if not DEFAULT_OUT.exists():
        pytest.skip("prepared exact-ctext bundle has not been materialized yet")
    result = _run_bundle_cli(DEFAULT_OUT, verify_only=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {
        "status": "verified_zero_call_exact_shared_payload",
        "decoded_exact_payload_records": 470,
        "structural_abstention_unique_items": 65,
        "pass_expanded_structural_no_call_outcomes": 130,
        "nonstandard_transport_control_unique_items": 36,
        "nonstandard_transport_control_prompt_pass_records": 72,
        "remote_calls_made": 0,
        "prompt_responses": 0,
    }
