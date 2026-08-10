import json
from pathlib import Path

import pytest

from methods.metric_seam.science_claims_v2 import addressed_pipeline as v7
from methods.metric_seam.science_claims_v2 import addressed_pipeline_v8 as v8
from methods.metric_seam.science_claims_v2 import addressed_runner_v8 as runner
from methods.metric_seam.science_claims_v2 import evaluate_addressed_v8 as evaluator


def _bundle(tmp_path: Path) -> Path:
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({
            "paper_id": "p0",
            "abstract": "Method A improves by 20 percent.",
            "body": "Experiments show Method A improves by 20 percent over B.",
            "y": "forbidden",
        }) + "\n" + json.dumps({
            "paper_id": "p1",
            "abstract": "Missing body control.",
            "body": "",
            "y": 1,
        }) + "\n",
        encoding="utf-8",
    )
    v7_bundle = tmp_path / "v7"
    v7.prepare(source, v7.DEFAULT_SPEC, v7.DEFAULT_MODEL, v7_bundle)
    comparator = tmp_path / "historical_code.json"
    comparator.write_text(json.dumps({
        "schema_version": "test-old-code",
        "provenance": "manual_test_fixture",
        "pipeline_status": "selected_fixture",
        "input": {"path": str(source), "sha256": v8.hash_file(source)},
        "records": [
            {"paper_id": "p0", "certificate_count": 1},
            {"paper_id": "p1", "certificate_count": 0},
        ],
    }), encoding="utf-8")
    bundle = tmp_path / "v8"
    v8.prepare(
        source, v8.DEFAULT_SPEC, v8.DEFAULT_MODEL, v7_bundle, bundle, comparator
    )
    return bundle


def _content():
    return json.dumps({
        "paper_id": "p0",
        "selections": [{
            "claim_sentence_id": "A0001",
            "evidence_sentence_id": "B0001",
            "decision": "supported",
            "relation": "numeric",
            "quantity_state": "aligned",
            "comparison_state": "not_required",
            "evidence_kind": "numeric_relation",
            "quantity_count": 1,
            "comparison_present": False,
        }],
    })


def _provider_response():
    return {
        "model": "z-ai/glm-4.7",
        "provider": "mock-provider",
        "choices": [{
            "message": {"content": _content()},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 7},
        },
    }


def test_runner_mock_retry_binds_physical_attempts_and_resumes_without_network(tmp_path):
    bundle = _bundle(tmp_path)
    output = tmp_path / "results.jsonl"
    failures = tmp_path / "failures.jsonl"
    calls = []

    def flaky(endpoint, api_key, payload, timeout):
        calls.append((endpoint, api_key, payload, timeout))
        if len(calls) == 1:
            raise TimeoutError("mock timeout")
        return 200, _provider_response()

    summary = runner.run_serial(
        bundle, output, failures, api_key="mock-key", max_requests=1,
        max_attempts=2, sender=flaky,
    )
    assert summary == {
        "already_completed": 0,
        "logical_requests_launched": 1,
        "successful_results_appended": 1,
        "terminal_failures": 0,
        "physical_attempt_count_including_retries": 2,
    }
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["schema_version"] == v8.RESULT_SCHEMA
    assert row["telemetry"]["physical_attempt_count"] == 2
    assert row["telemetry"]["reasoning"] == {
        "requested": False,
        "reported_reasoning_tokens": 7,
        "provider_returned_reasoning_field": False,
        "trace_retained": False,
    }
    assert row["api_payload_sha256"] == v8.hash_value(calls[-1][2])

    def forbidden_sender(*args):
        raise AssertionError("resume attempted a new network call")

    resumed = runner.run_serial(
        bundle, output, failures, api_key="mock-key", max_requests=1,
        sender=forbidden_sender,
    )
    assert resumed["already_completed"] == 1
    assert resumed["logical_requests_launched"] == 0
    assert resumed["physical_attempt_count_including_retries"] == 0


def test_runner_validates_all_prior_result_bindings_before_resume(tmp_path):
    bundle = _bundle(tmp_path)
    output = tmp_path / "results.jsonl"
    failures = tmp_path / "failures.jsonl"
    runner.run_serial(
        bundle, output, failures, api_key="mock-key", max_requests=1,
        sender=lambda *args: (200, _provider_response()),
    )
    row = json.loads(output.read_text(encoding="utf-8"))
    row["provider"] = "different-provider"
    output.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="provider"):
        runner.run_serial(
            bundle, output, failures, api_key="mock-key", max_requests=1,
            sender=lambda *args: (_ for _ in ()).throw(
                AssertionError("network must not be reached")
            ),
        )


def test_runner_ledgers_invalid_2xx_parse_attempt_then_retries(tmp_path):
    bundle = _bundle(tmp_path)
    output = tmp_path / "results.jsonl"
    failures = tmp_path / "failures.jsonl"
    responses = [
        {
            "model": "z-ai/glm-4.7", "provider": "mock-provider",
            "choices": [{"message": {"content": "not-json"}, "finish_reason": "stop"}],
            "usage": {},
        },
        _provider_response(),
    ]

    def sender(*args):
        return 200, responses.pop(0)

    summary = runner.run_serial(
        bundle, output, failures, api_key="mock-key", max_requests=1,
        sender=sender,
    )
    assert summary["successful_results_appended"] == 1
    assert summary["physical_attempt_count_including_retries"] == 2
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["telemetry"]["attempts"][0] == {
        "attempt_index": 1,
        "outcome": "error",
        "http_status": 200,
        "error_type": "ValueError",
    }
    assert row["telemetry"]["attempts"][1]["outcome"] == "success"
    assert not failures.exists()


def test_runner_terminal_invalid_2xx_is_count_only_failure_with_every_attempt(tmp_path):
    bundle = _bundle(tmp_path)
    output = tmp_path / "results.jsonl"
    failures = tmp_path / "failures.jsonl"
    invalid = {
        "model": "z-ai/glm-4.7", "provider": "mock-provider",
        "choices": [{
            "message": {"content": json.dumps({
                "paper_id": "p0",
                "selections": [{
                    "claim_sentence_id": "A9999",
                    "evidence_sentence_id": "B0001",
                    "decision": "supported",
                    "relation": "numeric",
                    "quantity_state": "aligned",
                    "comparison_state": "not_required",
                    "evidence_kind": "numeric_relation",
                    "quantity_count": 1,
                    "comparison_present": False,
                }],
            })},
            "finish_reason": "stop",
        }],
        "usage": {},
    }
    summary = runner.run_serial(
        bundle, output, failures, api_key="mock-key", max_requests=1,
        sender=lambda *args: (200, invalid),
    )
    assert summary["successful_results_appended"] == 0
    assert summary["terminal_failures"] == 1
    assert summary["physical_attempt_count_including_retries"] == 2
    assert not output.exists()
    failure = json.loads(failures.read_text(encoding="utf-8"))
    assert failure["physical_attempt_count"] == 2
    assert [row["http_status"] for row in failure["attempts"]] == [200, 200]
    assert failure["response_content_retained"] is False
    assert "response" not in failure and "content" not in failure
    for key in (
        "request_sha256", "model_manifest_sha256", "bundle_manifest_sha256",
        "api_payload_sha256", "runner_sha256", "provider", "model",
    ):
        assert failure[key]


def test_api_payload_has_exact_schema_and_reasoning_off(tmp_path):
    bundle = _bundle(tmp_path)
    manifest, requests, _ = v8.verify_bundle(bundle)
    request = next(iter(requests.values()))
    payload = runner.api_payload_for_request(
        request, manifest["model_manifest"]["identity"]
    )
    assert payload["reasoning"] == {"effort": "none"}
    assert payload["provider"] == {"require_parameters": True}
    assert payload["response_format"]["type"] == "json_schema"
    schema = payload["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False
    assert v8.PROMPT_CERTIFICATE_TYPE in request["system_prompt"]


def test_estimation_gates_tiny_and_constant_support():
    tiny = evaluator._binary_comparison([False, True], [False, True])
    assert tiny["estimate_status"] == "not_estimated"
    assert tiny["estimate_reason"].startswith("tiny_support")
    constant = evaluator._binary_comparison(
        [False] * 20, [False] * 20
    )
    assert constant["estimate_status"] == "not_estimated"
    assert "constant" in constant["estimate_reason"]
    estimated = evaluator._binary_comparison(
        [False, True] * 10, [False, True] * 10
    )
    assert estimated["estimate_status"] == "estimated"
    assert estimated["phi"] == 1.0


def test_evaluator_uses_code_as_comparator_not_ground_truth_and_tiny_gate(tmp_path):
    bundle = _bundle(tmp_path)
    results = tmp_path / "results.jsonl"
    failures = tmp_path / "failures.jsonl"
    runner.run_serial(
        bundle, results, failures, api_key="mock-key", max_requests=1,
        sender=lambda *args: (200, _provider_response()),
    )
    normalized = tmp_path / "normalized.jsonl"
    rejects = tmp_path / "rejects.jsonl"
    assert v8.ingest(bundle, results, normalized, rejects)["accepted_new"] == 1
    old = tmp_path / "old.json"
    bundle_manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    source_sha = bundle_manifest["input"]["source_file_sha256"]
    old.write_text(json.dumps({
        "schema_version": "test-old-code",
        "input": {"path": "fixture-source.jsonl", "sha256": source_sha},
        "records": [
            {"paper_id": "p0", "certificate_count": 1},
            {"paper_id": "p1", "certificate_count": 0},
        ],
    }), encoding="utf-8")
    old_sha = v8.hash_file(old)
    with pytest.raises(ValueError, match="arbitrary old-code path requires"):
        evaluator.evaluate(bundle, normalized, old)
    with pytest.raises(ValueError, match="SHA"):
        evaluator.evaluate(
            bundle, normalized, old,
            expected_old_code_sha256="0" * 64,
            expected_old_code_schema_version="test-old-code",
            expected_old_code_source_sha256=source_sha,
        )
    with pytest.raises(ValueError, match="schema"):
        evaluator.evaluate(
            bundle, normalized, old,
            expected_old_code_sha256=old_sha,
            expected_old_code_schema_version="wrong-schema",
            expected_old_code_source_sha256=source_sha,
        )
    wrong_source = tmp_path / "old_wrong_source.json"
    wrong_source.write_text(json.dumps({
        "schema_version": "test-old-code",
        "input": {"path": "fixture-source.jsonl", "sha256": "0" * 64},
        "records": [
            {"paper_id": "p0", "certificate_count": 1},
            {"paper_id": "p1", "certificate_count": 0},
        ],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="payload input SHA"):
        evaluator.evaluate(
            bundle, normalized, wrong_source,
            expected_old_code_sha256=v8.hash_file(wrong_source),
            expected_old_code_schema_version="test-old-code",
            expected_old_code_source_sha256=source_sha,
        )
    payload = evaluator.evaluate(
        bundle, normalized, old,
        expected_old_code_sha256=old_sha,
        expected_old_code_schema_version="test-old-code",
        expected_old_code_source_sha256=source_sha,
    )
    assert payload["objective"].endswith("no_external_anchor")
    assert payload["bindings"]["old_code_expected_sha256"] == old_sha
    assert payload["bindings"]["old_code_expected_schema_version"] == "test-old-code"
    assert payload["bindings"]["old_code_expected_source_sha256"] == source_sha
    assert "not supervised ground truth" in payload["interpretation"][
        "old_corrected_code"
    ]
    assert "same-evidence-content" in payload["interpretation"]["old_corrected_code"]
    assert "full/input-representation isomorphism is unavailable" in (
        payload["interpretation"]["isomorphism"]
    )
    assert payload["support"] == {
        "corpus_total": 2,
        "prompt_eligible_total": 1,
        "structural_abstention_total": 1,
        "completed_prompt_results": 1,
        "uncompleted_prompt_eligible": 0,
    }
    assert all(
        value["estimate_status"] == "not_estimated"
        for value in payload["comparisons"].values()
    )
    assert set(payload["comparisons"]) == {
        "prompt_assertion_vs_old_corrected_code_comparator"
    }
    fidelity = payload["conditional_prompt_certificate_construct_fidelity"]
    assert fidelity["diagnostic_status"] == (
        "descriptive_conditional_not_isomorphism_estimate"
    )
    assert fidelity["right_only_is_structurally_undefined"] is True
    assert fidelity["phi_is_not_reported"] is True
    assert fidelity["prompt_asserted_certificate_count"] == 1
    assert fidelity["code_parser_verified_count"] == 1
    hybrid = payload["prompt_selected_code_confirmed_hybrid_witnesses"]
    assert hybrid["count"] == 1
    assert hybrid["scope"] == "document_local_relation_local_parser_scoped"
    assert hybrid["external_scientific_truth"] is False
    assert hybrid["effect_on_prompt_acceptance"] == "none_non_gating"
    assert "separately code-confirmed conditional on prompt-selected spans" in (
        hybrid["claim_licensed"]
    )
    provenance = payload["old_fullpaper_code_comparator_provenance"]
    assert provenance["original_decomposition_discovery"] == "manual_historical"
    assert provenance["v8_analysis_pipeline_status"] == "selected"
    assert provenance["selection_mode"] == "retrospective_seed"
    assert provenance["automatically_discovered_by_v8"] is False
