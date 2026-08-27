import json
from pathlib import Path

import pytest

from methods.metric_seam.science_claims_v2.articulability_api_runner import (
    api_payload,
    call,
    provider_diagnostic,
    response_text,
)
from methods.metric_seam.science_claims_v2.articulability_pipeline import (
    MANIFEST_SCHEMA,
    RESULT_SCHEMA,
    _allowed_projection,
    evaluate,
    hash_file,
    ingest,
    pending,
    prepare,
    verify_bundle,
)


class Trap(dict):
    def get(self, key, default=None):
        if key == "y":
            raise AssertionError("label accessed")
        return super().get(key, default)


def _files(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({
            "paper_id": "p1", "y": "DO_NOT_EMIT_7e13", "abstract": "Method A improves by 20 percent.",
            "body": "Experiments show Method A improves by 20 percent over B."
        }) + "\n" + json.dumps({
            "paper_id": "p2", "y": 1, "abstract": "No result is stated.", "body": ""
        }) + "\n"
    )
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "input_allowlist": ["paper_id", "abstract", "body"],
        "external_knowledge": "forbidden",
        "system_prompt": "Use supplied text only.",
        "decision_semantics": {"supported": "distinct body relation agrees"},
        "user_template": "PAPER_ID: {paper_id}\n\nABSTRACT:\n{abstract}\n\nBODY:\n{body}",
        "output_schema": {"paper_id": "string"},
    }))
    model = tmp_path / "model.json"
    model.write_text(json.dumps({
        "schema_version": "science-articulability-model-v1", "backend": "test",
        "protocol": "test", "model": "fixed-model", "temperature": 0.0,
        "max_output_tokens": 1000, "system_prompt_transport": "system",
        "response_transport": "text",
    }))
    return source, spec, model


def _response(paper_id="p1"):
    quantity = {"start": 21, "end": 31, "raw": "20 percent", "value": 20,
                "unit": "percent"}
    certificate = {
        "decision": "supported", "witness_kind": "relation_certificate", "reason": "same relation",
        "claim": {"sentence_index": 0, "text": "Method A improves by 20 percent.",
                  "relation": "numeric", "quantities": [quantity], "comparison": None},
        "evidence": {"sentence_index": 0, "start": 0, "end": 59,
                     "text": "Experiments show Method A improves by 20 percent over B.",
                     "quantities": [quantity], "comparison": None},
        "checks": {"bm25": None, "claim_term_coverage": None, "quantity_matches": 1,
                   "quantity_required": 1, "relation_state": "aligned"},
    }
    return {
        "paper_id": paper_id, "status": "supported", "reason": "certificate", "claim_count": 1,
        "certificate_count": 1, "evidence_link_count": 0, "certificates": [certificate],
        "evidence_links": [], "matches": [certificate],
        "graph": {"claim_nodes": 1, "evidence_nodes": 1, "edges": None,
                  "matched_edges": 1, "matching": "prompt_relation_matching"},
    }


def _bound(request, response=None, bundle_manifest_sha256=None, **updates):
    row = {
        "schema_version": RESULT_SCHEMA,
        "request_id": request["request_id"],
        "request_sha256": request["request_sha256"],
        "model_manifest_sha256": request["model_manifest_sha256"],
        "bundle_manifest_sha256": bundle_manifest_sha256,
        "response": response if response is not None else _response(request["paper_id"]),
    }
    row.update(updates)
    return row


def test_projection_never_requests_label():
    projected = _allowed_projection(Trap(paper_id="p", y=10, abstract="a", body="b"), 1)
    assert projected == {"paper_id": "p", "abstract": "a", "body": "b"}


def test_prepare_binds_allowlisted_input_prompt_and_model(tmp_path):
    source, spec, model = _files(tmp_path)
    bundle = tmp_path / "bundle"
    manifest = prepare(source, spec, model, bundle)
    assert manifest["schema_version"] == MANIFEST_SCHEMA
    assert manifest["requests"]["count"] == manifest["requests"]["api_call_count"] == 2
    assert manifest["requests"]["sha256"] == hash_file(bundle / "requests.jsonl")
    text = (bundle / "requests.jsonl").read_text()
    assert "DO_NOT_EMIT_7e13" not in text
    _, requests = verify_bundle(bundle)
    assert len(requests) == 2
    first = next(iter(requests.values()))
    assert "FROZEN_DECISION_SEMANTICS" in first["system_prompt"]
    assert "FROZEN_OUTPUT_SCHEMA" in first["system_prompt"]
    assert '"paper_id": "string"' in first["system_prompt"]
    assert "Copy every evidence.text exactly and verbatim" in first["system_prompt"]


def test_verify_detects_request_tampering(tmp_path):
    source, spec, model = _files(tmp_path)
    bundle = tmp_path / "bundle"
    prepare(source, spec, model, bundle)
    with (bundle / "requests.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_bundle(bundle)


def test_ingest_is_resumable_and_bound(tmp_path):
    source, spec, model = _files(tmp_path)
    bundle = tmp_path / "bundle"
    prepare(source, spec, model, bundle)
    _, requests = verify_bundle(bundle)
    bundle_sha = hash_file(bundle / "manifest.json")
    req = next(iter(requests.values()))
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps(_bound(req, bundle_manifest_sha256=bundle_sha)) + "\n")
    normalized, rejects = bundle / "normalized.jsonl", bundle / "rejects.jsonl"
    first = ingest(bundle, raw, normalized, rejects)
    second = ingest(bundle, raw, normalized, rejects)
    assert first == {"accepted_new": 1, "already_present": 0, "rejected": 0, "remaining": 1}
    assert second == {"accepted_new": 0, "already_present": 1, "rejected": 0, "remaining": 1}
    assert len(normalized.read_text().splitlines()) == 1


def test_ingest_rejects_binding_and_ungrounded_witnesses(tmp_path):
    source, spec, model = _files(tmp_path)
    bundle = tmp_path / "bundle"
    prepare(source, spec, model, bundle)
    _, requests = verify_bundle(bundle)
    bundle_sha = hash_file(bundle / "manifest.json")
    req = next(iter(requests.values()))
    bad_ground = _response()
    bad_ground["certificates"][0]["evidence"]["text"] = "A hallucinated experiment."
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        json.dumps(_bound(req, bundle_manifest_sha256=bundle_sha,
                          request_sha256="0" * 64)) + "\n" +
        json.dumps(_bound(req, response=bad_ground, bundle_manifest_sha256=bundle_sha,
                          request_id=req["request_id"] + "_x")) + "\n"
    )
    summary = ingest(bundle, raw, bundle / "normalized.jsonl", bundle / "rejects.jsonl")
    assert summary["accepted_new"] == 0
    assert summary["rejected"] == 2


@pytest.mark.parametrize("collection", ["evidence_links", "matches"])
def test_ingest_rejects_ungrounded_noncertificate_witnesses(tmp_path, collection):
    source, spec, model = _files(tmp_path)
    bundle = tmp_path / "bundle"
    prepare(source, spec, model, bundle)
    _, requests = verify_bundle(bundle)
    bundle_sha = hash_file(bundle / "manifest.json")
    req = next(iter(requests.values()))
    response = _response()
    witness = {
        "decision": "evidence_link", "witness_kind": "evidence_link",
        "shape": {
            "claim": response["certificates"][0]["claim"],
            "evidence": {"text": "A hallucinated experiment."},
            "checks": {},
        },
    }
    response["certificates"] = []
    response["certificate_count"] = 0
    response["evidence_links"] = [witness] if collection == "evidence_links" else []
    response["evidence_link_count"] = len(response["evidence_links"])
    response["matches"] = [witness] if collection == "matches" else []
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps(_bound(
        req, response=response, bundle_manifest_sha256=bundle_sha
    )) + "\n")
    summary = ingest(bundle, raw, bundle / "normalized.jsonl", bundle / "rejects.jsonl")
    assert summary["accepted_new"] == 0
    assert summary["rejected"] == 1
    assert "not a verbatim whitespace-canonical span" in (
        bundle / "rejects.jsonl"
    ).read_text()


def test_ingest_accepts_flattened_grounded_evidence_link_shape(tmp_path):
    source, spec, model = _files(tmp_path)
    bundle = tmp_path / "bundle"
    prepare(source, spec, model, bundle)
    _, requests = verify_bundle(bundle)
    bundle_sha = hash_file(bundle / "manifest.json")
    req = next(iter(requests.values()))
    response = _response()
    certificate = response["certificates"][0]
    link = {
        "decision": "evidence_link", "witness_kind": "evidence_link",
        "claim": certificate["claim"], "evidence": certificate["evidence"],
        "checks": certificate["checks"],
    }
    response["certificates"] = []
    response["certificate_count"] = 0
    response["evidence_links"] = [link]
    response["evidence_link_count"] = 1
    response["matches"] = [link]
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps(_bound(
        req, response=response, bundle_manifest_sha256=bundle_sha
    )) + "\n")
    summary = ingest(bundle, raw, bundle / "normalized.jsonl", bundle / "rejects.jsonl")
    assert summary["accepted_new"] == 1
    assert summary["rejected"] == 0


def test_ingest_allows_only_source_layout_whitespace_canonicalization(tmp_path):
    source, spec, model = _files(tmp_path)
    source.write_text(json.dumps({
        "paper_id": "p1", "abstract": "Method A improves\nby 20 percent.",
        "body": "Experiments show Method A improves\n\tby 20 percent over B.",
    }) + "\n")
    bundle = tmp_path / "bundle"
    prepare(source, spec, model, bundle)
    _, requests = verify_bundle(bundle)
    bundle_sha = hash_file(bundle / "manifest.json")
    req = next(iter(requests.values()))
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps(_bound(
        req, response=_response(), bundle_manifest_sha256=bundle_sha
    )) + "\n")
    summary = ingest(bundle, raw, bundle / "normalized.jsonl", bundle / "rejects.jsonl")
    assert summary["accepted_new"] == 1
    assert summary["rejected"] == 0


@pytest.mark.parametrize(
    ("field", "variant"),
    [
        ("claim", "method A improves by 20 percent."),
        ("claim", "Method A improves by 20 percent!"),
        ("claim", "Method A improves by 20-percent."),
        ("evidence", "Experiments show method A improves by 20 percent over B."),
        ("evidence", "Experiments show Method A improves by 20 percent over B!"),
        ("evidence", "Experiments show Method A improves by 20-percent over B."),
    ],
)
def test_ingest_rejects_case_punctuation_and_token_normalized_variants(
    tmp_path, field, variant
):
    source, spec, model = _files(tmp_path)
    bundle = tmp_path / "bundle"
    prepare(source, spec, model, bundle)
    _, requests = verify_bundle(bundle)
    bundle_sha = hash_file(bundle / "manifest.json")
    req = next(iter(requests.values()))
    response = _response()
    response["certificates"][0][field]["text"] = variant
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps(_bound(
        req, response=response, bundle_manifest_sha256=bundle_sha
    )) + "\n")
    summary = ingest(bundle, raw, bundle / "normalized.jsonl", bundle / "rejects.jsonl")
    assert summary["accepted_new"] == 0
    assert summary["rejected"] == 1
    assert "not a verbatim whitespace-canonical span" in (
        bundle / "rejects.jsonl"
    ).read_text()


def test_ingest_rejects_grounded_but_qualitative_strong_certificate(tmp_path):
    source, spec, model = _files(tmp_path)
    bundle = tmp_path / "bundle"
    prepare(source, spec, model, bundle)
    _, requests = verify_bundle(bundle)
    req = next(iter(requests.values()))
    response = _response()
    for witness in (response["certificates"][0], response["matches"][0]):
        witness["claim"]["relation"] = "qualitative"
        witness["claim"]["quantities"] = []
        witness["evidence"]["quantities"] = []
        witness["checks"]["quantity_matches"] = 0
        witness["checks"]["quantity_required"] = 0
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps(_bound(
        req, response=response, bundle_manifest_sha256=hash_file(bundle / "manifest.json")
    )) + "\n")
    summary = ingest(bundle, raw, bundle / "normalized.jsonl", bundle / "rejects.jsonl")
    assert summary["accepted_new"] == 0 and summary["rejected"] == 1
    assert "relation must be numeric or comparative" in (
        bundle / "rejects.jsonl"
    ).read_text()


def test_ingest_rejects_numeric_certificate_without_quantity_payload(tmp_path):
    source, spec, model = _files(tmp_path)
    bundle = tmp_path / "bundle"
    prepare(source, spec, model, bundle)
    _, requests = verify_bundle(bundle)
    req = next(iter(requests.values()))
    response = _response()
    for witness in (response["certificates"][0], response["matches"][0]):
        witness["claim"]["quantities"] = []
        witness["evidence"]["quantities"] = []
        witness["checks"]["quantity_matches"] = 0
        witness["checks"]["quantity_required"] = 0
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps(_bound(
        req, response=response, bundle_manifest_sha256=hash_file(bundle / "manifest.json")
    )) + "\n")
    summary = ingest(bundle, raw, bundle / "normalized.jsonl", bundle / "rejects.jsonl")
    assert summary["accepted_new"] == 0 and summary["rejected"] == 1
    assert "numeric certificate has no quantity relation" in (
        bundle / "rejects.jsonl"
    ).read_text()


def test_ingest_rejects_wrong_bundle_manifest_binding(tmp_path):
    source, spec, model = _files(tmp_path)
    bundle = tmp_path / "bundle"
    prepare(source, spec, model, bundle)
    _, requests = verify_bundle(bundle)
    req = next(iter(requests.values()))
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        json.dumps(_bound(req, bundle_manifest_sha256="0" * 64)) + "\n"
    )
    summary = ingest(bundle, raw, bundle / "normalized.jsonl", bundle / "rejects.jsonl")
    assert summary["accepted_new"] == 0
    assert summary["rejected"] == 1
    assert "bundle_manifest_sha256 mismatch" in (bundle / "rejects.jsonl").read_text()


def test_ingest_banks_malformed_jsonl_row_as_rejection(tmp_path):
    source, spec, model = _files(tmp_path)
    bundle = tmp_path / "bundle"
    prepare(source, spec, model, bundle)
    raw = tmp_path / "raw.jsonl"
    raw.write_text('{"truncated":')
    summary = ingest(bundle, raw, bundle / "normalized.jsonl", bundle / "rejects.jsonl")
    assert summary["rejected"] == 1
    assert "Expecting value" in (bundle / "rejects.jsonl").read_text()


def test_pending_emits_only_unfinished_requests(tmp_path):
    source, spec, model = _files(tmp_path)
    bundle = tmp_path / "bundle"
    prepare(source, spec, model, bundle)
    _, requests = verify_bundle(bundle)
    bundle_sha = hash_file(bundle / "manifest.json")
    req = next(iter(requests.values()))
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps(_bound(req, bundle_manifest_sha256=bundle_sha)) + "\n")
    normalized = bundle / "normalized.jsonl"
    ingest(bundle, raw, normalized, bundle / "rejects.jsonl")
    summary = pending(bundle, normalized, bundle / "pending.jsonl")
    assert summary == {"completed": 1, "pending": 1, "total": 2}
    assert len((bundle / "pending.jsonl").read_text().splitlines()) == 1


def test_evaluate_treats_code_as_comparator_not_truth(tmp_path):
    source, spec, model = _files(tmp_path)
    bundle = tmp_path / "bundle"
    prepare(source, spec, model, bundle)
    _, requests = verify_bundle(bundle)
    bundle_sha = hash_file(bundle / "manifest.json")
    req = next(iter(requests.values()))
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps(_bound(req, bundle_manifest_sha256=bundle_sha)) + "\n")
    normalized = bundle / "normalized.jsonl"
    ingest(bundle, raw, normalized, bundle / "rejects.jsonl")
    code = tmp_path / "code.json"
    code.write_text(json.dumps({"records": [_response()]}))
    payload = evaluate(bundle, normalized, code, bundle / "evaluation.json",
                       bundle / "REPORT.md", require_complete=False)
    assert payload["external_supervision"] == "none"
    assert payload["normalization_evaluation_instrument"]["sha256"] == hash_file(
        Path(__file__).with_name("articulability_pipeline.py")
    )
    assert payload["isomorphism"]["status"] == "non_estimating_descriptive_comparison"
    assert payload["isomorphism"]["estimating"] is False
    assert "fewer_than_two_shared_papers" in payload["isomorphism"][
        "non_estimating_reasons"
    ]
    assert payload["isomorphism"]["matched_witnesses"] == 1
    assert "not ground truth" in payload["isomorphism"]["interpretation"]


def test_evaluate_reports_weaker_evidence_link_overlap_separately(tmp_path):
    source, spec, model = _files(tmp_path)
    bundle = tmp_path / "bundle"
    prepare(source, spec, model, bundle)
    _, requests = verify_bundle(bundle)
    bundle_sha = hash_file(bundle / "manifest.json")
    req = next(iter(requests.values()))
    response = _response()
    link = dict(response["certificates"][0])
    link["decision"] = "evidence_link"
    link["witness_kind"] = "evidence_link"
    response["status"] = "evidence_link"
    response["certificates"] = []
    response["certificate_count"] = 0
    response["evidence_links"] = [link]
    response["evidence_link_count"] = 1
    response["matches"] = [link]
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps(_bound(
        req, response=response, bundle_manifest_sha256=bundle_sha
    )) + "\n")
    normalized = bundle / "normalized.jsonl"
    ingest(bundle, raw, normalized, bundle / "rejects.jsonl")
    code = tmp_path / "code.json"
    code.write_text(json.dumps({"records": [response]}))
    payload = evaluate(
        bundle, normalized, code, bundle / "evaluation.json", bundle / "REPORT.md",
        require_complete=False,
    )
    iso = payload["isomorphism"]
    assert iso["prompt_witnesses"] == iso["code_witnesses"] == 0
    assert iso["prompt_evidence_links"] == iso["code_evidence_links"] == 1
    assert iso["matched_evidence_links"] == 1
    assert iso["prompt_evidence_link_match_rate"] == 1.0


def test_evaluate_records_bounded_transport_without_hidden_reasoning(tmp_path):
    source, spec, model = _files(tmp_path)
    model_payload = json.loads(model.read_text())
    model_payload["reasoning"] = {"effort": "none"}
    model.write_text(json.dumps(model_payload))
    bundle = tmp_path / "bundle"
    prepare(source, spec, model, bundle)
    _, requests = verify_bundle(bundle)
    bundle_sha = hash_file(bundle / "manifest.json")
    req = next(iter(requests.values()))
    raw = bundle / "raw.jsonl"
    bound = _bound(req, bundle_manifest_sha256=bundle_sha)
    bound["provider_metadata"] = {
        "model": "fixed-model-version", "stop_reason": "stop",
        "usage": {"completion_tokens_details": {"reasoning_tokens": 7}},
    }
    raw.write_text(json.dumps(bound) + "\n")
    normalized, rejects = bundle / "normalized.jsonl", bundle / "rejects.jsonl"
    ingest(bundle, raw, normalized, rejects)
    rejects.write_text("")
    payload = evaluate(
        bundle, normalized, None, bundle / "evaluation.json", bundle / "REPORT.md",
        require_complete=False, raw_results_path=raw, rejects_path=rejects,
    )
    smoke = payload["execution_smoke"]
    assert smoke["raw_results"]["attempted_unique_requests"] == 1
    assert smoke["validation"] == {
        "valid_normalized": 1, "rejected": 0, "valid_rate_among_attempted": 1.0,
        "accepted_rejected_partition_complete": True, "rejection_reasons": {},
    }
    provider = smoke["provider_observation"]
    assert provider["requested_reasoning"] == {"effort": "none"}
    assert provider["reported_reasoning_tokens_total"] == 7
    assert provider["reasoning_request_observed_honored_on_all_responses"] is False
    assert provider["hidden_reasoning_text_retained"] is False
    assert "fixed-model-version" in provider["observed_models"]
    assert "secret_trace_7e13" not in json.dumps(smoke).lower()


def test_evaluate_can_require_completion(tmp_path):
    source, spec, model = _files(tmp_path)
    bundle = tmp_path / "bundle"
    prepare(source, spec, model, bundle)
    with pytest.raises(ValueError, match="incomplete results"):
        evaluate(bundle, bundle / "missing.jsonl", None, bundle / "out.json",
                 bundle / "report.md", require_complete=True)


def test_optional_api_adapter_preserves_separate_system_channel():
    row = {"system_prompt": "system", "user_prompt": "user"}
    model = {"protocol": "anthropic_messages", "system_prompt_transport": "system",
             "model": "fixed", "max_output_tokens": 12, "temperature": 0.0}
    payload = api_payload(row, model)
    assert payload == {"model": "fixed", "max_tokens": 12, "temperature": 0.0,
                       "system": "system", "messages": [{"role": "user", "content": "user"}]}
    assert response_text({"content": [{"type": "text", "text": "{\"ok\":true}"}]}) == '{"ok":true}'


def test_optional_api_adapter_supports_documented_json_mode():
    row = {"system_prompt": "system", "user_prompt": "user"}
    model = {"protocol": "openai_chat_completions", "system_prompt_transport": "system",
             "model": "fixed", "max_output_tokens": 12, "temperature": 0.0,
             "provider_require_parameters": True,
             "reasoning": {"effort": "none"}}
    payload = api_payload(row, model)
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["provider"] == {"require_parameters": True}
    assert payload["reasoning"] == {"effort": "none"}
    assert payload["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    raw = {"choices": [{"message": {"content": '{"ok":true}'}}]}
    assert response_text(raw, "openai_chat_completions") == '{"ok":true}'


def test_no_text_diagnostic_records_shape_not_reasoning_text():
    raw = {
        "id": "generation-id", "model": "fixed", "provider": "example",
        "choices": [{"finish_reason": "length", "native_finish_reason": "max_tokens",
                     "message": {"content": "", "reasoning": "private trace"}}],
        "usage": {"completion_tokens": 12},
    }
    diagnostic = provider_diagnostic(raw)
    assert diagnostic["content_characters"] == 0
    assert diagnostic["reasoning_characters"] == 13
    assert "private trace" not in json.dumps(diagnostic)
    with pytest.raises(ValueError, match="no text content") as caught:
        response_text(raw, "openai_chat_completions")
    assert "private trace" not in str(caught.value)


def test_call_passes_explicit_remaining_deadline(monkeypatch):
    observed = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(_request, timeout):
        observed["timeout"] = timeout
        return Response()

    times = iter([10.0, 10.25])
    monkeypatch.setattr(
        "methods.metric_seam.science_claims_v2.articulability_api_runner.time.monotonic",
        lambda: next(times),
    )
    monkeypatch.setattr(
        "methods.metric_seam.science_claims_v2.articulability_api_runner.urlrequest.urlopen",
        fake_urlopen,
    )
    result = call(
        "https://example.invalid", "secret", {"x": 1}, "openai_chat_completions",
        request_timeout_seconds=30, tries=2,
    )
    assert result == {"ok": True}
    assert observed["timeout"] == pytest.approx(29.75)
