import json
from pathlib import Path

import pytest

from methods.metric_seam.science_claims_v2 import addressed_pipeline as v7
from methods.metric_seam.science_claims_v2 import addressed_pipeline_v8 as v8
from methods.metric_seam.science_claims_v2 import addressed_runner_v8 as runner


class LabelTrap(dict):
    def get(self, key, default=None):
        if key in {"y", "label", "acceptance", "judgement"}:
            raise AssertionError("projection indexed a label field")
        return super().get(key, default)


def _source(tmp_path: Path, *, one_eligible=False) -> Path:
    rows = [
        {
            "paper_id": "p0",
            "abstract": "Method A improves by 20 percent.",
            "body": "Experiments show Method A improves by 20 percent over B.",
            "y": "LABEL_SENTINEL_MUST_NOT_ESCAPE",
        },
        {
            "paper_id": "p1",
            "abstract": "Method C is better than D.",
            "body": "",
            "y": 1,
        },
    ]
    if not one_eligible:
        rows.append({
            "paper_id": "p2",
            "abstract": "Method E is better than F.",
            "body": "Results show Method E is better than F.",
            "y": 0,
        })
    path = tmp_path / "source.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def _historical_comparator(tmp_path: Path, source: Path) -> Path:
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    path = tmp_path / "historical_code.json"
    path.write_text(json.dumps({
        "schema_version": "test-old-code",
        "provenance": "manual_test_fixture",
        "pipeline_status": "selected_fixture",
        "input": {"path": str(source), "sha256": v8.hash_file(source)},
        "records": [
            {"paper_id": row["paper_id"], "certificate_count": 0}
            for row in rows
        ],
    }), encoding="utf-8")
    return path


def _bundle(tmp_path: Path, *, one_eligible=False) -> Path:
    source = _source(tmp_path, one_eligible=one_eligible)
    comparator = _historical_comparator(tmp_path, source)
    v7_bundle = tmp_path / "v7"
    v7.prepare(source, v7.DEFAULT_SPEC, v7.DEFAULT_MODEL, v7_bundle)
    bundle = tmp_path / "v8"
    v8.prepare(
        source, v8.DEFAULT_SPEC, v8.DEFAULT_MODEL, v7_bundle, bundle, comparator
    )
    return bundle


def _selection(**updates):
    value = {
        "claim_sentence_id": "A0001",
        "evidence_sentence_id": "B0001",
        "decision": "supported",
        "relation": "numeric",
        "quantity_state": "aligned",
        "comparison_state": "not_required",
        "evidence_kind": "numeric_relation",
        "quantity_count": 1,
        "comparison_present": False,
    }
    value.update(updates)
    return value


def _response(*selections, paper_id="p0"):
    return {"paper_id": paper_id, "selections": list(selections or [_selection()])}


def _telemetry(attempts=1):
    values = [
        {
            "attempt_index": index,
            "outcome": "error" if index < attempts else "success",
            "http_status": None if index < attempts else 200,
            "error_type": "TimeoutError" if index < attempts else None,
        }
        for index in range(1, attempts + 1)
    ]
    return {
        "physical_attempt_count": attempts,
        "attempts": values,
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        "finish_reason": "stop",
        "reasoning": {
            "requested": False,
            "reported_reasoning_tokens": None,
            "provider_returned_reasoning_field": False,
            "trace_retained": False,
        },
        "provider_response_model": "z-ai/glm-4.7",
        "provider_name": "test-provider",
    }


def _bound(bundle: Path, request: dict, response=None, telemetry=None):
    manifest, _, _ = v8.verify_bundle(bundle)
    parsed = response if response is not None else _response()
    payload = runner.api_payload_for_request(
        request, manifest["model_manifest"]["identity"]
    )
    return {
        "schema_version": v8.RESULT_SCHEMA,
        "request_id": request["request_id"],
        "request_sha256": request["request_sha256"],
        "model_manifest_sha256": manifest["model_manifest"]["canonical_sha256"],
        "bundle_manifest_sha256": v8.hash_file(bundle / "manifest.json"),
        "runner_sha256": v8.hash_file(Path(runner.__file__)),
        "api_payload_sha256": v8.hash_value(payload),
        "provider": manifest["model_manifest"]["identity"]["backend"],
        "model": manifest["model_manifest"]["identity"]["model"],
        "response": json.dumps(parsed),
        "parsed_response_sha256": v8.hash_value(parsed),
        "telemetry": telemetry or _telemetry(),
    }


def _first_request(bundle: Path):
    manifest, requests, abstentions = v8.verify_bundle(bundle)
    request = min(requests.values(), key=lambda row: row["source_index"])
    return manifest, request, abstentions


def test_projection_deserializes_but_never_indexes_or_retains_labels():
    row = LabelTrap(
        paper_id="p", abstract="a", body="b", y=1, label=2, acceptance=3
    )
    assert v8.allowed_projection(row, 1) == {
        "paper_id": "p", "abstract": "a", "body": "b"
    }


def test_prepare_separates_strata_preserves_v7_addresses_and_binds_all_fields(tmp_path):
    bundle = _bundle(tmp_path)
    manifest, requests, abstentions = v8.verify_bundle(bundle)
    assert manifest["status"] == "prepared_not_run_no_api_calls"
    assert manifest["execution_policy"]["api_calls_made_by_prepare"] == 0
    assert manifest["execution_policy"]["gpu_used"] is False
    assert manifest["strata"]["observed"] == {
        "corpus_records": 3,
        "body_present_prompt_eligible": 2,
        "missing_body_structural_abstentions": 1,
    }
    assert len(requests) == 2 and len(abstentions) == 1
    assert next(iter(abstentions.values()))["reason"] == "missing_fullpaper_body"
    assert next(iter(abstentions.values()))["api_call_required"] is False
    request = min(requests.values(), key=lambda row: row["source_index"])
    assert request["source_map"] == v7.build_source_map(request["paper_input"])
    assert request["source_map"]["abstract"][0]["sentence_id"] == "A0001"
    assert "V8_TYPE_GUARD" in request["system_prompt"]
    assert "LABEL_SENTINEL_MUST_NOT_ESCAPE" not in (
        bundle / v8.REQUESTS_NAME
    ).read_text(encoding="utf-8")
    assert set(manifest["files"]["source_crosswalk"]["span_status_counts"]) == {
        "matched", "ambiguous", "unmatched"
    }
    report = (bundle / v8.REPORT_NAME).read_text(encoding="utf-8")
    assert "2" in report and "Missing-body deterministic structural abstentions" in report
    assert v8.PROMPT_CERTIFICATE_TYPE in report
    old = manifest["historical_code_comparator"]
    assert old["sha256"] == v8.hash_file(v8._resolve_recorded_path(old["path"]))
    assert old["schema_version"] == "test-old-code"
    assert old["input_source_sha256"] == manifest["input"]["source_file_sha256"]
    assert old["selection_mode"] == "retrospective_seed"
    assert old["automatically_discovered_by_v8"] is False


def test_prepare_requires_exact_v7_paper_input_not_only_segmented_surface(tmp_path):
    source = _source(tmp_path)
    v7_bundle = tmp_path / "v7"
    v7.prepare(source, v7.DEFAULT_SPEC, v7.DEFAULT_MODEL, v7_bundle)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    # Trailing whitespace is intentionally absent from v7 addressed spans, so a
    # source-map-only check could miss this exact-input drift.
    rows[0]["body"] += "   \n"
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exact v7 paper_input/current projection drift"):
        v8.prepare(
            source, v8.DEFAULT_SPEC, v8.DEFAULT_MODEL, v7_bundle, tmp_path / "v8"
        )


@pytest.mark.parametrize(
    ("path", "new_value", "message"),
    [
        (("objective",), "supervised", "objective changed"),
        (("execution_policy", "gpu_used"), True, "execution policy"),
        (("input", "record_count"), 99, "full recomputation mismatch"),
        (("request_statistics", "prompt_characters", "total"), 0, "full recomputation mismatch"),
        (("implementation_dependencies", "corrected_code", "sha256"), "0" * 64,
         "full recomputation mismatch"),
    ],
)
def test_verify_recomputes_entire_manifest_not_just_file_hashes(
    tmp_path, path, new_value, message
):
    bundle = _bundle(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = manifest
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = new_value
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        v8.verify_bundle(bundle)


def test_verify_rejects_rehashed_request_or_crosswalk_tampering(tmp_path):
    bundle = _bundle(tmp_path)
    requests = v8._read_jsonl(bundle / v8.REQUESTS_NAME)
    requests[0]["user_prompt"] += " changed"
    v8._write_jsonl(bundle / v8.REQUESTS_NAME, requests)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["requests"]["sha256"] = v8.hash_file(bundle / v8.REQUESTS_NAME)
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="deterministic regeneration"):
        v8.verify_bundle(bundle)


def test_code_audit_is_non_gating_and_prompt_certificate_is_explicit(tmp_path):
    bundle = _bundle(tmp_path)
    _, request, _ = _first_request(bundle)
    verified = v8.hydrate_response(_response(), request)
    cert = verified["prompt_asserted_relation_certificates"][0]
    assert cert["witness_kind"] == v8.PROMPT_CERTIFICATE_TYPE
    assert cert["code_relation_audit"]["status"] == "verified"
    assert cert["code_relation_audit"]["non_gating"] is True
    assert cert["hybrid_witness_kind"] == (
        "prompt_selected_code_confirmed_hybrid_witness"
    )
    assert verified["prompt_selected_code_confirmed_hybrid_witness_count"] == 1

    changed_request = json.loads(json.dumps(request))
    changed_request["source_map"]["body"][0]["text"] = (
        "Experiments show Method A improves by 30 percent over B."
    )
    changed_request["source_map"]["body"][0]["text_sha256"] = v8.hash_value(
        changed_request["source_map"]["body"][0]["text"]
    )
    diverged = v8.hydrate_response(_response(), changed_request)
    assert diverged["prompt_asserted_relation_certificate_count"] == 1
    assert diverged["matches"][0]["code_relation_audit"]["status"] == "diverged"
    assert diverged["matches"][0]["hybrid_witness_kind"] is None
    assert diverged["prompt_selected_code_confirmed_hybrid_witness_count"] == 0
    assert diverged["seam"]["code_audit_can_reject_prompt_response"] is False


def test_numeric_code_audit_is_decision_aware_and_separates_parser_state():
    base = _selection()
    aligned = v8.code_relation_audit(
        base,
        "Method A improves by 20 percent.",
        "Method A improves by 20 percent.",
    )
    mismatch = v8.code_relation_audit(
        base,
        "Method A improves by 20 percent.",
        "Method A improves by 30 percent.",
    )
    assert aligned["status"] == aligned["prompt_audit_status"] == "verified"
    assert aligned["quantity_parser"]["parser_relation_state"] == "aligned"
    assert mismatch["status"] == "diverged"
    assert mismatch["quantity_parser"]["parser_relation_state"] == "mismatch"

    # Numeric contradiction is not emitted by the current prompt schema, but the audit
    # primitive remains symmetric if that schema is later widened.
    contradicted = dict(base, decision="contradicted")
    agrees = v8.code_relation_audit(
        contradicted,
        "Method A improves by 20 percent.",
        "Method A improves by 30 percent.",
    )
    disagrees = v8.code_relation_audit(
        contradicted,
        "Method A improves by 20 percent.",
        "Method A improves by 20 percent.",
    )
    assert agrees["status"] == "verified"
    assert agrees["quantity_parser"]["parser_relation_state"] == "mismatch"
    assert disagrees["status"] == "diverged"
    assert disagrees["quantity_parser"]["parser_relation_state"] == "aligned"


def test_numeric_contradiction_symmetry_is_primitive_only_in_v8_contract(tmp_path):
    bundle = _bundle(tmp_path)
    manifest, request, _ = _first_request(bundle)
    scope = manifest["semantic_contract"]["contradiction_scope"]
    assert scope["accepted_response_relation"] == "comparative_only"
    assert scope["numeric_contradiction_symmetry"].startswith("primitive_only")
    numeric_contradiction = _selection(
        decision="contradicted", quantity_state="mismatch"
    )
    with pytest.raises(ValueError, match="contradiction requires a comparative relation"):
        v8.hydrate_response(_response(numeric_contradiction), request)


def test_comparative_code_audit_is_symmetric_for_support_and_contradiction():
    supported = _selection(
        relation="comparative", quantity_count=0, quantity_state="not_required",
        comparison_state="aligned", comparison_present=True,
        evidence_kind="comparative_relation",
    )
    contradicted = dict(
        supported, decision="contradicted", comparison_state="direction_mismatch"
    )
    aligned_claim = "Alpha model is better than Beta model."
    aligned_evidence = "Alpha model is better than Beta model."
    reversed_evidence = "Alpha model is worse than Beta model."
    support_agrees = v8.code_relation_audit(
        supported, aligned_claim, aligned_evidence
    )
    support_diverges = v8.code_relation_audit(
        supported, aligned_claim, reversed_evidence
    )
    contradiction_agrees = v8.code_relation_audit(
        contradicted, aligned_claim, reversed_evidence
    )
    contradiction_diverges = v8.code_relation_audit(
        contradicted, aligned_claim, aligned_evidence
    )
    assert support_agrees["status"] == "verified"
    assert support_diverges["status"] == "diverged"
    assert contradiction_agrees["status"] == "verified"
    assert contradiction_diverges["status"] == "diverged"
    assert contradiction_agrees["comparison_parser"]["parser_relation_state"] in {
        "direction_mismatch", "reversed_roles"
    }


def test_comparative_audit_derives_quantity_obligation_from_source_not_prompt(tmp_path):
    bundle = _bundle(tmp_path)
    _, request, _ = _first_request(bundle)
    changed = json.loads(json.dumps(request))
    claim = "Alpha model is 20 percent better than Beta model."
    evidence = "Alpha model is 30 percent better than Beta model."
    changed["source_map"]["abstract"][0]["text"] = claim
    changed["source_map"]["abstract"][0]["text_sha256"] = v8.hash_value(claim)
    changed["source_map"]["body"][0]["text"] = evidence
    changed["source_map"]["body"][0]["text_sha256"] = v8.hash_value(evidence)
    selection = _selection(
        relation="comparative", quantity_count=0, quantity_state="not_required",
        comparison_state="aligned", comparison_present=True,
        evidence_kind="comparative_relation",
    )
    result = v8.hydrate_response(_response(selection), changed)
    audit = result["matches"][0]["code_relation_audit"]
    assert audit["comparison_parser"]["parser_relation_state"] == "aligned"
    assert audit["quantity_parser"]["claim_quantity_count"] == 1
    assert audit["quantity_parser"]["evidence_quantity_count"] == 1
    assert audit["quantity_parser"]["parser_relation_state"] == "mismatch"
    assert audit["status"] == "diverged"
    assert result["prompt_selected_code_confirmed_hybrid_witness_count"] == 0


def test_crosswalk_reports_boundary_ambiguity_without_changing_v7_surface():
    text = 'Result holds. "Next!"'
    source_map = v7.segment_source(text, section="body")
    corrected = v8.corrected_code.segment_sentences(text)
    rows = [v8.crosswalk_span(span, corrected) for span in source_map]
    assert [row["sentence_id"] for row in rows] == ["B0001", "B0002"]
    assert any(row["status"] in {"matched", "ambiguous"} for row in rows)
    assert [span["text"] for span in source_map] == ["Result holds.", '"Next!"']


def test_ingest_changed_raw_resume_conflict_is_rejected(tmp_path):
    bundle = _bundle(tmp_path)
    _, request, _ = _first_request(bundle)
    raw = tmp_path / "raw.jsonl"
    normalized = tmp_path / "normalized.jsonl"
    rejects = tmp_path / "rejects.jsonl"
    raw.write_text(json.dumps(_bound(bundle, request)) + "\n", encoding="utf-8")
    first = v8.ingest(bundle, raw, normalized, rejects)
    assert first["accepted_new"] == 1

    changed = _response(_selection(
        decision="insufficient", evidence_sentence_id=None,
        evidence_kind="none", quantity_state="missing", quantity_count=0,
    ))
    raw.write_text(
        json.dumps(_bound(bundle, request, response=changed)) + "\n", encoding="utf-8"
    )
    second = v8.ingest(bundle, raw, normalized, rejects)
    assert second["accepted_new"] == 0 and second["rejected"] == 1
    assert "changed-raw resume conflict" in rejects.read_text(encoding="utf-8")


def test_ingest_rejects_same_response_with_changed_retry_telemetry(tmp_path):
    bundle = _bundle(tmp_path)
    _, request, _ = _first_request(bundle)
    raw = tmp_path / "raw.jsonl"
    normalized = tmp_path / "normalized.jsonl"
    rejects = tmp_path / "rejects.jsonl"
    raw.write_text(json.dumps(_bound(bundle, request)) + "\n", encoding="utf-8")
    assert v8.ingest(bundle, raw, normalized, rejects)["accepted_new"] == 1
    changed_transport = _bound(bundle, request, telemetry=_telemetry(2))
    raw.write_text(json.dumps(changed_transport) + "\n", encoding="utf-8")
    summary = v8.ingest(bundle, raw, normalized, rejects)
    assert summary["already_present_exact"] == 0
    assert summary["rejected"] == 1
    assert "full bound transport result differs" in rejects.read_text(encoding="utf-8")


def test_normalized_resume_replays_hydration_and_code_audit(tmp_path):
    bundle = _bundle(tmp_path)
    _, request, _ = _first_request(bundle)
    raw = tmp_path / "raw.jsonl"
    normalized = tmp_path / "normalized.jsonl"
    rejects = tmp_path / "rejects.jsonl"
    raw.write_text(json.dumps(_bound(bundle, request)) + "\n", encoding="utf-8")
    v8.ingest(bundle, raw, normalized, rejects)
    row = json.loads(normalized.read_text(encoding="utf-8"))
    row["result"]["matches"][0]["code_relation_audit"]["status"] = "diverged"
    row["result_sha256"] = v8.hash_value(row["result"])
    normalized.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hydration/code-audit replay mismatch"):
        v8.ingest(bundle, raw, normalized, rejects)


def test_normalized_resume_requires_exact_keys_and_replays_bound_transport(tmp_path):
    bundle = _bundle(tmp_path)
    manifest, request, _ = _first_request(bundle)
    raw = tmp_path / "raw.jsonl"
    normalized = tmp_path / "normalized.jsonl"
    rejects = tmp_path / "rejects.jsonl"
    raw.write_text(json.dumps(_bound(bundle, request)) + "\n", encoding="utf-8")
    v8.ingest(bundle, raw, normalized, rejects)
    row = json.loads(normalized.read_text(encoding="utf-8"))
    assert set(row) == v8._NORMALIZED_KEYS
    row["decorative_unchecked_field"] = True
    with pytest.raises(ValueError, match="exact contract"):
        v8._verify_normalized_row(
            row, request=request, manifest=manifest,
            bundle_manifest_sha256=v8.hash_file(bundle / "manifest.json"),
        )

    row.pop("decorative_unchecked_field")
    row["bound_transport_result"]["telemetry"]["physical_attempt_count"] = 2
    row["transport_result_sha256"] = v8.hash_value(row["bound_transport_result"])
    with pytest.raises(ValueError, match="attempt ledger length mismatch"):
        v8._verify_normalized_row(
            row, request=request, manifest=manifest,
            bundle_manifest_sha256=v8.hash_file(bundle / "manifest.json"),
        )


def test_bound_result_rejects_attempt_count_or_api_payload_drift(tmp_path):
    bundle = _bundle(tmp_path)
    manifest, request, _ = _first_request(bundle)
    row = _bound(bundle, request)
    row["telemetry"]["physical_attempt_count"] = 2
    with pytest.raises(ValueError, match="attempt ledger length mismatch"):
        v8.verify_bound_result(
            row, request=request, manifest=manifest,
            bundle_manifest_sha256=v8.hash_file(bundle / "manifest.json"),
        )
    row = _bound(bundle, request)
    row["api_payload_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="api_payload_sha256"):
        v8.verify_bound_result(
            row, request=request, manifest=manifest,
            bundle_manifest_sha256=v8.hash_file(bundle / "manifest.json"),
        )


@pytest.mark.parametrize(
    ("telemetry", "message"),
    [
        (_telemetry(3), "exceeds frozen max_attempts"),
        ({
            **_telemetry(2),
            "attempts": [
                {"attempt_index": 1, "outcome": "success", "http_status": 200,
                 "error_type": None},
                {"attempt_index": 2, "outcome": "success", "http_status": 200,
                 "error_type": None},
            ],
        }, "only the final physical attempt"),
        ({
            **_telemetry(1),
            "attempts": [
                {"attempt_index": 1, "outcome": "success", "http_status": 500,
                 "error_type": None},
            ],
        }, "requires HTTP 2xx"),
        ({
            **_telemetry(1),
            "attempts": [
                {"attempt_index": 2, "outcome": "success", "http_status": 200,
                 "error_type": None},
            ],
        }, "consecutively numbered"),
    ],
)
def test_bound_result_enforces_frozen_physical_attempt_shape(
    tmp_path, telemetry, message
):
    bundle = _bundle(tmp_path)
    manifest, request, _ = _first_request(bundle)
    row = _bound(bundle, request, telemetry=telemetry)
    with pytest.raises(ValueError, match=message):
        v8.verify_bound_result(
            row, request=request, manifest=manifest,
            bundle_manifest_sha256=v8.hash_file(bundle / "manifest.json"),
        )
