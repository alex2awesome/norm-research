import json
from pathlib import Path

import pytest

from methods.metric_seam.science_claims_v2.addressed_pipeline import (
    MANIFEST_SCHEMA,
    NORMALIZED_SCHEMA,
    RESULT_SCHEMA,
    _allowed_projection,
    hash_file,
    hydrate_response,
    ingest,
    prepare,
    segment_source,
    verify_bundle,
)


class LabelTrap(dict):
    def get(self, key, default=None):
        if key in {"y", "label", "acceptance"}:
            raise AssertionError("supervised label accessed")
        return super().get(key, default)


def _files(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({
            "paper_id": "p1",
            "y": "DO_NOT_EMIT_82f1",
            "abstract": "Method A improves by 20 percent. It outperforms B.",
            "body": (
                "Experiments show Method A improves by 20 percent over B. "
                "Table 2 confirms the gain."
            ),
        })
        + "\n"
        + json.dumps({
            "paper_id": "p2",
            "y": 1,
            "abstract": "No result is stated.",
            "body": "Background only.",
        })
        + "\n",
        encoding="utf-8",
    )
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "schema_version": "test-addressed-prompt-v1",
        "input_allowlist": ["paper_id", "abstract", "body"],
        "external_knowledge": "forbidden",
        "max_claims": 5,
        "system_prompt": "Return source addresses only.",
        "decision_semantics": {"supported": "same numeric relation"},
        "typed_relation_semantics": {"relation": ["numeric"]},
        "output_schema": {"paper_id": "string", "selections": []},
    }), encoding="utf-8")
    model = tmp_path / "model.json"
    model.write_text(json.dumps({
        "schema_version": "test-addressed-model-v1",
        "backend": "test",
        "protocol": "test",
        "model": "fixed-test-model",
        "temperature": 0.0,
        "max_output_tokens": 1000,
        "system_prompt_transport": "system",
        "response_transport": "text",
        "execution_status": "prepared_not_run",
    }), encoding="utf-8")
    return source, spec, model


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


def _response(*selections, paper_id="p1"):
    return {"paper_id": paper_id, "selections": list(selections or [_selection()])}


def _bound(bundle: Path, request: dict, response=None, **updates):
    row = {
        "schema_version": RESULT_SCHEMA,
        "request_id": request["request_id"],
        "request_sha256": request["request_sha256"],
        "model_manifest_sha256": request["model_manifest_sha256"],
        "bundle_manifest_sha256": hash_file(bundle / "manifest.json"),
        "response": response if response is not None else _response(),
    }
    row.update(updates)
    return row


def _prepared(tmp_path: Path, name="bundle"):
    source, spec, model = _files(tmp_path)
    bundle = tmp_path / name
    prepare(source, spec, model, bundle)
    _, requests = verify_bundle(bundle)
    request = sorted(requests.values(), key=lambda row: row["sequence_index"])[0]
    return bundle, request


def _ingest_one(tmp_path: Path, bundle: Path, row: dict, stem="case"):
    raw = tmp_path / f"{stem}.raw.jsonl"
    normalized = tmp_path / f"{stem}.normalized.jsonl"
    rejects = tmp_path / f"{stem}.rejects.jsonl"
    raw.write_text(json.dumps(row) + "\n", encoding="utf-8")
    summary = ingest(bundle, raw, normalized, rejects)
    return summary, normalized, rejects


def test_projection_never_requests_supervised_fields():
    source = LabelTrap(
        paper_id="p", abstract="claim", body="evidence", y=1, label=2, acceptance=3
    )
    assert _allowed_projection(source, 1) == {
        "paper_id": "p", "abstract": "claim", "body": "evidence"
    }


def test_segmenter_is_exact_addressed_and_stable():
    text = '  Accuracy is 3.14.  E.g. this stays together.\n"Final!"  '
    first = segment_source(text, section="abstract")
    second = segment_source(text, section="abstract")
    assert first == second
    assert [row["sentence_id"] for row in first] == ["A0001", "A0002", "A0003"]
    assert first[0]["text"] == "Accuracy is 3.14."
    assert first[1]["text"] == "E.g. this stays together."
    assert first[2]["text"] == '"Final!"'
    cursor = 0
    for row in first:
        assert not text[cursor:row["start"]].strip()
        assert text[row["start"]:row["end"]] == row["text"]
        cursor = row["end"]
    assert not text[cursor:].strip()
    assert segment_source("", section="body") == []
    assert segment_source(" \n\t ", section="abstract") == []


def test_prepare_binds_source_segmenter_prompt_model_and_requests(tmp_path):
    source, spec, model = _files(tmp_path)
    bundle = tmp_path / "bundle"
    manifest = prepare(source, spec, model, bundle)
    assert manifest["schema_version"] == MANIFEST_SCHEMA
    assert manifest["status"] == "prepared_not_run_no_api_calls"
    assert manifest["api_calls_made_by_prepare"] == 0
    assert manifest["gpu_used"] is False
    assert manifest["requests"]["count"] == 2
    assert manifest["requests"]["sha256"] == hash_file(bundle / "requests.jsonl")
    text = (bundle / "requests.jsonl").read_text(encoding="utf-8")
    assert "DO_NOT_EMIT_82f1" not in text
    assert "A0001" in text and "B0001" in text
    assert "TRANSPORT_GUARD" in text
    assert manifest["segmentation_contract"]["canonical_sha256"]
    assert manifest["segmentation_contract"]["identity"]["implementation_file_sha256"]
    verified, requests = verify_bundle(bundle)
    assert verified == manifest
    assert len(requests) == 2


def test_preparation_requests_and_hydration_replay_stably(tmp_path):
    source, spec, model = _files(tmp_path)
    left, right = tmp_path / "left", tmp_path / "right"
    prepare(source, spec, model, left)
    prepare(source, spec, model, right)
    assert (left / "requests.jsonl").read_bytes() == (right / "requests.jsonl").read_bytes()
    _, left_requests = verify_bundle(left)
    _, right_requests = verify_bundle(right)
    left_request = sorted(left_requests.values(), key=lambda row: row["sequence_index"])[0]
    right_request = sorted(right_requests.values(), key=lambda row: row["sequence_index"])[0]
    assert hydrate_response(_response(), left_request) == hydrate_response(
        _response(), right_request
    )


def test_verify_rejects_request_tampering(tmp_path):
    bundle, _ = _prepared(tmp_path)
    with (bundle / "requests.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="requests file hash mismatch"):
        verify_bundle(bundle)


@pytest.mark.parametrize("bad_sequence", ["0", True, -1])
def test_verify_validates_sequence_type_before_id_formatting(tmp_path, bad_sequence):
    bundle, _ = _prepared(tmp_path)
    rows = [json.loads(line) for line in (bundle / "requests.jsonl").read_text().splitlines()]
    rows[0]["sequence_index"] = bad_sequence
    (bundle / "requests.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    manifest["requests"]["sha256"] = hash_file(bundle / "requests.jsonl")
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="sequence_index must be a nonnegative integer"):
        verify_bundle(bundle)


def test_verify_requires_contiguous_sequence_indices(tmp_path):
    bundle, _ = _prepared(tmp_path)
    rows = [json.loads(line) for line in (bundle / "requests.jsonl").read_text().splitlines()]
    rows[1]["sequence_index"] = 2
    rows[1]["request_id"] = (
        f"science_addressed_0002_{rows[1]['request_sha256'][:16]}"
    )
    (bundle / "requests.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    manifest["requests"]["sha256"] = hash_file(bundle / "requests.jsonl")
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="contiguous from zero"):
        verify_bundle(bundle)


def test_ingest_hydrates_exact_spans_and_derives_bookkeeping(tmp_path):
    bundle, request = _prepared(tmp_path)
    summary, normalized, rejects = _ingest_one(tmp_path, bundle, _bound(bundle, request))
    assert summary == {
        "accepted_new": 1, "already_present": 0, "rejected": 0, "remaining": 1
    }
    assert not rejects.exists()
    row = json.loads(normalized.read_text(encoding="utf-8"))
    assert row["schema_version"] == NORMALIZED_SCHEMA
    result = row["result"]
    assert result["status"] == "supported"
    assert result["claim_count"] == result["certificate_count"] == 1
    assert result["evidence_link_count"] == 0
    certificate = result["certificates"][0]
    assert certificate["claim"]["text"] == "Method A improves by 20 percent."
    assert certificate["evidence"]["text"] == (
        "Experiments show Method A improves by 20 percent over B."
    )
    assert request["paper_input"]["abstract"][
        certificate["claim"]["start"]:certificate["claim"]["end"]
    ] == certificate["claim"]["text"]
    assert request["paper_input"]["body"][
        certificate["evidence"]["start"]:certificate["evidence"]["end"]
    ] == certificate["evidence"]["text"]
    assert result["transport"]["model_returned_source_text"] is False


@pytest.mark.parametrize(
    ("field", "bad_id", "message"),
    [
        ("claim_sentence_id", "A9999", "claim address is out of range"),
        ("evidence_sentence_id", "B9999", "evidence address is out of range"),
    ],
)
def test_ingest_rejects_out_of_range_addresses(tmp_path, field, bad_id, message):
    bundle, request = _prepared(tmp_path)
    response = _response(_selection(**{field: bad_id}))
    summary, _, rejects = _ingest_one(
        tmp_path, bundle, _bound(bundle, request, response=response), stem=field
    )
    assert summary["accepted_new"] == 0 and summary["rejected"] == 1
    assert message in rejects.read_text(encoding="utf-8")


def test_ingest_rejects_abstract_address_as_body_evidence(tmp_path):
    bundle, request = _prepared(tmp_path)
    response = _response(_selection(evidence_sentence_id="A0002"))
    summary, _, rejects = _ingest_one(
        tmp_path, bundle, _bound(bundle, request, response=response), stem="leakage"
    )
    assert summary["accepted_new"] == 0 and summary["rejected"] == 1
    assert "not body" in rejects.read_text(encoding="utf-8")


def test_ingest_rejects_duplicate_claim_and_evidence_addresses(tmp_path):
    bundle, request = _prepared(tmp_path)
    duplicate_claim = _response(
        _selection(),
        _selection(evidence_sentence_id="B0002", decision="insufficient",
                   evidence_kind="none", quantity_state="missing"),
    )
    summary, _, rejects = _ingest_one(
        tmp_path, bundle, _bound(bundle, request, response=duplicate_claim),
        stem="duplicate_claim",
    )
    assert summary["rejected"] == 1
    assert "duplicate claim address" in rejects.read_text(encoding="utf-8")

    duplicate_evidence = _response(
        _selection(),
        _selection(claim_sentence_id="A0002", decision="insufficient",
                   evidence_kind="none", quantity_state="not_required"),
    )
    summary, _, rejects = _ingest_one(
        tmp_path, bundle, _bound(bundle, request, response=duplicate_evidence),
        stem="duplicate_evidence",
    )
    assert summary["rejected"] == 1
    assert "duplicate evidence address" in rejects.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        (
            _selection(
                relation="qualitative", decision="supported", quantity_count=0,
                quantity_state="not_required", comparison_present=False,
                comparison_state="not_required", evidence_kind="qualitative_link",
            ),
            "only numeric/comparative relations may be certificates",
        ),
        (
            _selection(quantity_count=0),
            "requires at least one aligned quantity",
        ),
        (
            _selection(
                relation="comparative", quantity_count=0, quantity_state="not_required",
                comparison_state="aligned", comparison_present=False,
                evidence_kind="comparative_relation",
            ),
            "requires aligned comparative evidence",
        ),
    ],
)
def test_strong_certificate_requires_executable_numeric_or_comparative_data(
    tmp_path, selection, message
):
    bundle, request = _prepared(tmp_path)
    summary, _, rejects = _ingest_one(
        tmp_path, bundle, _bound(bundle, request, response=_response(selection)),
        stem="strong_guard_" + str(abs(hash(message))),
    )
    assert summary["accepted_new"] == 0 and summary["rejected"] == 1
    assert message in rejects.read_text(encoding="utf-8")


def test_qualitative_zero_quantity_null_comparison_is_weak_not_strong(tmp_path):
    bundle, request = _prepared(tmp_path)
    weak = _selection(
        relation="qualitative", decision="evidence_link", quantity_count=0,
        quantity_state="not_required", comparison_present=False,
        comparison_state="not_required", evidence_kind="qualitative_link",
    )
    summary, normalized, _ = _ingest_one(
        tmp_path, bundle, _bound(bundle, request, response=_response(weak)), stem="weak"
    )
    assert summary["accepted_new"] == 1
    result = json.loads(normalized.read_text(encoding="utf-8"))["result"]
    assert result["certificate_count"] == 0
    assert result["evidence_link_count"] == 1


@pytest.mark.parametrize("case", ["paper", "request_binding", "model_binding", "bundle_binding"])
def test_ingest_rejects_wrong_paper_or_cryptographic_binding(tmp_path, case):
    bundle, request = _prepared(tmp_path)
    response = _response(paper_id="wrong") if case == "paper" else _response()
    updates = {}
    if case == "request_binding":
        updates["request_sha256"] = "0" * 64
    elif case == "model_binding":
        updates["model_manifest_sha256"] = "1" * 64
    elif case == "bundle_binding":
        updates["bundle_manifest_sha256"] = "2" * 64
    summary, _, rejects = _ingest_one(
        tmp_path, bundle, _bound(bundle, request, response=response, **updates), stem=case
    )
    assert summary["accepted_new"] == 0 and summary["rejected"] == 1
    reason = rejects.read_text(encoding="utf-8")
    assert ("paper_id" in reason) if case == "paper" else ("binding" in reason)


def test_response_cannot_smuggle_copied_source_text(tmp_path):
    bundle, request = _prepared(tmp_path)
    response = _response()
    response["selections"][0]["claim_text"] = "Method A improves by 20 percent."
    summary, _, rejects = _ingest_one(
        tmp_path, bundle, _bound(bundle, request, response=response), stem="copied_text"
    )
    assert summary["rejected"] == 1
    assert "exactly the frozen typed keys" in rejects.read_text(encoding="utf-8")


def test_stable_ingest_is_resumable_without_duplicate_output(tmp_path):
    bundle, request = _prepared(tmp_path)
    raw = tmp_path / "raw.jsonl"
    normalized = tmp_path / "normalized.jsonl"
    rejects = tmp_path / "rejects.jsonl"
    raw.write_text(json.dumps(_bound(bundle, request)) + "\n", encoding="utf-8")
    first = ingest(bundle, raw, normalized, rejects)
    original = normalized.read_bytes()
    second = ingest(bundle, raw, normalized, rejects)
    assert first["accepted_new"] == 1
    assert second == {
        "accepted_new": 0, "already_present": 1, "rejected": 0, "remaining": 1
    }
    assert normalized.read_bytes() == original


def test_malformed_raw_jsonl_is_logged_and_valid_next_line_is_ingested(tmp_path):
    bundle, request = _prepared(tmp_path)
    raw = tmp_path / "mixed.raw.jsonl"
    normalized = tmp_path / "mixed.normalized.jsonl"
    rejects = tmp_path / "mixed.rejects.jsonl"
    raw.write_text(
        '{"request_id": broken\n' + json.dumps(_bound(bundle, request)) + "\n",
        encoding="utf-8",
    )
    summary = ingest(bundle, raw, normalized, rejects)
    assert summary["accepted_new"] == 1 and summary["rejected"] == 1
    rejection = json.loads(rejects.read_text(encoding="utf-8"))
    assert rejection["line_number"] == 1
    assert "malformed JSONL result" in rejection["reason"]
    assert rejection["raw_line_sha256"]


@pytest.mark.parametrize(
    "tamper",
    [
        "request_sha256", "paper_input_sha256", "source_map_sha256",
        "segmentation_contract_sha256", "prompt_spec_sha256",
        "model_manifest_sha256", "bundle_manifest_sha256", "result",
    ],
)
def test_resume_rechecks_all_bindings_and_deterministic_hydration(tmp_path, tamper):
    bundle, request = _prepared(tmp_path)
    raw = tmp_path / "resume.raw.jsonl"
    normalized = tmp_path / "resume.normalized.jsonl"
    rejects = tmp_path / "resume.rejects.jsonl"
    raw.write_text(json.dumps(_bound(bundle, request)) + "\n", encoding="utf-8")
    assert ingest(bundle, raw, normalized, rejects)["accepted_new"] == 1
    row = json.loads(normalized.read_text(encoding="utf-8"))
    if tamper == "result":
        row["result"]["status"] = "contradicted"
    else:
        row[tamper] = "f" * 64
    normalized.write_text(json.dumps(row) + "\n", encoding="utf-8")
    expected = "hydration mismatch" if tamper == "result" else "binding mismatch"
    with pytest.raises(ValueError, match=expected):
        ingest(bundle, raw, normalized, rejects)


def test_incoherent_certificate_is_rejected_before_hydration(tmp_path):
    bundle, request = _prepared(tmp_path)
    response = _response(_selection(quantity_state="mismatch"))
    summary, _, rejects = _ingest_one(
        tmp_path, bundle, _bound(bundle, request, response=response), stem="incoherent"
    )
    assert summary["rejected"] == 1
    assert "requires at least one aligned quantity" in rejects.read_text(encoding="utf-8")
