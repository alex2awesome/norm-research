import json

from methods.metric_seam.verifiers.construct_probe import (
    PATENT_ANTECEDENT_PROPOSAL,
    compile_request,
    compile_sample_requests,
    parse_response,
)


def test_full_context_probe_is_frozen_and_hash_sampled() -> None:
    rows = [{"item_key": f"i{i}", "ctext": f"line {i}"} for i in range(40)]
    first = compile_sample_requests(PATENT_ANTECEDENT_PROPOSAL, rows, model="test-model")
    second = compile_sample_requests(PATENT_ANTECEDENT_PROPOSAL, list(reversed(rows)), model="test-model")
    assert [r["request_sha256"] for r in first] == [r["request_sha256"] for r in second]
    assert len(first) == 32
    assert all(r["proposal"]["authorship_constraints"]["detector_source_seen"] is False for r in first)


def test_parser_accepts_one_embedded_fence_and_binds_document_lines() -> None:
    raw = 'Result:\n```json\n{"applies":true,"violated":false,"witnesses":[{"path":"document.txt","start_line":1,"end_line":1}]}\n```'
    verdict, mode = parse_response(raw, ctext="claim text")
    assert verdict.state == "satisfied"
    assert mode == "single_json_fence"


def test_request_digest_binds_full_proposal() -> None:
    request = compile_request(PATENT_ANTECEDENT_PROPOSAL, item_key="x", ctext="one\ntwo", model="m")
    assert request["ctext"] == "one\ntwo"
    assert request["proposal"]["relation"] in request["user_prompt"]
    assert "000001|one\n000002|two" in request["user_prompt"]
    assert request["response_contract"]["transport_version"] == "line-numbered-full-context.v2"
