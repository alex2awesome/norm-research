import json

from scripts.tools.silver_match_v3.adjudicate_gemma import (
    batched_work,
    build_item_prompt,
    ordered_candidates,
    parse_response,
    prompt_equivalence_groups,
    scan_candidate_input,
)


def test_match_must_be_candidate_constrained():
    value, error = parse_response(
        '{"decision":"MATCH","metric_id":"a9","confidence":"high","reason":"fits"}',
        {"a1", "a2"},
    )
    assert value is None
    assert error == "metric_not_in_candidates"


def test_abstention_must_not_carry_metric():
    value, error = parse_response(
        '{"decision":"CONTEXT_NEEDED","metric_id":"a1","confidence":"high","reason":"it"}',
        {"a1"},
    )
    assert value is None
    assert error == "metric_on_abstention"


def test_valid_abstention():
    value, error = parse_response(
        '{"decision":"GENERIC_VERDICT","metric_id":null,"confidence":"medium","reason":"only holistic"}',
        {"a1"},
    )
    assert error is None
    assert value["decision"] == "GENERIC_VERDICT"


def test_numeric_confidence_is_deterministically_normalized():
    value, error = parse_response(
        '{"decision":"MATCH","metric_id":"a1","confidence":0.84,"reason":"fits"}',
        {"a1"},
    )
    assert error is None
    assert value["confidence"] == "high"
    assert value["confidence_raw"] == 0.84


def test_json_parser_handles_braces_inside_reason_string():
    value, error = parse_response(
        '```json\n{"decision":"CONTEXT_NEEDED","metric_id":null,'
        '"confidence":"high","reason":"what does {} : {} mean?"}\n```',
        {"a1"},
    )
    assert error is None
    assert value["decision"] == "CONTEXT_NEEDED"


def test_json_parser_repairs_literal_latex_backslash_in_reason():
    value, error = parse_response(
        '```json\n{"decision":"MATCH","metric_id":"a2","confidence":"high",'
        '"reason":"Prefer `\\|` over `||` for notation."}\n```',
        {"a2"},
    )
    assert error is None
    assert value == {
        "decision": "MATCH",
        "metric_id": "a2",
        "confidence": "high",
        "reason": "Prefer `\\|` over `||` for notation.",
    }


def test_hashed_order_is_stable():
    rows = [{"metric_id": "a1"}, {"metric_id": "a2"}, {"metric_id": "a3"}]
    assert ordered_candidates(rows, "hashed", "f" * 64) == ordered_candidates(
        rows, "hashed", "f" * 64
    )


def test_prompt_includes_grounded_context_but_not_aspect_as_evidence():
    prompt = build_item_prompt(
        "instructions",
        {
            "task": "peer-review",
            "norm": "this is unclear",
            "context": "The derivation in section two is unclear because a premise is missing.",
            "aspect": "clarity",
            "polarity": "neg",
        },
        [{"metric_id": "a0"}],
        {
            "a0": {
                "metric_id": "a0",
                "name": "Derivation clarity",
                "description": "The derivation is clear and complete.",
                "examples": [],
            }
        },
    )
    assert "EVIDENCE PASSAGE FROM THE HUMAN FEEDBACK" in prompt
    assert "EXTRACTION ASPECT HINT (weak evidence only)" in prompt


def test_prompt_equivalence_requires_identical_prompt_and_ordered_ids():
    candidates = [{"metric_id": "a1"}, {"metric_id": "a2"}]
    batch = [
        ({}, {"norm_uid": "u1"}, candidates, "same"),
        ({}, {"norm_uid": "u2"}, list(candidates), "same"),
        ({}, {"norm_uid": "u3"}, list(reversed(candidates)), "same"),
        ({}, {"norm_uid": "u4"}, candidates, "different"),
    ]
    representatives, representative_for, group_sizes = prompt_equivalence_groups(batch)
    assert representatives == [0, 2, 3]
    assert representative_for == [0, 0, 2, 3]
    assert group_sizes == {0: 2, 2: 1, 3: 1}


def test_candidate_scan_is_streaming_shard_and_resume_aware(tmp_path):
    path = tmp_path / "candidates.jsonl"
    rows = [
        {"norm_uid": "0" * 64, "corpus": "c"},
        {"norm_uid": "1" * 64, "corpus": "c"},
        {"norm_uid": "2" * 64, "corpus": "d"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    corpora, eligible = scan_candidate_input(
        path, done={"0" * 64}, shard_id=0, num_shards=1
    )
    assert corpora == {"c", "d"}
    assert eligible == 2


def test_batched_work_never_materializes_more_than_requested():
    batches = list(batched_work(iter(range(5)), 2))
    assert batches == [[0, 1], [2, 3], [4]]
