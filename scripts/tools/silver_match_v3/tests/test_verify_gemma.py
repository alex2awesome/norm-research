from scripts.tools.silver_match_v3.verify_gemma import (
    parse_response,
    verification_prompt_equivalence_groups,
)


def test_parse_confirm_requires_primary_metric():
    parsed, error = parse_response(
        '{"decision":"CONFIRM_MATCH","metric_id":"a1","confidence":"high","reason":"wins"}',
        "a1",
        {"a2"},
    )
    assert error is None
    assert parsed["metric_id"] == "a1"

    parsed, error = parse_response(
        '{"decision":"CONFIRM_MATCH","metric_id":"a2","confidence":"high","reason":"wins"}',
        "a1",
        {"a2"},
    )
    assert parsed is None
    assert error == "confirm_metric_mismatch"


def test_parse_better_candidate_is_restricted_to_alternatives():
    parsed, error = parse_response(
        '{"decision":"BETTER_CANDIDATE","metric_id":"a2","confidence":"medium","reason":"closer"}',
        "a1",
        {"a2"},
    )
    assert error is None
    assert parsed["decision"] == "BETTER_CANDIDATE"

    parsed, error = parse_response(
        '{"decision":"BETTER_CANDIDATE","metric_id":"a3","confidence":"medium","reason":"closer"}',
        "a1",
        {"a2"},
    )
    assert parsed is None
    assert error == "better_metric_not_alternative"


def test_parse_ambiguous_requires_null_metric():
    parsed, error = parse_response(
        '{"decision":"AMBIGUOUS_MATCH","metric_id":null,"confidence":"medium","reason":"siblings tie"}',
        "a1",
        {"a2"},
    )
    assert error is None
    assert parsed["metric_id"] is None


def test_verifier_dedup_requires_same_proposal_alternatives_and_prompt():
    primary = {"metric_id": "a1"}
    alternatives = [{"metric_id": "a2"}]
    batch = [
        ({}, primary, {"norm_uid": "u1"}, alternatives, "same"),
        ({}, dict(primary), {"norm_uid": "u2"}, list(alternatives), "same"),
        ({}, {"metric_id": "a3"}, {"norm_uid": "u3"}, alternatives, "same"),
        ({}, primary, {"norm_uid": "u4"}, alternatives, "different"),
    ]
    representatives, representative_for, sizes = (
        verification_prompt_equivalence_groups(batch)
    )
    assert representatives == [0, 2, 3]
    assert representative_for == [0, 0, 2, 3]
    assert sizes == {0: 2, 2: 1, 3: 1}
