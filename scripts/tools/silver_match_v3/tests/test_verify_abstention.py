from scripts.tools.silver_match_v3.verify_abstention_gemma import (
    build_prompt,
    parse_response,
)


def test_parse_typed_abstention_and_possible_match():
    parsed, error = parse_response(
        'prefix {"decision":"NO_CANDIDATE_FITS","metric_id":null,"confidence":"high","reason":"explicit but absent"}'
    )
    assert error is None
    assert parsed["decision"] == "NO_CANDIDATE_FITS"
    parsed, error = parse_response(
        '{"decision":"POSSIBLE_EXACT_BANK_MATCH","metric_id":null,"confidence":"medium","reason":"specific criterion"}'
    )
    assert error is None
    assert parsed["decision"] == "POSSIBLE_EXACT_BANK_MATCH"


def test_parse_rejects_metric_and_unknown_type():
    assert parse_response(
        '{"decision":"NOISE","metric_id":"a1","confidence":"high","reason":"x"}'
    )[1] == "metric_on_abstention_verification"
    assert parse_response(
        '{"decision":"MATCH","metric_id":null,"confidence":"high","reason":"x"}'
    )[1] == "unknown_decision"


def test_prompt_contains_exhaustive_coverage_and_trial_evidence():
    prompt = build_prompt(
        "SYSTEM",
        {"task": "t", "norm": "too vague", "context": "The answer is too vague.", "kind": "critique", "polarity": "negative"},
        {
            "rescue_bank_count": 104,
            "provisional_decision": "NO_CANDIDATE_FITS",
            "vote_counts": {"NO_CANDIDATE_FITS": 3},
            "trial_results": [{"trial": 0, "decision": "NO_CANDIDATE_FITS", "confidence": "high", "reason": "no exact card"}],
        },
    )
    assert "FROZEN BANK SIZE: 104" in prompt
    assert "BANK COVERAGE: exhaustive" in prompt
    assert "trial 0" in prompt
    assert "too vague" in prompt
