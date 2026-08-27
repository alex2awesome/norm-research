from __future__ import annotations

import json

import pytest

from methods.metric_seam.verifiers.math_a12_llm_contract import (
    PARSER_VERSION,
    REQUEST_SCHEMA,
    RationalExpressionPair,
    compile_request,
    parse_response,
    validate_response_envelope,
)
from methods.metric_seam.verifiers.schema import SchemaError, Span


PAIR = RationalExpressionPair(
    pair_id="opaque-pair-017",
    lhs_display=r"(x^2 - 1)/(x - 1)",
    rhs_display=r"x + 1",
    lhs_span=Span("answers/item-017.tex", 12, 12),
    rhs_span=Span("answers/item-017.tex", 14, 14),
)


def _raw(*, applies: bool, violated: bool, witnesses: list[Span]) -> str:
    return json.dumps(
        {
            "applies": applies,
            "violated": violated,
            "witnesses": [span.to_json_value() for span in witnesses],
        }
    )


def test_request_is_deterministic_pass_specific_and_split_bound() -> None:
    one = compile_request(
        pair=PAIR, pass_index=1, model="claude-pinned", split="compiler_train"
    )
    again = compile_request(
        pair=PAIR, pass_index=1, model="claude-pinned", split="compiler_train"
    )
    second_pass = compile_request(
        pair=PAIR, pass_index=2, model="claude-pinned", split="compiler_train"
    )
    heldout = compile_request(
        pair=PAIR, pass_index=1, model="claude-pinned", split="sealed_heldout"
    )

    assert one == again
    assert one["schema"] == REQUEST_SCHEMA
    assert one["split"] == "compiler_train"
    assert len({one["request_sha256"], second_pass["request_sha256"], heldout["request_sha256"]}) == 3
    assert one["response_contract"] == {
        "parser_version": PARSER_VERSION,
        "floats_allowed": False,
        "applicable_witnesses": "exact_supplied_pair_spans",
    }
    assert '"confidence"' not in one["system_prompt"]


@pytest.mark.parametrize(
    ("violated", "state"),
    [(False, "satisfied"), (True, "violated")],
)
def test_applicable_response_accepts_exact_pair_spans(
    violated: bool, state: str
) -> None:
    parsed = parse_response(
        _raw(
            applies=True,
            violated=violated,
            witnesses=[PAIR.lhs_span, PAIR.rhs_span],
        ),
        pair=PAIR,
    )
    assert parsed.verdict.state == state
    assert parsed.parser_version == PARSER_VERSION
    assert parsed.parse_mode == "strict_json"


def test_nonapplicable_requires_no_witness_and_fenced_json_is_disclosed() -> None:
    raw = _raw(applies=False, violated=False, witnesses=[])
    parsed = parse_response(f"```json\n{raw}\n```", pair=PAIR)
    assert parsed.verdict.state == "not_applicable"
    assert parsed.parse_mode == "fence_unwrapped"


def test_single_json_fence_with_explanatory_prose_is_recovered() -> None:
    raw = _raw(
        applies=True,
        violated=True,
        witnesses=[PAIR.lhs_span, PAIR.rhs_span],
    )
    parsed = parse_response(f"Explanation that is not retained.\n```json\n{raw}\n```", pair=PAIR)
    assert parsed.verdict.state == "violated"
    assert parsed.parse_mode == "embedded_fence"


def test_multiple_fenced_blocks_are_rejected_as_ambiguous() -> None:
    raw = _raw(applies=False, violated=False, witnesses=[])
    with pytest.raises(SchemaError, match="multiple fenced"):
        parse_response(f"```json\n{raw}\n```\n```json\n{raw}\n```", pair=PAIR)


@pytest.mark.parametrize(
    "witnesses",
    [
        [],
        [PAIR.lhs_span],
        [PAIR.rhs_span],
        [PAIR.lhs_span, Span("answers/other.tex", 14, 14)],
        [PAIR.lhs_span, PAIR.rhs_span, PAIR.rhs_span],
    ],
)
def test_applicable_response_must_cite_both_and_only_supplied_spans(
    witnesses: list[Span],
) -> None:
    with pytest.raises(SchemaError):
        parse_response(
            _raw(applies=True, violated=True, witnesses=witnesses), pair=PAIR
        )


def test_response_rejects_floats_confidence_and_rationale() -> None:
    with pytest.raises(SchemaError):
        parse_response(
            '{"applies":true,"violated":false,"witnesses":[],"confidence":0.9}',
            pair=PAIR,
        )
    with pytest.raises(SchemaError):
        parse_response(
            '{"applies":false,"violated":false,"witnesses":[],"rationale":"x"}',
            pair=PAIR,
        )


def test_envelope_binds_digest_pair_and_split() -> None:
    request = compile_request(
        pair=PAIR, pass_index=2, model="claude-pinned", split="compiler_train"
    )
    raw = _raw(
        applies=True,
        violated=False,
        witnesses=[PAIR.lhs_span, PAIR.rhs_span],
    )
    validated = validate_response_envelope(
        {"request_sha256": request["request_sha256"], "raw_response": raw},
        request,
    )
    assert validated["pair_id"] == PAIR.pair_id
    assert validated["pass_index"] == 2
    assert validated["split"] == "compiler_train"

    with pytest.raises(SchemaError, match="digest mismatch"):
        validate_response_envelope(
            {"request_sha256": "0" * 64, "raw_response": raw}, request
        )


def test_pair_record_is_bounded_and_file_line_qualified() -> None:
    with pytest.raises(ValueError, match="bounded"):
        RationalExpressionPair(
            "opaque",
            "x" * 4097,
            "x",
            Span("answers/a.tex", 1, 1),
            Span("answers/a.tex", 1, 1),
        )
    with pytest.raises(ValueError, match="file/line"):
        RationalExpressionPair(
            "opaque",
            "x",
            "x",
            Span("answers/a.tex", 1, 1, node_id="lhs"),
            Span("answers/a.tex", 1, 1),
        )


def test_coincident_pair_spans_require_one_unique_witness() -> None:
    pair = RationalExpressionPair(
        "same-line",
        "x + 0",
        "x",
        Span("answers/a.tex", 9, 9),
        Span("answers/a.tex", 9, 9),
    )
    parsed = parse_response(
        _raw(applies=True, violated=False, witnesses=[pair.lhs_span]), pair=pair
    )
    assert parsed.verdict.state == "satisfied"
