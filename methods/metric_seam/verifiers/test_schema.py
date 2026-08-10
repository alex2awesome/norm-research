from __future__ import annotations

import pytest

from methods.metric_seam.verifiers.schema import (
    SchemaError,
    Span,
    Verdict,
    load_json_no_floats,
    validate_json_no_floats,
)


def test_verdict_round_trip_and_invariants() -> None:
    verdict = Verdict.from_json(
        '{"applies":true,"violated":true,"witnesses":'
        '[{"path":"src/check.py","start_line":2,"end_line":4,'
        '"node_id":"call-1"}]}'
    )
    assert verdict == Verdict(
        True, True, (Span("src/check.py", 2, 4, node_id="call-1"),)
    )
    assert verdict.to_json_value() == {
        "applies": True,
        "violated": True,
        "witnesses": [
            {
                "path": "src/check.py",
                "start_line": 2,
                "end_line": 4,
                "node_id": "call-1",
            }
        ],
    }
    assert verdict.witnesses[0].lines() == {
        ("src/check.py", 2),
        ("src/check.py", 3),
        ("src/check.py", 4),
    }
    assert verdict.state == "violated"

    satisfied = Verdict(True, False, (Span("src/check.py", 8, 8),))
    assert satisfied.state == "satisfied"
    assert satisfied.witnesses
    assert Verdict(False, False).state == "not_applicable"

    with pytest.raises(SchemaError, match="cannot be violated"):
        Verdict(False, True, (Span("src/check.py", 1, 1),))
    with pytest.raises(SchemaError, match="cannot carry witnesses"):
        Verdict(False, False, (Span("src/check.py", 1, 1),))
    with pytest.raises(SchemaError, match="requires at least one witness"):
        Verdict(True, True)
    with pytest.raises(SchemaError, match="requires at least one witness"):
        Verdict(True, False)


@pytest.mark.parametrize(
    "start,end",
    [(0, 1), (3, 2), (True, 2), (1, False)],
)
def test_span_bounds_and_integer_types(start: object, end: object) -> None:
    with pytest.raises(SchemaError):
        Span("src/check.py", start, end)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/abs/file.py",
        "../escape.py",
        "src/../escape.py",
        "src//file.py",
        "a\\b.py",
        "C:/absolute.py",
        "~/expanded.py",
        " a.py",
    ],
)
def test_span_requires_safe_nonempty_relative_path(path: str) -> None:
    with pytest.raises(SchemaError, match="path"):
        Span(path, 1, 1)


def test_optional_node_id_must_be_nonempty_and_safe() -> None:
    assert Span("src/check.py", 1, 1, node_id=None).node_id is None
    with pytest.raises(SchemaError, match="node_id"):
        Span("src/check.py", 1, 1, node_id="")
    with pytest.raises(SchemaError, match="node_id"):
        Span("src/check.py", 1, 1, node_id="bad\nnode")


@pytest.mark.parametrize(
    "raw",
    [
        '0.0',
        '{"outer":[{"score":0.5}]}',
        'NaN',
        'Infinity',
        '{"applies":true,"violated":false,"witnesses":[],"score":1.0}',
    ],
)
def test_no_floats_at_any_json_depth(raw: str) -> None:
    with pytest.raises(SchemaError, match="floating-point"):
        load_json_no_floats(raw)


def test_schema_is_strict_about_keys_and_json_container_types() -> None:
    with pytest.raises(SchemaError, match="key mismatch"):
        Verdict.from_json(
            {"applies": True, "violated": False, "witnesses": [], "extra": 1}
        )
    with pytest.raises(SchemaError, match="not JSON-compatible"):
        Verdict.from_json({"applies": True, "violated": False, "witnesses": ()})
    with pytest.raises(SchemaError, match="not JSON-compatible"):
        validate_json_no_floats({"not": object()})
