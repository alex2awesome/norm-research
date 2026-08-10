import json

import pytest

from scripts.tools.silver_match_v3.ingest_claude_structured_labels import (
    apply_overrides,
    collect,
)


def _session(path, labels):
    event = {
        "type": "assistant",
        "message": {
            "model": "claude-test",
            "content": [
                {
                    "type": "tool_use",
                    "name": "StructuredOutput",
                    "input": {"labels": labels},
                }
            ],
        },
    }
    path.write_text(json.dumps(event) + "\n", encoding="utf-8")


def test_collect_preserves_expected_order(tmp_path):
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    _session(first, [{"norm_uid": "b", "decision": "MATCH"}])
    _session(second, [{"norm_uid": "a", "decision": "NOISE"}])
    rows, models = collect([first, second], [{"norm_uid": "a"}, {"norm_uid": "b"}])
    assert [row["norm_uid"] for row in rows] == ["a", "b"]
    assert models[str(first)] == ["claude-test"]


def test_collect_rejects_incomplete_sessions(tmp_path):
    path = tmp_path / "session.jsonl"
    _session(path, [{"norm_uid": "a"}])
    with pytest.raises(ValueError, match="do not equal expected"):
        collect([path], [{"norm_uid": "a"}, {"norm_uid": "b"}])


def test_apply_overrides_preserves_order_and_replaces_whole_label():
    rows = [
        {"norm_uid": "a", "decision": "NOISE", "metric_id": None},
        {"norm_uid": "b", "decision": "MATCH", "metric_id": "a1"},
    ]
    overrides = [{"norm_uid": "a", "decision": "MATCH", "metric_id": "a2"}]
    result = apply_overrides(rows, overrides)
    assert result == [overrides[0], rows[1]]


def test_apply_overrides_rejects_unknown_uid():
    with pytest.raises(ValueError, match="absent"):
        apply_overrides([{"norm_uid": "a"}], [{"norm_uid": "b"}])
