import json
from pathlib import Path

from scripts.tools.silver_match_v3.build_manifest import (
    _context_window,
    _segment_record_counts,
    scored_deploy_norms,
)


def _jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_scored_deploy_alignment_restores_metadata(tmp_path):
    deploy = tmp_path / "deploy.jsonl"
    score = tmp_path / "score.jsonl"
    passage = "Full passage with line one and more context"
    _jsonl(
        deploy,
        [
            {"unit_id": "doc0", "signals": []},
            {
                "unit_id": "doc1",
                "signals": [
                    {
                        "signal_text": "more context is needed",
                        "passage_text": passage,
                        "signal_type": "suggestion",
                        "polarity": "negative",
                    },
                    {
                        "signal_text": "bare fact",
                        "passage_text": "bare fact",
                        "signal_type": "observation",
                        "polarity": "neutral",
                    },
                ],
            },
        ],
    )
    _jsonl(
        score,
        [
            {
                "unit_id": "doc1",
                "scored": [
                    {
                        "signal_text": "more context is needed",
                        "passage_text": passage[:200],
                        "faithful": 1,
                        "valid": 1,
                        "reason": "specific recommendation",
                    },
                    {
                        "signal_text": "bare fact",
                        "passage_text": "bare fact",
                        "faithful": 1,
                        "valid": 0,
                        "reason": "not evaluative",
                    },
                ],
            }
        ],
    )
    rows = list(scored_deploy_norms("fixture", "code-review", [(deploy, score)]))
    assert len(rows) == 2
    assert rows[0]["context"] == passage
    assert rows[0]["polarity"] == "negative"
    assert rows[0]["kind"] == "suggestion"
    assert rows[0]["source_record_row"] == 0
    assert rows[0]["source_signal_index"] == 0
    assert rows[0]["extraction_valid"] == 1
    assert rows[1]["norm"] == "bare fact"
    assert rows[1]["extraction_valid"] == 0


def test_segment_counts_detect_live_append_only_score(tmp_path):
    deploy = tmp_path / "deploy.jsonl"
    score = tmp_path / "score.jsonl"
    _jsonl(
        deploy,
        [
            {"unit_id": "empty", "signals": []},
            {"unit_id": "one", "signals": [{"signal_text": "one"}]},
            {"unit_id": "two", "signals": [{"signal_text": "two"}]},
        ],
    )
    _jsonl(score, [{"unit_id": "one", "scored": []}])
    assert _segment_record_counts(deploy, score) == (2, 1)


def test_context_grounding_allows_only_whitespace_normalization():
    source = "prefix\nThe high-level  idea is novel\n(suffix)"
    assert "The high-level idea is novel" in _context_window(
        source, "The high-level idea is novel"
    )
    assert _context_window(source, "a paraphrase not in the source") == ""
