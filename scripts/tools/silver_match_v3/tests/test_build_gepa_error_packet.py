import json

import pytest

from scripts.tools.silver_match_v3.build_gepa_error_packet import build_packet
from scripts.tools.silver_match_v3.common import write_jsonl


def _fixture(tmp_path, *, split="train"):
    bank = tmp_path / "bank.json"
    bank.write_text(
        json.dumps(
            {
                "metrics": [
                    {"metric_id": "a1", "name": "gold", "description": "g"},
                    {"metric_id": "a2", "name": "pred", "description": "p"},
                ]
            }
        )
    )
    norms = tmp_path / "norms.jsonl"
    write_jsonl(
        norms,
        [
            {
                "norm_uid": "u",
                "task": "t",
                "corpus": "c",
                "row": 0,
                "norm": "criterion",
                "context": "context",
            }
        ],
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "banks": {"t": {"path": str(bank)}},
                "corpora": {"c": {"task": "t", "path": str(norms)}},
            }
        )
    )
    truth = tmp_path / "truth.jsonl"
    write_jsonl(
        truth,
        [
            {
                "norm_uid": "u",
                "task": "t",
                "corpus": "c",
                "row": 0,
                "split": split,
                "decision": "MATCH",
                "metric_id": "a1",
                "reason": "truth",
            }
        ],
    )
    original = tmp_path / "original.jsonl"
    hashed = tmp_path / "hashed.jsonl"
    prediction = [
        {
            "norm_uid": "u",
            "task": "t",
            "decision": "MATCH",
            "metric_id": "a2",
            "reason": "prediction",
        }
    ]
    write_jsonl(original, prediction)
    write_jsonl(hashed, prediction)
    candidates = tmp_path / "candidates.jsonl"
    write_jsonl(
        candidates,
        [
            {
                "norm_uid": "u",
                "task": "t",
                "candidates": [
                    {"metric_id": "a2", "rank": 1},
                    {"metric_id": "a1", "rank": 2},
                ],
            }
        ],
    )
    return manifest, truth, original, hashed, candidates


def test_builds_strict_wrong_train_packet(tmp_path):
    manifest, truth, original, hashed, candidates = _fixture(tmp_path)
    packet, summary = build_packet(
        manifest_path=manifest,
        task="t",
        truth_path=truth,
        original_path=original,
        hashed_path=hashed,
        candidates_path=candidates,
    )
    assert len(packet) == 1
    assert packet[0]["gold_metric_id"] == "a1"
    assert packet[0]["predicted_metric_id"] == "a2"
    assert summary["strict_wrong"] == 1


def test_rejects_non_train_panel(tmp_path):
    manifest, truth, original, hashed, candidates = _fixture(tmp_path, split="dev")
    with pytest.raises(ValueError, match="restricted.*train"):
        build_packet(
            manifest_path=manifest,
            task="t",
            truth_path=truth,
            original_path=original,
            hashed_path=hashed,
            candidates_path=candidates,
        )
