import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.build_verifier_author_training_packet import (
    build_packet,
)


def _json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload) + "\n")
    return path


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_exact_targets_and_optimize_only_contract(tmp_path):
    common = {
        "task": "press-releases",
        "gepa_role": "optimize",
        "predeclared_split": "train",
        "prompt_gradient_eligible": True,
    }
    truth = _jsonl(
        tmp_path / "truth.jsonl",
        [
            {**common, "norm_uid": "a", "source_group": "ga", "decision": "MATCH", "metric_id": "m1"},
            {**common, "norm_uid": "b", "source_group": "gb", "decision": "MATCH", "metric_id": "m2"},
            {**common, "norm_uid": "c", "source_group": "gc", "decision": "NO_CANDIDATE_FITS", "metric_id": None},
        ],
    )
    items = _jsonl(
        tmp_path / "items.jsonl",
        [
            {"norm_uid": uid, "task": "press-releases", "corpus": "c", "source_group": f"g{uid}", "norm": uid}
            for uid in "abc"
        ],
    )
    proposals = _jsonl(
        tmp_path / "proposals.jsonl",
        [
            {"norm_uid": "a", "task": "press-releases", "decision": "MATCH", "metric_id": "m1"},
            {"norm_uid": "b", "task": "press-releases", "decision": "MATCH", "metric_id": "m1"},
            {"norm_uid": "c", "task": "press-releases", "decision": "MATCH", "metric_id": "m2"},
        ],
    )
    bank = _json(
        tmp_path / "bank.json",
        {"task": "press-releases", "metrics": [{"metric_id": "m1"}, {"metric_id": "m2"}]},
    )
    report = build_packet(
        task="press-releases",
        truth_path=truth,
        items_path=items,
        bank_path=bank,
        proposals_path=proposals,
        output_root=tmp_path / "out",
    )
    assert report["target_counts"] == {"CONFIRM_MATCH": 1, "REJECT": 2}
    assert report["fresh_verifier_dev_truth_read"] is False


def test_non_optimize_truth_fails_closed(tmp_path):
    truth = _jsonl(
        tmp_path / "truth.jsonl",
        [{"norm_uid": "a", "task": "t", "gepa_role": "verifier_dev", "predeclared_split": "train", "prompt_gradient_eligible": True}],
    )
    items = _jsonl(tmp_path / "items.jsonl", [{"norm_uid": "a"}])
    proposals = _jsonl(tmp_path / "proposals.jsonl", [{"norm_uid": "a"}])
    bank = _json(tmp_path / "bank.json", {"task": "t", "metrics": [{"metric_id": "m"}]})
    with pytest.raises(ValueError, match="optimize-only"):
        build_packet(
            task="t",
            truth_path=truth,
            items_path=items,
            bank_path=bank,
            proposals_path=proposals,
            output_root=tmp_path / "out",
        )
