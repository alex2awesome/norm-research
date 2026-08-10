import json

import pytest

from scripts.tools.silver_match_v3.score_optimize_two_order_adjudicator import score


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_scores_only_optimize_role(tmp_path):
    truth = tmp_path / "truth.jsonl"
    original = tmp_path / "original.jsonl"
    hashed = tmp_path / "hashed.jsonl"
    base = {
        "norm_uid": "u1",
        "task": "press-releases",
        "decision": "MATCH",
        "metric_id": "m1",
    }
    _jsonl(
        truth,
        [
            {
                **base,
                "gepa_role": "optimize",
                "split": "train",
                "prompt_gradient_eligible": True,
                "evaluation_only": False,
                "current_bank_source_sha256": "bank",
            }
        ],
    )
    prediction = {
        **base,
        "candidate_bank_source_sha256": "bank",
        "prompt_sha256": "prompt",
    }
    _jsonl(original, [prediction])
    _jsonl(hashed, [{**prediction, "decision": "NO_CANDIDATE_FITS", "metric_id": None}])
    report, errors = score(truth, original, hashed)
    assert report["role"] == "optimize_prompt_gradient_only"
    assert report["metrics"]["truth_match_count"] == 1
    assert errors[0]["error_kind"] == "order_unstable"

    rows = [json.loads(line) for line in truth.read_text().splitlines()]
    rows[0]["gepa_role"] = "select"
    _jsonl(truth, rows)
    with pytest.raises(ValueError, match="optimize-role"):
        score(truth, original, hashed)
