import json

import pytest

from scripts.tools.silver_match_v3.prepare_ce_eligible_truth import partition


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _row(uid, decision, metric_id=None, **extra):
    return {
        "norm_uid": uid, "task": "humor", "source_group": f"g-{uid}",
        "split": "train", "decision": decision, "metric_id": metric_id,
        "current_bank_source_sha256": "bank", **extra,
    }


def test_unanchored_family_is_typed_only_not_guessed(tmp_path):
    path = tmp_path / "truth.jsonl"
    _write(path, [
        _row("exact", "MATCH", "m0"),
        _row("family-no-anchor", "MATCH_FAMILY_ONLY"),
        _row("family-anchor", "MATCH_FAMILY_ONLY", family_metric_ids=["m1"]),
        _row("reject", "NO_CANDIDATE_FITS"),
    ])
    eligible, excluded, report = partition(path)
    assert {row["norm_uid"] for row in eligible} == {"exact", "family-anchor", "reject"}
    assert [row["norm_uid"] for row in excluded] == ["family-no-anchor"]
    assert excluded[0]["gemma_typed_eligible"] is True
    assert report["policy"]["family_anchor_inference_from_free_text_reason"] is False


def test_source_group_crossing_fails(tmp_path):
    path = tmp_path / "truth.jsonl"
    rows = [_row("a", "MATCH", "m0"), _row("b", "NOISE")]
    rows[0]["source_group"] = rows[1]["source_group"] = "same"
    rows[1]["split"] = "dev"
    _write(path, rows)
    with pytest.raises(ValueError, match="cross splits"):
        partition(path)
