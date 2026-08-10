import json

import pytest

from scripts.tools.silver_match_v3.audit_false_abstentions import (
    audit_false_abstentions,
    clopper_pearson_lower,
    clopper_pearson_upper,
)


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_zero_of_sixty_supports_under_five_percent_one_sided():
    assert clopper_pearson_upper(0, 60) < 0.05
    assert clopper_pearson_upper(0, 58) > 0.05


def test_nonzero_exact_upper_bound_is_conservative():
    upper = clopper_pearson_upper(2, 100)
    assert 0.05 < upper < 0.07


def test_all_correct_match_precision_has_exact_lower_bound():
    assert clopper_pearson_lower(30, 30) > 0.90
    assert clopper_pearson_lower(20, 20) < 0.90


def test_conditional_false_abstention_claim(tmp_path):
    gold_path, pred_path = tmp_path / "gold.jsonl", tmp_path / "pred.jsonl"
    gold, pred = [], []
    # Sixty audited abstentions are genuinely nonmatches, which is just enough
    # for a zero-failure one-sided 95% upper bound below 5%.
    for i in range(60):
        base = {"norm_uid": f"n{i}", "task": "t", "corpus": "c"}
        gold.append({**base, "decision": "NOISE", "metric_id": None})
        pred.append({**base, "decision": "NOISE", "metric_id": None})
    # Also establish exact-match accuracy independently of abstention risk.
    gold.append(
        {
            "norm_uid": "m",
            "task": "t",
            "corpus": "c",
            "decision": "MATCH",
            "metric_id": "a1",
        }
    )
    pred.append(
        {
            "norm_uid": "m",
            "task": "t",
            "corpus": "c",
            "decision": "MATCH",
            "metric_id": "a1",
        }
    )
    _write(gold_path, gold)
    _write(pred_path, pred)
    report = audit_false_abstentions([gold_path], [pred_path])
    assert report["overall"]["predicted_abstentions"] == 60
    assert report["overall"]["false_abstentions"] == 0
    assert report["overall"]["claim_supported"] is True
    assert report["overall"]["gold_match_exact_accuracy"] == 1.0
    assert report["overall"]["predicted_matches"] == 1
    assert report["overall"]["predicted_match_exact_precision"] == 1.0
    assert report["overall"]["predicted_match_precision_claim_supported"] is False


def test_false_abstention_is_detected(tmp_path):
    gold_path, pred_path = tmp_path / "gold.jsonl", tmp_path / "pred.jsonl"
    gold = [
        {
            "norm_uid": "u",
            "task": "t",
            "corpus": "c",
            "decision": "MATCH",
            "metric_id": "a1",
        }
    ]
    pred = [
        {
            "norm_uid": "u",
            "task": "t",
            "corpus": "c",
            "decision": "NO_CANDIDATE_FITS",
            "metric_id": None,
        }
    ]
    _write(gold_path, gold)
    _write(pred_path, pred)
    report = audit_false_abstentions([gold_path], [pred_path])
    assert report["overall"]["false_abstentions"] == 1
    assert report["overall"]["false_abstention_probability"] == 1.0
    assert report["overall"]["claim_supported"] is False


def test_typed_abstention_accuracy_has_an_exact_lower_bound(tmp_path):
    gold_path, pred_path = tmp_path / "gold.jsonl", tmp_path / "pred.jsonl"
    gold, pred = [], []
    for i in range(60):
        base = {"norm_uid": f"u{i}", "task": "t", "corpus": "c"}
        decision = "NOISE" if i < 30 else "NO_CANDIDATE_FITS"
        gold.append({**base, "decision": decision, "metric_id": None})
        pred.append({**base, "decision": decision, "metric_id": None})
    _write(gold_path, gold)
    _write(pred_path, pred)
    summary = audit_false_abstentions([gold_path], [pred_path])["overall"]
    assert summary["typed_abstention_exact_correct"] == 60
    assert summary["typed_abstention_exact_accuracy"] == 1.0
    assert summary["typed_abstention_exact_accuracy_lower_bound"] > 0.90


def test_missing_prediction_fails_closed(tmp_path):
    gold_path, pred_path = tmp_path / "gold.jsonl", tmp_path / "pred.jsonl"
    _write(
        gold_path, [{"norm_uid": "u", "task": "t", "corpus": "c", "decision": "NOISE"}]
    )
    _write(pred_path, [])
    with pytest.raises(ValueError, match="predictions miss"):
        audit_false_abstentions([gold_path], [pred_path])


def test_analysis_exclusions_are_bound_and_cannot_overlap_gold(tmp_path):
    gold_path, pred_path, exclusion = (
        tmp_path / "gold.jsonl",
        tmp_path / "pred.jsonl",
        tmp_path / "exclude.jsonl",
    )
    row = {"norm_uid": "u", "task": "t", "corpus": "c", "decision": "NOISE"}
    _write(gold_path, [row])
    _write(pred_path, [row])
    _write(exclusion, [{"norm_uid": "other"}])
    report = audit_false_abstentions(
        [gold_path], [pred_path], analysis_exclusion_paths=[exclusion]
    )
    assert report["analysis_exclusions"]["count"] == 1
    _write(exclusion, [{"norm_uid": "u"}])
    with pytest.raises(ValueError, match="overlaps analysis exclusions"):
        audit_false_abstentions(
            [gold_path], [pred_path], analysis_exclusion_paths=[exclusion]
        )


def test_match_errors_separate_wrong_leaf_from_false_positive(tmp_path):
    gold_path, pred_path = tmp_path / "gold.jsonl", tmp_path / "pred.jsonl"
    gold = [
        {
            "norm_uid": "leaf",
            "task": "t",
            "corpus": "c",
            "decision": "MATCH",
            "metric_id": "a1",
        },
        {
            "norm_uid": "fp",
            "task": "t",
            "corpus": "c",
            "decision": "NOISE",
            "metric_id": None,
        },
    ]
    pred = [
        {
            "norm_uid": "leaf",
            "task": "t",
            "corpus": "c",
            "decision": "MATCH",
            "metric_id": "a2",
        },
        {
            "norm_uid": "fp",
            "task": "t",
            "corpus": "c",
            "decision": "MATCH",
            "metric_id": "a1",
        },
    ]
    _write(gold_path, gold)
    _write(pred_path, pred)
    summary = audit_false_abstentions([gold_path], [pred_path])["overall"]
    assert summary["predicted_match_wrong_leaf"] == 1
    assert summary["predicted_match_false_positive"] == 1
    assert summary["predicted_match_wrong_leaf_rate"] == 0.5
    assert summary["predicted_match_false_positive_rate"] == 0.5
