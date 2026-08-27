import json

from scripts.tools.silver_match_v3.audit_cross_encoder_metric_priors import build_audit


def test_confirms_saturated_metric_prior_collapse(tmp_path):
    bank = tmp_path / "bank.json"
    bank.write_text(
        json.dumps({"metrics": [{"metric_id": "a1"}, {"metric_id": "a2"}, {"metric_id": "a3"}]})
    )
    pairs = tmp_path / "pairs.jsonl"
    rows = [
        {"metric_id": "a1", "label": 1.0, "kind": "positive"},
        {"metric_id": "a1", "label": 1.0, "kind": "positive"},
        {"metric_id": "a2", "label": 0.0, "kind": "hard_negative"},
    ]
    pairs.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "task": "humor",
                "frozen_test_consumed": False,
                "base_dev": {
                    "top_score_quantiles": {"0.5": 0.2},
                    "margin_quantiles": {"0.5": 0.04},
                    "ungated_exact_recall_at_50": 0.5,
                },
                "selected_dev": {
                    "top_score_quantiles": {"0.5": 0.9998},
                    "margin_quantiles": {"0.5": 0.00005},
                    "ungated_exact_recall_at_50": 0.2,
                    "predicted_match_count": 0,
                },
            }
        )
    )
    audit = build_audit(report, pairs, bank, "humor")
    assert audit["status"] == "CONFIRMED_METRIC_CARD_PRIOR_COLLAPSE"
    assert audit["pair_audit"]["positive_only_metric_count"] == 1
    assert audit["pair_audit"]["unexposed_metric_count"] == 1
    assert audit["role_audit"]["permanent_blind_consumed"] is False


def test_does_not_confirm_without_score_saturation(tmp_path):
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"metrics": [{"metric_id": "a1"}, {"metric_id": "a2"}]}))
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(
        json.dumps({"metric_id": "a1", "label": 1.0, "kind": "positive"})
        + "\n"
        + json.dumps({"metric_id": "a2", "label": 0.0, "kind": "negative"})
        + "\n"
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "task": "humor",
                "frozen_test_consumed": False,
                "base_dev": {
                    "top_score_quantiles": {"0.5": 0.2},
                    "margin_quantiles": {"0.5": 0.04},
                    "ungated_exact_recall_at_50": 0.5,
                },
                "selected_dev": {
                    "top_score_quantiles": {"0.5": 0.7},
                    "margin_quantiles": {"0.5": 0.02},
                    "ungated_exact_recall_at_50": 0.6,
                    "predicted_match_count": 1,
                },
            }
        )
    )
    audit = build_audit(report, pairs, bank, "humor")
    assert audit["status"] == "DIAGNOSIS_NOT_CONFIRMED"
