from scripts.tools.silver_match_v3.audit_frozen_anchor_labels import audit, wilson


def test_audit_counts_exact_metric_and_typed_decisions():
    labels = [
        {"norm_uid": "a", "decision": "MATCH", "metric_id": "a1"},
        {"norm_uid": "b", "decision": "MATCH_FAMILY_ONLY", "metric_id": None},
        {"norm_uid": "c", "decision": "NOISE", "metric_id": None},
    ]
    gold = [
        {"norm_uid": "a", "decision": "MATCH", "metric_id": "a1", "stratum": "x"},
        {"norm_uid": "b", "decision": "MATCH", "metric_id": "a2", "stratum": "x"},
        {"norm_uid": "c", "decision": "NOISE", "metric_id": None, "stratum": "y"},
    ]
    report = audit(labels, gold)
    assert report["decision_correct"] == 2
    assert report["exact_correct"] == 2
    assert report["confusion"]["MATCH->MATCH_FAMILY_ONLY"] == 1
    assert report["by_stratum"]["x"]["n"] == 2


def test_wilson_is_bounded_and_nonempty_for_empty_sample():
    assert wilson(0, 0) == [0.0, 1.0]
    low, high = wilson(7, 10)
    assert 0.0 < low < 0.7 < high < 1.0
