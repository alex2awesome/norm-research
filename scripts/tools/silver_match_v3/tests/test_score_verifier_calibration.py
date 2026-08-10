from scripts.tools.silver_match_v3.score_verifier_calibration import score


def test_high_precision_filter_metrics():
    truth = [
        {"norm_uid": "good", "metric_id": "a1"},
        {"norm_uid": "bad", "metric_id": "a2"},
    ]
    primary = [
        {"norm_uid": "good", "metric_id": "a1"},
        {"norm_uid": "bad", "metric_id": "a1"},
    ]
    verified = [
        {"norm_uid": "good", "decision": "CONFIRM_MATCH", "metric_id": "a1", "confidence": "high"},
        {"norm_uid": "bad", "decision": "BETTER_CANDIDATE", "metric_id": "a2", "confidence": "high"},
    ]
    report, errors = score(truth, primary, verified)
    assert report["retained_precision"] == 1.0
    assert report["retained_precision_wilson_95"][0] < 1.0
    assert report["wrong_proposal_rejection_rate"] == 1.0
    assert report["conflict_exact_correction_rate"] == 1.0
    assert errors == []
