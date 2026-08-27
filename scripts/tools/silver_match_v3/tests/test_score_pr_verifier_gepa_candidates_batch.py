from scripts.tools.silver_match_v3.score_pr_verifier_gepa_candidates_batch import _score


def test_two_order_exact_high_policy() -> None:
    targets = {
        "p": {"target": "CONFIRM_MATCH"},
        "n": {"target": "REJECT"},
        "r": {"target": "CONFIRM_MATCH"},
    }
    primary = {uid: {"metric_id": "a1"} for uid in targets}
    original = {
        "p": {"decision": "CONFIRM_MATCH", "metric_id": "a1", "confidence": "high", "parse_error": None},
        "n": {"decision": "CONFIRM_MATCH", "metric_id": "a1", "confidence": "high", "parse_error": None},
        "r": {"decision": "CONFIRM_MATCH", "metric_id": "a1", "confidence": "medium", "parse_error": None},
    }
    hashed = {
        "p": {"decision": "CONFIRM_MATCH", "metric_id": "a1", "confidence": "high", "parse_error": None},
        "n": {"decision": "NO_CANDIDATE_FITS", "metric_id": None, "confidence": "high", "parse_error": None},
        "r": {"decision": "CONFIRM_MATCH", "metric_id": "a1", "confidence": "high", "parse_error": None},
    }
    result = _score(targets, primary, original, hashed)
    assert result["retained"] == 1
    assert result["retained_true"] == 1
    assert result["false_retained"] == 0
    assert result["retained_precision"] == 1.0
    assert result["retained_recall_of_correct_proposals"] == 0.5
