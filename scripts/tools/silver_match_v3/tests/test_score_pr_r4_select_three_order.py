from scripts.tools.silver_match_v3.score_pr_r4_select_three_order import summarize_three


def test_three_order_strict_consensus_is_exact_and_typed():
    truth = [
        {"norm_uid": "m", "decision": "MATCH", "metric_id": "a1", "corpus": "c"},
        {"norm_uid": "n", "decision": "NO_CANDIDATE_FITS", "metric_id": None, "corpus": "c"},
        {"norm_uid": "u", "decision": "MATCH", "metric_id": "a2", "corpus": "c"},
    ]
    predictions = {
        order: {
            "m": {"decision": "MATCH", "metric_id": "a1"},
            "n": {"decision": "NO_CANDIDATE_FITS", "metric_id": None},
            "u": {
                "decision": "MATCH",
                "metric_id": "a2" if order != "reverse" else "a3",
            },
        }
        for order in ("original", "hashed", "reverse")
    }
    report = summarize_three(truth, predictions)
    strict = report["strict_all_three_consensus"]
    assert strict["confirmed_match_count"] == 1
    assert strict["correct_exact_id_count"] == 1
    assert strict["strict_typed_abstention_correct"] == 1
    assert report["order"]["any_exact_disagreement_count"] == 1
