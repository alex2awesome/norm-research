from scripts.tools.silver_match_v3.score_teacher_verification_audit import join_key, score


def test_scores_retained_precision_and_injected_exact_rate():
    rows = [
        {
            "norm_uid": "a", "gemma_outcome": "retained",
            "proposal_retrieval_status": "natural_top_k",
            "audit_stratum": ["retained", "natural_top_k", "common"],
            "manual_decision": "EXACT_MATCH", "manual_reason": "same construct",
        },
        {
            "norm_uid": "b", "gemma_outcome": "retained",
            "proposal_retrieval_status": "injected_for_verification",
            "audit_stratum": ["retained", "injected_for_verification", "rare"],
            "manual_decision": "WRONG_METRIC", "manual_reason": "different leaf",
        },
        {
            "norm_uid": "c", "gemma_outcome": "other_rejection",
            "proposal_retrieval_status": "injected_for_verification",
            "audit_stratum": ["other_rejection", "injected_for_verification", "rare"],
            "manual_decision": "EXACT_MATCH", "manual_reason": "proposal was right",
        },
    ]
    report = score(rows)
    assert report["retained_exact_precision"]["estimate"] == 0.5
    assert report["retained_exact_precision_design_weighted"]["estimate"] == 0.5
    assert report["rejected_wrong_proposal_rate"]["estimate"] == 0.0
    assert report["exact_proposal_injected_rate"]["estimate"] == 0.5


def test_joins_blind_labels_to_machine_key():
    labels = [{"norm_uid": "a", "manual_decision": "EXACT_MATCH"}]
    key = [{"norm_uid": "a", "gemma_outcome": "retained"}]
    assert join_key(labels, key)[0]["gemma_outcome"] == "retained"
