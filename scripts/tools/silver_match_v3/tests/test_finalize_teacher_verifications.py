from scripts.tools.silver_match_v3.finalize_teacher_verifications import finalize


def test_two_order_high_precision_finalization():
    proposals = [
        {"norm_uid": "keep", "metric_id": "a1", "task": "t", "corpus": "c"},
        {"norm_uid": "contrast", "metric_id": "a1", "task": "t", "corpus": "c"},
    ]
    first = [
        {"norm_uid": "keep", "decision": "CONFIRM_MATCH", "metric_id": "a1", "confidence": "high", "order_mode": "hashed", "prompt_sha256": "prompt"},
        {"norm_uid": "contrast", "decision": "BETTER_CANDIDATE", "metric_id": "a2", "confidence": "high", "order_mode": "hashed", "prompt_sha256": "prompt"},
    ]
    second = [
        {"norm_uid": "keep", "decision": "CONFIRM_MATCH", "metric_id": "a1", "confidence": "medium", "order_mode": "reverse", "prompt_sha256": "prompt"},
        {"norm_uid": "contrast", "decision": "BETTER_CANDIDATE", "metric_id": "a2", "confidence": "medium", "order_mode": "reverse", "prompt_sha256": "prompt"},
    ]
    retained, contrasts, rejected, report = finalize(
        proposals,
        first,
        second,
        require_one_high=True,
        retrieval_injected={"keep": True, "contrast": False},
        selected_prompt_sha256="prompt",
        requires_independent_audit=True,
        calibration_power_status="underpowered",
    )
    assert [row["norm_uid"] for row in retained] == ["keep"]
    assert contrasts[0]["preferred_metric_id"] == "a2"
    assert [row["norm_uid"] for row in rejected] == ["contrast"]
    assert report["counts"]["retained"] == 1
    assert retained[0]["proposal_retrieval_status"] == "injected_for_verification"
    assert retained[0]["gradient_eligible"] is False
    assert contrasts[0]["proposal_retrieval_status"] == "natural_top_k"
    assert report["proposal_retrieval_counts"] == {
        "injected_for_verification": 1,
        "natural_top_k": 1,
    }
    assert report["order_stability"] == {
        "decision_agreement": 1.0,
        "decision_metric_agreement": 1.0,
    }
    assert report["calibration_power_status"] == "underpowered"
