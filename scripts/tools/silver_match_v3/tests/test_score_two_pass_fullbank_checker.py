from scripts.tools.silver_match_v3.score_two_pass_fullbank_checker import score_rows


def _label(uid: str, metric: str, confidence: str = "high") -> dict:
    return {
        "norm_uid": uid,
        "decision": "MATCH",
        "metric_id": metric,
        "confidence": confidence,
    }


def test_two_pass_rule_requires_same_high_exact_proposal() -> None:
    uids = [f"u{i}" for i in range(21)]
    primary = {uid: _label(uid, "a1") for uid in uids}
    truth = {uid: _label(uid, "a1") for uid in uids}
    truth["u20"] = _label("u20", "a2")
    labels_a = {uid: _label(uid, "a1") for uid in uids}
    labels_b = {uid: _label(uid, "a1") for uid in uids}
    labels_b["u20"] = _label("u20", "a2")
    result = score_rows(
        labels_a,
        labels_b,
        primary,
        truth,
        minimum_retained=20,
        minimum_point_precision=0.9,
        minimum_wilson_lower=0.8,
    )
    assert result["retained"] == 20
    assert result["retained_true"] == 20
    assert result["false_retained"] == 0
    assert result["all_gates_pass"] is True


def test_medium_confidence_does_not_enter_high_policy() -> None:
    uids = [f"u{i}" for i in range(20)]
    primary = {uid: _label(uid, "a1") for uid in uids}
    truth = {uid: _label(uid, "a1") for uid in uids}
    labels_a = {uid: _label(uid, "a1") for uid in uids}
    labels_b = {uid: _label(uid, "a1") for uid in uids}
    labels_b["u0"] = _label("u0", "a1", "medium")
    result = score_rows(
        labels_a,
        labels_b,
        primary,
        truth,
        minimum_retained=20,
        minimum_point_precision=0.9,
        minimum_wilson_lower=0.8,
    )
    assert result["retained"] == 19
    assert result["all_gates_pass"] is False
