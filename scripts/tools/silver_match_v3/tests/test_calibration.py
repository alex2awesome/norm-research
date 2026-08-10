from scripts.tools.silver_match_v3.make_calibration import (
    diverse_matches,
    split_for,
    split_group_for,
)
from scripts.tools.silver_match_v3.score_calibration import adjudication_score


def test_diverse_matches_round_robins_metrics():
    rows = [
        {"metric_id": "a0", "norm_uid": f"{i:064x}"} for i in range(10)
    ] + [{"metric_id": "a1", "norm_uid": f"{100 + i:064x}"} for i in range(2)]
    selected = diverse_matches(rows, 4)
    assert {row["metric_id"] for row in selected[:2]} == {"a0", "a1"}


def test_split_is_stable():
    uid = "b" * 64
    assert split_for(uid) == split_for(uid)


def test_split_group_prefers_paper_then_source():
    base = {"corpus": "peer", "norm_uid": "u", "source_id": "review-1"}
    assert split_group_for(base) == "peer:source:review-1"
    assert split_group_for({**base, "paper_id": "paper-1"}) == "peer:paper:paper-1"


def test_manual_typed_abstention_requires_same_type():
    teacher = {"decision": "NO_EXPLICIT_CRITERION"}
    assert adjudication_score(teacher, {"decision": "NO_EXPLICIT_CRITERION"}) == 1.0
    assert adjudication_score(teacher, {"decision": "GENERIC_VERDICT"}) == 0.0
