from __future__ import annotations

from scripts.tools.silver_match_v3.select_humor_fresh_release import (
    _select_adjudicator,
    _select_verifier,
    score_adjudicator,
    score_verifier,
)


def _truth(uid: str, decision: str, metric_id: str | None) -> dict:
    return {"norm_uid": uid, "decision": decision, "metric_id": metric_id}


def _proposal(uid: str, metric_id: str) -> dict:
    return {"norm_uid": uid, "decision": "MATCH", "metric_id": metric_id}


def _verify(uid: str, metric_id: str, keep: bool) -> dict:
    return {
        "norm_uid": uid,
        "decision": "CONFIRM_MATCH" if keep else "REJECT_MATCH",
        "metric_id": metric_id if keep else None,
        "confidence": "high",
        "parse_error": None,
    }


def test_adjudicator_score_uses_exact_leaf_and_typed_nonmatch() -> None:
    truth = {
        "a": _truth("a", "MATCH", "m1"),
        "b": _truth("b", "MATCH", "m2"),
        "c": _truth("c", "NO_CANDIDATE_FITS", None),
    }
    proposals = {
        "a": _proposal("a", "m1"),
        "b": _proposal("b", "m3"),
        "c": _proposal("c", "m4"),
    }
    score = score_adjudicator(truth, proposals)
    assert score["correct_exact_proposal_count"] == 1
    assert score["exact_proposal_precision"] == 1 / 3
    assert score["exact_proposal_recall"] == 1 / 2


def test_verifier_requires_three_high_exact_confirmations() -> None:
    truth = {
        "a": _truth("a", "MATCH", "m1"),
        "b": _truth("b", "MATCH", "m2"),
        "c": _truth("c", "NO_CANDIDATE_FITS", None),
    }
    proposals = {
        "a": _proposal("a", "m1"),
        "b": _proposal("b", "m9"),
        "c": _proposal("c", "m8"),
    }
    orders = {
        name: {
            "a": _verify("a", "m1", True),
            "b": _verify("b", "m9", name != "reverse"),
            "c": _verify("c", "m8", True),
        }
        for name in ("original", "hashed", "reverse")
    }
    score = score_verifier(
        truth,
        proposals,
        orders,
        thresholds={
            "minimum_retained": 1,
            "minimum_retained_exact_precision": 0.9,
            "minimum_retained_exact_precision_wilson_95_lower": 0.0,
        },
    )
    assert score["retained_count"] == 2
    assert score["retained_true_count"] == 1
    assert score["retained_exact_precision"] == 0.5
    assert score["eligible"] is False


def test_frozen_tie_breaks_are_lexicographically_smaller() -> None:
    adj = [
        {
            "name": name,
            "score": {
                "exact_proposal_precision_wilson_95_lower": 0.5,
                "exact_f_beta_0_5": 0.6,
                "exact_proposal_precision": 0.7,
                "exact_proposal_recall": 0.4,
            },
        }
        for name in ("z", "a")
    ]
    assert _select_adjudicator(adj)["name"] == "a"
    ver = [
        {
            "name": name,
            "score": {
                "eligible": True,
                "retained_exact_precision_wilson_95_lower": 0.8,
                "exact_f_beta_0_5": 0.7,
                "retained_exact_precision": 0.9,
                "retained_exact_recall_of_truth_matches": 0.5,
            },
        }
        for name in ("z", "a")
    ]
    assert _select_verifier(ver)["name"] == "a"
