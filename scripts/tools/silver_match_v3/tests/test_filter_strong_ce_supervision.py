import pytest

from scripts.tools.silver_match_v3.filter_strong_ce_supervision import (
    filter_strong_rows,
)


def _norm(uid: str, group: str) -> dict:
    return {
        "norm_uid": uid,
        "task": "math-stackexchange",
        "corpus": "math",
        "source_id": group,
    }


def _row(uid: str, decision: str, metric_id=None, **extra) -> dict:
    return {
        "norm_uid": uid,
        "task": "math-stackexchange",
        "current_bank_source_sha256": "bank",
        "decision": decision,
        "metric_id": metric_id,
        **extra,
    }


def test_filter_excludes_weak_forced_and_future_dev_groups():
    norms = {
        "a": _norm("a", "g-a"),
        "b": _norm("b", "g-b"),
        "c": _norm("c", "g-c"),
    }
    teachers = [
        _row("a", "MATCH", "m1", label_source="human"),
        _row("b", "MATCH", "m2", label_source="sonnet_forced_top3"),
        _row("c", "NO_CANDIDATE_FITS", None, label_source="human"),
    ]
    output, audit = filter_strong_rows(
        teacher_inputs=[("teachers", teachers)],
        reference_inputs=[("dev", [_row("c", "MATCH", "m1")])],
        norms=norms,
        task="math-stackexchange",
        bank_hash="bank",
        bank_ids={"m1", "m2"},
    )
    assert [row["norm_uid"] for row in output] == ["a"]
    assert audit["exclusions"]["excluded_weak_forced"] == 1
    assert audit["exclusions"]["excluded_future_dev_source_group"] == 1
    assert audit["weak_forced_rows_used_as_exact_positives"] == 0


def test_filter_rejects_conflicting_strong_exact_labels():
    norms = {"a": _norm("a", "g-a")}
    with pytest.raises(ValueError, match="conflicting strong supervision"):
        filter_strong_rows(
            teacher_inputs=[
                ("one", [_row("a", "MATCH", "m1")]),
                ("two", [_row("a", "MATCH", "m2")]),
            ],
            reference_inputs=[],
            norms=norms,
            task="math-stackexchange",
            bank_hash="bank",
            bank_ids={"m1", "m2"},
        )


def test_filter_rejects_nonmatch_with_metric_id():
    norms = {"a": _norm("a", "g-a")}
    with pytest.raises(ValueError, match="typed nonmatch carries metric ID"):
        filter_strong_rows(
            teacher_inputs=[("one", [_row("a", "NO_CANDIDATE_FITS", "m1")])],
            reference_inputs=[],
            norms=norms,
            task="math-stackexchange",
            bank_hash="bank",
            bank_ids={"m1"},
        )
