import pytest

from scripts.tools.silver_match_v3.subset_candidates_by_references import (
    reference_uids,
    select_candidate_rows,
)


def _reference(uid: str) -> dict:
    return {
        "norm_uid": uid,
        "task": "humor",
        "current_bank_source_sha256": "bank",
    }


def _candidate(uid: str, depth: int = 3) -> dict:
    return {
        "norm_uid": uid,
        "task": "humor",
        "bank_source_sha256": "bank",
        "candidates": [{"metric_id": f"m{i}"} for i in range(depth)],
    }


def test_selects_exact_union_and_validates_depth():
    wanted, counts = reference_uids(
        [("train", [_reference("b")]), ("dev", [_reference("a")])],
        task="humor",
        bank_hash="bank",
    )
    rows, audit = select_candidate_rows(
        [_candidate("unused"), _candidate("b"), _candidate("a")],
        wanted=wanted,
        task="humor",
        bank_hash="bank",
        expected_k=3,
    )
    assert counts == {"train": 1, "dev": 1}
    assert [row["norm_uid"] for row in rows] == ["a", "b"]
    assert audit["selected_rows"] == audit["requested_uids"] == 2
    assert audit["candidate_depth_counts"] == {"3": 2}


def test_rejects_reference_uid_across_inputs():
    with pytest.raises(ValueError, match="multiple inputs"):
        reference_uids(
            [("train", [_reference("x")]), ("dev", [_reference("x")])],
            task="humor",
            bank_hash="bank",
        )


def test_rejects_missing_candidate_uid():
    with pytest.raises(ValueError, match="misses 1"):
        select_candidate_rows(
            [_candidate("a")],
            wanted={"a", "b"},
            task="humor",
            bank_hash="bank",
            expected_k=3,
        )


def test_rejects_duplicate_topk_metric():
    row = _candidate("a")
    row["candidates"][2]["metric_id"] = "m1"
    with pytest.raises(ValueError, match="empty/duplicated"):
        select_candidate_rows(
            [row],
            wanted={"a"},
            task="humor",
            bank_hash="bank",
            expected_k=3,
        )
