import pytest

from scripts.tools.silver_match_v3.evaluate_retrieval import evaluate


def _relations():
    metrics = {
        "a0": {
            "equivalent_metric_ids": ["a0", "a1"],
            "family_ids": ["f0"],
            "family_metric_ids": ["a0", "a1", "a2"],
        },
        "a1": {
            "equivalent_metric_ids": ["a0", "a1"],
            "family_ids": ["f0"],
            "family_metric_ids": ["a0", "a1", "a2"],
        },
        "a2": {
            "equivalent_metric_ids": ["a2"],
            "family_ids": ["f0"],
            "family_metric_ids": ["a0", "a1", "a2"],
        },
        "a3": {
            "equivalent_metric_ids": ["a3"],
            "family_ids": [],
            "family_metric_ids": ["a3"],
        },
    }
    return {
        "relation_schema_version": "test-v1",
        "pair_labels_sha256": "pairs",
        "tasks": {
            "peer-review": {
                "bank_source_sha256": "bank",
                "metric_relations": metrics,
            }
        },
    }


def _teacher(uid="u0", metric="a0"):
    return {
        "norm_uid": uid,
        "corpus": "pr",
        "task": "peer-review",
        "decision": "MATCH",
        "metric_id": metric,
        "current_bank_source_sha256": "bank",
    }


def _candidate(uid, ids):
    return {
        "norm_uid": uid,
        "task": "peer-review",
        "bank_source_sha256": "bank",
        "candidates": [{"metric_id": value} for value in ids],
    }


def test_exact_equivalence_and_family_are_reported_separately():
    teachers = [_teacher("u0"), _teacher("u1"), _teacher("u2")]
    candidates = {
        "u0": _candidate("u0", ["a0"]),
        "u1": _candidate("u1", ["a1"]),
        "u2": _candidate("u2", ["a2"]),
    }
    report, _ = evaluate(teachers, candidates, _relations(), [1])
    overall = report["overall"]
    assert overall["exact_recall_at_1"] == pytest.approx(1 / 3)
    assert overall["equivalence_recall_at_1"] == pytest.approx(2 / 3)
    assert overall["family_recall_at_1"] == 1.0
    assert overall["family_credit_gain_at_1"] == pytest.approx(2 / 3)


def test_missing_candidate_counts_as_miss_and_lowers_coverage():
    report, _ = evaluate([_teacher("u0")], {}, _relations(), [1, 3])
    assert report["overall"]["candidate_coverage"] == 0.0
    assert report["overall"]["family_recall_at_3"] == 0.0


def test_candidate_bank_provenance_is_required():
    candidate = _candidate("u0", ["a0"])
    candidate.pop("bank_source_sha256")
    with pytest.raises(ValueError, match="lacks bank_source_sha256"):
        evaluate([_teacher("u0")], {"u0": candidate}, _relations(), [1])


def test_teacher_bank_provenance_is_required():
    teacher = _teacher("u0")
    teacher.pop("current_bank_source_sha256")
    with pytest.raises(ValueError, match="lacks current bank provenance"):
        evaluate([teacher], {"u0": _candidate("u0", ["a0"])}, _relations(), [1])


def test_unknown_candidate_metric_is_rejected():
    with pytest.raises(ValueError, match="outside peer-review bank"):
        evaluate(
            [_teacher("u0")],
            {"u0": _candidate("u0", ["a999"])},
            _relations(),
            [1],
        )


def test_duplicate_teacher_uid_is_rejected():
    with pytest.raises(ValueError, match="duplicate norm_uid"):
        evaluate([_teacher("u0"), _teacher("u0")], {}, _relations(), [1])
