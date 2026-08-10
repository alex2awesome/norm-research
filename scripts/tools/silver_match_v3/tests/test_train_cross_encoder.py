import numpy as np
import pytest

from scripts.tools.silver_match_v3.train_cross_encoder import (
    CELabel,
    audit_source_group_splits,
    build_explicit_split_map,
    gate_report,
    merge_teacher_rows,
    tune_gate,
)


def label(uid, decision, metric=None):
    return CELabel(uid, "c", "t", uid, "dev", uid, decision, metric, (), "strong", ())


def test_gate_report_counts_wrong_metric_as_false_positive_and_false_negative():
    labels = [label("u1", "MATCH", "a1"), label("u2", "NO_CANDIDATE_FITS")]
    scores = np.asarray([[0.7, 0.9], [0.8, 0.1]], dtype=np.float32)
    report = gate_report(labels, ["a1", "a2"], scores, 0.5, 0.0)
    assert report["confusion"] == {"tp": 0, "fp": 2, "fn": 1, "tn": 0}


def test_tune_gate_can_abstain_to_meet_precision_constraint():
    labels = [
        label("u1", "MATCH", "a1"),
        label("u2", "NO_CANDIDATE_FITS"),
        label("u3", "MATCH", "a1"),
    ]
    scores = np.asarray([[0.95, 0.1], [0.6, 0.2], [0.4, 0.3]], dtype=np.float32)
    report = tune_gate(
        labels,
        ["a1", "a2"],
        scores,
        min_precision=0.9,
        min_predictions=1,
        min_precision_lower=0.0,
    )
    assert report["precision_constraint_met"]
    assert report["exact_match_precision"] >= 0.9
    assert report["exact_match_recall"] > 0


def test_tune_gate_rejects_underpowered_perfect_precision():
    labels = [label("u1", "MATCH", "a1"), label("u2", "NO_CANDIDATE_FITS")]
    scores = np.asarray([[0.99, 0.01], [0.1, 0.09]], dtype=np.float32)
    report = tune_gate(
        labels,
        ["a1", "a2"],
        scores,
        min_precision=0.9,
        min_predictions=20,
        min_precision_lower=0.8,
    )
    assert report["precision_constraint_met"] is False
    assert report["predicted_match_count"] < 20


def test_explicit_split_map_preserves_predeclared_roles(tmp_path):
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    train.write_text('{"norm_uid":"u1"}\n{"norm_uid":"u2"}\n')
    dev.write_text('{"norm_uid":"u3"}\n')
    mapping, provenance = build_explicit_split_map(
        {"train": [train], "dev": [dev], "test": []}
    )
    assert mapping == {"u1": "train", "u2": "train", "u3": "dev"}
    assert provenance[str(train)]["role"] == "train"
    assert len(provenance[str(train)]["sha256"]) == 64


def test_explicit_split_map_rejects_uid_across_roles(tmp_path):
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    train.write_text('{"norm_uid":"u1"}\n')
    dev.write_text('{"norm_uid":"u1"}\n')
    with pytest.raises(ValueError, match="appears in explicit roles"):
        build_explicit_split_map({"train": [train], "dev": [dev]})


def test_source_group_audit_rejects_cross_role_family():
    values = [label("u1", "MATCH", "a1"), label("u2", "MATCH", "a1")]
    values[0] = CELabel(
        values[0].norm_uid,
        values[0].corpus,
        values[0].task,
        "shared-family",
        "train",
        values[0].query,
        values[0].decision,
        values[0].metric_id,
        values[0].acceptable_metric_ids,
        values[0].supervision_strength,
        values[0].teacher_sources,
    )
    values[1] = CELabel(
        values[1].norm_uid,
        values[1].corpus,
        values[1].task,
        "shared-family",
        "dev",
        values[1].query,
        values[1].decision,
        values[1].metric_id,
        values[1].acceptable_metric_ids,
        values[1].supervision_strength,
        values[1].teacher_sources,
    )
    with pytest.raises(ValueError, match="source groups cross CE roles"):
        audit_source_group_splits(values)


def test_optimize_truth_requires_explicit_ce_bridge():
    norms = {
        "u1": {
            "norm_uid": "u1",
            "task": "t",
            "corpus": "c",
            "row": 0,
            "source_id": "s1",
            "statement": "be clear",
        }
    }
    row = {
        "norm_uid": "u1",
        "task": "t",
        "decision": "MATCH",
        "metric_id": "a1",
        "gepa_role": "optimize",
    }
    with pytest.raises(ValueError, match="lacks a policy-bound CE bridge"):
        merge_teacher_rows(
            [("truth.jsonl", row)],
            norms,
            "t",
            {"a1"},
            "bank",
            split_seed=1,
            split_by_uid={"u1": "train"},
        )
    row["ce_training_eligible"] = True
    labels, _ = merge_teacher_rows(
        [("truth.jsonl", row)],
        norms,
        "t",
        {"a1"},
        "bank",
        split_seed=1,
        split_by_uid={"u1": "train"},
    )
    assert labels[0].split == "train"
