import numpy as np
import pytest

from scripts.tools.silver_match_v3 import train_cross_encoder as base
from scripts.tools.silver_match_v3 import train_cross_encoder_balanced as balanced


def _label(uid: str, metric_id: str) -> base.CELabel:
    return base.CELabel(
        norm_uid=uid,
        corpus="humor_multi",
        task="humor",
        source_group=f"group:{uid}",
        split="train",
        query=f"query {uid}",
        decision="MATCH",
        metric_id=metric_id,
        acceptable_metric_ids=(metric_id,),
        supervision_strength="strong",
        teacher_sources=("unit-test",),
    )


def _bank():
    return [
        {
            "metric_id": f"a{index}",
            "name": f"metric {index}",
            "definition": f"definition {index}",
            "examples": [],
        }
        for index in range(3)
    ]


def _config(minimum_negative: int = 1):
    return {
        "sampling_seed": 71317,
        "max_unique_positive_uids_per_metric": 2,
        "hard_negatives_per_match": 1,
        "global_balanced_negatives_per_match": 1,
        "hard_negatives_per_abstain": 1,
        "global_balanced_negatives_per_abstain": 1,
        "min_negative_exposure_per_bank_metric": minimum_negative,
        "target_negative_to_positive_pair_ratio": 1.0,
        "minimum_negative_to_positive_pair_ratio": 1.0,
        "maximum_positive_pair_fraction_per_metric": 0.5,
    }


def test_balanced_sampler_is_deterministic_and_passes_all_metric_gates(monkeypatch):
    labels = [
        _label("u0a", "a0"),
        _label("u0b", "a0"),
        _label("u1a", "a1"),
        _label("u1b", "a1"),
        _label("u2a", "a2"),
        _label("u2b", "a2"),
    ]
    monkeypatch.setattr(
        base,
        "lexical_rankings",
        lambda rows, cards: np.tile(np.arange(len(cards)), (len(rows), 1)),
    )
    candidates = {
        label.norm_uid: [metric_id for metric_id in ("a0", "a1", "a2")]
        for label in labels
    }
    balanced._CONFIG = _config()
    balanced._AUDIT_ONLY = False
    first = balanced.build_balanced_training_pairs(
        labels,
        _bank(),
        candidates,
        negatives_per_positive=2,
        negatives_per_abstain=2,
        strong_positive_repeats=1,
    )
    first_audit = balanced._PAIR_AUDIT
    balanced._CONFIG = _config()
    second = balanced.build_balanced_training_pairs(
        labels,
        _bank(),
        candidates,
        negatives_per_positive=2,
        negatives_per_abstain=2,
        strong_positive_repeats=1,
    )
    assert first == second
    assert first_audit["status"] == "PASS"
    assert first_audit["failed_metric_count"] == 0
    assert all(
        row["negative_pairs"] >= row["positive_pairs"]
        and row["positive_pair_fraction"] <= 0.5
        for row in first_audit["per_metric_exposure"]
    )


def test_balanced_sampler_fails_before_training_when_exposure_is_infeasible(monkeypatch):
    labels = [_label("u0", "a0"), _label("u1", "a1"), _label("u2", "a2")]
    monkeypatch.setattr(
        base,
        "lexical_rankings",
        lambda rows, cards: np.tile(np.arange(len(cards)), (len(rows), 1)),
    )
    balanced._CONFIG = _config(minimum_negative=10)
    balanced._AUDIT_ONLY = False
    with pytest.raises(ValueError, match="metric exposure gates failed before training"):
        balanced.build_balanced_training_pairs(
            labels,
            _bank(),
            {},
            negatives_per_positive=2,
            negatives_per_abstain=2,
            strong_positive_repeats=1,
        )
