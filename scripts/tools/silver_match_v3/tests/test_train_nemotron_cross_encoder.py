import numpy as np
import pytest
import torch

from scripts.tools.silver_match_v3.train_nemotron_cross_encoder import (
    BINARY_CLASS_TO_ID,
    CLASS_TO_ID,
    PairExample,
    DeterministicWeightedSampler,
    attention_mask_mean_pool,
    bidirectional_collate,
    class_quotas,
    deterministic_source_split,
    deterministic_weighted_indices,
    grouped_exact_gate_report,
    output_label_id,
    pair_example_from_row,
    sampling_weights,
    tune_dev_thresholds,
)


def _example(
    uid: str,
    metric_id: str,
    label: str,
    *,
    source_group: str | None = None,
) -> PairExample:
    return PairExample(
        norm_uid=uid,
        source_group=source_group or f"group:{uid}",
        metric_id=metric_id,
        norm_text=f"norm {uid}",
        evidence=f"evidence {uid}",
        metric_card=f"card {metric_id}",
        label=label,
    )


def _probabilities(exact_scores):
    values = []
    for exact in exact_scores:
        remainder = 1.0 - exact
        values.append([exact, remainder * 0.4, remainder * 0.6])
    return np.asarray(values, dtype=np.float64)


def test_attention_mask_mean_pool_excludes_padding_and_rejects_empty_rows():
    hidden = torch.tensor(
        [
            [[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]],
            [[2.0, 4.0], [50.0, 50.0], [50.0, 50.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    pooled = attention_mask_mean_pool(hidden, mask)
    assert torch.equal(pooled, torch.tensor([[2.0, 4.0], [2.0, 4.0]]))
    with pytest.raises(ValueError, match="all-padding"):
        attention_mask_mean_pool(hidden[:1], torch.zeros((1, 3), dtype=torch.long))


def test_weighted_sampler_is_exact_deterministic_and_rank_sharded():
    labels = ["EXACT", "EXACT", "FAMILY", "REJECT", "REJECT", "REJECT"]
    first = deterministic_weighted_indices(
        labels, num_samples=40, seed=713, epoch=2
    )
    second = deterministic_weighted_indices(
        labels, num_samples=40, seed=713, epoch=2
    )
    assert first == second
    assert class_quotas(40) == {"EXACT": 10, "FAMILY": 10, "REJECT": 20}
    observed = {name: 0 for name in ("EXACT", "FAMILY", "REJECT")}
    for index in first:
        observed[labels[index]] += 1
    assert observed == {"EXACT": 10, "FAMILY": 10, "REJECT": 20}

    rank0 = list(
        DeterministicWeightedSampler(
            labels, num_samples=40, seed=713, rank=0, world_size=2
        )
    )
    rank1 = list(
        DeterministicWeightedSampler(
            labels, num_samples=40, seed=713, rank=1, world_size=2
        )
    )
    reconstructed = [value for pair in zip(rank0, rank1) for value in pair]
    assert reconstructed == deterministic_weighted_indices(
        labels, num_samples=40, seed=713, epoch=0
    )


def test_source_disjoint_split_is_stable_and_keeps_groups_together():
    rows = []
    for group_index in range(80):
        group = f"source:{group_index}"
        rows.extend(
            [
                _example(f"u{group_index}a", "a0", "EXACT", source_group=group),
                _example(f"u{group_index}b", "a1", "REJECT", source_group=group),
            ]
        )
    train_a, dev_a, audit_a = deterministic_source_split(
        rows, seed=19, dev_fraction=0.2
    )
    train_b, dev_b, audit_b = deterministic_source_split(
        rows, seed=19, dev_fraction=0.2
    )
    assert train_a == train_b
    assert dev_a == dev_b
    assert audit_a == audit_b
    assert {row.source_group for row in train_a}.isdisjoint(
        {row.source_group for row in dev_a}
    )
    assert audit_a["source_group_overlap_count"] == 0


def test_grouped_gate_uses_top_candidate_margin():
    rows = [
        _example("u1", "a0", "EXACT"),
        _example("u1", "a1", "FAMILY"),
        _example("u2", "a0", "EXACT"),
        _example("u2", "a2", "REJECT"),
    ]
    probabilities = _probabilities([0.90, 0.89, 0.85, 0.40])
    report = grouped_exact_gate_report(
        rows, probabilities, score_threshold=0.5, margin_threshold=0.10
    )
    # u1 is correctly ranked but rejected because its candidate margin is .01;
    # u2 is retained with a .45 margin.
    assert report["confusion"] == {"tp": 1, "fp": 0, "fn": 1, "tn": 0}
    assert report["predicted_exact_count"] == 1
    assert report["exact_precision"] == 1.0


def test_threshold_tuning_meets_exact_precision_and_wilson_gate_deterministically():
    rows = [
        _example("u1", "a0", "EXACT"),
        _example("u1", "a1", "REJECT"),
        _example("u2", "a0", "EXACT"),
        _example("u2", "a2", "REJECT"),
        _example("u3", "a1", "REJECT"),
        _example("u3", "a2", "REJECT"),
    ]
    # Only u1 is a safe retained exact result.  u2's wrong candidate and u3's
    # abstention candidate have lower P(EXACT) and must be gated away.
    probabilities = _probabilities([0.96, 0.10, 0.70, 0.82, 0.65, 0.20])
    first = tune_dev_thresholds(
        rows,
        probabilities,
        min_exact_precision=1.0,
        min_wilson_lower=0.20,
        min_exact_predictions=1,
        score_grid_size=9,
        margin_grid_size=7,
    )
    second = tune_dev_thresholds(
        rows,
        probabilities,
        min_exact_precision=1.0,
        min_wilson_lower=0.20,
        min_exact_predictions=1,
        score_grid_size=9,
        margin_grid_size=7,
    )
    assert first == second
    assert first["precision_wilson_gate_met"] is True
    assert first["exact_precision"] == 1.0
    assert first["exact_precision_wilson_95_lower"] >= 0.20
    assert first["predicted_exact_count"] == 1
    assert first["confusion"]["tp"] == 1


def test_bidirectional_collate_builds_both_concatenation_orders_without_model():
    class FakeTokenizer:
        def __init__(self):
            self.first = None
            self.second = None

        def __call__(self, first, *, text_pair, **kwargs):
            self.first = list(first)
            self.second = list(text_pair)
            assert kwargs["max_length"] == 1024
            return {
                "input_ids": torch.arange(24).reshape(4, 6),
                "attention_mask": torch.ones((4, 6), dtype=torch.long),
            }

    tokenizer = FakeTokenizer()
    rows = [_example("u1", "a0", "EXACT"), _example("u2", "a1", "FAMILY")]
    batch = bidirectional_collate(tokenizer, max_length=1024)(rows)
    assert tokenizer.first == [
        rows[0].norm_evidence,
        rows[0].metric_card,
        rows[1].norm_evidence,
        rows[1].metric_card,
    ]
    assert tokenizer.second == [
        rows[0].metric_card,
        rows[0].norm_evidence,
        rows[1].metric_card,
        rows[1].norm_evidence,
    ]
    assert batch["input_ids"].shape == (2, 2, 6)
    assert batch["labels"].tolist() == [
        CLASS_TO_ID["EXACT"],
        CLASS_TO_ID["FAMILY"],
    ]


def test_pair_relation_overrides_norm_level_decision() -> None:
    row = pair_example_from_row(
        {
            "norm_uid": "u1",
            "source_group": "humor:source:one",
            "metric_id": "a7",
            "query": "The human explicitly dislikes cruelty.",
            "metric_card": "Timing and pacing",
            "relation": "REJECT",
            "decision": "MATCH",
        }
    )
    assert row.label == "REJECT"


def test_binary_mode_maps_exact_to_one_and_family_reject_to_zero() -> None:
    assert output_label_id("EXACT", "binary") == BINARY_CLASS_TO_ID["EXACT"] == 1
    assert output_label_id("FAMILY", "binary") == 0
    assert output_label_id("REJECT", "binary") == 0
    assert sampling_weights("binary", 0.5) == {"NON_EXACT": 0.5, "EXACT": 0.5}
    assert sampling_weights("binary", 0.25) == {"NON_EXACT": 0.75, "EXACT": 0.25}
    assert class_quotas(40, sampling_weights("binary", 0.25)) == {
        "NON_EXACT": 30,
        "EXACT": 10,
    }


def test_binary_sampling_is_deterministic_and_collapses_both_negative_types() -> None:
    binary_labels = ["EXACT", "NON_EXACT", "NON_EXACT"]
    weights = sampling_weights("binary", 0.5)
    first = deterministic_weighted_indices(
        binary_labels, num_samples=20, seed=99, weights=weights
    )
    second = deterministic_weighted_indices(
        binary_labels, num_samples=20, seed=99, weights=weights
    )
    assert first == second
    assert sum(binary_labels[index] == "EXACT" for index in first) == 10

    class FakeTokenizer:
        def __call__(self, first, *, text_pair, **kwargs):
            del text_pair, kwargs
            return {
                "input_ids": torch.zeros((len(first), 2), dtype=torch.long),
                "attention_mask": torch.ones((len(first), 2), dtype=torch.long),
            }

    rows = [
        _example("u1", "a0", "EXACT"),
        _example("u2", "a1", "FAMILY"),
        _example("u3", "a2", "REJECT"),
    ]
    batch = bidirectional_collate(
        FakeTokenizer(), max_length=8, classification_mode="binary"
    )(rows)
    assert batch["labels"].tolist() == [1, 0, 0]


def test_binary_norm_gate_uses_dev_threshold_without_argmax_requirement() -> None:
    rows = [
        _example("u1", "right", "EXACT"),
        _example("u1", "wrong", "REJECT"),
        _example("u2", "wrong", "REJECT"),
        _example("u2", "also-wrong", "FAMILY"),
    ]
    # The correct u1 candidate has P(EXACT)=.49, so NON_EXACT is argmax. Binary
    # mode may still retain it under a dev-selected .45 score/.20 margin policy.
    probabilities = np.asarray([[0.51, 0.49], [0.80, 0.20], [0.60, 0.40], [0.90, 0.10]])
    report = grouped_exact_gate_report(
        rows,
        probabilities,
        score_threshold=0.45,
        margin_threshold=0.20,
        classification_mode="binary",
    )
    assert report["confusion"] == {"tp": 1, "fp": 0, "fn": 0, "tn": 3}
    assert report["pair_three_way_accuracy"] is None
    tuned = tune_dev_thresholds(
        rows,
        probabilities,
        min_exact_precision=1.0,
        min_wilson_lower=0.0,
        min_exact_predictions=1,
        score_grid_size=5,
        margin_grid_size=5,
        classification_mode="binary",
    )
    assert tuned["classification_mode"] == "binary"
    assert tuned["exact_precision"] == 1.0
