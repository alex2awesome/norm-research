"""Reconstruction-MCQ experimental-design and selection-channel tests."""
from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pytest

from methods.metric_implementer.recon_channel import (
    _balanced_option_permutations,
    _candidate_sims,
    _exact_contrastive_example_indices,
    _mcq_recon,
    _permuted_label_examples,
    mcq_identity_channel,
    mcq_logit_values_from_precomputed_behaviors,
    mcq_value_from_precomputed_behavior,
    run_metric,
)


def _metric(metric_id: str, description: str, body: str | None = None):
    return SimpleNamespace(
        metric_id=metric_id,
        name=description,
        description=description,
        body=body or description,
        meta={},
    )


def test_distractor_statistics_cannot_see_heldout_behavior():
    target = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    design = np.arange(4)
    same_on_design = target.copy()
    same_on_design[4:] = 1 - same_on_design[4:]
    differs_on_design = target.copy()
    differs_on_design[:2] = 1 - differs_on_design[:2]
    pool = [
        (_metric("heldout_only", "heldout only"), same_on_design),
        (_metric("design_difference", "design difference"), differs_on_design),
    ]

    stats = {row["met"].metric_id: row for row in _candidate_sims(
        target, pool, "target", design_indices=design)}
    assert stats["heldout_only"]["n_disagree"] == 0
    assert stats["design_difference"]["n_disagree"] == 2

    # Arbitrarily changing the lockbox still cannot change a design statistic.
    pool[1] = (pool[1][0], np.r_[differs_on_design[:4], [1, 1, 1, 1]])
    changed = {row["met"].metric_id: row for row in _candidate_sims(
        target, pool, "target", design_indices=design)}
    assert changed["design_difference"]["n_disagree"] == 2
    assert changed["design_difference"]["kappa"] == pytest.approx(
        stats["design_difference"]["kappa"])


def test_exact_teaching_set_separates_every_distractor_and_balances_target():
    target = np.array([.9, .9, .1, .1, .9, .1, .9, .1, .9, .1])
    alternatives = [
        np.array([.1, .9, .1, .1, .1, .1, .9, .1, .9, .1]),
        np.array([.9, .1, .1, .1, .9, .9, .9, .1, .9, .1]),
        np.array([.9, .9, .9, .1, .9, .1, .1, .1, .9, .1]),
    ]
    selected, design = _exact_contrastive_example_indices(
        target, alternatives, n_examples=7, min_disagreements=2)

    assert len(selected) == 7
    assert design["per_distractor_disagreements_demonstrated"] == [2, 2, 2]
    assert design["target_example_counts"]["0"] > 0
    assert design["target_example_counts"]["1"] > 0
    again, _ = _exact_contrastive_example_indices(
        target, alternatives, n_examples=7, min_disagreements=2)
    assert np.array_equal(selected, again)


def test_teaching_set_fails_closed_for_behaviorally_unidentified_option():
    target = np.array([.1, .9, .1, .9, .1, .9, .1, .9])
    clone = target.copy()
    with pytest.raises(ValueError, match="lack the declared design-split separation"):
        _exact_contrastive_example_indices(
            target, [clone], n_examples=4, min_disagreements=1)


def test_option_permutations_are_exactly_counterbalanced_in_complete_blocks():
    permutations = _balanced_option_permutations(4, 12, seed=3)
    for canonical_option in range(4):
        positions = [int(np.flatnonzero(p == canonical_option)[0]) for p in permutations]
        assert [positions.count(pos) for pos in range(4)] == [3, 3, 3, 3]
    assert all(sorted(p.tolist()) == [0, 1, 2, 3] for p in permutations)


def test_shuffled_label_control_preserves_marginal_but_breaks_pairing():
    records = [(i, f"item {i}", score) for i, score in enumerate([0, 0, 1, 1])]
    rendered = _permuted_label_examples(records, seed=2)
    scores = [int(x) for x in re.findall(r"\[score=(\d)\]", rendered)]
    assert sorted(scores) == [0, 0, 1, 1]
    assert scores != [0, 0, 1, 1]


class _DescriptionSelector:
    """Select the displayed option containing TARGET, regardless of condition."""

    def generate_batch(self, prompts, **_kwargs):
        outputs = []
        for prompt in prompts:
            choice = next(
                int(number) for number, description in
                re.findall(r"(?m)^(\d+)\. (.+)$", prompt)
                if "TARGET" in description
            )
            outputs.append(f'{{"choice": {choice}}}')
        return outputs


class _BinaryExecutor:
    def generate_batch(self, prompts, **_kwargs):
        return ["YES" if "TARGET BODY" in prompt else "NO" for prompt in prompts]


class _LogitSelector:
    def score_choices(self, prompts, choices, **_kwargs):
        rows = []
        for prompt in prompts:
            target = next(
                int(number) - 1 for number, description in
                re.findall(r"(?m)^(\d+)\. (.+)$", prompt)
                if "TARGET" in description
            )
            probs = np.full(len(choices), 0.1)
            probs[target] = 0.7
            rows.append(probs.tolist())
        return rows


class _ConditionSensitiveLogitSelector:
    """Assign target probability .8 with annotations and .2 in both controls."""

    def score_choices(self, prompts, choices, **kwargs):
        rows = []
        seeds = kwargs.get("seed")
        if not isinstance(seeds, (list, tuple, np.ndarray)):
            seeds = [seeds] * len(prompts)
        for prompt, seed in zip(prompts, seeds):
            target = next(
                int(number) - 1 for number, description in
                re.findall(r"(?m)^(\d+)\. (.+)$", prompt)
                if "TARGET" in description
            )
            target_probability = 0.8 if int(seed) // 10_000 == 1 else 0.2
            remainder = (1.0 - target_probability) / (len(choices) - 1)
            probs = np.full(len(choices), remainder)
            probs[target] = target_probability
            rows.append(probs.tolist())
        return rows


def test_mcq_is_a_counterbalanced_selection_prompt_with_separate_behavioral_replay():
    options = [
        _metric("target", "TARGET criterion", "TARGET BODY"),
        _metric("d1", "first distractor", "DISTRACTOR 1"),
        _metric("d2", "second distractor", "DISTRACTOR 2"),
        _metric("d3", "third distractor", "DISTRACTOR 3"),
    ]
    records = [(0, "positive example", 1), (1, "negative example", 0)]
    recovered, picks, report = _mcq_recon(
        _BinaryExecutor(),
        "story",
        "[score=1]\npositive\n\n[score=0]\nnegative",
        records,
        options,
        0,
        ["heldout one", "heldout two"],
        500,
        8,
        0.7,
        recon=_DescriptionSelector(),
        run_controls=True,
    )

    assert report["identification_acc"] == 1.0
    assert report["no_demonstration_acc"] == 1.0
    assert report["shuffled_label_acc"] == 1.0
    assert report["annotation_lift_over_no_demonstration"] == 0.0
    assert report["target_position_counts"] == {"0": 2, "1": 2, "2": 2, "3": 2}
    assert all(pick["choice"] == 0 for pick in picks)
    assert np.all(recovered == 1.0)
    assert "candidate-option behaviors" not in " ".join(
        report["conditions"]["annotations"]["raw_outputs"])


def test_mcq_uses_normalized_choice_probabilities_when_backend_exposes_them():
    options = [
        _metric("target", "TARGET criterion", "TARGET BODY"),
        _metric("d1", "first distractor", "DISTRACTOR 1"),
        _metric("d2", "second distractor", "DISTRACTOR 2"),
        _metric("d3", "third distractor", "DISTRACTOR 3"),
    ]
    records = [(0, "positive example", 1), (1, "negative example", 0)]
    recovered, _, report = _mcq_recon(
        _BinaryExecutor(),
        "story",
        "[score=1]\npositive\n\n[score=0]\nnegative",
        records,
        options,
        0,
        ["heldout one", "heldout two"],
        500,
        8,
        0.7,
        recon=_LogitSelector(),
        run_controls=True,
        choice_readout="auto",
    )

    assert report["readout_kind"] == "normalized_choice_logits"
    assert report["identification_score"] == pytest.approx(0.7)
    assert report["identification_acc"] == 1.0
    assert report["annotation_lift_over_strongest_control"] == pytest.approx(0.0)
    assert np.all(recovered == 1.0)


def test_run_metric_persists_disjoint_design_and_replay_artifacts():
    n = 24
    texts = [f"item {i}" for i in range(n)]
    item_ids = [f"id-{i}" for i in range(n)]
    target_pyes = (np.arange(n) % 2).astype(float)
    target = _metric("target", "TARGET criterion", "TARGET BODY")
    distractors = [
        _metric("d1", "first distractor", "DISTRACTOR 1"),
        _metric("d2", "second distractor", "DISTRACTOR 2"),
        _metric("d3", "third distractor", "DISTRACTOR 3"),
    ]
    pool_pyes = [(metric, 1.0 - target_pyes) for metric in distractors]

    rows = run_metric(
        _BinaryExecutor(),
        "story",
        target,
        distractors,
        texts,
        target_pyes,
        R=8,
        n_train=12,
        mode="mcq",
        max_chars=500,
        distractor="contrastive",
        pool_pyes=pool_pyes,
        n_options=4,
        recon_backend=_DescriptionSelector(),
        mcq_n_examples=7,
        mcq_min_design_disagreements=2,
        mcq_min_demo_disagreements=2,
        item_ids=item_ids,
        mcq_choice_readout="sampled",
    )

    assert len(rows) == 1
    row = rows[0]
    design_ids = set(row["mcq_design"]["design_item_ids"])
    heldout_ids = set(row["mcq_design"]["heldout_item_ids"])
    assert design_ids.isdisjoint(heldout_ids)
    assert design_ids | heldout_ids == set(item_ids)
    assert set(row["mcq_design"]["example_item_ids_in_prompt_order"]) <= design_ids
    assert row["mcq_design"]["heldout_used_for_design"] is False
    assert row["primary_reconstruction_score"] == 1.0
    assert row["identification"]["position_counterbalanced"] is True
    assert len(row["mcq_raw"]["heldout_target_verdicts"]) == 12
    assert len(row["mcq_raw"]["heldout_recovered_verdicts"]) == 12


def test_identity_mutual_information_is_aggregated_across_randomized_targets():
    def row(target, probabilities):
        return {
            "metric_id": target,
            "identification": {
                "option_codebook": [{"metric_id": "a"}, {"metric_id": "b"}],
                "conditions": {
                    "annotations": {
                        "canonical_choice_probabilities": [probabilities],
                    },
                },
            },
        }

    perfect = mcq_identity_channel([row("a", [1.0, 0.0]), row("b", [0.0, 1.0])])
    independent = mcq_identity_channel([row("a", [0.5, 0.5]), row("b", [0.5, 0.5])])
    assert perfect["mutual_information_bits"] == pytest.approx(1.0)
    assert perfect["mean_target_recovery_probability"] == pytest.approx(1.0)
    assert independent["mutual_information_bits"] == pytest.approx(0.0)
    assert independent["mean_target_recovery_probability"] == pytest.approx(0.5)


def test_precomputed_prompt_value_is_anchor_free_and_keeps_degenerate_draws():
    n = 12
    constant_target = np.ones(n)
    distractors = [
        {"metric_id": f"d{i}", "description": f"distractor {i}",
         "scores": np.ones(n), "body": f"body {i}"}
        for i in range(3)
    ]
    value = mcq_value_from_precomputed_behavior(
        _ConditionSensitiveLogitSelector(),
        noun="story",
        candidate_prompt_text="candidate prompt",
        target_metric_id="target",
        target_description="TARGET criterion",
        target_scores=constant_target,
        probe_texts=[f"item {i}" for i in range(n)],
        distractors=distractors,
        design_indices=np.arange(n),
        codebook_frozen_before_prompt_search=True,
        n_examples=4,
        n_reconstruction_draws=4,
        choice_readout="logits",
    )
    assert value["raw_target_option_probability"] == pytest.approx(0.8)
    assert value["value_mark"] == pytest.approx(0.6)
    assert value["design"]["teaching_set"]["target_is_degenerate_on_design"] is True
    assert value["design"]["teaching_set"]["minimum_demonstrated_separation"] == 0
    assert value["design"]["uses_external_labels"] is False


def test_precomputed_prompt_value_requires_a_frozen_codebook():
    with pytest.raises(ValueError, match="codebook must be frozen"):
        mcq_value_from_precomputed_behavior(
            _ConditionSensitiveLogitSelector(),
            noun="story",
            candidate_prompt_text="candidate",
            target_metric_id="target",
            target_description="TARGET criterion",
            target_scores=np.tile([0.0, 1.0], 4),
            probe_texts=[f"item {i}" for i in range(8)],
            distractors=[{
                "metric_id": "d", "description": "distractor",
                "scores": np.tile([1.0, 0.0], 4),
            }],
            design_indices=np.arange(8),
            codebook_frozen_before_prompt_search=False,
            n_examples=4,
        )


def test_batched_logit_values_match_the_single_prompt_reference():
    n = 12
    rows = np.vstack([np.ones(n), np.tile([0.0, 1.0], n // 2)])
    distractors = [
        {"metric_id": f"d{i}", "description": f"distractor {i}",
         "scores": np.roll(rows[1], i + 1), "body": f"body {i}"}
        for i in range(3)
    ]
    common = dict(
        noun="story",
        target_metric_id="target",
        target_description="TARGET criterion",
        probe_texts=[f"item {i}" for i in range(n)],
        distractors=distractors,
        design_indices=np.arange(n),
        codebook_frozen_before_prompt_search=True,
        n_examples=4,
        n_reconstruction_draws=4,
    )
    selector = _ConditionSensitiveLogitSelector()
    reference = [
        mcq_value_from_precomputed_behavior(
            selector,
            candidate_prompt_text=f"candidate {i}",
            target_scores=row,
            choice_readout="logits",
            **common,
        )
        for i, row in enumerate(rows)
    ]
    batched = mcq_logit_values_from_precomputed_behaviors(
        selector,
        candidate_prompt_texts=["candidate 0", "candidate 1"],
        target_score_rows=rows,
        query_batch_size=3,
        **common,
    )
    assert [value["value_mark"] for value in batched] == pytest.approx(
        [value["value_mark"] for value in reference])
    assert [value["raw_target_option_probability"] for value in batched] == pytest.approx(
        [value["raw_target_option_probability"] for value in reference])
    assert [value["design"]["teaching_transcript_sha256"] for value in batched] == [
        value["design"]["teaching_transcript_sha256"] for value in reference]
