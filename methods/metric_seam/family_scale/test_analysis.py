from __future__ import annotations

import math

import pytest

from methods.metric_seam.family_scale.analysis import (
    BatchCalibrationObservation,
    ConcordanceObservation,
    FunnelCounts,
    G1Observation,
    G2ControlObservation,
    assemble_family_certificate_inputs,
    batching_calibration,
    c_index,
    clustered_bootstrap_c_index,
    reliability_ceiling_normalization,
    resolution_statistics,
    summarize_g1,
    summarize_g2,
)


def test_funnel_preserves_every_denominator_and_residual() -> None:
    funnel = FunnelCounts("symbolic", "math", "stackexchange", 100, 20, 60, 15, 40)
    assert funnel.after_base_rate == 80
    assert funnel.unauthored == 20
    assert funnel.after_gate == 45
    assert funnel.nonoperational_after_gate == 5
    assert funnel.operational_per_proposed == 0.4
    assert funnel.operational_per_authored == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="surviving the base-rate"):
        FunnelCounts("x", "d", "c", 10, 8, 3, 0, 0)


def test_resolution_reports_pairwise_ties_mode_distinct_and_entropy() -> None:
    result = resolution_statistics([0, 0, 1, 2])
    assert result.distinct == 3
    assert result.mode_fraction == 0.5
    assert result.tied_pair_count == 1
    assert result.pair_count == 6
    assert result.tie_fraction == pytest.approx(1 / 6)
    assert result.entropy_bits == 1.5
    assert result.normalized_entropy == 0.75


def test_c_index_uses_target_comparability_and_half_prediction_ties() -> None:
    result = c_index([0, 1, 2], [0, 0, 2])
    assert result.comparable_pairs == 3
    assert result.concordant_pairs == 2
    assert result.tied_prediction_pairs == 1
    assert result.value == pytest.approx(2.5 / 3)
    assert c_index([1, 1], [0, 1]).value is None


def test_two_pass_ceiling_normalizes_signed_concordance() -> None:
    result = reliability_ceiling_normalization(
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    )
    assert result.primary_c_index == 1
    assert result.pass_c_index == 1
    assert result.signed_reliability_ceiling == 1
    assert result.normalized_c_index == 1

    undefined = reliability_ceiling_normalization(
        [0, 1, 2], [0, 1, 2], [0, 1, 2], [2, 1, 0]
    )
    assert undefined.signed_pass_reliability == -1
    assert undefined.normalized_c_index is None


def test_hierarchical_cluster_bootstrap_is_seeded_and_uses_available_levels() -> None:
    rows = [
        ConcordanceObservation(
            target=float(document),
            prediction=float(document + (metric == "m2")),
            metric_id=metric,
            document_id=f"{metric}-d{document}",
            call_id=f"{metric}-d{document}-c1",
        )
        for metric in ("m1", "m2")
        for document in range(4)
    ]
    first = clustered_bootstrap_c_index(rows, draws=50, seed=9)
    repeat = clustered_bootstrap_c_index(rows, draws=50, seed=9)
    assert first == repeat
    assert first.cluster_levels == ("metric_id", "document_id", "call_id")
    assert first.draws_valid > 0
    assert 0 <= first.lower <= first.upper <= 1

    with pytest.raises(ValueError, match="partially populated"):
        clustered_bootstrap_c_index(
            [rows[0], ConcordanceObservation(1, 1)], draws=2
        )

    iid = clustered_bootstrap_c_index(
        [ConcordanceObservation(float(i), float(i)) for i in range(4)],
        draws=10,
        seed=2,
    )
    assert iid.cluster_levels == ("observation",)
    assert iid.draws_valid > 0


def test_batch_calibration_retains_bias_by_batch() -> None:
    rows = [
        BatchCalibrationObservation("a", "b1", 0, 1),
        BatchCalibrationObservation("b", "b1", 1, 2),
        BatchCalibrationObservation("c", "b2", 2, 2),
    ]
    result = batching_calibration(rows)
    assert result.mean_difference == pytest.approx(2 / 3)
    assert result.batch_mean_differences == {"b1": 1.0, "b2": 0.0}
    assert result.max_absolute_batch_mean_difference == 1
    assert result.pearson == pytest.approx(math.sqrt(3) / 2)
    assert result.concordance.value == pytest.approx(2.5 / 3)


def _g1_rows(corpus: str = "c1") -> list[G1Observation]:
    states = ["not_applicable", "satisfied", "violated", "satisfied", "violated"]
    rows = []
    for index, state in enumerate(states):
        witness = frozenset() if state == "not_applicable" else frozenset({f"p:{index}"})
        rows.append(
            G1Observation(
                "family",
                "domain",
                corpus,
                "metric",
                "unit",
                f"item-{index}",
                state,
                state,
                witness,
                witness,
            )
        )
    return rows


def _g2_rows(corpus: str = "c1") -> list[G2ControlObservation]:
    rows = []
    for implementation in ("code", "llm"):
        rows.extend(
            [
                G2ControlObservation(
                    "family", "domain", corpus, implementation, "positive",
                    "positive_true_violation", True,
                ),
                G2ControlObservation(
                    "family", "domain", corpus, implementation, "trap",
                    "negative_proxy_trap", False,
                ),
            ]
        )
    return rows


def test_g1_and_g2_are_separate_per_domain_corpus() -> None:
    g1 = summarize_g1(_g1_rows("c1") + _g1_rows("c2"))
    assert len(g1) == 2
    assert all(value.g1_ready for value in g1.values())
    assert all(value.mean_witness_jaccard == 1 for value in g1.values())

    g2 = summarize_g2(_g2_rows("c1") + _g2_rows("c2"))
    assert len(g2) == 2
    assert all(value.g2_ready for value in g2.values())
    assert all(value.proxy_trap_passed == 2 for value in g2.values())


def test_proxy_trap_failure_blocks_family_input_without_losing_denominator() -> None:
    funnel = FunnelCounts("family", "domain", "c1", 10, 2, 7, 1, 5)
    g1 = summarize_g1(_g1_rows())
    controls = _g2_rows()
    controls[-1] = G2ControlObservation(
        "family", "domain", "c1", "llm", "trap", "negative_proxy_trap", True
    )
    g2 = summarize_g2(controls)
    assembled = assemble_family_certificate_inputs([funnel], g1, g2)[funnel.key]
    assert not assembled.ready_for_family_certificate
    assert assembled.blockers == ("g2_not_ready",)
    assert assembled.funnel.proposed == 10
    assert assembled.funnel.operational == 5


def test_assembly_rejects_g1_or_g2_groups_without_a_proposed_denominator() -> None:
    with pytest.raises(ValueError, match="lack a proposed-funnel denominator"):
        assemble_family_certificate_inputs([], summarize_g1(_g1_rows()), {})
