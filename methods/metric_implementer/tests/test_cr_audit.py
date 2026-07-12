"""Finite-sample tests for the CR-3 best-single-prompt certificate."""
from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.stats import binom

from methods.metric_implementer.experiments.cr_audit import (
    all_finite_prompt_dpi_certificate,
    all_finite_prompt_population_certificate,
    classify_prompt_evolution,
    clopper_pearson_upper,
    cr3_certificate,
    dkw_expected_max_lower,
    dkw_expected_max_upper,
    empirical_bernstein_lower,
    empirical_bernstein_upper,
    prompt_articulation_certificate,
    stratified_split,
    zero_error_lockbox_plan,
)
from methods.metric_implementer.experiments.value_census import i_binary


def test_cp_upper_has_exact_one_sided_coverage():
    worst = 0.0
    n = 40
    for p in np.linspace(0.001, 0.999, 999):
        fail = sum(binom.pmf(z, n, p) for z in range(n + 1)
                   if clopper_pearson_upper(z, n, 0.05) < p)
        worst = max(worst, float(fail))
    assert worst <= 0.05 + 1e-12


def test_empirical_bernstein_direct_theorem_is_conservative_on_bernoulli_grid():
    worst = 0.0
    n = 30
    for p in np.linspace(0.01, 0.99, 199):
        fail = 0.0
        for z in range(n + 1):
            marks = np.r_[np.ones(z), np.zeros(n - z)]
            if empirical_bernstein_upper(marks, 1.0, 0.05) < p:
                fail += float(binom.pmf(z, n, p))
        worst = max(worst, fail)
    assert worst <= 0.05 + 1e-12


def test_empirical_bernstein_lower_is_conservative_on_bernoulli_grid():
    worst = 0.0
    n = 30
    for p in np.linspace(0.01, 0.99, 199):
        fail = 0.0
        for z in range(n + 1):
            marks = np.r_[np.ones(z), np.zeros(n - z)]
            if empirical_bernstein_lower(marks, 1.0, 0.05) > p:
                fail += float(binom.pmf(z, n, p))
        worst = max(worst, fail)
    assert worst <= 0.05 + 1e-12


def test_dkw_expected_max_bound_has_coverage_on_bernoulli_grid():
    worst = 0.0
    n = 30
    horizon = 3
    for p in np.linspace(0.01, 0.99, 199):
        true_expected_max = 1.0 - (1.0 - p) ** horizon
        fail = 0.0
        for z in range(n + 1):
            marks = np.r_[np.ones(z), np.zeros(n - z)]
            upper, _ = dkw_expected_max_upper(
                {"family": marks}, {"family": horizon}, 1.0, 0.05)
            if upper < true_expected_max - 1e-12:
                fail += float(binom.pmf(z, n, p))
        worst = max(worst, fail)
    assert worst <= 0.05 + 1e-12


def _world():
    rng = np.random.default_rng(4)
    n = 240
    y = np.tile([0, 1], n // 2).astype(int)
    base = y.copy()
    base[:72] = 1 - base[:72]
    better = y.copy()
    better[:24] = 1 - better[:24]
    perfect = y.copy()
    noise = rng.integers(0, 2, n)
    patterns = np.asarray([base, better, perfect, noise], np.uint8)
    probabilities = {
        "family_a": np.asarray([0.55, 0.25, 0.05, 0.15]),
        "family_b": np.asarray([0.45, 0.25, 0.10, 0.20]),
    }
    return y, patterns, probabilities


def _sample_audit(patterns, probabilities, n_per_family=500, seed=10):
    rng = np.random.default_rng(seed)
    rows, families = [], []
    for family, probs in probabilities.items():
        draws = rng.choice(len(patterns), n_per_family, p=probs)
        rows.extend(patterns[draws])
        families.extend([family] * n_per_family)
    return np.asarray(rows), families


def _exact_expected_max_gain(gains, probabilities, horizons):
    families = list(probabilities)
    slots = [family for family in families for _ in range(horizons[family])]
    total = 0.0
    for outcome in itertools.product(range(len(gains)), repeat=len(slots)):
        probability = 1.0
        for family, species in zip(slots, outcome):
            probability *= probabilities[family][species]
        total += probability * max(gains[species] for species in outcome)
    return total


def test_prompt_ceiling_covers_exact_finite_horizon_best_single_prompt():
    y, patterns, probabilities = _world()
    pool = patterns[[0]]
    audit, families = _sample_audit(patterns, probabilities)
    horizons = {"family_a": 2, "family_b": 2}
    cert = prompt_articulation_certificate(
        pool, audit, y, families,
        family_names=list(probabilities),
        horizon_per_family=horizons,
        alpha=0.05,
        debug_internals=True,
    )
    c = cert["certified"]
    r_pool = i_binary(y, pool[0])
    gains = np.maximum(0.0, np.asarray([i_binary(y, p) for p in patterns]) - r_pool)
    true_mean = {f: float(np.dot(probabilities[f], gains)) for f in probabilities}
    for family in probabilities:
        assert c["per_family"][family]["mean_gain_UCB_bits"] >= true_mean[family]
    true_expected_best_gain = _exact_expected_max_gain(gains, probabilities, horizons)
    assert c["finite_horizon_expected_prompt_ceiling_UCB_bits"] >= r_pool + true_expected_best_gain - 1e-9
    assert c["finite_horizon_expected_prompt_ceiling_UCB_bits"] <= c["DPI_cap_bits"] + 1e-9
    assert c["simultaneous_confidence"] == pytest.approx(0.95)
    assert c["alpha_allocation"]["primary_claims"] == 3


def test_dkw_expected_max_lower_and_upper_bracket_degenerate_future_value():
    marks = {
        "family_a": np.full(300, 0.2),
        "family_b": np.full(300, 0.2),
    }
    horizon = {"family_a": 20, "family_b": 20}
    lower, _ = dkw_expected_max_lower(marks, horizon, 1.0, 0.01)
    upper, _ = dkw_expected_max_upper(marks, horizon, 1.0, 0.01)
    assert lower <= 0.2 <= upper


def test_status_evidence_is_simultaneous_and_ordered():
    y, patterns, probabilities = _world()
    audit, families = _sample_audit(patterns, probabilities, n_per_family=300)
    cert = prompt_articulation_certificate(
        patterns[[0]], audit, y, families,
        family_names=list(probabilities),
        horizon_per_family={"family_a": 2, "family_b": 2},
        alpha=0.05,
    )
    evidence = cert["status_evidence"]
    assert evidence["simultaneous_confidence"] == pytest.approx(0.95)
    mass_lo, mass_hi = evidence["behavioral_missing_mass_interval"]
    gain_lo, gain_hi = evidence["finite_horizon_expected_best_gain_interval_bits"]
    assert 0.0 <= mass_lo <= mass_hi <= 1.0
    assert 0.0 <= gain_lo <= gain_hi
    assert len(evidence["alpha_allocation"]["claims"]) == 3


def test_external_anchor_free_reconstruction_values_do_not_assume_species_substitutability():
    # Every prompt has the same executor behavior, but Reconstruction-MCQ values differ. Exact
    # behavior support is exhausted; value support is not, because wording can affect recovery.
    behavior = np.tile([0, 1], 60).astype(float)
    pool = np.vstack([behavior, behavior])
    audit = np.vstack([behavior for _ in range(120)])
    families = ["family_a"] * 60 + ["family_b"] * 60
    pool_values = np.array([0.30, 0.40])
    audit_values = np.r_[np.full(60, 0.45), np.full(60, 0.55)]
    cert = prompt_articulation_certificate(
        pool,
        audit,
        None,
        families,
        family_names=["family_a", "family_b"],
        horizon_per_family={"family_a": 1, "family_b": 1},
        pool_values=pool_values,
        audit_values=audit_values,
        value_cap=1.0,
        value_name="Reconstruction-MCQ target-option probability",
        value_unit="probability",
        value_determined_by_exact_behavior=False,
        p_min=0.5,
    )
    assert cert["estimand"]["target_mode"] == "supplied bounded anchor-free reconstruction values"
    assert cert["estimand"]["value_unit"] == "probability"
    assert cert["certified"]["pool_best_prompt_value"] == pytest.approx(0.40)
    assert cert["certified"]["finite_horizon_expected_best_gain_UCB"] >= 0.15
    support = cert["assumption_dependent"]["exact_support"]
    assert support["support_exhausted"] is True
    assert support["value_support_exhausted"] is False
    assert support["pool_union_support_prompt_ceiling_UCB"] == pytest.approx(1.0)
    assert cert["scope"]["external_supervision_used"] is False


def test_external_values_require_complete_bounded_marks():
    behavior = np.tile([0, 1], 20).astype(float)
    pool = behavior[None, :]
    audit = np.vstack([behavior, behavior])
    with pytest.raises(ValueError, match="supplied together"):
        prompt_articulation_certificate(
            pool, audit, None, ["family", "family"], family_names=["family"],
            pool_values=np.array([0.5]), value_cap=1.0)
    with pytest.raises(ValueError, match=r"\[0, value_cap\]"):
        prompt_articulation_certificate(
            pool, audit, None, ["family", "family"], family_names=["family"],
            pool_values=np.array([0.5]), audit_values=np.array([0.2, 1.2]), value_cap=1.0)


def test_deterministic_mcq_value_can_promote_exact_support_exhaustion():
    behavior = np.tile([0, 1], 40).astype(float)
    pool = behavior[None, :]
    audit = np.vstack([behavior for _ in range(240)])
    cert = prompt_articulation_certificate(
        pool,
        audit,
        None,
        ["family_a"] * 120 + ["family_b"] * 120,
        family_names=["family_a", "family_b"],
        horizon_per_family=1,
        pool_values=np.array([0.4]),
        audit_values=np.full(240, 0.4),
        value_cap=0.8,
        value_unit="probability",
        value_determined_by_exact_behavior=True,
        p_min=0.5,
    )
    support = cert["assumption_dependent"]["exact_support"]
    assert support["support_exhausted"] is True
    assert support["value_support_exhausted"] is True
    assert support["pool_union_support_prompt_ceiling_UCB"] == pytest.approx(0.4)


def test_exact_behavior_value_premise_fails_closed_on_inconsistent_values():
    behavior = np.tile([0, 1], 20).astype(float)
    with pytest.raises(ValueError, match="identical exact behaviors"):
        prompt_articulation_certificate(
            behavior[None, :],
            np.vstack([behavior, behavior]),
            None,
            ["family", "family"],
            family_names=["family"],
            pool_values=np.array([0.4]),
            audit_values=np.array([0.4, 0.5]),
            value_cap=0.8,
            value_determined_by_exact_behavior=True,
        )


def test_exact_teaching_transcript_partition_can_tighten_value_support():
    pool_behavior = np.tile([0, 1], 30).astype(float)
    novel_behaviors = np.asarray([
        np.roll(pool_behavior, shift) for shift in range(1, 121)
    ])
    families = ["family_a"] * 60 + ["family_b"] * 60
    cert = prompt_articulation_certificate(
        pool_behavior[None, :],
        novel_behaviors,
        None,
        families,
        family_names=["family_a", "family_b"],
        horizon_per_family=1,
        pool_values=np.array([0.4]),
        audit_values=np.full(120, 0.4),
        value_cap=0.8,
        pool_value_species=["transcript-a"],
        audit_value_species=["transcript-a"] * 120,
        value_p_min=0.5,
    )
    assert cert["certified"]["exact_value_state_missing_mass_U0"] < 0.5
    support = cert["assumption_dependent"]["exact_value_state_support"]
    assert support["support_exhausted"] is True
    assert support["pool_union_support_prompt_ceiling_UCB"] == pytest.approx(0.4)


def test_value_species_partition_fails_closed_if_state_does_not_determine_value():
    behavior = np.tile([0, 1], 20).astype(float)
    with pytest.raises(ValueError, match="value species has inconsistent"):
        prompt_articulation_certificate(
            behavior[None, :],
            np.vstack([behavior, behavior]),
            None,
            ["family", "family"],
            family_names=["family"],
            pool_values=np.array([0.4]),
            audit_values=np.array([0.4, 0.5]),
            value_cap=0.8,
            pool_value_species=["same"],
            audit_value_species=["same", "same"],
        )


def _status_certificate(mass_interval, gain_interval):
    return {
        "status_evidence": {
            "simultaneous_confidence": 0.95,
            "behavioral_missing_mass_interval": list(mass_interval),
            "finite_horizon_expected_best_gain_interval_bits": list(gain_interval),
        },
        "certified": {
            "pool_best_prompt_recovery_bits": 0.4,
            "future_draws_per_family": {"family_a": 100, "family_b": 100},
        },
    }


def test_confirmation_status_distinguishes_plateau_rise_and_unsaturated():
    common = dict(
        confirmation_is_never_absorbed=True,
        stopping_rule_frozen_before_confirmation=True,
        plateau_epsilon_bits=0.02,
        saturation_missing_mass=0.10,
    )
    plateau = classify_prompt_evolution(
        _status_certificate((0.01, 0.08), (0.0, 0.015)), **common)
    rising = classify_prompt_evolution(
        _status_certificate((0.20, 0.35), (0.04, 0.12)), **common)
    unsaturated = classify_prompt_evolution(
        _status_certificate((0.20, 0.35), (0.0, 0.10)), **common)
    assert plateau["headline_status"] == "CERTIFIED_PROCESS_SATURATED_AND_PLATEAUED"
    assert rising["headline_status"] == "CERTIFIED_PROCESS_RISING"
    assert rising["behavior_status"] == "CERTIFIED_UNSATURATED"
    assert unsaturated["headline_status"] == "CERTIFIED_BEHAVIORALLY_UNSATURATED_VALUE_UNRESOLVED"
    assert plateau["scope"]["axis"] == "prompt evolution at one fixed executor"


def test_prompt_status_refuses_monitor_or_adaptive_confirmation():
    cert = _status_certificate((0.01, 0.08), (0.0, 0.015))
    with pytest.raises(ValueError, match="never-absorbed"):
        classify_prompt_evolution(
            cert,
            confirmation_is_never_absorbed=False,
            stopping_rule_frozen_before_confirmation=True,
        )
    with pytest.raises(ValueError, match="stopping rule"):
        classify_prompt_evolution(
            cert,
            confirmation_is_never_absorbed=True,
            stopping_rule_frozen_before_confirmation=False,
        )


def test_all_finite_prompt_identity_witness_attains_the_global_dpi_cap():
    y, patterns, _ = _world()
    candidates = np.vstack([patterns[0], y])
    cert = all_finite_prompt_dpi_certificate(
        candidates,
        y,
        candidate_labels=["atomic", "target_definition"],
        identity_witness_index=1,
        identity_witness_is_target_definition=True,
    )
    c = cert["certificate"]
    assert c["status"] == "PROVABLY_OPTIMAL_IDENTITY"
    assert c["all_prompt_DPI_upper_bound_bits"] == pytest.approx(1.0)
    assert c["best_evaluated_lower_bound_bits"] == pytest.approx(1.0)
    assert c["certified_optimization_gap_UCB_bits"] == pytest.approx(0.0)
    assert cert["proof_scope"]["population_exact_by_construction"] is True
    assert cert["estimand"]["prompt_class"] == "all finite prompts Sigma*; no prompt-length budget"


def test_all_finite_prompt_certificate_reports_the_honest_unresolved_gap():
    y, patterns, _ = _world()
    cert = all_finite_prompt_dpi_certificate(
        patterns[[0]], y, candidate_labels=["atomic"], epsilon_bits=0.01)
    c = cert["certificate"]
    achieved = i_binary(y, patterns[0])
    assert c["status"] == "GLOBAL_GAP_EXCEEDS_TARGET_EPSILON"
    assert c["best_evaluated_lower_bound_bits"] == pytest.approx(achieved)
    assert c["all_prompt_DPI_upper_bound_bits"] == pytest.approx(1.0)
    assert c["certified_optimization_gap_UCB_bits"] == pytest.approx(1.0 - achieved)
    assert cert["proof_scope"]["population_exact_by_construction"] is False


def test_all_finite_prompt_certificate_rejects_a_false_identity_claim():
    y, patterns, _ = _world()
    with pytest.raises(ValueError, match="does not reproduce M"):
        all_finite_prompt_dpi_certificate(
            patterns[[0]], y,
            identity_witness_index=0,
            identity_witness_is_target_definition=True,
        )


def test_information_optimal_complement_is_valid_and_reported_as_a_diagnostic_relation():
    y, _, _ = _world()
    cert = all_finite_prompt_dpi_certificate((1 - y)[None, :], y)
    c = cert["certificate"]
    assert c["DPI_attained_on_frozen_panel"] is True
    assert c["status"] == "PROVABLY_OPTIMAL_DPI_ATTAINED_FIXED_PANEL"
    assert c["best_candidate_target_relation"] == "EXACT_COMPLEMENT"
    assert c["best_candidate_target_mismatch_rate"] == 1.0


def test_population_certificate_requires_a_frozen_candidate_channel():
    y, patterns, _ = _world()
    with pytest.raises(ValueError, match="frozen before"):
        all_finite_prompt_population_certificate(
            y, patterns[0], candidate_frozen_before_lockbox=False)


def test_population_certificate_tightens_for_a_zero_error_lockbox():
    y = np.tile([0, 1], 2000)
    cert = all_finite_prompt_population_certificate(
        y, y,
        candidate_frozen_before_lockbox=True,
        alpha=0.05,
        epsilon_bits=0.03,
    )
    c = cert["certificate"]
    assert c["polarity_invariant_error_rate_observed"] == 0.0
    assert c["candidate_recovery_LCB_bits"] <= 1.0
    assert c["all_prompt_DPI_upper_bound_UCB_bits"] == pytest.approx(1.0)
    assert c["certified_optimization_gap_UCB_bits"] < 0.03
    assert c["status"] == "PROVABLY_EPSILON_OPTIMAL_POPULATION"


def test_population_gap_bound_has_exact_component_coverage_in_binary_symmetric_world():
    # In this world M~Bernoulli(1/2), and an independent Bernoulli(e) error bit
    # gives the exact true optimization gap h(e).  Prevalence and error counts are
    # independent binomials, so enumerate the complete finite-sample failure rate.
    n = 24
    alpha = 0.05
    for error_rate in np.linspace(0.01, 0.49, 25):
        true_gap = float(-error_rate * np.log2(error_rate)
                         - (1.0 - error_rate) * np.log2(1.0 - error_rate))
        failure = 0.0
        for positives in range(n + 1):
            p_mass = float(binom.pmf(positives, n, 0.5))
            target = np.r_[np.ones(positives), np.zeros(n - positives)]
            for errors in range(n + 1):
                e_mass = float(binom.pmf(errors, n, error_rate))
                candidate = target.copy()
                candidate[:errors] = 1 - candidate[:errors]
                cert = all_finite_prompt_population_certificate(
                    target,
                    candidate,
                    candidate_frozen_before_lockbox=True,
                    alpha=alpha,
                )
                if cert["certificate"]["certified_optimization_gap_UCB_bits"] < true_gap - 1e-12:
                    failure += p_mass * e_mass
        assert failure <= alpha + 1e-10


def test_zero_error_lockbox_plan_is_minimal_and_matches_certificate():
    plan = zero_error_lockbox_plan(0.02, alpha=0.05)
    n = plan["n_lockbox_required"]
    assert 2200 < n < 2400
    y = np.resize(np.asarray([0, 1]), n)
    cert = all_finite_prompt_population_certificate(
        y, y, candidate_frozen_before_lockbox=True, alpha=0.05)
    assert cert["certificate"]["certified_optimization_gap_UCB_bits"] <= 0.02
    previous = np.resize(np.asarray([0, 1]), n - 1)
    cert_previous = all_finite_prompt_population_certificate(
        previous, previous, candidate_frozen_before_lockbox=True, alpha=0.05)
    assert cert_previous["certificate"]["certified_optimization_gap_UCB_bits"] > 0.02


def test_exact_partition_plus_external_pmin_can_certify_support_exhaustion():
    y, patterns, probabilities = _world()
    audit, families = _sample_audit(patterns, probabilities, n_per_family=300, seed=12)
    cert = prompt_articulation_certificate(
        patterns, audit, y, families,
        family_names=list(probabilities),
        horizon_per_family=10,
        p_min=0.05,
    )
    support = cert["assumption_dependent"]["exact_support"]
    assert cert["certified"]["exact_pattern_missing_mass_U0"] < 0.05
    assert support["support_exhausted"]
    assert support["pool_union_support_prompt_ceiling_UCB_bits"] == cert["certified"]["pool_best_prompt_recovery_bits"]

    weak_floor = prompt_articulation_certificate(
        patterns, audit, y, families,
        family_names=list(probabilities),
        horizon_per_family=10,
        p_min=0.005,
    )["assumption_dependent"]["exact_support"]
    assert not weak_floor["support_exhausted"]


def test_missing_or_unexpected_family_fails_closed():
    y, patterns, _ = _world()
    with pytest.raises(ValueError, match="every declared family"):
        prompt_articulation_certificate(
            patterns[[0]], patterns[[1, 2]], y, ["family_a", "family_a"],
            family_names=["family_a", "family_b"],
        )


def test_strict_rule_is_monotone_against_one_frozen_leader_map():
    y, patterns, probabilities = _world()
    audit, families = _sample_audit(patterns, probabilities, n_per_family=80, seed=14)
    cert = prompt_articulation_certificate(
        patterns[[0]], audit, y, families,
        family_names=list(probabilities), tau=0.90, tau_strict=0.95,
        debug_internals=True,
    )
    flags = cert["_internals"]["flags_base"]
    strict = cert["_internals"]["flags_strict"]
    assert cert["marginal_diagnostics"]["strict_behavioral_missing_mass"]["U0"] >= 0.0
    for family in probabilities:
        assert all((not base) or stricter for base, stricter in zip(flags[family], strict[family]))
        assert all(isinstance(x, (bool, np.bool_)) for x in flags[family])


def test_stratified_split_is_deterministic_and_rejects_missing_families():
    tags = ["glm_a"] * 30 + ["glm_b"] * 30 + ["glm_c"] * 30
    first = stratified_split(tags, family_prefixes=("glm_a", "glm_b", "glm_c"), seed=3)
    second = stratified_split(tags, family_prefixes=("glm_a", "glm_b", "glm_c"), seed=3)
    assert np.array_equal(first["audit"], second["audit"])
    with pytest.raises(ValueError, match="at least two"):
        stratified_split(tags, family_prefixes=("glm_a", "missing"), seed=3)


def test_legacy_wrapper_is_explicitly_conditional_not_proven_iid():
    y, patterns, _ = _world()
    sigs = np.vstack([patterns[i % len(patterns)] for i in range(90)])
    tags = [f"glm_{'abc'[i % 3]}" for i in range(90)]
    cert = cr3_certificate(sigs, y, tags, horizon_mult=2, split_seed=1)
    assert cert["schema"] == "cr3-prompt-articulation-v5"
    assert cert["scope"]["iid_provenance_established"] is False
    assert "legacy" in cert["scope"]["source"]
