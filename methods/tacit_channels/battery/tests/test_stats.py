"""Adversarial tests for battery/stats.py — every statistic is attacked with synthetic
agents of KNOWN profile, including one agent per audit failure mode (the tests that would
have caught the 2026-07-23 defects before first reporting)."""
import numpy as np
import pytest

from methods.tacit_channels.battery.stats import (
    ITEM_AGREEMENT_CHANCE, composition_rho, conf_acc_stats, confidence_scale_valid,
    holistic_residual, leak_stats, not_gap, zrank,
)

RNG = np.random.default_rng(42)
N = 400
MASK = np.ones(N, dtype=bool)


def _policy():
    return RNG.normal(size=N)


def test_leak_compliant_vs_rigid_agent():
    tf = {"canonical": _policy()}
    others = [_policy() for _ in range(5)]
    # compliant inverter: exclusion = reversed own judgment
    compliant = leak_stats({"canonical": -tf["canonical"]}, tf, others, MASK)
    assert compliant["leak_self"] < -0.95 and compliant["leak_specific"] < -0.9
    # rigid agent: exclusion identical to tf
    rigid = leak_stats({"canonical": tf["canonical"] + RNG.normal(0, .05, N)}, tf,
                       others, MASK)
    assert rigid["leak_self"] > 0.9 and rigid["leak_specific"] > 0.85


def test_leak_generic_factor_agent_yields_zero_specific():
    """THE audit-#2 agent: one shared vector everywhere -> huge leak_self, but the
    cross-cell null absorbs it; leak_specific ~ 0. This test would have caught the
    '+.93 rigidity' over-reading before publication."""
    shared = _policy()
    noise = lambda: shared + RNG.normal(0, 0.3, N)
    tf = {"canonical": noise()}
    others = [noise() for _ in range(6)]
    r = leak_stats({"canonical": noise()}, tf, others, MASK)
    assert r["leak_self"] > 0.75              # looks rigid...
    assert abs(r["leak_specific"]) < 0.15     # ...but it is generic structure, not leak


def test_not_gap_perfect_and_ignoring_agents():
    t = _policy()
    tf = [t + RNG.normal(0, 0.2, N)]
    perfect = not_gap(tf, [-t + RNG.normal(0, 0.2, N)], t, MASK)
    assert abs(perfect["not_gap"]) < 0.1      # inverts cleanly -> gap ~ 0
    ignorer = not_gap(tf, [t + RNG.normal(0, 0.2, N)], t, MASK)
    assert ignorer["not_gap"] > 1.5           # ~ 2 * tf_rho
    assert ignorer["not_gap"] == pytest.approx(
        perfect["tf_rho"] + ignorer["tf_rho"], abs=0.35)


def test_composition_min_composer_agent():
    a, b = _policy(), _policy()
    truth = np.minimum(zrank(a), zrank(b))
    composer = composition_rho([truth + RNG.normal(0, 0.1, N)], [a, b], MASK)
    assert composer > 0.9
    single_member_only = composition_rho([zrank(a)], [a, b], MASK)
    assert single_member_only < composer      # ignoring B must score lower


def test_holistic_recovers_span_and_fails_closed():
    X = RNG.normal(size=(N, 30))
    w = RNG.normal(size=30)
    y = X @ w + RNG.normal(0, 0.5, N)
    fit, ev = np.arange(N) % 2 == 0, np.arange(N) % 2 == 1
    r = holistic_residual(y, X, fit, ev)
    assert r["verdict"] == "ok" and r["oos_r2"] > 0.8
    # y OUTSIDE the span -> low R2 but still a valid number
    r2 = holistic_residual(RNG.normal(size=N), X, fit, ev)
    assert r2["verdict"] == "ok" and r2["oos_r2"] < 0.2
    # audit failure mode: floor-collapsed y -> verdict, NEVER a number
    y_floor = np.zeros(N); y_floor[:4] = 1.0
    RNG.shuffle(y_floor)
    r3 = holistic_residual(y_floor * 0.01, X, fit, ev)
    assert r3["verdict"] == "degenerate" and r3["oos_r2"] is None


def test_confidence_scale_gate_catches_both_w1b_degeneracies():
    spread = RNG.uniform(0, 100, (90, 400))
    assert confidence_scale_valid(spread)["valid"]
    constant85 = np.full((90, 400), 85.0)     # the 7B-base failure
    constant85[:, :20] = 100.0
    assert not confidence_scale_valid(constant85)["valid"]
    binary = RNG.choice([0.0, 100.0], size=(90, 400), p=[.77, .23])  # the adapter failure
    assert not confidence_scale_valid(binary)["valid"]


def test_conf_acc_dienes_and_chance_anchoring():
    agree = np.clip(RNG.normal(0.75, 0.1, N), 0, 1)
    tracking = conf_acc_stats(agree * 100 + RNG.normal(0, 2, N), agree)
    assert tracking["conf_acc_corr"] > 0.8    # explicit-knowledge agent
    independent = conf_acc_stats(RNG.uniform(0, 100, N), agree)
    assert abs(independent["conf_acc_corr"]) < 0.15   # zero-corr signature
    assert independent["guess_agreement_minus_chance"] == pytest.approx(
        independent["guess_agreement"] - ITEM_AGREEMENT_CHANCE)
    degenerate = conf_acc_stats(np.full(N, 85.0), agree)
    assert degenerate["degenerate_confidence"]        # constant conf -> flagged, no corr


# ---- 2026-07-25 expansion: agents for the failure modes the first suite left open ------


def test_leak_partial_inverter_ordering():
    """Mixture agents lambda in {0 rigid, .5 coin-flip, 1 compliant}: leak_specific must
    order strictly rigid > mixed ~ 0 > inverter — the statistic is a DIAL, not a flag."""
    tf = {"canonical": _policy()}
    others = [_policy() for _ in range(5)]
    out = {}
    for lam in (0.0, 0.5, 1.0):
        e = (1 - 2 * lam) * tf["canonical"] + RNG.normal(0, 0.3, N)
        out[lam] = leak_stats({"canonical": e}, tf, others, MASK)["leak_specific"]
    assert out[0.0] > 0.85 and out[1.0] < -0.85
    assert abs(out[0.5]) < 0.2
    assert out[0.0] > out[0.5] > out[1.0]


def test_leak_mask_is_actually_applied():
    """Rigidity planted ONLY on masked-OUT items must not reach the statistic."""
    t = _policy()
    e = np.where(np.arange(N) < N // 2, t, _policy())   # rigid on first half only
    others = [_policy() for _ in range(4)]
    full = leak_stats({"canonical": e}, {"canonical": t}, others, MASK)
    clean_half = leak_stats({"canonical": e}, {"canonical": t}, others,
                            np.arange(N) >= N // 2)
    assert full["leak_self"] > 0.35
    assert abs(clean_half["leak_self"]) < 0.15


def test_not_gap_adverse_is_min_over_forms():
    """One NOT-ignoring form among three must dominate (adverse = worst-form semantics,
    same min-over-forms law as the P1 gate). All-perfect control stays ~0."""
    t = _policy()
    tf = [t + RNG.normal(0, 0.2, N)]
    perfect_form = lambda: -t + RNG.normal(0, 0.2, N)
    mostly = not_gap(tf, [perfect_form(), perfect_form(), t + RNG.normal(0, 0.2, N)],
                     t, MASK)
    assert mostly["not_gap"] > 1.5            # the single ignorer drives the readout
    all_perfect = not_gap(tf, [perfect_form() for _ in range(3)], t, MASK)
    assert abs(all_perfect["not_gap"]) < 0.15


def test_composition_v1_single_reference_and_or_composer():
    a, b = _policy(), _policy()
    # v1 mode: one target-composed reference vector, no blend construction
    target_composed = zrank(a)
    assert composition_rho([target_composed + RNG.normal(0, 0.1, N)],
                           [target_composed], MASK) > 0.9
    # OR-composer (max blend) must score well below a min-composer vs the AND reference
    min_score = composition_rho(
        [np.minimum(zrank(a), zrank(b)) + RNG.normal(0, 0.1, N)], [a, b], MASK)
    or_score = composition_rho(
        [np.maximum(zrank(a), zrank(b)) + RNG.normal(0, 0.1, N)], [a, b], MASK)
    assert or_score < 0.65 and min_score - or_score > 0.25


def test_holistic_correlated_wrong_span_caveat():
    """ENCODES the cycle-2 calibration finding (binding Act-3 caveat): a wrong-but-
    CORRELATED predictor span still recovers high R^2, so unnamed_share = 1 - R^2 is a
    SPAN-RELATIVE LOWER BOUND — never evidence that the named span is the true one. If
    this test ever fails, the estimator's semantics changed and the caveat wording in
    the prereg must be revisited."""
    X = RNG.normal(size=(N, 20))
    y = X @ RNG.normal(size=20) + RNG.normal(0, 0.5, N)
    fit, ev = np.arange(N) % 2 == 0, np.arange(N) % 2 == 1
    true_r2 = holistic_residual(y, X, fit, ev)["oos_r2"]
    wrong_corr = 0.8 * X + 0.6 * RNG.normal(size=(N, 20))     # correlated wrong span
    wrong_r2 = holistic_residual(y, wrong_corr, fit, ev)["oos_r2"]
    assert true_r2 > 0.9
    assert wrong_r2 > 0.4                     # high despite being the WRONG span


def test_holistic_eval_side_degeneracy_fails_closed():
    X = RNG.normal(size=(N, 10))
    fit, ev = np.arange(N) % 2 == 0, np.arange(N) % 2 == 1
    y = RNG.normal(size=N)
    y[ev] = 3.14                              # variance only on the fit side
    assert holistic_residual(y, X, fit, ev)["verdict"] == "degenerate"


def test_conf_acc_small_n_and_nan_handling():
    agree = np.clip(RNG.normal(0.75, 0.1, N), 0, 1)
    conf = agree * 100 + RNG.normal(0, 2, N)
    assert conf_acc_stats(conf[:40], agree[:40]) is None      # < 50 items -> None
    half_nan = conf.copy()
    half_nan[::2] = np.nan                                    # 200 finite remain
    assert conf_acc_stats(half_nan, agree)["conf_acc_corr"] > 0.8


def test_scale_gate_unique_boundary_and_sparse_rows():
    # 7 distinct levels with healthy spread: fails on n_unique alone (bar is 8)
    levels7 = RNG.choice([0., 15., 30., 45., 60., 75., 90.], size=(20, 400))
    r = confidence_scale_valid(levels7)
    assert r["n_unique"] == 7 and not r["valid"]
    # sparse rows (< 10 finite) are excluded: their variance must not rescue a
    # constant-on-dense-rows load
    conf = np.full((5, 400), np.nan)
    conf[0, :5] = [0, 25, 50, 75, 100]        # flashy but sparse -> excluded
    conf[1, :5] = [10, 30, 50, 70, 90]
    conf[2:, :] = 85.0                        # dense rows constant
    r2 = confidence_scale_valid(conf)
    assert r2["median_cell_std"] == 0.0 and not r2["valid"]
