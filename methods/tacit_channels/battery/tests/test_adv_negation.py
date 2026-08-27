"""Adversarial tests for battery/stats.py::not_gap ONLY (the negation-probe statistic).

Complements methods/tacit_channels/battery/tests/test_stats.py, which already covers the
perfect-inverter-vs-ignorer pair and adverse-min-over-neg-forms with one ignorer among three.
Every test here targets a DIFFERENT known-profile agent or code-path than those two -- no
duplication. Agents are synthetic, deterministic (fixed-seed default_rng), CPU-only.
"""
import numpy as np
import pytest

from methods.tacit_channels.battery.stats import not_gap
from methods.tacit_channels.channels.common import spearman

RNG = np.random.default_rng(4242)
N = 400
MASK = np.ones(N, dtype=bool)


def _policy():
    return RNG.normal(size=N)


def test_not_gap_pure_noise_interpretation_hazard():
    """DOCUMENTED HAZARD: a pure-noise agent (tf AND negated both statistically unrelated
    to the target) also lands at not_gap ~ 0 -- the SAME signature the docstring assigns to
    "perfect explicit NOT" (stats.py:59). not_gap alone cannot distinguish "applies NOT
    cleanly" from "has no installed policy to negate in the first place": this agent's dict
    is shape-indistinguishable from test_stats.py::test_not_gap_perfect_and_ignoring_agents's
    `perfect` agent. Any tally consuming this dict MUST gate on tf_rho being materially
    positive before reading not_gap as evidence of explicit-NOT ability.
    kills: nothing in the code (this is a semantics hazard, not a defect) -- documents why
    tf_rho must always be reported alongside not_gap, never dropped from a downstream tally."""
    t = _policy()
    tf_noise = _policy()          # uncorrelated with target -> no installed policy to speak of
    neg_noise = _policy()         # uncorrelated with reversed target either
    r = not_gap([tf_noise], [neg_noise], t, MASK)
    assert abs(r["tf_rho"]) < 0.15         # no real signal on the plain leg...
    assert abs(r["not_gap"]) < 0.15        # ...yet the gap alone reads as "perfect NOT"


def test_not_gap_anti_installed_agent_sign_convention():
    """Anti-installed agent: the PLAIN judgment is already the target's mirror image
    (tf = -target), and the NEGATED judgment mirrors back onto the target (negated =
    +target). By hand: tf_rho = rho(-target, target) ~ -1; neg_rho = rho(+target,
    -target) ~ -1 too. not_gap = tf_rho - neg_rho ~ 0 -- this agent reads as a clean
    explicit-NOT applier despite an inverted install, because both legs are equally
    upside-down and cancel.
    kills: a sign-convention regression scoring the negated leg against +t_ref instead of
    -t_ref (stats.py:65) -- verified inline: that swap turns this agent's gap from ~0 into
    ~ -1.95 instead of leaving it near zero."""
    t = _policy()
    tf = [-t + RNG.normal(0, 0.2, N)]
    neg = [t + RNG.normal(0, 0.2, N)]
    r = not_gap(tf, neg, t, MASK)
    buggy_neg_rho = spearman(neg[0], t)              # what a +t_ref mutant would compute
    assert r["tf_rho"] < -0.8 and r["neg_rho_vs_reversed"] < -0.8
    assert abs(r["not_gap"]) < 0.15
    assert abs(r["tf_rho"] - buggy_neg_rho) > 1.5    # the mutant's gap would NOT cancel


def test_not_gap_adverse_over_tf_forms_flips_gap_negative():
    """Adverse (min-over-forms) semantics apply on BOTH legs, not only the negated one.
    Two good tf forms (rho ~0.95 vs target) plus one garbage tf form (independent noise,
    rho ~0) drag tf_rho down to ~0 via the min. A single perfect negated form (rho ~0.98 vs
    the REVERSED target) is untouched. The composite not_gap therefore lands strongly
    NEGATIVE (~-1.0) despite the model applying NOT perfectly on the one negated form it
    was given.
    kills: an implementation that applies adverse-min on the negated leg only and averages
    (or takes vecs[0]) on the tf leg -- that mutation would report tf_rho near the good
    forms' ~0.95 and a near-zero gap, hiding the true negative-gap signal asserted here."""
    t = _policy()
    good1 = t + RNG.normal(0, 0.2, N)
    good2 = t + RNG.normal(0, 0.2, N)
    garbage = _policy()
    perfect_neg = -t + RNG.normal(0, 0.2, N)
    r = not_gap([good1, good2, garbage], [perfect_neg], t, MASK)
    assert r["neg_rho_vs_reversed"] > 0.9
    assert abs(r["tf_rho"]) < 0.15
    assert r["not_gap"] < -0.5


def test_not_gap_constant_vectors_dropped_and_all_degenerate_returns_none():
    """Constant vectors correlate with nothing (spearman returns nan when std==0) and must
    be DROPPED from the adverse-min pool, not treated as rho=0 and not crashing min(). One
    constant tf form among three real ones must vanish, leaving the min of the two real
    forms untouched. If EVERY form on a leg is constant, that leg's adverse() has nothing
    left and not_gap must return None outright -- on either leg, and on both.
    kills: (a) treating a dropped-NaN form as rho=0 inside the min (would corrupt this
    exact case, since the real forms' rhos are both positive); (b) removing the
    `if vals else None` guard in adverse() (stats.py:63), which raises ValueError on min()
    of an empty sequence for the all-constant legs."""
    t = _policy()
    tf_a = t + RNG.normal(0, 0.2, N)
    tf_b = t + RNG.normal(0, 0.3, N)
    tf_const = np.full(N, 5.0)
    neg = -t + RNG.normal(0, 0.2, N)
    r = not_gap([tf_a, tf_b, tf_const], [neg], t, MASK)
    expected = min(spearman(tf_a, t), spearman(tf_b, t))
    assert r["tf_rho"] == pytest.approx(expected, abs=1e-9)
    assert not_gap([np.full(N, 1.0), np.full(N, 2.0)], [neg], t, MASK) is None
    assert not_gap([tf_a], [np.full(N, 1.0), np.full(N, 7.0)], t, MASK) is None


def test_not_gap_hyperinverter_scale_invariance():
    """Hyper-inverter: the negated judgment overshoots magnitude by 2.5x (negated =
    -2.5*target + tiny noise) instead of matching -target exactly. Spearman correlates
    RANKS, so it is scale-free: the adverse rho on the negated leg -- and hence not_gap --
    must come out essentially identical to an unscaled -1x inverter given the SAME noise
    draw.
    kills: any switch from rank correlation to a raw covariance/dot-product (unnormalized)
    statistic -- verified inline: cov(-2.5*target+eps, -target) is ~2.5x cov(-target+eps,
    -target) on this draw (2.38 vs 0.95), so a covariance-based mutant would blow the
    2.5x-scaled case's gap far away from the 1x-scaled case's, failing the near-equality
    assertion below."""
    t = _policy()
    tf = [t + RNG.normal(0, 0.2, N)]
    eps = RNG.normal(0, 0.05, N)
    scale1 = not_gap(tf, [-t + eps], t, MASK)
    scale2p5 = not_gap(tf, [-2.5 * t + eps], t, MASK)
    cov_scale1 = float(np.cov(-t + eps, -t)[0, 1])
    cov_scale2p5 = float(np.cov(-2.5 * t + eps, -t)[0, 1])
    assert cov_scale2p5 / cov_scale1 == pytest.approx(2.5, abs=0.1)   # sanity: cov DOES scale
    assert scale1["neg_rho_vs_reversed"] > 0.95
    assert scale2p5["neg_rho_vs_reversed"] > 0.95
    assert scale1["not_gap"] == pytest.approx(scale2p5["not_gap"], abs=0.05)


def test_not_gap_degenerate_reference_on_mask_returns_none():
    """The target reference can be degenerate WITHIN the analysis mask even though it
    varies fine over the full domain (e.g. an item subgroup that happens to share one
    label). Masking to a constant-valued region of t_ref means spearman's std==0 guard
    fires for every form on both legs -> both adverse() calls return None -> not_gap must
    return None outright, never a dict with NaN entries. The SAME vectors under the full
    (non-degenerate) mask must work normally, isolating the failure to the mask-restricted
    reference, not the vectors themselves.
    kills: a version that checks tf_vecs/neg_vecs for degeneracy but forgets the reference
    itself can be degenerate post-mask -- e.g. computing std() on the unmasked t_ref --
    which would silently use the fine, non-constant reference and return a normal-looking
    but WRONG-domain dict instead of None."""
    t = _policy()
    t_const_region = t.copy()
    t_const_region[:50] = 7.0
    degenerate_mask = np.arange(N) < 50
    tf = t + RNG.normal(0, 0.2, N)
    neg = -t + RNG.normal(0, 0.2, N)
    assert not_gap([tf], [neg], t_const_region, degenerate_mask) is None
    full = not_gap([tf], [neg], t_const_region, MASK)
    assert full is not None and full["tf_rho"] > 0.5


def test_not_gap_adverse_min_is_order_independent():
    """adverse() takes min() over a list -- min is order-independent by definition, but a
    plausible off-by-one refactor (e.g. reading vals[0] or vals[-1] as a stand-in for "the
    worst form") is NOT. Three tf forms of deliberately different quality (near-zero,
    moderate, near-perfect rho vs target) must yield the IDENTICAL not_gap regardless of
    the order they are passed in.
    kills: any substitution of min(vals) with vals[0]/vals[-1]/mean(vals) -- the first two
    are order-sensitive (this test permutes the same three forms three ways), the third
    moves the result off the true minimum; either would break the exact equality asserted
    here."""
    t = _policy()
    worst = _policy()
    medium = 0.5 * t + RNG.normal(0, 1, N)
    best = t + RNG.normal(0, 0.1, N)
    neg = -t + RNG.normal(0, 0.1, N)
    orders = [[worst, medium, best], [best, medium, worst], [medium, worst, best]]
    results = [not_gap(o, [neg], t, MASK)["not_gap"] for o in orders]
    assert results[0] == results[1] == results[2]


def test_not_gap_single_outlier_robust_not_pearson():
    """A single astronomically large value in one tf form (1e8 against an otherwise
    N(0,1)-scaled policy) shifts that one item's RANK and nothing else -- Spearman should
    barely move. The equivalent PEARSON correlation on the same contaminated data
    collapses from ~0.99 to strongly negative, because Pearson is magnitude-sensitive.
    kills: a rewrite of adverse()'s correlation call from the rank-based `spearman` helper
    to `np.corrcoef` on raw values -- that substitution would crater tf_rho on the
    outlier-contaminated form well below the 0.9 floor asserted here."""
    t = _policy()
    tf_clean = t + RNG.normal(0, 0.1, N)
    neg = -t + RNG.normal(0, 0.1, N)
    tf_outlier = tf_clean.copy()
    tf_outlier[0] = 1e8
    pearson_outlier = np.corrcoef(tf_outlier, t)[0, 1]
    assert pearson_outlier < 0.5           # sanity: Pearson really is wrecked by the outlier
    r = not_gap([tf_outlier], [neg], t, MASK)
    assert r["tf_rho"] > 0.9                # Spearman is not


def test_not_gap_does_not_mutate_caller_target_reference():
    """The negated leg needs the REVERSED target (-t_ref). Building it must allocate a new
    array (unary `-`), never mutate the caller's t_ref in place -- callers reuse the same
    reference vector across the tf leg (computed first, stats.py:64) and potentially across
    other cells/statistics in the same tally pass.
    kills: an in-place negation (`t_ref *= -1` or `np.negative(t_ref, out=t_ref)`)
    substituted for `-t_ref` in the neg_rho call (stats.py:65) -- that would leave the
    caller's array flipped after the function returns, corrupting whatever runs next."""
    t = _policy()
    t_before = t.copy()
    tf = t + RNG.normal(0, 0.2, N)
    neg = -t + RNG.normal(0, 0.2, N)
    not_gap([tf], [neg], t, MASK)
    assert np.array_equal(t, t_before)


def test_not_gap_empty_form_list_returns_none():
    """An upstream row-builder can legitimately hand not_gap an empty forms list for one
    leg (e.g. every negated-form row got filtered out for a cell). This must degrade to
    None cleanly on either leg -- and on both -- never raise (min() of an empty sequence)
    and never fabricate a 0/NaN rho.
    kills: dropping the `if vals else None` guard in adverse() (stats.py:63), which would
    raise ValueError("min() arg is an empty sequence") the moment a form list for either
    leg is empty."""
    t = _policy()
    tf = t + RNG.normal(0, 0.2, N)
    neg = -t + RNG.normal(0, 0.2, N)
    assert not_gap([], [neg], t, MASK) is None
    assert not_gap([tf], [], t, MASK) is None
    assert not_gap([], [], t, MASK) is None
