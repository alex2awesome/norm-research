"""Adversarial tests for holistic_residual (methods/tacit_channels/battery/stats.py),
round 2. The first suite (test_stats.py) already covers: true-span recovery, outside-span
low R^2, global floor-collapse -> degenerate, the correlated-wrong-span caveat (unnamed_
share is a SPAN-RELATIVE LOWER BOUND, not proof the named span is the true one), and
eval-side-only degeneracy. Do NOT duplicate any of those here.

This file attacks five angles NOT in that list:
  1. the ESTIMAND itself (nonlinear/interaction use of named constructs reads as "unnamed"),
  2. the [0,1] framing of unnamed_share (it is unclipped and can exceed 1, or go to +-inf),
  3. the exact y_std_floor boundary (strict '<', both at the floor and just under it),
  4. a zero-variance predictor column (the 1e-9 z-score guard),
  5. fit_mask/eval_mask overlap (the function trusts the caller for disjointness),
plus five original angles: an int-dtype 0/1 mask (numpy fancy-indexing vs boolean-masking
mismatch), an all-False fit_mask (a NaN-comparison loophole in the degeneracy guard), a
caller-supplied y_std_floor=0.0 (the guard's threshold itself can be degenerate), column-
order invariance (a sanity/regression check), and a single NaN inside X (silent whole-
column poisoning via mean/std).

Every test uses a local, fixed-seed rng — no shared module state, no network/GPU.
"""
import numpy as np
import pytest

from methods.tacit_channels.battery.stats import holistic_residual


def _even_odd(n):
    idx = np.arange(n)
    return idx % 2 == 0, idx % 2 == 1


# ---- 1. ESTIMAND HAZARD: nonlinear use of named constructs reads as "unnamed" ----------


def test_holistic_interaction_term_reads_as_unnamed():
    """y is a PURE, fully-determined function of two NAMED columns (X0 * X1) -- zero truly
    unnamed structure exists. Linear ridge cannot represent a product term as a linear
    combination of X, so oos_r2 collapses near/below 0 and unnamed_share reads near 1.

    kills: any interpretation of unnamed_share as "the fraction of judgment attributable
    to unnamed constructs" -- it also silently swallows NONLINEAR use of named ones.
    """
    rng = np.random.default_rng(101)
    n = 400
    X = rng.normal(size=(n, 10))
    y = X[:, 0] * X[:, 1]                       # 100% determined by named cols 0 and 1
    fit, ev = _even_odd(n)
    r = holistic_residual(y, X, fit, ev)
    assert r["verdict"] == "ok"
    # DOCUMENTED HAZARD: fully-named-determined y still reads as almost entirely unnamed.
    assert r["oos_r2"] < 0.1
    assert r["unnamed_share"] > 0.9


# ---- 2. unnamed_share is unclipped: can exceed 1, or hit +-inf ------------------------


def test_holistic_unnamed_share_can_exceed_one_on_concept_shift():
    """Construct a concept-shift y: on FIT indices y is a strong, learnable function of X;
    on EVAL indices y is INDEPENDENT noise (same scale, unrelated to X). Ridge legitimately
    learns real weights from the fit half; applied to the eval half those weights inject an
    extra, uncorrelated noise source on top of eval's own noise, so oos_r2 goes solidly
    negative and unnamed_share = 1 - oos_r2 exceeds 1.

    FIXED 2026-07-25: unnamed_share is now clipped to [0, 1] for reporting while oos_r2
    stays raw (still goes strongly negative here).
    kills: removing the clip, or wrongly clipping oos_r2 as well.
    """
    rng = np.random.default_rng(202)
    n = 400
    p = 20
    fit, ev = _even_odd(n)
    X = rng.normal(size=(n, p))
    w = rng.normal(size=p) * 3
    y = np.zeros(n)
    y[fit] = X[fit] @ w + rng.normal(0, 0.1, fit.sum())      # strong signal on FIT
    y[ev] = rng.normal(size=ev.sum()) * y[fit].std()          # unrelated noise on EVAL
    r = holistic_residual(y, X, fit, ev)
    assert r["verdict"] == "ok"
    assert r["oos_r2"] < -0.5                 # raw R^2 keeps the concept-shift signal
    assert r["unnamed_share"] == 1.0          # reported share clipped at the boundary


# ---- 3. y_std_floor boundary: strict '<' on BOTH subsets ------------------------------


def test_holistic_std_floor_boundary_strict_inequality():
    """y std EXACTLY equal to y_std_floor on both the fit and eval subsets must NOT be
    flagged degenerate (the guard is strict '<'); a value epsilon below the same floor on
    both subsets MUST be flagged. The per-subset std (not the whole-sample std) is what
    the guard actually checks. Std is built from a signed +-v two-value array (half +v,
    half -v) rather than a normalize-then-rescale chain, since the latter is off by ~1
    ULP of floating-point noise at this boundary and would make the "exactly at the
    floor" case flaky by construction, not by the code under test.

    kills: an accidental '<' -> '<=' flip (would wrongly reject legitimate floor-boundary
    data) and a refactor that checks whole-sample std instead of the two subset stds.
    """
    rng = np.random.default_rng(303)
    n = 300                                        # 150 fit / 150 eval -> exact +-v split
    floor = 0.05
    fit, ev = _even_odd(n)
    X = rng.normal(size=(n, 5))

    def _y_with_subset_std(v):
        y = np.zeros(n)
        for mask in (fit, ev):
            idx = np.where(mask)[0]
            half = len(idx) // 2
            y[idx] = np.array([v] * half + [-v] * (len(idx) - half))
        return y

    y_at = _y_with_subset_std(floor)
    assert y_at[fit].std() == floor and y_at[ev].std() == floor      # bit-exact boundary
    r_at = holistic_residual(y_at, X, fit, ev, y_std_floor=floor)
    assert r_at["verdict"] == "ok"

    y_below = _y_with_subset_std(floor - 1e-6)
    r_below = holistic_residual(y_below, X, fit, ev, y_std_floor=floor)
    assert r_below["verdict"] == "degenerate"
    assert r_below["oos_r2"] is None


# ---- 4. constant column hits the 1e-9 guard, not a crash ------------------------------


def test_holistic_constant_column_matches_dropped_column():
    """A constant column (std 0) forces the z-score guard's 1e-9 epsilon denominator
    instead of a true divide-by-zero. Since the column carries no across-item variance, it
    should contribute nothing to the fit -- i.e. the result should be numerically
    indistinguishable from simply DROPPING that column outright.

    kills: any refactor that lets a zero-variance predictor NaN/Inf the whole run, or that
    (via the 1e-9 residual scale) assigns it spurious predictive weight.
    """
    rng = np.random.default_rng(404)
    n = 400
    fit, ev = _even_odd(n)
    X = rng.normal(size=(n, 6))
    X_const = X.copy()
    X_const[:, 3] = 7.0                          # constant column, no signal possible
    w = rng.normal(size=6)
    w[3] = 0.0
    y = X @ w + rng.normal(0, 0.3, n)

    r_with = holistic_residual(y, X_const, fit, ev)
    assert np.isfinite(r_with["oos_r2"])
    X_dropped = np.delete(X_const, 3, axis=1)
    r_dropped = holistic_residual(y, X_dropped, fit, ev)
    assert r_with["oos_r2"] == pytest.approx(r_dropped["oos_r2"], abs=1e-9)
    assert r_with["alpha"] == r_dropped["alpha"]


# ---- 5. fit_mask/eval_mask overlap: no disjointness check at all ----------------------


def test_holistic_overlapping_masks_inflate_r2():
    """The function performs NO check that fit_mask and eval_mask are disjoint. Make
    eval_mask a SUBSET of fit_mask (every eval item was also trained on) and compare
    against the honest disjoint even/odd split on the identical (y, X): the leaked variant
    must score at least as well as the honest one.

    kills: any refactor that assumes holistic_residual itself enforces train/eval
    separation -- that responsibility is entirely on the caller, silently.
    """
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 15))
    w = rng.normal(size=15)
    y = X @ w + rng.normal(0, 3.0, n)
    fit_disjoint, ev = _even_odd(n)
    r_disjoint = holistic_residual(y, X, fit_disjoint, ev)

    fit_overlap = np.ones(n, dtype=bool)          # eval indices are ALSO in fit
    r_overlap = holistic_residual(y, X, fit_overlap, ev)
    # DOCUMENTED HAZARD: leakage is not caught -- overlap score >= honest disjoint score.
    assert r_overlap["oos_r2"] > r_disjoint["oos_r2"]


# ---- 6. int-dtype 0/1 mask: fancy-indexing vs boolean-masking mismatch ----------------


def test_holistic_int_mask_dtype_crashes_instead_of_being_rejected():
    """fit_mask/eval_mask are implicitly assumed boolean. Cast otherwise-identical masks
    to int (0/1) -- a plausible bug if masks round-trip through JSON/npz. numpy treats an
    int array as FANCY indexing for the direct 'y[fit_mask]'/'y[eval_mask]' reads (picking
    elements at POSITIONS 0/1, not by truth value) while 'np.where(mask)' elsewhere in the
    same function correctly reinterprets it as boolean-like (nonzero => True). The
    resulting length mismatch between the two interpretations raises, rather than the
    function validating dtype up front.

    kills: any caller who assumes an int 0/1 mask is a safe drop-in for a boolean mask.
    """
    rng = np.random.default_rng(505)
    n = 400
    fit, ev = _even_odd(n)
    X = rng.normal(size=(n, 10))
    w = rng.normal(size=10)
    y = X @ w + rng.normal(0, 0.5, n)
    r_bool = holistic_residual(y, X, fit, ev)
    assert r_bool["verdict"] == "ok"                     # bool baseline works fine
    # DOCUMENTED HAZARD: int 0/1 masks are not rejected up front; they crash downstream.
    with pytest.raises(ValueError):
        holistic_residual(y, X, fit.astype(int), ev.astype(int))


# ---- 7. all-False fit_mask: NaN-comparison loophole in the degeneracy guard -----------


def test_holistic_all_false_fit_mask_fails_closed():
    """FIXED 2026-07-25 (was a documented hazard): an entirely-False fit_mask made
    y[fit_mask].std() NaN, and 'NaN < floor' is False -- the guard was bypassed and the
    function crashed downstream (best_alpha never assigned). Empty masks now return a
    clean degenerate verdict.

    kills: removing the count_nonzero/isfinite arm of the degeneracy guard.
    """
    rng = np.random.default_rng(606)
    n = 400
    _, ev = _even_odd(n)
    fit_empty = np.zeros(n, dtype=bool)
    X = rng.normal(size=(n, 10))
    w = rng.normal(size=10)
    y = X @ w + rng.normal(0, 0.5, n)
    r = holistic_residual(y, X, fit_empty, ev)
    assert r["verdict"] == "degenerate" and r["oos_r2"] is None
    r2 = holistic_residual(y, X, np.zeros(n, dtype=bool), np.zeros(n, dtype=bool))
    assert r2["verdict"] == "degenerate"      # both-empty also fails closed


# ---- 8. y_std_floor=0.0: the guard's own threshold can be degenerate ------------------


def test_holistic_zero_floor_constant_y_still_fails_closed():
    """FIXED 2026-07-25 (was a documented hazard): y_std_floor=0.0 with an EXACTLY
    constant y bypassed the strict '<' guard ('0.0 < 0.0' is False) and divided by zero
    downstream, returning verdict "ok" with oos_r2 = -inf. The guard now also fires on
    std == 0.0 regardless of the caller's floor, while the std == floor > 0 boundary
    semantics (strict '<', certified by the boundary test above) are unchanged.

    kills: dropping the explicit '== 0.0' arm of the degeneracy guard.
    """
    rng = np.random.default_rng(707)
    n = 400
    fit, ev = _even_odd(n)
    X = rng.normal(size=(n, 5))
    y_const = np.full(n, 3.14)
    assert y_const[fit].std() == 0.0 and y_const[ev].std() == 0.0
    r = holistic_residual(y_const, X, fit, ev, y_std_floor=0.0)
    assert r["verdict"] == "degenerate" and r["oos_r2"] is None


# ---- 9. column-order invariance (sanity / regression check) --------------------------


def test_holistic_column_permutation_invariance():
    """Permuting the columns of X, with everything else unchanged, must leave oos_r2,
    alpha, and unnamed_share EXACTLY unchanged -- the estimator has no legitimate
    dependence on column order or identity beyond the data itself.

    kills: an indexing bug that accidentally pins a specific column position (e.g. a
    hardcoded 'X[:, :5]' feature slice, or a construct-name-to-column manifest applied
    after code that has already reordered the columns).
    """
    rng = np.random.default_rng(808)
    n = 400
    p = 12
    fit, ev = _even_odd(n)
    X = rng.normal(size=(n, p))
    w = rng.normal(size=p)
    y = X @ w + rng.normal(0, 0.4, n)
    r1 = holistic_residual(y, X, fit, ev)
    perm = rng.permutation(p)
    r2 = holistic_residual(y, X[:, perm], fit, ev)
    assert r1["alpha"] == r2["alpha"]
    assert r1["oos_r2"] == pytest.approx(r2["oos_r2"], abs=1e-9)
    assert r1["unnamed_share"] == pytest.approx(r2["unnamed_share"], abs=1e-9)


# ---- 10. a single NaN in X silently poisons the WHOLE column -------------------------


def test_holistic_nan_in_predictor_raises_informative_error():
    """FIXED 2026-07-25 (was a documented hazard): a single NaN cell in X poisoned its
    ENTIRE z-scored column via X.mean(0)/X.std(0) and crashed later with an opaque
    TypeError (best_alpha never assigned). The estimator now rejects non-finite X up
    front with an informative ValueError naming the count.

    kills: removing the isfinite(X) precondition (reverting to the opaque late crash).
    """
    rng = np.random.default_rng(909)
    n = 400
    fit, ev = _even_odd(n)
    X = rng.normal(size=(n, 10))
    w = rng.normal(size=10)
    y = X @ w + rng.normal(0, 0.5, n)
    X_nan = X.copy()
    X_nan[5, 3] = np.nan                          # one missing cell, off both boundaries
    with pytest.raises(ValueError, match="non-finite"):
        holistic_residual(y, X_nan, fit, ev)
