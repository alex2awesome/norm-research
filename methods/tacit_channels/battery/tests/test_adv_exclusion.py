"""Adversarial tests for the exclusion/leak probe statistic (battery/stats.py::leak_stats).

Companion to test_stats.py's leak_* coverage (compliant-vs-rigid, generic-factor agent,
partial-inverter lambda-ordering, mask-application) — this file targets angles those tests
leave open: sign-flipped cross-cell nulls, median-vs-mean aggregation, degenerate/disjoint
inputs, NaN handling inside the Spearman core it calls, the empty-cross-tf fallback, and the
statistic's return-type contract. Each test uses its OWN locally-seeded
np.random.default_rng so it is reproducible independent of file/test ordering.
"""
import numpy as np
import pytest

from methods.tacit_channels.battery.stats import leak_stats
from methods.tacit_channels.channels.common import spearman

N = 400
MASK = np.ones(N, dtype=bool)


def test_leak_anti_generic_factor_can_exceed_self():
    """kills: an abs(leak_cross) or max(0, leak_cross) clamp in leak_specific's definition.

    Shared-factor agent whose OTHER-cell tf vectors carry the shared factor with a FLIPPED
    sign (e.g. an annotation-direction confound). leak_self is high (rigid-looking), but
    leak_cross is strongly NEGATIVE, so the correct leak_specific = self - cross must
    EXCEED leak_self, not merely fall below it (as it would under an abs()/clip() bug that
    treats "very negative cross" the same as "no generic overlap")."""
    rng = np.random.default_rng(1001)
    shared = rng.normal(size=N)
    tf = {"canonical": shared + rng.normal(0, 0.2, N)}
    e = shared + rng.normal(0, 0.2, N)
    others = [-shared + rng.normal(0, 0.2, N) for _ in range(5)]   # flipped-sign null
    r = leak_stats({"canonical": e}, tf, others, MASK)
    assert r["leak_cross"] < -0.5
    assert r["leak_self"] > 0.7
    assert r["leak_specific"] > r["leak_self"] + 0.5


def test_leak_self_is_median_over_forms_not_mean():
    """kills: swapping np.median for np.mean (or a sum/len average) when pooling same[].

    Three forms with individual leak {+~.97, +~.97, -~.97}: the median sits with the
    2-1 majority (~+.97); the mean would land near +0.32 — a materially different,
    less robust number. Cross-checks leak_self against BOTH candidates directly."""
    rng = np.random.default_rng(1002)
    t = rng.normal(size=N)
    tf = {f: t + rng.normal(0, 0.1, N) for f in ("f1", "f2", "f3")}
    excl = {"f1": t + rng.normal(0, 0.2, N), "f2": t + rng.normal(0, 0.2, N),
            "f3": -t + rng.normal(0, 0.2, N)}
    others = [rng.normal(size=N) for _ in range(4)]
    r = leak_stats(excl, tf, others, MASK)
    individual = [spearman(np.asarray(excl[f])[MASK], np.asarray(tf[f])[MASK])
                  for f in excl]
    mean_val, median_val = float(np.mean(individual)), float(np.median(individual))
    assert r["leak_self"] > 0.8
    assert abs(r["leak_self"] - mean_val) > 0.3           # rules out a mean-based swap
    assert r["leak_self"] == pytest.approx(median_val, abs=1e-9)


def test_leak_constant_form_dropped_all_constant_returns_none():
    """kills: a missing/relocated np.isnan filter on `same` (constant-form leak silently
    entering the median as 0 or crashing corrcoef upstream), and a fallback that returns a
    number instead of None when every form is degenerate.

    One p_yes-collapsed (constant) form among three must be transparently DROPPED — the
    result must equal computing leak_stats with that form omitted entirely, bit-for-bit.
    When ALL forms are constant, leak_stats must return None, never a number."""
    rng = np.random.default_rng(1003)
    t = rng.normal(size=N)
    tf = {f: t + rng.normal(0, 0.1, N) for f in ("f1", "f2", "f3")}
    others = [rng.normal(size=N) for _ in range(4)]
    valid_f1, valid_f3 = t + rng.normal(0, 0.1, N), t + rng.normal(0, 0.1, N)
    excl_mixed = {"f1": valid_f1, "f2": np.full(N, 3.0), "f3": valid_f3}
    r_mixed = leak_stats(excl_mixed, tf, others, MASK)
    r_dropped = leak_stats({"f1": valid_f1, "f3": valid_f3}, tf, others, MASK)
    assert r_mixed == r_dropped
    excl_allconst = {"f1": np.full(N, 1.0), "f2": np.full(N, 2.0), "f3": np.full(N, -5.0)}
    assert leak_stats(excl_allconst, tf, others, MASK) is None


def test_leak_empty_cross_tf_fails_closed_with_flag():
    """FIXED 2026-07-25 (was a documented hazard): with no cross-cell null vectors,
    leak_specific silently equalled the raw leak_self headline the module docstring
    forbids. Now leak_cross/leak_specific are None with n_cross=0 — callers cannot
    mistake "cross never computed" for "cross computed as ~0"; leak_self stays readable.
    kills: reverting the empty-cross branch to the 0.0 fallback."""
    rng = np.random.default_rng(1004)
    t = rng.normal(size=N)
    tf = {"canonical": t + rng.normal(0, 0.1, N)}
    e = t + rng.normal(0, 0.1, N)
    r = leak_stats({"canonical": e}, tf, [], MASK)
    assert r["leak_self"] > 0.5                # the raw signal stays inspectable...
    assert r["leak_cross"] is None             # ...but never masquerades as corrected
    assert r["leak_specific"] is None
    assert r["n_cross"] == 0
    with_cross = leak_stats({"canonical": e}, tf, [rng.normal(size=N)], MASK)
    assert with_cross["n_cross"] == 1 and with_cross["leak_specific"] is not None


def test_leak_monotone_transform_invariance():
    """kills: any switch from Spearman (rank-based) to Pearson correlation inside
    leak_stats' call chain. Applying a strictly increasing, rank-preserving squash
    (exp(x/3)) to every input vector — exclusion, own-tf, and all cross-tf vectors — must
    leave leak_self/leak_cross/leak_specific EXACTLY unchanged, since Spearman depends only
    on rank order. A Pearson-based statistic would shift under this nonlinear rescaling."""
    rng = np.random.default_rng(1005)
    t = rng.normal(size=N)
    tf = {"canonical": t + rng.normal(0, 0.15, N)}
    e = t + rng.normal(0, 0.15, N)
    others = [rng.normal(size=N) for _ in range(4)]
    base = leak_stats({"canonical": e}, tf, others, MASK)
    squash = lambda v: np.exp(v / 3.0)
    trans = leak_stats({"canonical": squash(e)}, {"canonical": squash(tf["canonical"])},
                        [squash(o) for o in others], MASK)
    for k in ("leak_self", "leak_cross", "leak_specific"):
        assert trans[k] == pytest.approx(base[k], abs=1e-9)


def test_leak_single_nan_item_silently_perturbs_rather_than_drops():
    """kills: nothing — this documents current behavior so a future FIX (dropping NaN
    items before ranking) is a deliberate, visible change to this test, not a silent one.

    DOCUMENTED HAZARD (methods/tacit_channels/channels/common.py:91-97 `spearman`, :78-88
    `_rankdata`): spearman() only guards against a WHOLE-VECTOR constant (`a.std() == 0`);
    it never checks for individual NaN entries. A single np.nan survives into
    _rankdata's argsort, which (via mergesort's NaN-sorts-last behavior) assigns it an
    ordinary FINITE extreme rank instead of being excluded — so one NaN item silently
    PERTURBS leak_self by a small amount rather than being dropped, raising, or
    propagating a NaN that leak_stats' own isnan-filter (stats.py:41) could catch."""
    rng = np.random.default_rng(1006)
    t = rng.normal(size=N)
    tf = {"canonical": t + rng.normal(0, 0.1, N)}
    e_clean = t + rng.normal(0, 0.1, N)
    others = [rng.normal(size=N) for _ in range(4)]
    clean = leak_stats({"canonical": e_clean}, tf, others, MASK)
    e_nan = e_clean.copy()
    e_nan[3] = np.nan
    nan_r = leak_stats({"canonical": e_nan}, tf, others, MASK)
    assert nan_r is not None and np.isfinite(nan_r["leak_self"])
    diff = abs(nan_r["leak_self"] - clean["leak_self"])
    assert 0 < diff < 0.01           # nonzero (not dropped) but small (not catastrophic)


def test_leak_disjoint_form_keys_returns_none_via_key_match_not_position():
    """kills: replacing the `if f in tf_by_form` key-match filter (stats.py:40) with a
    positional zip of excl_by_form/tf_by_form values — a mutant like that would silently
    pair mismatched forms and return a bogus number here instead of None.

    excl_by_form and tf_by_form share ZERO form names (a plumbing/naming-drift scenario,
    not numeric degeneracy) -> every entry is filtered out of `same` -> None."""
    rng = np.random.default_rng(1007)
    t = rng.normal(size=N)
    tf = {"canonical": t + rng.normal(0, 0.1, N)}
    e = t + rng.normal(0, 0.1, N)          # would score ~perfectly IF keys matched
    others = [rng.normal(size=N) for _ in range(3)]
    assert leak_stats({"totally_different_form_name": e}, tf, others, MASK) is None


def test_leak_all_false_mask_returns_none_not_a_nan_dict():
    """kills: a guard that only checks mask.any() implicitly via array truthiness (which
    raises ValueError on a non-empty boolean array) or a version that returns a
    NaN-filled dict instead of routing through the same[] isnan-filter to None."""
    rng = np.random.default_rng(1008)
    t = rng.normal(size=N)
    tf = {"canonical": t + rng.normal(0, 0.1, N)}
    e = t + rng.normal(0, 0.1, N)
    others = [rng.normal(size=N) for _ in range(3)]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = leak_stats({"canonical": e}, tf, others, np.zeros(N, dtype=bool))
    assert r is None


def test_leak_cross_side_nan_filtered_independently_of_same_side():
    """kills: removing the inline `if not np.isnan(s): cross.append(s)` filter in the
    cross-loop (stats.py:47-49) — e.g. a mutant that appends every score unconditionally,
    letting a degenerate other-cell vector inject a NaN (or a substituted 0.0) into the
    cross list and shift its median.

    A degenerate (constant) OTHER-cell tf vector mixed into cross_tf must be silently
    excluded from leak_cross's median — the result must exactly match computing
    leak_stats with that vector omitted from cross_tf altogether."""
    rng = np.random.default_rng(1009)
    t = rng.normal(size=N)
    tf = {"canonical": t + rng.normal(0, 0.1, N)}
    e = t + rng.normal(0, 0.1, N)
    good_others = [rng.normal(size=N), rng.normal(size=N)]
    const_other = np.full(N, 7.0)
    mixed = leak_stats({"canonical": e}, tf,
                       good_others[:1] + [const_other] + good_others[1:], MASK)
    only_good = leak_stats({"canonical": e}, tf, good_others, MASK)
    assert mixed == only_good


def test_leak_stats_returns_native_python_floats():
    """kills: dropping the explicit float(...) casts around np.median (stats.py:50-51),
    which would leak np.float64 into the returned dict. Downstream tallies persist these
    dicts via write_jsonl -> json.dumps (channels/common.py write_jsonl), which raises
    TypeError on a bare numpy scalar — isinstance(x, float) alone does NOT catch this
    regression since np.float64 subclasses the builtin float; type(x) is float does."""
    rng = np.random.default_rng(1010)
    t = rng.normal(size=N)
    tf = {"canonical": t + rng.normal(0, 0.1, N)}
    e = t + rng.normal(0, 0.1, N)
    others = [rng.normal(size=N) for _ in range(3)]
    r = leak_stats({"canonical": e}, tf, others, MASK)
    for k in ("leak_self", "leak_cross", "leak_specific"):
        assert type(r[k]) is float, f"{k} is {type(r[k])}, not a native float"
