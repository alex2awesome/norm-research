"""Adversarial tests for battery/stats.py's composition_rho (and zrank where it feeds the
composition semantics directly) — the composition-probe arm of the tacitness battery
("can the model judge 'A AND SIMULTANEOUSLY B' for two criteria it holds separately?").

Existing coverage (methods/tacit_channels/battery/tests/test_stats.py) already proves:
  - a true min-composer beats a single-member-only composer (test_composition_min_composer_agent)
  - the v1 single-target-reference path (member_refs of length 1) works
  - an OR-composer (max blend) scores far below a min-composer

This file does NOT repeat any of those. It attacks composition_rho with synthetic
composer-agents of KNOWN blending profiles along axes the first suite left open: scale
invariance of the min_z blend, 3-way (not just 2-way) reduction, adverse-min semantics
over MULTIPLE composed forms, the mode!="min_z" fallback footgun and its order-sensitivity,
a constant/trivial-member degeneracy in zrank, partial-NaN corruption, Spearman's shape
invariance, and empty-mask fail-closed behavior.

Everything here is deterministic (np.random.default_rng with a fixed seed per test), CPU-only,
no I/O. Genuine defects are asserted as CURRENT behavior with a "DOCUMENTED HAZARD" comment —
none are fixed here.
"""
import warnings

import numpy as np
import pytest

from methods.tacit_channels.battery.stats import composition_rho, zrank
from methods.tacit_channels.channels.common import spearman


def test_composition_scale_invariance_of_member_zrank():
    """A composer's score against the min_z AND-reference must not change when a member
    reference is rescaled/shifted by an arbitrary (order-preserving) amount — zrank()
    ranks before blending, so raw magnitude is not supposed to matter.
    kills: a raw-z or unnormalized-min mutation (dropping the zrank() call inside
    composition_rho's min_z branch, e.g. np.minimum.reduce(member_refs) on raw values)
    would make the reference — and hence the score — sensitive to this rescaling."""
    rng = np.random.default_rng(101)
    N = 300
    mask = np.ones(N, dtype=bool)
    a, b = rng.normal(size=N), rng.normal(size=N)
    ref = np.minimum(zrank(a), zrank(b))
    composer = ref + rng.normal(0, 0.1, N)
    baseline = composition_rho([composer], [a, b], mask)
    a_rescaled = a * 1e6 + 5e8            # huge multiplicative + additive distortion
    rescaled = composition_rho([composer], [a_rescaled, b], mask)
    assert baseline == pytest.approx(rescaled, abs=1e-9)
    assert baseline > 0.9


def test_composition_three_member_reduce_beats_pairwise_only():
    """A true 3-way AND composer (tracks the reduce-min of all three members) must score
    clearly higher against the 3-way reference than a composer that only satisfies 2 of
    the 3 criteria and ignores the third entirely.
    kills: an implementation that reduces only the first two member_refs (e.g.
    np.minimum(zrank(m[0]), zrank(m[1])), silently dropping m[2:]) instead of
    np.minimum.reduce over ALL supplied members."""
    rng = np.random.default_rng(102)
    N = 300
    mask = np.ones(N, dtype=bool)
    a, b, c = rng.normal(size=N), rng.normal(size=N), rng.normal(size=N)
    true_ref3 = np.minimum.reduce([zrank(a), zrank(b), zrank(c)])
    full_composer = true_ref3 + rng.normal(0, 0.1, N)
    ignoring_c = np.minimum(zrank(a), zrank(b)) + rng.normal(0, 0.1, N)
    full_score = composition_rho([full_composer], [a, b, c], mask)
    partial_score = composition_rho([ignoring_c], [a, b, c], mask)
    assert full_score > 0.9
    assert partial_score < 0.85
    assert full_score - partial_score > 0.1


def test_composition_adverse_min_over_forms_is_worst_not_mean():
    """One garbage (unrelated) composed form among two good ones must drag the adverse
    statistic down to (essentially) the garbage form's own score, not average it away.
    kills: replacing the adverse `min` over comp_vecs with a mean/median reduction."""
    rng = np.random.default_rng(103)
    N = 300
    mask = np.ones(N, dtype=bool)
    a, b = rng.normal(size=N), rng.normal(size=N)
    ref = np.minimum(zrank(a), zrank(b))
    good1 = ref + rng.normal(0, 0.05, N)
    good2 = ref + rng.normal(0, 0.05, N)
    garbage = rng.normal(size=N)
    individual = [spearman(v[mask], ref[mask]) for v in (good1, good2, garbage)]
    adverse = composition_rho([good1, good2, garbage], [a, b], mask)
    mean_would_be = float(np.mean(individual))
    assert adverse == pytest.approx(min(individual), abs=1e-9)
    assert adverse < mean_would_be - 0.3


def test_composition_mode_other_than_min_z_falls_through_to_first_member():
    """DOCUMENTED HAZARD (stats.py composition_rho, `if mode == "min_z" and
    len(member_refs) >= 2: ... else: ref = np.asarray(member_refs[0], float)`): ANY mode
    string other than the literal "min_z" — even with 2+ member_refs supplied — skips the
    AND-blend entirely and silently uses member_refs[0] as the WHOLE reference. A caller
    who passes mode="target" (a plausible typo/variant name) expecting an error, a
    different named reference, or at least a warning that member_refs[1:] are ignored
    gets none of that: every member past the first is dropped with zero signal, and its
    actual content is completely irrelevant to the result.
    kills: any assumption that an unrecognized `mode` is validated or warned about —
    asserts the CURRENT silently-ignoring behavior stays exactly this way."""
    rng = np.random.default_rng(104)
    N = 300
    mask = np.ones(N, dtype=bool)
    a, b = rng.normal(size=N), rng.normal(size=N)
    a_only_composer = zrank(a) + rng.normal(0, 0.05, N)
    score_ab = composition_rho([a_only_composer], [a, b], mask, mode="target")
    score_a_alone = composition_rho([a_only_composer], [a], mask, mode="target")
    unrelated_b = rng.normal(size=N)
    score_ab_diff_b = composition_rho([a_only_composer], [a, unrelated_b], mask,
                                      mode="target")
    assert score_ab == pytest.approx(score_a_alone, abs=1e-9)
    assert score_ab == pytest.approx(score_ab_diff_b, abs=1e-9)   # b's CONTENT is irrelevant
    assert score_ab > 0.9


def test_composition_member_refs_order_matters_in_fallback_path():
    """DOCUMENTED HAZARD (same fallback line as above): in the mode != "min_z" path,
    member_refs=[a, b] and member_refs=[b, a] are NOT equivalent calls — the reference
    silently becomes whichever member happens to sit at index 0. A caller that reorders
    its member list (e.g. after a dict-iteration-order change upstream) gets a
    completely different statistic with no error or warning.
    kills: any refactor that treats member_refs as order-independent outside the min_z
    branch (e.g. a set, or a reduce) — asserts the CURRENT order sensitivity."""
    rng = np.random.default_rng(105)
    N = 300
    mask = np.ones(N, dtype=bool)
    a, b = rng.normal(size=N), rng.normal(size=N)
    a_only_composer = zrank(a) + rng.normal(0, 0.05, N)
    score_ab = composition_rho([a_only_composer], [a, b], mask, mode="target")
    score_ba = composition_rho([a_only_composer], [b, a], mask, mode="target")
    assert score_ab > 0.9        # ref == a: composer matches it near-perfectly
    assert score_ba < 0.3        # ref == b (unrelated to the composer): score collapses


def test_composition_min_z_reduce_is_permutation_invariant_by_contrast():
    """Direct counterpoint to the two order-dependence hazards above: inside the REAL
    mode="min_z" path, np.minimum.reduce is commutative, so any permutation of the same
    member_refs list must give an identical statistic — the fallback's order-sensitivity
    is specific to that footgun, not a property of composition_rho in general.
    kills: a min_z implementation that folds refs sequentially with a non-commutative
    op, or that (like the fallback) accidentally keys off list position."""
    rng = np.random.default_rng(106)
    N = 300
    mask = np.ones(N, dtype=bool)
    a, b, c = rng.normal(size=N), rng.normal(size=N), rng.normal(size=N)
    comp3 = np.minimum.reduce([zrank(a), zrank(b), zrank(c)]) + rng.normal(0, 0.05, N)
    perms = [[a, b, c], [c, a, b], [b, c, a], [c, b, a], [a, c, b], [b, a, c]]
    scores = [composition_rho([comp3], refs, mask, mode="min_z") for refs in perms]
    assert all(s == pytest.approx(scores[0], abs=1e-9) for s in scores)
    assert scores[0] > 0.9


def test_composition_constant_member_ref_clamps_the_and_reference():
    """DOCUMENTED HAZARD (stats.py zrank: ranks of a constant array are all tied, so the
    std-guard `s if s > 0 else 1.0` divides a zero numerator by 1.0 -> zrank(constant) is
    the all-zero vector; that zero vector then enters composition_rho's
    np.minimum.reduce). A trivially-always-satisfied criterion — modeled as a CONSTANT
    member vector — is therefore NOT a no-op when folded into the min-AND blend:
    np.minimum(zrank(real), 0) clamps away the entire positive half of the real member's
    ranking. Even a composer that PERFECTLY tracks the one substantive criterion cannot
    reach rho=1.0 once a constant member is added to member_refs.
    kills: an unstated assumption that constant/inert members are harmless to include —
    asserts the CURRENT below-ceiling score instead of a fixed/guarded one."""
    rng = np.random.default_rng(107)
    N = 300
    mask = np.ones(N, dtype=bool)
    a = rng.normal(size=N)
    const_b = np.full(N, 5.0)
    perfect_a_composer = zrank(a).copy()
    score_single = composition_rho([perfect_a_composer], [a], mask)
    score_with_constant = composition_rho([perfect_a_composer], [a, const_b], mask)
    assert score_single == pytest.approx(1.0, abs=1e-9)     # no constant member: true ceiling
    assert score_with_constant < 0.97                       # HAZARD: ceiling not reached
    assert score_with_constant > 0.8                        # but not destroyed either


def test_composition_partial_nan_in_composed_vector_is_not_dropped():
    """DOCUMENTED HAZARD (stats.py composition_rho's `vals = [x for x in vals if not
    np.isnan(x)]` filter + common.py spearman's `if a.std() == 0` guard): that filter only
    catches a comp vector whose ENTIRE spearman result comes back NaN (a wholly-constant
    or wholly-degenerate vector). A vector with a handful of NaNs buried among otherwise-
    valid values slips through instead: std() over an array containing NaN is itself NaN
    (never equal to 0), so spearman's degenerate-input guard never fires; _rankdata's
    argsort then places each NaN at the END of the ranking (as if it were the maximum
    value), silently returning a real, WRONG correlation rather than being excluded.
    kills: an assumption that any NaN contamination in a comp vector gets cleanly
    dropped by the existing filter — asserts the CURRENT silent-corruption behavior."""
    rng = np.random.default_rng(108)
    N = 300
    mask = np.ones(N, dtype=bool)
    a, b = rng.normal(size=N), rng.normal(size=N)
    ref = np.minimum(zrank(a), zrank(b))
    clean_composer = ref.copy()
    corrupted_composer = ref.copy()
    corrupted_composer[[3, 7, 19, 42, 88]] = np.nan     # a handful of bad items, in-mask
    clean_score = composition_rho([clean_composer], [a, b], mask)
    corrupted_score = composition_rho([corrupted_composer], [a, b], mask)
    assert clean_score == pytest.approx(1.0, abs=1e-9)
    assert corrupted_score is not None and not np.isnan(corrupted_score)  # NOT dropped
    assert corrupted_score < clean_score - 0.01                           # but silently wrong


def test_composition_monotonic_transform_of_composer_is_rank_invariant():
    """A composer's raw output SCALE/SHAPE must never matter, only its ranking — e.g. a
    model that answers along a heavily nonlinear (but strictly monotonic) confidence
    curve is exactly as good a composer as one with a linear curve, provided the
    ORDERING matches the AND-reference.
    kills: swapping the adverse statistic from Spearman to a raw Pearson correlation
    (np.corrcoef on untransformed values) — Pearson is NOT shape-invariant, so a
    nonlinear-but-monotonic composer would (wrongly) score well below 1.0 under it,
    exactly as the raw-Pearson contrast below demonstrates."""
    rng = np.random.default_rng(109)
    N = 300
    mask = np.ones(N, dtype=bool)
    a, b = rng.normal(size=N), rng.normal(size=N)
    ref = np.minimum(zrank(a), zrank(b))
    identity_score = composition_rho([ref.copy()], [a, b], mask)
    shifted = ref - ref.min() + 1.0
    cubed = shifted ** 3                       # strictly increasing, wildly nonlinear
    transformed_score = composition_rho([cubed], [a, b], mask)
    raw_pearson = float(np.corrcoef(cubed, ref)[0, 1])
    assert identity_score == pytest.approx(1.0, abs=1e-9)
    assert transformed_score == pytest.approx(1.0, abs=1e-9)   # Spearman: untouched
    assert raw_pearson < 0.97                                  # Pearson WOULD have moved


def test_composition_empty_mask_fails_closed_to_none():
    """An all-False mask (zero surviving items) must degrade gracefully to None via the
    same NaN-drop path used for degenerate vectors elsewhere in this module — not crash
    with a numpy divide-by-zero / empty-slice exception.
    kills: any refactor that indexes into the masked arrays without going through the
    std()==0 / NaN-filtering guard, which would raise instead of returning None."""
    rng = np.random.default_rng(110)
    N = 300
    a, b = rng.normal(size=N), rng.normal(size=N)
    ref = np.minimum(zrank(a), zrank(b))
    composer = ref + rng.normal(0, 0.1, N)
    empty_mask = np.zeros(N, dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # empty-slice noise, not a failure
        result = composition_rho([composer], [a, b], empty_mask)
    assert result is None
