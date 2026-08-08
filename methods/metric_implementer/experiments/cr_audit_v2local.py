"""CR-3: executor-indexed prompt-articulation certificates.

The scientific target is a best SINGLE prompt, not a checklist assembled from an
unbounded number of atomic criteria.  On a frozen probe panel X, target verdict M,
executor E, and prompt p, define

    R_E(p) = I(M ; binarize(E(p, X))).

For a frozen discovery pool Omega and future prompts drawn independently within
predeclared proposer families, this module certifies three quantities:

1. classifier-relative behavioral novelty mass;
2. exact-pattern novelty mass, which can support a population-size/exhaustion
   conversion only under an external minimum-mass assumption p_min; and
3. an upper bound on the expected best-prompt recovery after a fixed future draw
   budget.  If G(p) = max(0, R_E(p) - max_{q in Omega} R_E(q)), then

       E[max future recovery] <= R_Omega + sum_f m_f E[G | family=f].

Per-family Clopper-Pearson, empirical-Bernstein, and DKW bounds make all three
primary upper statements simultaneous at level 1-alpha.  No fuzzy-species substitutability,
submodularity, Good-Turing value extrapolation, or optimizer-asymptote assumption is
used.  All recovery statements are conditional on the frozen empirical probe panel;
lifting them to a deployment distribution requires an independent probe-lockbox
generalization layer.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Dict, Sequence

import numpy as np
from scipy.stats import beta as _beta

from .cr_horizon import leader_species
from .value_census import i_binary
from .. import vinfo

FAMILY_PREFIX_DEFAULT = ("glm_a", "glm_b", "glm_c")
SCHEMA_VERSION = "cr3-prompt-articulation-v2"
ALL_PROMPT_SCHEMA_VERSION = "all-finite-prompt-dpi-v1"


def clopper_pearson_lower(z: int, n: int, alpha: float) -> float:
    """One-sided exact lower bound for a Binomial(n, p) parameter."""
    _validate_binomial_args(z, n, alpha)
    if z == 0:
        return 0.0
    return float(_beta.ppf(alpha, z, n - z + 1))


def clopper_pearson_upper(z: int, n: int, alpha: float) -> float:
    """One-sided exact upper bound for a Binomial(n, p) parameter."""
    _validate_binomial_args(z, n, alpha)
    if z == n:
        return 1.0
    return float(_beta.ppf(1.0 - alpha, z + 1, n - z))


def clopper_pearson_interval(z: int, n: int, alpha: float) -> tuple[float, float]:
    """Two-sided exact ``1-alpha`` Clopper-Pearson interval."""
    _validate_binomial_args(z, n, alpha)
    return (
        clopper_pearson_lower(z, n, alpha / 2.0),
        clopper_pearson_upper(z, n, alpha / 2.0),
    )


def _validate_binomial_args(z: int, n: int, alpha: float) -> None:
    if n <= 0 or z < 0 or z > n:
        raise ValueError(f"invalid binomial count z={z}, n={n}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")


def empirical_bernstein_upper(y: np.ndarray, b_cap: float, alpha: float) -> float:
    """Maurer-Pontil one-sided UCB for the mean of independent marks in [0, b_cap].

    This uses the theorem directly.  It deliberately does not choose the minimum of
    several same-data confidence bounds without a multiplicity allocation.
    """
    y = np.asarray(y, float)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    if not np.isfinite(b_cap) or b_cap < 0.0:
        raise ValueError(f"b_cap must be finite and nonnegative, got {b_cap}")
    if np.any(~np.isfinite(y)) or np.any(y < -1e-12) or np.any(y > b_cap + 1e-12):
        raise ValueError("marks must be finite and lie in [0, b_cap]")
    if b_cap == 0.0:
        return 0.0
    n = len(y)
    if n < 2:
        return float(b_cap)
    var = float(np.var(y, ddof=1))
    ln = float(np.log(2.0 / alpha))
    ucb = float(y.mean() + np.sqrt(2.0 * var * ln / n)
                + 7.0 * b_cap * ln / (3.0 * (n - 1)))
    return float(np.clip(ucb, 0.0, b_cap))


def dkw_expected_max_upper(
    marks: Mapping[str, np.ndarray],
    horizon: Mapping[str, int],
    b_cap: float,
    component_alpha: float,
) -> tuple[float, dict[str, float]]:
    """DKW upper confidence bound for the expected future maximum mark.

    With probability at least ``1-sum_f component_alpha``, every family CDF obeys
    ``F_f(t) >= max(0, Fhat_f(t)-eps_f)`` simultaneously for all ``t``.  Independence
    of future draws then gives an upper envelope for the survival function of their
    maximum, which is integrated exactly between observed mark values.
    """
    if not np.isfinite(b_cap) or b_cap < 0.0:
        raise ValueError("b_cap must be finite and nonnegative")
    if not 0.0 < component_alpha < 1.0:
        raise ValueError("component_alpha must lie in (0, 1)")
    families = tuple(marks)
    if set(horizon) != set(families):
        raise ValueError("horizon and marks must have identical family keys")
    arrays: dict[str, np.ndarray] = {}
    eps: dict[str, float] = {}
    for family in families:
        y = np.asarray(marks[family], float)
        if y.ndim != 1 or len(y) == 0:
            raise ValueError(f"family {family!r} needs a nonempty one-dimensional mark sample")
        if np.any(~np.isfinite(y)) or np.any(y < -1e-12) or np.any(y > b_cap + 1e-12):
            raise ValueError("marks must be finite and lie in [0, b_cap]")
        if int(horizon[family]) < 0:
            raise ValueError("future horizon counts must be nonnegative")
        arrays[family] = np.clip(y, 0.0, b_cap)
        eps[family] = float(np.sqrt(np.log(2.0 / component_alpha) / (2.0 * len(y))))
    if b_cap == 0.0 or not any(int(horizon[f]) for f in families):
        return 0.0, eps

    points = sorted({0.0, b_cap, *(float(v) for y in arrays.values() for v in y)})
    upper = 0.0
    for left, right in zip(points[:-1], points[1:]):
        if right <= left:
            continue
        probe = (left + right) / 2.0
        cdf_max_lower = 1.0
        for family in families:
            m = int(horizon[family])
            if m == 0:
                continue
            fhat = float(np.mean(arrays[family] <= probe))
            cdf_lower = max(0.0, fhat - eps[family])
            cdf_max_lower *= cdf_lower ** m
        upper += (right - left) * (1.0 - cdf_max_lower)
    return float(np.clip(upper, 0.0, b_cap)), eps


def stratified_split(tags: Sequence[str], *, family_prefixes: Sequence[str],
                     audit_frac: float = 1.0 / 3.0, seed: int = 0) -> Dict[str, np.ndarray]:
    """Random discovery/audit split within every predeclared family.

    Splitting cannot manufacture independence.  Certificate validity still requires
    the original draws to have been independent conditional on family.
    """
    if not 0.0 < audit_frac < 1.0:
        raise ValueError("audit_frac must lie in (0, 1)")
    prefixes = tuple(str(x) for x in family_prefixes)
    if not prefixes or len(set(prefixes)) != len(prefixes):
        raise ValueError("family_prefixes must be nonempty and unique")
    rng = np.random.default_rng(seed)
    disc: list[int] = []
    aud: list[int] = []
    fam_of: dict[int, str] = {}
    assigned: set[int] = set()
    for fp in prefixes:
        idx = np.array([i for i, t in enumerate(tags) if str(t).startswith(fp)], int)
        if idx.size < 2:
            raise ValueError(f"family {fp!r} needs at least two draws, got {idx.size}")
        overlap = assigned.intersection(map(int, idx))
        if overlap:
            raise ValueError(f"overlapping family prefixes assign indices twice: {sorted(overlap)[:3]}")
        assigned.update(map(int, idx))
        perm = rng.permutation(idx)
        n_aud = min(len(idx) - 1, max(1, int(round(audit_frac * len(idx)))))
        a, d = perm[:n_aud], perm[n_aud:]
        aud.extend(map(int, a))
        disc.extend(map(int, d))
        fam_of.update({int(i): fp for i in a})
    return {
        "discovery": np.asarray(sorted(disc), int),
        "audit": np.asarray(sorted(aud), int),
        "family_of": fam_of,
    }


def _agree_max(col: np.ndarray, leader_cols: np.ndarray) -> float:
    if leader_cols.shape[0] == 0:
        return 0.0
    return float((leader_cols == col[None, :]).mean(axis=1).max())


def _binary_matrix(x: np.ndarray, *, name: str) -> np.ndarray:
    a = np.asarray(x, float)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty 2-D matrix")
    if np.any(~np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite scores")
    return (a > 0.5).astype(np.uint8)


def _weights(families: Sequence[str], family_weights: Mapping[str, float] | None) -> dict[str, float]:
    if family_weights is None:
        return {f: 1.0 / len(families) for f in families}
    if set(family_weights) != set(families):
        raise ValueError("family_weights keys must exactly match family_names")
    out = {f: float(family_weights[f]) for f in families}
    if any(not np.isfinite(w) or w <= 0.0 for w in out.values()):
        raise ValueError("family weights must be finite and positive")
    total = sum(out.values())
    return {f: w / total for f, w in out.items()}


def _horizon_counts(families: Sequence[str], horizon_per_family: int | Mapping[str, int]) -> dict[str, int]:
    if isinstance(horizon_per_family, Mapping):
        if set(horizon_per_family) != set(families):
            raise ValueError("horizon keys must exactly match family_names")
        out = {f: int(horizon_per_family[f]) for f in families}
    else:
        out = {f: int(horizon_per_family) for f in families}
    if any(v < 0 for v in out.values()):
        raise ValueError("future horizon counts must be nonnegative")
    return out


def all_finite_prompt_dpi_certificate(
    candidate_sigs: np.ndarray,
    M: np.ndarray,
    *,
    candidate_labels: Sequence[str] | None = None,
    identity_witness_index: int | None = None,
    identity_witness_is_target_definition: bool = False,
    epsilon_bits: float = 0.02,
    atol: float = 1e-12,
    scope: Mapping[str, object] | None = None,
) -> dict:
    """Certify a candidate against the target-indexed DPI over all finite prompts.

    The implemented estimand is the hard-verdict fixed-target objective

        A^*_{b,E} = sup_{p in Sigma*} I(M_b; binarize(E(p, X))).

    On the frozen panel ``M_b`` is deterministic given ``X``, so ``H(M_b)`` is a
    valid upper bound for *every* finite prompt.  The best evaluated candidate is
    therefore globally epsilon-optimal with certified gap at most ``H(M_b)-R``.

    ``identity_witness_is_target_definition`` is stronger than empirical equality:
    it declares that the indexed candidate is literally the one-form prompt used
    to define ``M_b`` under the same frozen executor and readout.  The function
    checks equality on the supplied panel and records the constructional premise;
    callers must establish the prompt/executor/readout identity in provenance.
    Capture-recapture quantities are deliberately absent because finite black-box
    sampling cannot tighten an all-``Sigma*`` upper bound.
    """
    candidates = _binary_matrix(candidate_sigs, name="candidate_sigs")
    raw_target = np.asarray(M, float)
    if raw_target.ndim != 1 or len(raw_target) != candidates.shape[1]:
        raise ValueError("M and candidate_sigs must share the probe dimension")
    if np.any(~np.isfinite(raw_target)):
        raise ValueError("M contains non-finite scores")
    if not np.isfinite(epsilon_bits) or epsilon_bits < 0.0:
        raise ValueError("epsilon_bits must be finite and nonnegative")
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and nonnegative")

    labels = ([f"candidate_{i}" for i in range(len(candidates))]
              if candidate_labels is None else [str(x) for x in candidate_labels])
    if len(labels) != len(candidates):
        raise ValueError("candidate_labels length must equal the number of candidates")
    if identity_witness_is_target_definition and identity_witness_index is None:
        raise ValueError("a declared target-definition identity needs an identity_witness_index")
    if identity_witness_index is not None and not 0 <= int(identity_witness_index) < len(candidates):
        raise ValueError("identity_witness_index is out of range")

    y = (raw_target > 0.5).astype(np.uint8)
    h_m = float(vinfo._h_bits(float(y.mean()))) if np.unique(y).size > 1 else 0.0
    recovery = np.asarray([i_binary(y, row) for row in candidates], float)
    best_index = int(np.argmax(recovery))
    best_recovery = float(recovery[best_index])
    gap = float(max(0.0, h_m - best_recovery))
    panel_attained = bool(gap <= atol)
    mismatch = np.mean(candidates != y[None, :], axis=1)
    best_mismatch = float(mismatch[best_index])
    if best_mismatch <= atol:
        best_relation = "EXACT_MATCH"
    elif best_mismatch >= 1.0 - atol:
        best_relation = "EXACT_COMPLEMENT"
    else:
        best_relation = "MIXED_ERRORS"

    identity = None
    if identity_witness_index is not None:
        witness_index = int(identity_witness_index)
        witness_matches = bool(np.array_equal(candidates[witness_index], y))
        if identity_witness_is_target_definition and not witness_matches:
            raise ValueError("declared target-definition witness does not reproduce M on the panel")
        identity = {
            "candidate_index": witness_index,
            "candidate_label": labels[witness_index],
            "matches_target_on_frozen_panel": witness_matches,
            "is_literal_target_defining_prompt_under_same_executor_readout": bool(
                identity_witness_is_target_definition),
        }

    structural_identity = bool(identity_witness_is_target_definition)
    if h_m <= atol:
        status = "DEGENERATE_CONSTANT_TARGET"
    elif structural_identity:
        status = "PROVABLY_OPTIMAL_IDENTITY"
    elif panel_attained:
        status = "PROVABLY_OPTIMAL_DPI_ATTAINED_FIXED_PANEL"
    elif gap <= epsilon_bits + atol:
        status = "PROVABLY_EPSILON_OPTIMAL_FIXED_PANEL"
    else:
        status = "GLOBAL_GAP_EXCEEDS_TARGET_EPSILON"

    return {
        "schema": ALL_PROMPT_SCHEMA_VERSION,
        "estimand": {
            "name": "executor-indexed unrestricted promptable articulation",
            "definition": "A^*_{b,E} = sup_{p in Sigma*} I(M_b; binarize(E(p, X)))",
            "prompt_class": "all finite prompts Sigma*; no prompt-length budget",
            "target": "the frozen hard operationalization M_b",
            "unit": "Shannon bits",
        },
        "certificate": {
            "status": status,
            "all_prompt_DPI_upper_bound_bits": h_m,
            "best_evaluated_lower_bound_bits": best_recovery,
            "articulation_value_identified_interval_bits": [best_recovery, h_m],
            "certified_optimization_gap_UCB_bits": gap,
            "target_epsilon_bits": float(epsilon_bits),
            "meets_target_epsilon": bool(gap <= epsilon_bits + atol),
            "best_candidate_index": best_index,
            "best_candidate_label": labels[best_index],
            "best_candidate_target_mismatch_rate": best_mismatch,
            "best_candidate_target_relation": best_relation,
            "DPI_attained_on_frozen_panel": panel_attained,
            "identity_witness": identity,
        },
        "proof_scope": {
            **dict(scope or {}),
            "fixed_panel": True,
            "population_exact_by_construction": structural_identity,
            "population_note": (
                "Exact wherever M_b is defined by this same one-form prompt, executor, and hard readout; this is an identity theorem, not an empirical generalization claim."
                if structural_identity else
                "The numerical recovery and gap are fixed-panel quantities; a population gap needs an independent iid lockbox and simultaneous upper/lower confidence bounds."
            ),
            "assumptions": [
                "M_b is the deterministic hard target induced on X",
                "candidate verdicts use the same frozen executor and readout protocol",
                "M_b -> X -> candidate verdict is the declared channel",
            ],
            "not_used": [
                "capture-recapture",
                "proposer saturation",
                "submodularity",
                "a finite prompt-length manifest",
            ],
        },
    }


def all_finite_prompt_population_certificate(
    target: np.ndarray,
    candidate: np.ndarray,
    *,
    candidate_frozen_before_lockbox: bool,
    alpha: float = 0.05,
    epsilon_bits: float = 0.02,
    scope: Mapping[str, object] | None = None,
) -> dict:
    """Population epsilon-global certificate from a fresh iid hard-verdict lockbox.

    The objective is polarity-invariant MI; neither view is an anchor label.  Define

        e_pm = min(P(candidate != target), P(1-candidate != target)).

    Flipping a binary candidate does not change MI, and after the better population
    flip the error bit and candidate determine the target.  A two-sided exact binomial
    interval for the raw mismatch probability therefore gives an upper confidence
    bound on ``e_pm`` without selecting polarity from the lockbox.  Candidate
    suboptimality obeys

        A^*_{b,E} - I(target; candidate) <= H(target | candidate) <= h(e),

    so its optimization-gap UCB does not pay additional target-prevalence
    uncertainty.  Prevalence is still needed to report a confidence interval for
    the *value* ``A^*_{b,E}`` in bits.

    Candidate selection, prompt editing, and threshold selection must be frozen before
    the lockbox.  Polarity is quotiented out by the estimand itself, not treated as an
    adaptively selected semantic label.  This function refuses an adaptive candidate
    rather than silently treating a reused search panel as population evidence.
    """
    if not candidate_frozen_before_lockbox:
        raise ValueError("candidate and readout must be frozen before the iid lockbox")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if not np.isfinite(epsilon_bits) or epsilon_bits < 0.0:
        raise ValueError("epsilon_bits must be finite and nonnegative")

    target_raw = np.asarray(target, float)
    candidate_raw = np.asarray(candidate, float)
    if (target_raw.ndim != 1 or candidate_raw.ndim != 1
            or target_raw.shape != candidate_raw.shape or len(target_raw) < 2):
        raise ValueError("target and candidate must be aligned one-dimensional lockbox vectors")
    if np.any(~np.isfinite(target_raw)) or np.any(~np.isfinite(candidate_raw)):
        raise ValueError("target and candidate contain non-finite scores")
    y = (target_raw > 0.5).astype(np.uint8)
    z = (candidate_raw > 0.5).astype(np.uint8)

    n = len(y)
    positives = int(y.sum())
    raw_mismatches = int(np.sum(y != z))
    # Half of alpha covers a two-sided target-prevalence interval; half covers
    # the two-sided raw-mismatch interval.  Their union failure is <= alpha.
    alpha_prevalence = alpha / 2.0
    alpha_error = alpha / 2.0
    p_lo, p_hi = clopper_pearson_interval(positives, n, alpha_prevalence)
    mismatch_lo, mismatch_hi = clopper_pearson_interval(raw_mismatches, n, alpha_error)
    if mismatch_lo <= 0.5 <= mismatch_hi:
        e_pm_u = 0.5
    elif mismatch_hi < 0.5:
        e_pm_u = mismatch_hi
    else:
        e_pm_u = 1.0 - mismatch_lo

    h_p_lo = float(vinfo._h_bits(p_lo))
    h_p_hi = float(vinfo._h_bits(p_hi))
    h_target_lo = float(min(h_p_lo, h_p_hi))
    h_target_u = (1.0 if p_lo <= 0.5 <= p_hi
                  else float(max(h_p_lo, h_p_hi)))
    # h(e) increases only to 1/2.  If the confidence set crosses 1/2, the
    # assumption-free entropy upper bound for the error bit is one full bit.
    h_error_u = (1.0 if e_pm_u >= 0.5 else float(vinfo._h_bits(e_pm_u)))
    gap_u = float(min(h_target_u, h_error_u))
    recovery_lo = float(max(0.0, h_target_lo - gap_u))
    status = ("PROVABLY_EPSILON_OPTIMAL_POPULATION"
              if gap_u <= epsilon_bits else
              "POPULATION_GAP_EXCEEDS_TARGET_EPSILON")

    return {
        "schema": "all-finite-prompt-population-v1",
        "estimand": {
            "definition": "A^*_{b,E} = sup_{p in Sigma*} I(M_b; binarize(E(p, X)))",
            "prompt_class": "all finite prompts Sigma*; no prompt-length budget",
            "population": "the iid lockbox item distribution",
        },
        "certificate": {
            "status": status,
            "simultaneous_confidence": 1.0 - alpha,
            "all_prompt_DPI_upper_bound_UCB_bits": h_target_u,
            "candidate_recovery_LCB_bits": recovery_lo,
            "articulation_value_confidence_interval_bits": [recovery_lo, h_target_u],
            "certified_optimization_gap_UCB_bits": gap_u,
            "target_epsilon_bits": float(epsilon_bits),
            "meets_target_epsilon": bool(gap_u <= epsilon_bits),
            "target_prevalence_interval": [p_lo, p_hi],
            "target_entropy_interval_bits": [h_target_lo, h_target_u],
            "raw_mismatch_rate_observed": raw_mismatches / n,
            "raw_mismatch_probability_interval": [mismatch_lo, mismatch_hi],
            "polarity_invariant_error_rate_observed": min(
                raw_mismatches / n, 1.0 - raw_mismatches / n),
            "polarity_invariant_error_rate_UCB": e_pm_u,
            "conditional_entropy_UCB_bits": h_error_u,
            "polarity_treatment": "invariant; exact complements are equally informative",
            "n_lockbox": n,
        },
        "proof_scope": {
            **dict(scope or {}),
            "alpha_allocation": {
                "target_prevalence_interval": alpha_prevalence,
                "raw_mismatch_probability_interval": alpha_error,
            },
            "inequality": (
                "A^*_{b,E}-I(M_b;Z) <= H(M_b)-I(M_b;Z) "
                "= H(M_b|Z) <= h(min(P[M_b!=Z], P[M_b!=1-Z]))"
            ),
            "assumptions": [
                "iid lockbox items from the claimed population",
                "hard binary target and candidate verdicts",
                "candidate prompt and readout threshold frozen before lockbox access",
                "lockbox was not used for prompt mining, selection, stopping, or revision",
            ],
        },
    }


def zero_error_lockbox_plan(
    epsilon_bits: float,
    *,
    alpha: float = 0.05,
    max_n: int = 10_000_000,
) -> dict:
    """Smallest lockbox size whose zero-error gap UCB is at most ``epsilon_bits``.

    This is a prospective power calculation for
    :func:`all_finite_prompt_population_certificate`, using the same half-alpha
    allocation for the one-sided error bound.  It assumes the frozen candidate will
    make zero observed errors; any error requires recomputing the achieved bound.
    """
    if not np.isfinite(epsilon_bits) or not 0.0 < epsilon_bits < 1.0:
        raise ValueError("epsilon_bits must lie strictly between 0 and 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if int(max_n) < 2:
        raise ValueError("max_n must be at least 2")
    alpha_error = alpha / 2.0

    def bound(n: int) -> tuple[float, float]:
        _, mismatch_hi = clopper_pearson_interval(0, n, alpha_error)
        e_u = mismatch_hi
        gap_u = 1.0 if e_u >= 0.5 else float(vinfo._h_bits(e_u))
        return e_u, gap_u

    lo, hi = 2, 2
    while hi < max_n and bound(hi)[1] > epsilon_bits:
        hi = min(int(max_n), hi * 2)
    if bound(hi)[1] > epsilon_bits:
        raise ValueError(f"max_n={max_n} is insufficient for epsilon_bits={epsilon_bits}")
    while lo < hi:
        mid = (lo + hi) // 2
        if bound(mid)[1] <= epsilon_bits:
            hi = mid
        else:
            lo = mid + 1
    e_u, gap_u = bound(lo)
    return {
        "schema": "zero-error-lockbox-plan-v1",
        "epsilon_bits": float(epsilon_bits),
        "alpha": float(alpha),
        "confidence": float(1.0 - alpha),
        "n_lockbox_required": int(lo),
        "zero_error_rate_UCB": e_u,
        "zero_error_gap_UCB_bits": gap_u,
        "assumptions": [
            "zero observed errors from the frozen oriented candidate",
            "iid lockbox items",
            "the same alpha allocation as all_finite_prompt_population_certificate",
        ],
    }


def prompt_articulation_certificate(
    pool_sigs: np.ndarray,
    audit_sigs: np.ndarray,
    M: np.ndarray,
    audit_families: Sequence[str],
    *,
    family_names: Sequence[str],
    family_weights: Mapping[str, float] | None = None,
    horizon_per_family: int | Mapping[str, int] = 100,
    tau: float = 0.90,
    tau_strict: float = 0.95,
    alpha: float = 0.05,
    p_min: float | None = None,
    scope: Mapping[str, object] | None = None,
    debug_internals: bool = False,
) -> dict:
    """Issue a simultaneous process-relative best-single-prompt certificate.

    ``p_min`` refers to the probability of every exact behavior pattern under the
    declared proposer mixture.  It is never estimated here.
    """
    pool = _binary_matrix(pool_sigs, name="pool_sigs")
    audit = _binary_matrix(audit_sigs, name="audit_sigs")
    y = (np.asarray(M, float) > 0.5).astype(np.uint8)
    if y.ndim != 1 or len(y) != pool.shape[1] or audit.shape[1] != pool.shape[1]:
        raise ValueError("M, pool_sigs, and audit_sigs must share the probe dimension")
    if np.any(~np.isfinite(np.asarray(M, float))):
        raise ValueError("M contains non-finite scores")
    families = tuple(str(f) for f in family_names)
    if not families or len(set(families)) != len(families):
        raise ValueError("family_names must be nonempty and unique")
    af = [str(f) for f in audit_families]
    if len(af) != len(audit):
        raise ValueError("audit_families length must equal the number of audit rows")
    counts = Counter(af)
    if set(counts) != set(families) or any(counts[f] <= 0 for f in families):
        raise ValueError(f"audit must contain every declared family exactly; got {dict(counts)}")
    weights = _weights(families, family_weights)
    horizon = _horizon_counts(families, horizon_per_family)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if not 0.0 <= tau <= tau_strict <= 1.0:
        raise ValueError("require 0 <= tau <= tau_strict <= 1")
    if p_min is not None and not 0.0 < float(p_min) <= 1.0:
        raise ValueError("p_min must lie in (0, 1]")

    h_m = float(vinfo._h_bits(float(y.mean()))) if np.unique(y).size > 1 else 0.0
    pool_recovery = np.asarray([i_binary(y, col) for col in pool], float)
    best_idx = int(np.argmax(pool_recovery))
    r_pool = float(pool_recovery[best_idx])
    residual_cap = float(max(0.0, h_m - r_pool))

    species = leader_species(pool, list(range(len(pool))), agree_tau=tau)
    leaders = np.asarray(sorted(species), int)
    leader_cols = pool[leaders]
    exact_pool = {row.tobytes() for row in pool}

    flags_base: dict[str, list[bool]] = {f: [] for f in families}
    flags_strict: dict[str, list[bool]] = {f: [] for f in families}
    flags_exact: dict[str, list[bool]] = {f: [] for f in families}
    gain_marks: dict[str, list[float]] = {f: [] for f in families}
    audit_recovery = np.asarray([i_binary(y, col) for col in audit], float)
    for col, fam, rec in zip(audit, af, audit_recovery):
        agree = _agree_max(col, leader_cols)
        flags_base[fam].append(agree < tau)
        flags_strict[fam].append(agree < tau_strict)
        flags_exact[fam].append(col.tobytes() not in exact_pool)
        gain_marks[fam].append(float(np.clip(rec - r_pool, 0.0, residual_cap)))

    # Three primary upper claims share alpha equally.  The gain claim is split
    # between a mean/sum bound and a uniform-CDF expected-maximum bound so taking
    # their minimum remains valid.  Bonferroni within family makes every component
    # simultaneous.
    claim_alpha = alpha / 3.0
    mass_component_alpha = claim_alpha / len(families)
    gain_component_alpha = claim_alpha / (2.0 * len(families))

    def mass_upper(flags: Mapping[str, Sequence[bool]], *, component_delta: float) -> tuple[float, dict[str, float]]:
        per = {f: clopper_pearson_upper(int(sum(flags[f])), len(flags[f]), component_delta)
               for f in families}
        return float(sum(weights[f] * per[f] for f in families)), per

    u_base, u_base_f = mass_upper(flags_base, component_delta=mass_component_alpha)
    u_exact, u_exact_f = mass_upper(flags_exact, component_delta=mass_component_alpha)
    gain_u_f = {
        f: empirical_bernstein_upper(
            np.asarray(gain_marks[f], float), residual_cap, gain_component_alpha)
        for f in families
    }
    one_draw_gain_u = float(sum(weights[f] * gain_u_f[f] for f in families))
    mean_sum_horizon_u = float(min(
        residual_cap, sum(horizon[f] * gain_u_f[f] for f in families)))
    dkw_horizon_u, dkw_eps_f = dkw_expected_max_upper(
        {f: np.asarray(gain_marks[f], float) for f in families},
        horizon,
        residual_cap,
        gain_component_alpha,
    )
    horizon_gain_u = float(min(mean_sum_horizon_u, dkw_horizon_u))
    horizon_ceiling_u = float(min(h_m, r_pool + horizon_gain_u))

    # Marginal diagnostics are intentionally outside the simultaneous primary bundle.
    marginal_component_alpha = alpha / len(families)
    u_strict, u_strict_f = mass_upper(flags_strict, component_delta=marginal_component_alpha)

    def mass_lower(flags: Mapping[str, Sequence[bool]]) -> tuple[float, dict[str, float]]:
        per = {f: clopper_pearson_lower(int(sum(flags[f])), len(flags[f]), marginal_component_alpha)
               for f in families}
        return float(sum(weights[f] * per[f] for f in families)), per

    l_base, l_base_f = mass_lower(flags_base)
    l_exact, l_exact_f = mass_lower(flags_exact)
    l_strict, l_strict_f = mass_lower(flags_strict)

    support = None
    if p_min is not None:
        pm = float(p_min)
        exhausted = bool(u_exact < pm)
        support = {
            "assumption": f"every exact behavior pattern in the proposer-mixture support has mass >= {pm}",
            "p_min": pm,
            "max_patterns_in_pool_union_support": int(
                len(exact_pool) + np.floor(u_exact / pm)),
            "support_exhausted": exhausted,
            "pool_union_support_prompt_ceiling_UCB_bits": r_pool if exhausted else h_m,
            "interpretation": (
                "all proposer-support behavior patterns are represented in the pool; the union-class ceiling equals the pool optimum"
                if exhausted else
                "support exhaustion is not certified; the support-wide ceiling remains the DPI cap"
            ),
        }

    payload = {
        "schema": SCHEMA_VERSION,
        "estimand": {
            "name": "executor-indexed promptable articulation",
            "definition": "A_E(M; P) = sup_{p in P} I(M; binarize(E(p, X)))",
            "recovery_unit": "Shannon bits on the frozen empirical probe distribution",
            "prompt_class": "single prompts in the frozen pool union the declared proposer-process support",
        },
        "certified": {
            "simultaneous_confidence": 1.0 - alpha,
            "DPI_cap_bits": h_m,
            "pool_best_prompt_recovery_bits": r_pool,
            "pool_best_prompt_index": best_idx,
            "behavioral_missing_mass_U0": u_base,
            "exact_pattern_missing_mass_U0": u_exact,
            "one_draw_expected_gain_UCB_bits": one_draw_gain_u,
            "future_draws_per_family": horizon,
            "finite_horizon_expected_best_gain_UCB_bits": horizon_gain_u,
            "finite_horizon_expected_prompt_ceiling_UCB_bits": horizon_ceiling_u,
            "finite_horizon_gain_components_bits": {
                "mean_sum_UCB": mean_sum_horizon_u,
                "DKW_expected_max_UCB": dkw_horizon_u,
            },
            "expected_novel_draws_at_horizon_UCB": sum(
                horizon[f] * u_base_f[f] for f in families),
            "per_family": {
                f: {
                    "n_audit": counts[f],
                    "n_behaviorally_novel": int(sum(flags_base[f])),
                    "behavioral_mass_U0": u_base_f[f],
                    "n_exact_novel": int(sum(flags_exact[f])),
                    "exact_mass_U0": u_exact_f[f],
                    "mean_gain_mark_bits": float(np.mean(gain_marks[f])),
                    "mean_gain_UCB_bits": gain_u_f[f],
                    "DKW_epsilon": dkw_eps_f[f],
                }
                for f in families
            },
            "alpha_allocation": {
                "total_alpha": alpha,
                "primary_claims": 3,
                "alpha_per_claim": claim_alpha,
                "mass_alpha_per_family_component": mass_component_alpha,
                "gain_alpha_per_method_family_component": gain_component_alpha,
                "claims": [
                    "behavioral missing-mass upper bound",
                    "exact-pattern missing-mass upper bound",
                    "gain bounds and their finite-horizon expected-maximum consequence",
                ],
            },
        },
        "assumption_dependent": {"exact_support": support},
        "marginal_diagnostics": {
            "confidence_each": 1.0 - alpha,
            "behavioral_missing_mass_L0": l_base,
            "exact_pattern_missing_mass_L0": l_exact,
            "strict_behavioral_missing_mass": {"L0": l_strict, "U0": u_strict},
            "per_family_lower": {
                f: {
                    "behavioral_L0": l_base_f[f],
                    "exact_L0": l_exact_f[f],
                    "strict_L0": l_strict_f[f],
                    "strict_U0": u_strict_f[f],
                }
                for f in families
            },
            "n_discovery_leaders": len(leaders),
            "n_discovery_exact_patterns": len(exact_pool),
            "strict_rule": f"same frozen leaders; acceptance threshold {tau_strict}",
        },
        "scope": {
            **dict(scope or {}),
            "families": list(families),
            "family_weights": weights,
            "tau": tau,
            "tau_strict": tau_strict,
            "n_probes": int(len(y)),
            "n_pool_prompts": int(len(pool)),
            "conditional_on": [
                "audit prompts are independent conditional on proposer family",
                "the proposer configurations, family weights, executor, target metric, and probe panel were frozen before audit",
                "each prompt has one content-addressed executor signature on the frozen panel",
                "the gain ceiling concerns a best single prompt, not an unrestricted multi-prompt combiner",
                "deployment-distribution generalization is not claimed without an independent probe lockbox",
            ],
        },
    }
    if debug_internals:
        payload["_internals"] = {
            "pool_recovery": pool_recovery,
            "audit_recovery": audit_recovery,
            "gain_marks": {f: np.asarray(gain_marks[f], float) for f in families},
            "flags_base": flags_base,
            "flags_strict": flags_strict,
            "flags_exact": flags_exact,
            "leader_cols": leader_cols,
        }
    return payload


def cr3_certificate(
    sigs: np.ndarray,
    M: np.ndarray,
    tags: Sequence[str],
    *,
    family_prefixes: Sequence[str] = FAMILY_PREFIX_DEFAULT,
    tau: float = 0.90,
    tau_strict: float = 0.95,
    audit_frac: float = 1.0 / 3.0,
    horizon_mult: float = 10.0,
    alpha: float = 0.05,
    split_seed: int = 0,
    p_min: float | None = None,
    debug_internals: bool = False,
) -> dict:
    """Legacy-stream wrapper around :func:`prompt_articulation_certificate`.

    The resulting bounds remain conditional on iid-within-family provenance, which
    old checkpoint files generally do not establish.
    """
    b = _binary_matrix(sigs, name="sigs")
    if len(tags) != len(b):
        raise ValueError("tags length must equal the number of signature rows")
    sp = stratified_split(tags, family_prefixes=family_prefixes, audit_frac=audit_frac,
                          seed=split_seed)
    disc, audit = sp["discovery"], sp["audit"]
    total_future = max(0, int(round(float(horizon_mult) * len(b))))
    base, rem = divmod(total_future, len(family_prefixes))
    horizon = {f: base + int(i < rem) for i, f in enumerate(family_prefixes)}
    payload = prompt_articulation_certificate(
        b[disc], b[audit], M,
        [sp["family_of"][int(i)] for i in audit],
        family_names=family_prefixes,
        horizon_per_family=horizon,
        tau=tau,
        tau_strict=tau_strict,
        alpha=alpha,
        p_min=p_min,
        scope={
            "source": "retrospective legacy stream split",
            "iid_provenance_established": False,
            "warning": "legacy artifacts do not establish independent per-draw generation; treat as conditional sensitivity only",
            "split_seed": split_seed,
            "audit_fraction": audit_frac,
        },
        debug_internals=debug_internals,
    )
    if debug_internals:
        payload["_internals"].update({"discovery_indices": disc, "audit_indices": audit})
    return payload
