"""Generalized M-fluctuation parameter-instability test (Zeileis & Hornik 2007).

This is the node-level test at the heart of model-based recursive partitioning (MOB /
``partykit::glmtree``). At a node we fit a logistic GLM ``y ~ X`` (metric levels) and ask,
for each partitioning covariate ``z``, whether the fitted coefficients are *unstable* when
observations are ordered/grouped by ``z``. Instability => the metric->label relationship
differs across the subpopulation defined by ``z`` => split there.

Faithful-to-R statistics, permutation p-values
----------------------------------------------
The test statistic is built from the per-observation **estimating functions** (scores)

    psi_i = (y_i - p_hat_i) * x_i          (logistic regression; x_i includes the intercept)

decorrelated by an estimate ``J`` of their covariance. The empirical fluctuation process is

    efp(t) = J^{-1/2} n^{-1/2} sum_{i<=floor(n t)} psi_(i)         (psi ordered by z)

aggregated into a scalar:

- numeric ``z``  -> **sup-LM** (Andrews):  max_t ||efp(t)||^2 / (t (1-t))
- categorical z  -> **chi-squared**:        sum_c (1/n_c) dS_c^T J^{-1} dS_c

These match ``strucchange``'s ``supLM`` / ``catL2BB`` functionals (up to a monotone scaling
that does not affect a permutation p-value). Rather than port strucchange's asymptotic
Brownian-bridge null, we compute the p-value by **permutation**: under H0 the scores are
exchangeable w.r.t. ``z``. For numeric ``z`` the sup-LM null depends only on ``psi`` and
``n`` (not on the values of ``z``), so a single permutation null is shared across all
numeric variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------------------
# Node GLM + score contributions
# --------------------------------------------------------------------------------------

def fit_node_glm(
    X: np.ndarray,
    y: np.ndarray,
    ridge_C: float = 1e6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a (near-unpenalized) logistic GLM ``y ~ X`` at a node.

    Returns ``(beta, p_hat, X_design)`` where ``X_design`` has a leading intercept column.
    A very large ``C`` keeps the fit close to the unpenalized MLE so that the score
    residuals behave like the strucchange estimating functions (which sum to ~0).
    """
    from sklearn.linear_model import LogisticRegression

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = X.shape[0]
    X_design = np.column_stack([np.ones(n), X])

    # Degenerate label distribution: no within-node model is identifiable.
    if len(np.unique(y)) < 2:
        p = np.full(n, float(y.mean()) if n else 0.5)
        beta = np.zeros(X_design.shape[1])
        beta[0] = np.log((p[0] + 1e-9) / (1 - p[0] + 1e-9)) if n else 0.0
        return beta, p, X_design

    clf = LogisticRegression(
        penalty="l2", C=ridge_C, solver="lbfgs", max_iter=2000, fit_intercept=True,
    )
    clf.fit(X, y)
    beta = np.concatenate([clf.intercept_, clf.coef_.ravel()])
    p = clf.predict_proba(X)[:, 1]
    return beta, p, X_design


def score_contributions(X_design: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-observation estimating functions psi_i = (y_i - p_i) * x_i, shape (n, k).

    Columns are centered so they sum to ~0 (the MLE property that makes the cumulative
    process a bridge); harmless when the fit is already at the MLE.
    """
    psi = (y - p)[:, None] * X_design
    psi = psi - psi.mean(axis=0, keepdims=True)
    return psi


def cov_inverse(psi: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """Inverse of the outer-product covariance estimate J = (1/n) sum psi psi^T.

    Uses a pseudo-inverse with a small ridge for numerical stability when metrics are
    collinear. ``J`` is invariant to the ordering of ``psi``, so this is computed once per
    node and reused across every candidate ``z`` and every permutation.
    """
    n, k = psi.shape
    J = (psi.T @ psi) / max(n, 1)
    J = J + ridge * np.eye(k)
    return np.linalg.pinv(J)


# --------------------------------------------------------------------------------------
# Aggregation functionals
# --------------------------------------------------------------------------------------

def _suplm_statistic(
    psi_ordered: np.ndarray, Jinv: np.ndarray, trim: float
) -> Tuple[float, Optional[int]]:
    """sup-LM statistic over the trimming window; returns (stat, argmax index j in 1..n-1).

    ``j`` is the ordered position after which instability peaks (a split-location hint;
    glmtree does its own exhaustive cutpoint search).
    """
    n = psi_ordered.shape[0]
    cum = np.cumsum(psi_ordered, axis=0)          # cum[j-1] = sum of first j scores
    Q = np.einsum("ij,ij->i", cum @ Jinv, cum)    # quadratic forms, length n
    j = np.arange(1, n + 1)
    lo = max(1, int(np.floor(n * trim)))
    hi = min(n - 1, int(np.ceil(n * (1.0 - trim))))
    if hi < lo:
        return 0.0, None
    idx = np.arange(lo, hi + 1)                    # j values (1-based) in the window
    denom = j[idx - 1] * (n - j[idx - 1])
    weighted = (n * Q[idx - 1]) / denom            # n/(j(n-j)) * ||cum_j||^2_{Jinv}
    a = int(np.argmax(weighted))
    return float(weighted[a]), int(idx[a])


def _cat_statistic(psi: np.ndarray, Jinv: np.ndarray, codes: np.ndarray, n_levels: int) -> float:
    """Categorical chi-squared functional: sum_c (1/n_c) dS_c^T Jinv dS_c."""
    stat = 0.0
    for c in range(n_levels):
        mask = codes == c
        nc = int(mask.sum())
        if nc == 0:
            continue
        dS = psi[mask].sum(axis=0)
        stat += float(dS @ Jinv @ dS) / nc
    return stat


# --------------------------------------------------------------------------------------
# Permutation nulls
# --------------------------------------------------------------------------------------

def _numeric_null(
    psi: np.ndarray, Jinv: np.ndarray, trim: float, n_perm: int, rng: np.random.Generator
) -> np.ndarray:
    """Shared sup-LM null: statistic of ``psi`` under random orderings.

    Valid for *every* numeric ``z`` because sup-LM depends only on the ordering of scores,
    not on the magnitudes of ``z``.
    """
    n = psi.shape[0]
    out = np.empty(n_perm)
    for b in range(n_perm):
        perm = rng.permutation(n)
        stat, _ = _suplm_statistic(psi[perm], Jinv, trim)
        out[b] = stat
    return out


def _cat_null(
    psi: np.ndarray, Jinv: np.ndarray, codes: np.ndarray, n_levels: int,
    n_perm: int, rng: np.random.Generator,
) -> np.ndarray:
    """chi-squared null for one categorical ``z``: permute item->category assignment."""
    n = psi.shape[0]
    out = np.empty(n_perm)
    for b in range(n_perm):
        perm = rng.permutation(n)
        out[b] = _cat_statistic(psi, Jinv, codes[perm], n_levels)
    return out


def _perm_pvalue(observed: float, null: np.ndarray) -> float:
    """Empirical p-value with the +1 correction: (1 + #{null >= obs}) / (1 + B)."""
    B = len(null)
    return float((1 + int(np.sum(null >= observed))) / (1 + B))


def _null_z(observed: float, null: np.ndarray) -> float:
    """Standardized distance of the observed statistic from its own permutation null.

    Comparable across numeric (sup-LM) and categorical (chi-squared) covariates, whose raw
    statistics live on different scales. Used to rank variables whose permutation p-values are
    tied at the resolution floor 1/(B+1)."""
    sd = float(np.std(null))
    if sd < 1e-12:
        return 0.0
    return float((observed - float(np.mean(null))) / sd)


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------

@dataclass
class FluctResult:
    """Outcome of the instability test for a single partitioning covariate ``z``."""

    variable: str
    kind: str                     # "numeric" | "categorical"
    statistic: float
    pvalue: float                 # permutation p-value (pre-Bonferroni)
    adj_pvalue: float             # Bonferroni-adjusted across the tested covariates
    n: int
    null_z: float = 0.0           # (stat - null mean) / null sd -- comparable across z kinds
    split_index: Optional[int] = None   # numeric: ordered position hint for the split
    split_value: Optional[float] = None # numeric: covariate value at the instability peak


def test_node(
    psi: np.ndarray,
    z_frame: Dict[str, Tuple[np.ndarray, str]],
    *,
    trim: float = 0.1,
    n_perm: int = 999,
    bonferroni: bool = True,
    rng: Optional[np.random.Generator] = None,
    Jinv: Optional[np.ndarray] = None,
) -> List[FluctResult]:
    """Run the M-fluctuation test for every covariate in ``z_frame``.

    Parameters
    ----------
    psi
        Score-contribution matrix (n, k) from :func:`score_contributions`.
    z_frame
        Mapping ``name -> (values, kind)`` where ``kind`` is ``"numeric"`` or
        ``"categorical"``. Numeric values are floats; categorical values are integer codes
        (or any array coercible to category codes).
    trim, n_perm, bonferroni
        Test hyperparameters (see :class:`~..config.InfillConfig`).
    rng
        Seeded ``numpy`` Generator (reproducibility).
    Jinv
        Optional precomputed covariance inverse; otherwise derived from ``psi``.

    Returns a list of :class:`FluctResult`, one per covariate, sorted by adjusted p-value.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = psi.shape[0]
    if Jinv is None:
        Jinv = cov_inverse(psi)

    m = len(z_frame)
    numeric_null: Optional[np.ndarray] = None  # computed lazily, shared across numeric z
    results: List[FluctResult] = []

    for name, (values, kind) in z_frame.items():
        values = np.asarray(values)
        if kind == "numeric":
            order = np.argsort(values, kind="mergesort")
            stat, j = _suplm_statistic(psi[order], Jinv, trim)
            if numeric_null is None:
                numeric_null = _numeric_null(psi, Jinv, trim, n_perm, rng)
            p = _perm_pvalue(stat, numeric_null)
            z = _null_z(stat, numeric_null)
            split_value = None
            if j is not None and 1 <= j < n:
                sorted_vals = values[order]
                # threshold halfway between the two ordered values straddling the break
                split_value = float((sorted_vals[j - 1] + sorted_vals[j]) / 2.0)
            results.append(
                FluctResult(name, "numeric", stat, p, p, n, null_z=z,
                            split_index=j, split_value=split_value)
            )
        else:
            codes, n_levels = _as_codes(values)
            stat = _cat_statistic(psi, Jinv, codes, n_levels)
            null = _cat_null(psi, Jinv, codes, n_levels, n_perm, rng)
            p = _perm_pvalue(stat, null)
            z = _null_z(stat, null)
            results.append(FluctResult(name, "categorical", stat, p, p, n, null_z=z))

    if bonferroni and m > 1:
        for r in results:
            r.adj_pvalue = min(1.0, r.pvalue * m)

    # rank by significance, breaking p-floor ties by the null-standardized score (which IS
    # comparable across numeric/categorical, unlike the raw statistic)
    results.sort(key=lambda r: (r.adj_pvalue, -r.null_z))
    return results


test_node.__test__ = False  # public API named test_*; not a pytest test


def _as_codes(values: np.ndarray) -> Tuple[np.ndarray, int]:
    """Coerce a categorical covariate to contiguous integer codes 0..C-1."""
    uniq, codes = np.unique(values, return_inverse=True)
    return codes.astype(int), len(uniq)
