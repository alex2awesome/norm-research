"""V-information of metric recovery: I_V(x -> m_recovered)  (2026-06-16).

The unsupervised articulability objective (no ground-truth label Y). For a metric m recovered
through an articulation->re-execution channel, we measure the V-usable information the datapoint
carries about the *recovered verdict*:

    I_V(x -> m_recovered) = H(verdict) - H(verdict | item)
                          = H(p_bar) - (1/N) sum_i H(p_i)

where p_i is the recovered-verdict distribution for item i (estimated from K independent recovery
instances) and p_bar is the item-uniform mean. High only when the recovered metric *uses its range
across items* (H(p_bar) high) AND is *stable within an item across recovery instances* (H(p_i)
low) -- so range-compression and within-item recovery noise both drive it down automatically.
H(p_bar) is the within-family no-information baseline, so I_V already nets out the marginal; the
separate no-rubric vacuous baseline is only needed for the different quantity dI(rubric) = how much
the rubric adds over no rubric.

Units: bits (binary verdicts) in [0, H(p_bar)] <= 1. `iv_norm` = I_V / H(p_bar) is the fraction of
the used range that is reliably recovered.

Same-f pairing (prompt-optimality theory, notes/2026-06-18): the Shannon estimators above have TVD
twins -- `tvd_transmission` (I_TVD(I;V), max 1/2 = cap_TVD) and `tvd_recovery` (I_TVD(m;m̂)) -- built
from the SAME per-item means, so the data-processing guardrail R_TVD <= T_TVD holds termwise (Jensen)
and A_TVD = T_TVD - R_TVD is an honest same-f gap. Use `tvd_guardrail` for the cell-level T/R/A bundle.
DPI is SAME-DISTRIBUTION (2026-06-22): the bounding T is the HELD-OUT T_test = I_TVD(M̂_test; X_test)
(`tvd_transmission` on the held-out `recovered`), NOT the in-sample consistency T_train of
`channel_consistency` -- those live on different splits and DPI does not chain across them. `cap_TVD`
itself is a channel-capacity SANITY CHECK (`cap_sanity_check`), not a proximity-to-optimum KPI (§3.1).
TVD is the bounded f-divergence (Robertson-Koyejo): gaming-robust, the right f when optimizing prompts
AGAINST the measure. Shannon `iv_*` and TVD `tvd_*` must never be mixed across the R<=T inequality.

Plug-in MI over K=5 passes is upward-biased; we report Miller-Madow-debiased `iv_mm` and a bootstrap
CI (resample items, optionally passes). The synthetic channel plants metrics with a *known* analytic
I_V (`analytic_iv`) so the estimator can be calibrated (E0 kill-switch) before any real number is
trusted.

Recovery channels (the part to experiment with):
  * consistency    -- a fixed articulated rubric re-applied; K temperature passes = recovery noise.
                      `channel_consistency` reads the existing sampled long table. Zero GPU.
  * reconstruction -- reconstruct the rule from m's behavior, fresh executor re-applies to held-out
                      x. `iv_from_reconstruction` consumes the re-scored verdicts; the LLM run is the
                      caller's job (reuses measures.reconstruct + a small executor; tiny 1-GPU).
  * synthetic      -- planted-thickness metrics with known I_V; `channel_synthetic`. Zero GPU.
  * mixtures       -- combine the above by hand in the notebook.

numpy-only core; pandas imported lazily only in the long-table adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_EPS = 1e-12


# --------------------------------------------------------------------------------------------
# entropy + plug-in / Miller-Madow MI for binary verdicts
# --------------------------------------------------------------------------------------------

def _h_bits(p: np.ndarray | float) -> np.ndarray | float:
    """Binary entropy in bits; 0 at p in {0,1}. Vectorized."""
    p = np.asarray(p, dtype=float)
    out = np.zeros_like(p)
    m = (p > _EPS) & (p < 1.0 - _EPS)
    pm = p[m] if p.ndim else p
    if p.ndim == 0:
        if not m:
            return 0.0
        return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))
    out[m] = -(pm * np.log2(pm) + (1 - pm) * np.log2(1 - pm))
    return out


def _per_item(V: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """From an (N_items, K_passes) binary matrix (NaN = inapplicable pass) return
    (p_i, k_i, obs_bins_i): per-item YES-frequency, applicable-pass count, #distinct outcomes."""
    V = np.asarray(V, dtype=float)
    if V.ndim == 1:                              # (N,) of YES-fractions, no pass structure
        return V.astype(float), np.full(V.shape, np.nan), np.full(V.shape, 2)
    k = np.sum(np.isfinite(V), axis=1)
    with np.errstate(invalid="ignore"):
        p = np.nansum(V, axis=1) / np.maximum(k, 1)
    # distinct observed outcomes per item (1 if all passes agree, 2 if both 0 and 1 seen)
    any1 = np.nansum(V == 1.0, axis=1) > 0
    any0 = np.nansum(V == 0.0, axis=1) > 0
    obs = (any1.astype(int) + any0.astype(int)).clip(min=1)
    return p, k, obs


# --------------------------------------------------------------------------------------------
# Fixed-target prompt ceiling. These are the quantities used by the M_omega DPI theorem.
# --------------------------------------------------------------------------------------------

def _binary_soft_joint(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Joint law of conditionally independent Bernoulli channels over a shared item draw.

    ``left[i]`` and ``right[i]`` are P(YES | item i) for the fixed target and candidate prompt.
    The returned 2x2 table is ordered (NO, YES) on both axes.
    """
    q = np.asarray(left, dtype=float)
    p = np.asarray(right, dtype=float)
    if q.shape != p.shape or q.ndim != 1:
        raise ValueError("left and right must be aligned one-dimensional probability vectors")
    if q.size == 0 or not (np.isfinite(q).all() and np.isfinite(p).all()):
        raise ValueError("left and right must be finite and non-empty")
    if not ((0.0 <= q).all() and (q <= 1.0).all() and
            (0.0 <= p).all() and (p <= 1.0).all()):
        raise ValueError("channel probabilities must lie in [0, 1]")
    return np.array([
        [np.mean((1.0 - q) * (1.0 - p)), np.mean((1.0 - q) * p)],
        [np.mean(q * (1.0 - p)), np.mean(q * p)],
    ], dtype=float)


def binary_soft_channel_mi(target: Sequence[float], candidate: Sequence[float]) -> dict:
    """Same-distribution Shannon and TVD MI for a fixed binary target and candidate channel.

    The Markov model is ``M_omega <- X -> M_candidate``: conditional on the item, target and
    candidate verdict draws are independent Bernoulli variables with probabilities ``target`` and
    ``candidate``. This is the actual fixed-target recovery object; it does not binarize a soft
    target or mix divergences.
    """
    q = np.asarray(target, dtype=float)
    p = np.asarray(candidate, dtype=float)
    joint = _binary_soft_joint(q, p)
    row = joint.sum(axis=1)
    col = joint.sum(axis=0)
    prod = np.outer(row, col)
    nz = joint > 0
    shannon = float(np.sum(joint[nz] * np.log2(joint[nz] / prod[nz])))
    tvd = float(0.5 * np.abs(joint - prod).sum())
    # For binary conditionally-independent channels TVD MI also equals twice the covariance.
    tvd_cov = float(2.0 * abs(np.mean(q * p) - np.mean(q) * np.mean(p)))
    return {"shannon": max(shannon, 0.0), "tvd": tvd,
            "tvd_cov": tvd_cov, "joint": joint.tolist(), "n_items": int(q.size)}


def target_channel_ceiling(target: Sequence[float], *, delta: Optional[float] = None) -> dict:
    """All-prompt DPI ceiling for a fixed binary target channel.

    For an item-uniform frozen probe set the empirical quantities are exact:

      Shannon: ``I(M_omega; X) = H(mean q_i) - mean H(q_i)``
      TVD:     ``I_TVD(M_omega; X) = mean |q_i - mean q_i|``.

    If ``delta`` is supplied, the input must be a one-dimensional vector of exact/frozen channel
    probabilities on iid items. The returned one-sided population upper bounds hold simultaneously
    with probability at least ``1-delta`` by Hoeffding plus a union bound. They do not cover channel
    probabilities estimated from finitely many stochastic passes.
    """
    raw = np.asarray(target, dtype=float)
    estimated_from_passes = raw.ndim == 2
    q, _, _ = _per_item(raw)
    q = q[np.isfinite(q)]
    if q.size < 2 or not ((0.0 <= q).all() and (q <= 1.0).all()):
        raise ValueError("target must contain at least two finite probabilities in [0, 1]")
    qbar = float(np.mean(q))
    h_cond = float(np.mean(_h_bits(q)))
    shannon = float(max(0.0, _h_bits(qbar) - h_cond))
    tvd = float(np.mean(np.abs(q - qbar)))
    out = {
        "empirical": {"shannon": shannon, "tvd": tvd},
        "n_items": int(q.size),
        "scope": "frozen empirical probe distribution",
        "probabilities_estimated_from_passes": estimated_from_passes,
        "population": None,
    }
    if delta is None:
        return out
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    if estimated_from_passes:
        out["population_error"] = "finite-pass probability uncertainty is not covered"
        return out

    n = q.size
    # Three simultaneous events: target mean, mean conditional entropy, and target MAD about
    # the true mean. The mean event is two-sided; the other two are one-sided.
    r_mean = float(np.sqrt(np.log(6.0 / delta) / (2.0 * n)))
    r_one = float(np.sqrt(np.log(3.0 / delta) / (2.0 * n)))
    mu_lo, mu_hi = max(0.0, qbar - r_mean), min(1.0, qbar + r_mean)
    if mu_lo <= 0.5 <= mu_hi:
        h_marg_hi = 1.0
    else:
        h_marg_hi = float(max(_h_bits(mu_lo), _h_bits(mu_hi)))
    shannon_hi = float(min(1.0, max(0.0, h_marg_hi - max(0.0, h_cond - r_one))))
    tvd_hi = float(min(0.5, tvd + r_mean + r_one))
    out["population"] = {
        "confidence": float(1.0 - delta), "delta": float(delta),
        "shannon_upper": shannon_hi, "tvd_upper": tvd_hi,
        "assumptions": ["iid items", "fixed target channel", "exact per-item probabilities"],
    }
    return out


def fixed_target_channel_certificate(target: np.ndarray, candidate: np.ndarray, *,
                                     population_delta: Optional[float] = None,
                                     candidate_frozen: bool = False) -> dict:
    """Certificate bundle for ``M_omega <- X -> M_candidate`` on one held-out distribution.

    ``target`` and ``candidate`` may be per-item probabilities or item x pass matrices. The empirical
    DPI inequalities are algebraic for the induced channels. A population TVD ceiling/gap is issued
    only when both channels are one-dimensional exact probabilities and the candidate was frozen
    before the iid evaluation items were observed.
    """
    target_raw = np.asarray(target, dtype=float)
    candidate_raw = np.asarray(candidate, dtype=float)
    q, _, _ = _per_item(target_raw)
    p, _, _ = _per_item(candidate_raw)
    keep = np.isfinite(q) & np.isfinite(p)
    q, p = q[keep], p[keep]
    if q.size < 4:
        return {"valid": False, "error": "too_few_aligned_items", "n_items": int(q.size)}
    rec = binary_soft_channel_mi(q, p)
    target_cap = target_channel_ceiling(q)
    candidate_cap = target_channel_ceiling(p)
    ts, tt = target_cap["empirical"]["shannon"], target_cap["empirical"]["tvd"]
    cs, ct = candidate_cap["empirical"]["shannon"], candidate_cap["empirical"]["tvd"]
    shannon_u, tvd_u = min(ts, cs), min(tt, ct)
    out = {
        "valid": True, "scope": "frozen empirical held-out distribution",
        "markov_assumption": "target and candidate verdict draws are independent conditional on item",
        "n_items": int(q.size),
        "shannon": {"R": rec["shannon"], "T_target": ts, "T_candidate": cs,
                    "dpi_upper": shannon_u, "dpi_ok": bool(rec["shannon"] <= shannon_u + 1e-10),
                    "target_headroom_upper": float(max(0.0, ts - rec["shannon"]))},
        "tvd": {"R": rec["tvd"], "T_target": tt, "T_candidate": ct,
                "dpi_upper": tvd_u, "dpi_ok": bool(rec["tvd"] <= tvd_u + 1e-10),
                "target_headroom_upper": float(max(0.0, tt - rec["tvd"]))},
        "tightness_established": False,
        "tightness_note": "DPI is an upper bound; equality additionally requires a realizable sufficient readout",
        "population": None,
    }
    if population_delta is None:
        return out
    if not candidate_frozen:
        out["population_error"] = "candidate was not declared frozen before evaluation"
        return out
    if target_raw.ndim != 1 or candidate_raw.ndim != 1:
        out["population_error"] = "finite-pass probability uncertainty is not covered"
        return out

    # Spend half the error budget on the target ceiling and half on a covariance lower bound.
    cap = target_channel_ceiling(q, delta=population_delta / 2.0)["population"]
    delta_r = population_delta / 2.0
    rad = float(np.sqrt(np.log(6.0 / delta_r) / (2.0 * q.size)))
    cov_hat = float(np.mean(q * p) - np.mean(q) * np.mean(p))
    r_tvd_lo = float(2.0 * max(0.0, abs(cov_hat) - 3.0 * rad))
    out["population"] = {
        "confidence": float(1.0 - population_delta), "delta": float(population_delta),
        "tvd_target_upper": cap["tvd_upper"], "tvd_recovery_lower": r_tvd_lo,
        "tvd_gap_upper": float(max(0.0, cap["tvd_upper"] - r_tvd_lo)),
        "assumptions": ["iid evaluation items", "fixed target and candidate channels",
                        "exact per-item probabilities", "candidate frozen before evaluation"],
    }
    return out


# --------------------------------------------------------------------------------------------
# TVD-MI: total-variation mutual information between two verdict VIEWS over the same items.
# The gaming-robust estimator for the peer/perturbation channels (reliability = two passes;
# consistency = base vs nuisance-transformed; reconstruction = original vs induced rubric).
# TVD-MI = D_TV( P(a,b) || P(a)P(b) ) on a median-split contingency table = the information the
# two views SHARE about the item. It is 0 for INDEPENDENT views AND for COLLAPSED/CONSTANT views
# (a judge that scores everything the same gets 0, not a spurious "perfect agreement") -- which
# is exactly the anti-gaming property that rank/agreement estimators lack. Robertson & Koyejo,
# "Let's Measure Information Step-by-Step" (TMLR 2026); here the low-dim plug-in form, no critic.
# --------------------------------------------------------------------------------------------

def _binize(x: np.ndarray, n_bins: int, rng=None) -> np.ndarray:
    """Bin a [0,1] array into balanced quantile bins so TVD-MI measures dependence, not marginal
    skew. A constant view -> a single bin (-> TVD-MI 0). For n_bins=2 a *rank* median split is
    used (lower half rank -> 0, upper half -> 1) so heavy ties at the median don't collapse the
    split, which `x > median` would (a 2-level metric whose median equals its majority value).

    Ties MUST be broken at random, independently per vector (``rng``): breaking them by stable
    sort order makes bin membership a function of item POSITION for heavily-tied vectors, and
    since both sides of a tvd_mi call share the same item order, two INDEPENDENT 90%-tied
    vectors read as ~0.7-0.8 "dependence" (the permutation floor can't see it — the permutation
    destroys exactly the order coupling it should be calibrating). Found 2026-07-04 via the
    seam bridge-calibration slice; equivalent to independent infinitesimal jitter."""
    x = np.asarray(x, dtype=float)
    if x.max() - x.min() < _EPS:                     # constant view -> single bin
        return np.zeros(len(x), dtype=int)
    if n_bins <= 2:
        out = np.zeros(len(x), dtype=int)
        if rng is None:                              # deterministic fallback (untied data only)
            order = np.argsort(x, kind="stable")
        else:                                        # random tie-break WITHIN equal values
            order = np.lexsort((rng.random(len(x)), x))
        out[order[len(x) // 2:]] = 1
        return out
    qs = np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    return np.digitize(x, qs)


def _tvd_mi_raw(a_bin: np.ndarray, b_bin: np.ndarray, n_bins: int) -> float:
    """D_TV(joint || product-of-marginals) = 1/2 * sum |P(a,b) - P(a)P(b)| over the table."""
    joint = np.zeros((n_bins, n_bins))
    for ai, bi in zip(a_bin, b_bin):
        joint[ai, bi] += 1.0
    joint /= len(a_bin)
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    return 0.5 * float(np.abs(joint - pa * pb).sum())


def tvd_mi(a: Sequence[float], b: Sequence[float], *, n_bins: int = 2, debias: bool = True,
           n_perm: int = 32, seed: int = 0, normalize: bool = True) -> float:
    """TVD mutual information between two aligned verdict views over items, normalized to [0,1].

    ``a``, ``b``: per-item scores in [0,1] (NaN = inapplicable, dropped pairwise). High when the
    two views' high/low verdicts track each other across items; 0 for INDEPENDENT or for CONSTANT
    views. ``debias`` subtracts the finite-sample floor (mean over ``n_perm`` shuffles of ``b``),
    clamped at 0. ``normalize`` divides by the perfect-dependence maximum (n_bins-1)/n_bins.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 4:
        return float("nan")
    # independent tie-break streams per side — see _binize docstring (2026-07-04 fix)
    ab = _binize(a, n_bins, np.random.default_rng((seed, 17)))
    bb = _binize(b, n_bins, np.random.default_rng((seed, 29)))
    if len(np.unique(ab)) < 2 or len(np.unique(bb)) < 2:     # a view carries no variation -> 0
        return 0.0
    raw = _tvd_mi_raw(ab, bb, n_bins)
    if debias and n_perm > 0:
        rng = np.random.default_rng(seed)
        floor = float(np.mean([_tvd_mi_raw(ab, rng.permutation(bb), n_bins)
                               for _ in range(n_perm)]))
        raw = max(0.0, raw - floor)
    if normalize:
        scale = (n_bins - 1) / n_bins
        raw = raw / scale if scale > _EPS else raw
    return float(min(max(raw, 0.0), 1.0))


def tvd_mi_passes(V: np.ndarray, **kw) -> float:
    """Mean pairwise TVD-MI across the passes (columns) of an (N_items, K_passes) verdict matrix
    -- the reliability channel: information reproduced across independent recovery instances."""
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or V.shape[1] < 2:
        return float("nan")
    vals = [tvd_mi(V[:, i], V[:, j], **kw)
            for i in range(V.shape[1]) for j in range(i + 1, V.shape[1])]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------------------------
# TVD TRANSMISSION  I_TVD(I; V)  -- the same-f twin of the Shannon transmission `iv_from_passes`.
# The prompt-optimality theory (notes/2026-06-18) requires T̂ and R̂ in the SAME f-divergence: the
# DPI guardrail R = I(m;m̂) <= I(I;m̂) = T and the gap A = T - R are only meaningful when both legs
# use one f. The shipped `iv_from_passes` is Shannon; this is its TVD counterpart, so pairing it
# with `tvd_recovery` keeps A_TVD = T_TVD - R_TVD an honest same-f difference. Binary verdicts.
#
#   I_TVD(I;V) = (1/2N) sum_i ||p_i - p_bar||_1 = (1/N) sum_i |s_i - s_bar|   (binary)
#
# Max over a balanced deterministic split = 1 - 1/min(N,K) = 1/2 (binary) = cap_TVD (theory §3.1).
# --------------------------------------------------------------------------------------------

def _tvd_t_point(p_i: np.ndarray) -> Tuple[float, float, float]:
    """Binary I_TVD(I;V) from per-item YES-frequencies p_i: returns (raw, s_bar, gini_max), where
    gini_max = 2 s_bar(1-s_bar) is the marginal-constrained maximum (the TVD analog of H(p_bar))."""
    s_bar = float(np.mean(p_i))
    return float(np.mean(np.abs(p_i - s_bar))), s_bar, 2.0 * s_bar * (1.0 - s_bar)


def tvd_transmission(V: np.ndarray, *, debias: bool = True, n_perm: int = 32,
                     n_boot: int = 500, ci: float = 0.90, seed: int = 0) -> dict:
    """I_TVD(I; V) -- TVD transmission/consistency: the bounded-f-divergence twin of the Shannon
    `iv_from_passes`. ``V`` is an (N_items, K_passes) binary matrix (NaN = inapplicable pass) or a
    1-D vector of per-item YES-fractions. ``debias`` subtracts the finite-pass floor (mean statistic
    under verdict-independent-of-item, estimated by permuting the pooled per-pass verdicts across
    items, preserving per-item pass counts and the marginal), clamped at 0. ``tvd_t_norm`` divides
    by 2 s_bar(1-s_bar) = fraction of the marginal-constrained Gini max realized (TVD twin of
    ``iv_norm``)."""
    V = np.asarray(V, dtype=float)
    if V.ndim == 1:
        V = V[:, None]
    p, k, _ = _per_item(V)
    keep = np.isfinite(p) & (np.nan_to_num(k, nan=1) >= 1)
    n_items = int(keep.sum())
    if n_items < 4:
        return {"tvd_t": float("nan"), "tvd_t_norm": float("nan"), "tvd_t_raw": float("nan"),
                "s_bar": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n_items": n_items, "error": "too_few_items"}
    Vk, p, k = V[keep], p[keep], k[keep]
    raw, s_bar, gini = _tvd_t_point(p)

    floor = 0.0
    has_pass = bool(np.all(np.isfinite(k))) and Vk.shape[1] >= 2
    if debias and has_pass and n_perm > 0:
        rng = np.random.default_rng(seed)
        finite = np.isfinite(Vk)
        pool = Vk[finite]
        ks = finite.sum(axis=1).astype(int)
        fl = []
        for _ in range(n_perm):
            perm = rng.permutation(pool)
            s_null, idx = np.empty(n_items), 0
            for i, ki in enumerate(ks):
                s_null[i] = perm[idx:idx + ki].mean() if ki > 0 else s_bar
                idx += ki
            fl.append(float(np.mean(np.abs(s_null - float(np.mean(s_null))))))
        floor = float(np.mean(fl))
    # PRIMARY = raw plug-in: this is the DPI-coherent quantity (R_raw <= T_raw termwise by Jensen),
    # so the guardrail and A use it. The permutation floor is reported SEPARATELY as a significance
    # reference (tvd_t_sig = excess over verdict-independent-of-item); subtracting it from the point
    # estimate would strip real signal (a deterministic metric has zero finite-pass noise yet a large
    # floor), which is why it must NOT enter the DPI check.
    tvd_t = raw

    rng = np.random.default_rng(seed + 1)
    K = Vk.shape[1]
    vals = []
    for _ in range(n_boot):
        Vb = Vk[rng.integers(0, n_items, n_items)]
        if K > 1:
            Vb = Vb[:, rng.integers(0, K, K)]
        pb, _, _ = _per_item(Vb)
        pb = pb[np.isfinite(pb)]
        if pb.size >= 4:
            vals.append(_tvd_t_point(pb)[0])
    vals = np.array([v for v in vals if np.isfinite(v)])
    return {"tvd_t_raw": float(raw), "tvd_floor": float(floor), "tvd_t": float(tvd_t),
            "tvd_t_sig": float(max(0.0, raw - floor)), "s_bar": float(s_bar), "gini_max": float(gini),
            "tvd_t_norm": float(tvd_t / gini) if gini > _EPS else float("nan"),
            "ci_lo": float(np.quantile(vals, (1 - ci) / 2)) if vals.size else float("nan"),
            "ci_hi": float(np.quantile(vals, 1 - (1 - ci) / 2)) if vals.size else float("nan"),
            "n_items": n_items, "k_mean": float(np.nanmean(k)) if has_pass else float("nan")}


def iv_from_passes(V: np.ndarray) -> dict:
    """I_V(x -> m_recovered) from an (N_items, K_passes) binary recovery matrix.

    Returns plug-in and Miller-Madow-debiased I_V (bits), the entropy parts, and book-keeping.
    NaN entries (inapplicable passes) are dropped per item; items with 0 applicable passes drop.
    """
    p, k, obs = _per_item(V)
    keep = np.isfinite(p) & (np.nan_to_num(k, nan=1) >= 1)
    p, k, obs = p[keep], k[keep], obs[keep]
    n_items = int(keep.sum())
    if n_items < 2:
        return {"iv_plugin": float("nan"), "iv_mm": float("nan"), "n_items": n_items,
                "error": "too_few_items"}

    p_bar = float(np.mean(p))                      # item-uniform marginal
    h_marg = float(_h_bits(p_bar))
    h_cond = float(np.mean(_h_bits(p)))
    iv_plugin = h_marg - h_cond

    # Miller-Madow: H_MM = H_plugin + (#bins - 1)/(2 * n_samples)
    has_pass = np.isfinite(k).all() and np.all(k >= 1)
    if has_pass:
        m_total = float(np.sum(k))                 # pooled applicable verdicts
        marg_bins = 1 + int((p_bar > _EPS) and (p_bar < 1 - _EPS))  # 2 unless collapsed
        h_marg_mm = h_marg + (marg_bins - 1) / (2.0 * max(m_total, 1.0))
        h_cond_mm = float(np.mean(_h_bits(p) + (obs - 1) / (2.0 * np.maximum(k, 1.0))))
        iv_mm = h_marg_mm - h_cond_mm
    else:                                          # soft probs, no count structure -> no MM
        h_marg_mm, h_cond_mm, iv_mm = h_marg, h_cond, iv_plugin

    return {"iv_plugin": float(iv_plugin), "iv_mm": float(max(iv_mm, 0.0)),
            "iv_norm": float(max(iv_mm, 0.0) / h_marg) if h_marg > _EPS else float("nan"),
            "h_marg": h_marg, "h_cond": h_cond, "p_bar": p_bar, "n_items": n_items,
            "k_mean": float(np.nanmean(k)) if has_pass else float("nan")}


def iv_from_probs(p: Sequence[float]) -> dict:
    """Soft-readout I_V: p_i is a per-item recovered-verdict probability (e.g. logprob P(YES),
    one value per item). H(verdict|item) is taken as the model's own predictive entropy H_b(p_i).
    Semantics differ from the pass-based estimate (model confidence vs recovery stochasticity); use
    as the cross-readout check, not as a drop-in replacement. No MM (no count structure)."""
    p = np.asarray([x for x in p if x is not None and np.isfinite(x)], dtype=float)
    if p.size < 2:
        return {"iv_plugin": float("nan"), "iv_mm": float("nan"), "n_items": int(p.size),
                "error": "too_few_items"}
    p_bar = float(np.mean(p))
    h_marg = float(_h_bits(p_bar))
    h_cond = float(np.mean(_h_bits(p)))
    iv = max(h_marg - h_cond, 0.0)
    return {"iv_plugin": float(h_marg - h_cond), "iv_mm": float(iv),
            "iv_norm": float(iv / h_marg) if h_marg > _EPS else float("nan"),
            "h_marg": h_marg, "h_cond": h_cond, "p_bar": p_bar, "n_items": int(p.size),
            "readout": "soft"}


# --------------------------------------------------------------------------------------------
# degenerate-cell flags + bootstrap CI + full cell report
# --------------------------------------------------------------------------------------------

def degenerate_flags(rep: dict, *, min_items: int = 20, range_floor: float = 0.05) -> List[str]:
    flags = []
    if rep.get("error"):
        flags.append(rep["error"])
    if rep.get("n_items", 0) < min_items:
        flags.append("few_items")
    h_marg = rep.get("h_marg", float("nan"))
    if np.isfinite(h_marg) and h_marg < range_floor:
        flags.append("collapsed_no_range")     # recovered metric barely uses its range -> I_V ~ 0
    pb = rep.get("p_bar", float("nan"))
    if np.isfinite(pb) and (pb < 0.02 or pb > 0.98):
        flags.append("near_constant_verdict")
    return flags


def bootstrap_iv(V: np.ndarray, *, n_boot: int = 1000, ci: float = 0.90,
                 resample_passes: bool = True, seed: int = 0) -> dict:
    """Percentile bootstrap CI for iv_mm. Resamples items (and optionally passes within item).
    The point estimate stays the MM estimate on the full data; the CI quantifies sampling spread."""
    V = np.asarray(V, dtype=float)
    if V.ndim == 1:
        V = V[:, None]
    n_items, K = V.shape
    rng = np.random.default_rng(seed)
    pt = iv_from_passes(V).get("iv_mm", float("nan"))
    if not np.isfinite(pt) or n_items < 2:
        return {"iv_mm": pt, "ci_lo": float("nan"), "ci_hi": float("nan"), "n_boot": 0}
    vals = np.empty(n_boot)
    for b in range(n_boot):
        items = rng.integers(0, n_items, n_items)
        Vb = V[items]
        if resample_passes and K > 1:
            cols = rng.integers(0, K, K)
            Vb = Vb[:, cols]
        vals[b] = iv_from_passes(Vb).get("iv_mm", np.nan)
    vals = vals[np.isfinite(vals)]
    lo = float(np.quantile(vals, (1 - ci) / 2)) if vals.size else float("nan")
    hi = float(np.quantile(vals, 1 - (1 - ci) / 2)) if vals.size else float("nan")
    return {"iv_mm": float(pt), "ci_lo": lo, "ci_hi": hi, "ci": ci, "n_boot": int(vals.size)}


def cell_report(V: np.ndarray, *, n_boot: int = 1000, ci: float = 0.90, seed: int = 0,
                min_items: int = 20) -> dict:
    """Full per-cell readout: I_V (plug-in + MM), entropy parts, bootstrap CI, degeneracy flags."""
    rep = iv_from_passes(V)
    boot = bootstrap_iv(V, n_boot=n_boot, ci=ci, seed=seed)
    rep.update({"ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"], "n_boot": boot["n_boot"]})
    # same-f TVD transmission alongside the Shannon I_V, so every cell carries T in both f's
    t = tvd_transmission(V, n_boot=n_boot, ci=ci, seed=seed)
    rep.update({"tvd_t": t.get("tvd_t"), "tvd_t_norm": t.get("tvd_t_norm"),
                "tvd_t_ci_lo": t.get("ci_lo"), "tvd_t_ci_hi": t.get("ci_hi")})
    rep["flags"] = degenerate_flags(rep, min_items=min_items)
    return rep


# --------------------------------------------------------------------------------------------
# synthetic channel  (E0 calibration: known analytic I_V)
# --------------------------------------------------------------------------------------------

def analytic_iv(q: Sequence[float]) -> float:
    """Ground-truth I_V for a cell whose items have TRUE recovered-verdict probabilities q_i
    (item-uniform): H(mean q) - mean H(q_i). The target the estimator must recover on E0."""
    q = np.asarray(q, dtype=float)
    return float(_h_bits(float(np.mean(q))) - np.mean(_h_bits(q)))


def simulate_cell(q: Sequence[float], n_passes: int, rng: np.random.Generator) -> np.ndarray:
    """Draw an (N_items, n_passes) binary recovery matrix with per-item true probability q_i."""
    q = np.asarray(q, dtype=float)
    return (rng.random((q.size, n_passes)) < q[:, None]).astype(float)


def planted_probs(kind: str, n_items: int, rng: np.random.Generator, *, base: float = 0.5,
                  eps: float = 0.02, k: int = 1, K: int = 4) -> np.ndarray:
    """Per-item TRUE probabilities for a planted metric:
      'deterministic' -- q_i in {eps, 1-eps} (a near-code metric fully recovered) -> I_V ~ max.
      'noise'         -- q_i = base for all items (verdict independent of x) -> I_V ~ 0.
      'k_of_K'        -- verdict = OR/threshold of k recovered of K latent binary criteria; the
                         unrecovered K-k act as label noise -> intermediate, known via analytic_iv.
      'compressed'    -- items truly differ but recovery collapses to ~base (range compression) ->
                         true discrimination exists yet I_V must read LOW.
    """
    if kind == "deterministic":
        return np.where(rng.random(n_items) < 0.5, eps, 1 - eps)
    if kind == "noise":
        return np.full(n_items, base)
    if kind == "k_of_K":
        crit = rng.random((n_items, K)) < 0.5            # latent criteria
        recovered = crit[:, :k].any(axis=1)              # rubric only captures the first k
        return np.where(recovered, 1 - eps, eps)
    if kind == "compressed":
        # genuine spread destroyed by recovery: probabilities squeezed toward base
        return base + (rng.random(n_items) - 0.5) * 2 * eps
    raise ValueError(f"unknown planted kind: {kind}")


def channel_synthetic(kind: str, *, n_items: int = 60, n_passes: int = 5, seed: int = 0,
                      **kw) -> dict:
    """Build a planted cell and report estimated vs analytic I_V (the E0 calibration unit).

    NOTE: this calibrates the *item-conditional* estimator `iv_from_passes` = I(X;m̂). The headline
    articulability number is `iv_transmission` = I(m;m̂), a DIFFERENT estimator — calibrate it with
    `channel_synthetic_transmission` below.
    """
    rng = np.random.default_rng(seed)
    q = planted_probs(kind, n_items, rng, **kw)
    V = simulate_cell(q, n_passes, rng)
    rep = cell_report(V, seed=seed, min_items=0)
    rep["kind"] = kind
    rep["analytic_iv"] = analytic_iv(q)
    return rep


# --------------------------------------------------------------------------------------------
# transmission-mode E0 calibration: known I(m; m̂)  (validates iv_transmission, the headline)
# --------------------------------------------------------------------------------------------

def analytic_transmission(m_true: Sequence[float], *, channel: str = "bsc",
                          eps: float = 0.15, eps01: Optional[Tuple[float, float]] = None) -> float:
    """Ground-truth I(m; m̂) in bits when the recovered verdict m̂ is the true metric m passed
    through a KNOWN noise channel. m_true: per-item true binary verdicts.
      'bsc'  -- binary symmetric: m̂ = m flipped w.p. eps. I = H(p̄) - H(eps).
      'asym' -- eps01 = (e0, e1): P(m̂=1|m=0)=e0, P(m̂=0|m=1)=e1.
    This is what iv_transmission must recover on E0; eps=0 -> I=H(m) (perfect), eps=0.5 -> I=0.
    """
    m = np.asarray(m_true, dtype=float)
    pi = float(np.mean(m))
    if channel == "bsc":
        e = float(eps)
        p_marg = pi * (1 - e) + (1 - pi) * e
        return float(_h_bits(p_marg) - _h_bits(e))
    if channel == "asym":
        e0, e1 = eps01  # type: ignore[misc]
        p_marg = pi * (1 - e1) + (1 - pi) * e0
        h_cond = pi * _h_bits(1 - e1) + (1 - pi) * _h_bits(e0)
        return float(_h_bits(p_marg) - h_cond)
    raise ValueError(f"unknown channel: {channel}")


def analytic_tvd_transmission(q: Sequence[float]) -> float:
    """Ground-truth I_TVD(I; m̂) = (1/2N) sum_i ||q_i - q_bar||_1 for binary true per-item probs q_i
    = (1/N) sum_i |q_i - q_bar|. The target `tvd_transmission` must recover on E0."""
    q = np.asarray(q, dtype=float)
    return float(np.mean(np.abs(q - float(np.mean(q)))))


def analytic_tvd_recovery(m_true: Sequence[float], *, channel: str = "bsc", eps: float = 0.15,
                          eps01: Optional[Tuple[float, float]] = None) -> float:
    """Ground-truth I_TVD(m; m̂) = D_TV(2x2 joint || product) through a known channel. m_true: per-item
    m in {0,1}. bsc: m̂ = m flipped w.p. eps (-> 2π(1-π)|1-2eps|, =Gini at eps=0, 0 at eps=0.5)."""
    m = np.asarray(m_true, dtype=float)
    pi = float(np.mean(m))
    if channel == "bsc":
        e0 = e1 = float(eps)
    elif channel == "asym":
        e0, e1 = eps01  # type: ignore[misc]
    else:
        raise ValueError(f"unknown channel: {channel}")
    j = np.array([[(1 - pi) * (1 - e0), (1 - pi) * e0],     # m=0 row: [m̂=0, m̂=1]
                  [pi * e1,             pi * (1 - e1)]])     # m=1 row
    pm = j.sum(axis=1, keepdims=True)
    pmh = j.sum(axis=0, keepdims=True)
    return 0.5 * float(np.abs(j - pm * pmh).sum())


def simulate_transmission(m_true: Sequence[float], n_passes: int, rng: np.random.Generator, *,
                          channel: str = "bsc", eps: float = 0.15,
                          eps01: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """(N_items x n_passes) recovered-verdict matrix from m_true through the channel."""
    m = np.asarray(m_true, dtype=float)
    if channel == "bsc":
        flip = rng.random((m.size, n_passes)) < eps
        return np.where(flip, 1.0 - m[:, None], m[:, None]).astype(float)
    if channel == "asym":
        e0, e1 = eps01  # type: ignore[misc]
        p1 = np.where(m[:, None] == 1.0, 1 - e1, e0)   # P(m̂=1 | m)
        return (rng.random((m.size, n_passes)) < p1).astype(float)
    raise ValueError(f"unknown channel: {channel}")


def channel_synthetic_transmission(*, n_items: int = 200, n_passes: int = 5, pi: float = 0.5,
                                   channel: str = "bsc", eps: float = 0.15,
                                   eps01: Optional[Tuple[float, float]] = None,
                                   seed: int = 0) -> dict:
    """E0 calibration for the TRANSMISSION estimator: plant m (marginal pi) and m̂ = channel(m) with
    a KNOWN I(m;m̂); check iv_transmission recovers it. Also reports iv_from_passes = I(X;m̂) to
    verify the data-processing inequality  I(m;m̂) <= I(X;m̂)  holds on the estimates."""
    rng = np.random.default_rng(seed)
    m = (rng.random(n_items) < pi).astype(float)
    rec = simulate_transmission(m, n_passes, rng, channel=channel, eps=eps, eps01=eps01)
    rep = iv_transmission(rec, m, seed=seed)
    analytic = analytic_transmission(m, channel=channel, eps=eps, eps01=eps01)
    self_disc = iv_from_passes(rec).get("iv_mm", float("nan"))   # I(X;m̂), the DPI upper bound
    rep.update({
        "kind": f"transmission_{channel}",
        "eps": eps if channel == "bsc" else eps01,
        "analytic_transmission": analytic,
        "abs_err": abs(rep.get("iv_mm", float("nan")) - analytic),
        "in_ci": bool(rep.get("ci_lo", 1) <= analytic <= rep.get("ci_hi", -1)),
        "iv_self_IXmhat": self_disc,
        "dpi_ok": bool(rep.get("iv_mm", float("inf")) <= self_disc + 1e-9),
    })
    # same-f (TVD) calibration: estimator vs analytic I_TVD(m;m̂), plus the termwise DPI guardrail
    rec_tvd = tvd_recovery(rec, m, seed=seed)
    band = tvd_guardrail(rec, m, seed=seed)
    ana_tvd = analytic_tvd_recovery(m, channel=channel, eps=eps, eps01=eps01)
    rep.update({
        "tvd_recovery": rec_tvd["tvd_recovery"],
        "analytic_tvd_recovery": ana_tvd,
        "tvd_abs_err": abs(rec_tvd["tvd_recovery"] - ana_tvd),
        "tvd_in_ci": bool(rec_tvd["ci_lo"] <= ana_tvd <= rec_tvd["ci_hi"]),
        "T_tvd": band["T_tvd"], "A_tvd": band["A_tvd"], "dpi_tvd_ok": band["dpi_tvd_ok"],
    })
    return rep


# --------------------------------------------------------------------------------------------
# consistency channel  (existing sampled long table)
# --------------------------------------------------------------------------------------------

_DEFAULT_GROUP = ("task", "metric_id", "judge_model", "version_id", "token_cap")


def channel_consistency(df, group_cols: Sequence[str] = _DEFAULT_GROUP, *,
                        score_col: str = "score", item_col: str = "item_id",
                        pass_col: str = "pass", applicable_col: str = "applicable",
                        n_boot: int = 1000, ci: float = 0.90, min_items: int = 20,
                        require_applicable: bool = True):
    """Consistency-channel I_V for every cell of a sampled long table.

    Pivots each group to an (item x pass) binary matrix and runs `cell_report`. Returns a tidy
    pandas DataFrame: one row per cell with iv_mm/iv_plugin/iv_norm/CI/entropy parts/flags.
    """
    import pandas as pd
    group_cols = [c for c in group_cols if c in df.columns]
    d = df.copy()
    if require_applicable and applicable_col in d.columns:
        d.loc[~d[applicable_col].astype(bool), score_col] = np.nan
    rows = []
    for key, g in d.groupby(list(group_cols), dropna=False):
        mat = g.pivot_table(index=item_col, columns=pass_col, values=score_col, aggfunc="mean")
        V = mat.to_numpy(dtype=float)
        # binarize any soft means back to {0,1,NaN} only if values already binary; else keep soft
        rep = cell_report(V, n_boot=n_boot, ci=ci, min_items=min_items)
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row.update({k: rep.get(k) for k in
                    ("iv_mm", "iv_plugin", "iv_norm", "h_marg", "h_cond", "p_bar",
                     "n_items", "k_mean", "ci_lo", "ci_hi",
                     "tvd_t", "tvd_t_norm", "tvd_t_ci_lo", "tvd_t_ci_hi")})
        row["flags"] = ";".join(rep.get("flags", []))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# reconstruction channel  (consumes re-scored verdicts; LLM run is the caller's job)
# --------------------------------------------------------------------------------------------

def iv_transmission(recovered: np.ndarray, labels: Sequence[float], *, n_boot: int = 500,
                    ci: float = 0.90, seed: int = 0) -> dict:
    """I(m(x); m_recovered(x)) in bits -- the TRANSMISSION / articulability signal: how much the
    ORIGINAL metric's verdict (labels, binary) predicts the RECOVERED verdict (recovered = N_items x R
    binary draws). Robust to executor capability (unlike I(x->m_recovered), which rises with a
    stronger reader regardless of fidelity to m). 0 iff the recovered rule is independent of m.

        I = H(R) - H(R | m) = H(p_bar) - sum_c P(m=c) H(p_bar | m=c)

    `transmission_norm` = I / H(m) is the fraction of the metric's verdict-entropy that survives.
    """
    R = np.asarray(recovered, dtype=float)
    if R.ndim == 1:
        R = R[:, None]
    y = np.asarray(labels, dtype=float)
    p, k, _ = _per_item(R)
    keep = np.isfinite(y) & np.isfinite(p)
    p, k, y = p[keep], k[keep], y[keep]
    n = len(y)
    if n < 4 or np.unique(y).size < 2:
        return {"iv_mm": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "transmission_norm": float("nan"), "h_marg": float("nan"), "h_label": float("nan"),
                "n_items": int(n), "error": "degenerate_labels"}

    def _est(p, k, y):
        h_marg = _h_bits(float(np.mean(p)))
        h_cond = 0.0
        for c in (0.0, 1.0):
            g = (y == c)
            if g.any():
                h_cond += g.mean() * _h_bits(float(np.mean(p[g])))
        # light Miller-Madow: ~2 bins; marginal over sum(k) draws, each group over its own draws
        mm_marg = 1.0 / (2 * max(float(np.sum(k)), 1.0))
        mm_cond = sum((y == c).mean() / (2 * max(float(np.sum(k[y == c])), 1.0))
                      for c in (0.0, 1.0) if (y == c).any())
        return max((h_marg + mm_marg) - (h_cond + mm_cond), 0.0), h_marg

    iv, h_marg = _est(p, k, y)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(y[idx]).size < 2:
            continue
        vals.append(_est(p[idx], k[idx], y[idx])[0])
    vals = np.array([v for v in vals if np.isfinite(v)])
    h_label = float(_h_bits(float(np.mean(y))))
    return {"iv_mm": float(iv), "h_marg": float(h_marg), "h_label": h_label,
            "ci_lo": float(np.quantile(vals, (1 - ci) / 2)) if vals.size else float("nan"),
            "ci_hi": float(np.quantile(vals, 1 - (1 - ci) / 2)) if vals.size else float("nan"),
            "transmission_norm": float(iv / h_label) if h_label > _EPS else float("nan"),
            "n_items": int(n)}


def tvd_recovery(recovered: np.ndarray, labels: Sequence[float], *, n_boot: int = 500,
                 ci: float = 0.90, n_perm: int = 32, seed: int = 0) -> dict:
    """I_TVD(m; m_recovered) -- TVD recovery, the SAME-f twin of the Shannon `iv_transmission`.
    Closed form mirroring `iv_transmission`'s group-by-label structure:

        I_TVD(m;m̂) = 2 P(m=1) P(m=0) |p̄_1 - p̄_0|        (p̄_c = mean recovered YES-prob of m=c items)

    Because both this and `tvd_transmission` are per-item-MAD estimators, the data-processing
    guardrail  I_TVD(m;m̂) <= I_TVD(I;m̂)  holds **termwise by Jensen**, so `dpi_tvd_ok` from
    `tvd_guardrail` is a hard check, not a population-only hope. `recovered`: (N,R) binary draws or
    (N,) per-item means of m̂; `labels`: original verdict m in {0,1}. `tvd_recovery_norm` divides by
    the marginal max 2π(1-π) (perfect separation p̄_1=1, p̄_0=0). `debias` via label-permutation floor."""
    R = np.asarray(recovered, dtype=float)
    if R.ndim == 1:
        R = R[:, None]
    p, _, _ = _per_item(R)
    y = np.asarray(labels, dtype=float)
    keep = np.isfinite(y) & np.isfinite(p)
    p, y = p[keep], y[keep]
    n = len(y)
    if n < 4 or np.unique(y).size < 2:
        return {"tvd_recovery": float("nan"), "tvd_recovery_norm": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "n_items": int(n),
                "error": "degenerate_labels"}

    def _est(p, y):
        pi = float(np.mean(y))
        return 2.0 * pi * (1 - pi) * abs(float(np.mean(p[y == 1.0])) - float(np.mean(p[y == 0.0]))), pi

    raw, pi = _est(p, y)
    rng = np.random.default_rng(seed)
    floor = float(np.mean([_est(p, rng.permutation(y))[0] for _ in range(n_perm)])) if n_perm else 0.0
    rec = raw                                          # PRIMARY = raw (DPI-coherent with tvd_t)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(y[idx]).size >= 2:
            vals.append(_est(p[idx], y[idx])[0])
    vals = np.array([v for v in vals if np.isfinite(v)])
    gmax = 2 * pi * (1 - pi)
    return {"tvd_recovery": float(rec), "tvd_recovery_raw": float(raw), "tvd_floor": float(floor),
            "tvd_recovery_sig": float(max(0.0, raw - floor)),
            "tvd_recovery_norm": float(rec / gmax) if gmax > _EPS else float("nan"),
            "ci_lo": float(np.quantile(vals, (1 - ci) / 2)) if vals.size else float("nan"),
            "ci_hi": float(np.quantile(vals, 1 - (1 - ci) / 2)) if vals.size else float("nan"),
            "n_items": int(n)}


def tvd_guardrail(recovered: np.ndarray, labels: Sequence[float], **kw) -> dict:
    """Same-f, same-distribution TVD bundle for a binary fixed target and recovered prompt.

    ``T_tvd`` remains the historical alias for the CANDIDATE-side transmission ``I_TVD(X;mhat)``.
    The fixed-target prompt ceiling is ``T_target_tvd = I_TVD(m;X)``; both are valid DPI legs and
    ``U_dpi_tvd = min(T_target_tvd, T_candidate_tvd)`` is the sharp algebraic read available here.

    DPI is within ONE distribution (the 2026-06-22 correction): the bounding transmission must be the
    HELD-OUT T_test = I_TVD(M̂_test; X_test), computed from the SAME `recovered` verdicts R is. Both legs
    here are derived from `recovered` (held-out), so `dpi_tvd_ok` is a real check. Do NOT pass a train-
    split / consistency-channel transmission as the bound: in-sample consistency (`channel_consistency`,
    `tvd_transmission` on the prompt's own re-applied passes) lives on a DIFFERENT distribution than the
    held-out R and does NOT bound it via DPI -- it is a reliability readout, not a ceiling."""
    rec = tvd_recovery(recovered, labels, **{k: v for k, v in kw.items()
                                             if k in ("n_boot", "ci", "n_perm", "seed")})
    trn = tvd_transmission(recovered, **{k: v for k, v in kw.items()
                                         if k in ("n_boot", "ci", "n_perm", "seed", "debias")})
    target = target_channel_ceiling(np.asarray(labels, dtype=float))["empirical"]["tvd"]
    R, candidate = rec.get("tvd_recovery", float("nan")), trn.get("tvd_t", float("nan"))
    upper = min(target, candidate) if np.isfinite(target) and np.isfinite(candidate) else float("nan")
    return {"R_tvd": R,
            "T_tvd": candidate,  # backward-compatible alias; candidate side, not target ceiling
            "T_candidate_tvd": candidate, "T_target_tvd": target, "U_dpi_tvd": upper,
            "A_tvd": (candidate - R) if np.isfinite(R) and np.isfinite(candidate) else float("nan"),
            "target_headroom_tvd": (target - R) if np.isfinite(R) and np.isfinite(target) else float("nan"),
            "dpi_tvd_ok": bool(not (np.isfinite(R) and np.isfinite(upper)) or R <= upper + 1e-9),
            "R_ci": (rec.get("ci_lo"), rec.get("ci_hi")), "T_ci": (trn.get("ci_lo"), trn.get("ci_hi")),
            "n_items": rec.get("n_items")}


def cap_sanity_check(R_hat: float, *, n_bins: int = 2, near_frac: float = 0.9) -> dict:
    """Rung-1 cap as a CHANNEL-CAPACITY SANITY CHECK, not a proximity-to-optimum KPI (theory §3.1,
    downgraded 2026-06-22). cap_TVD = 1 - 1/min(N,K) = (n_bins-1)/n_bins is the maximum a K-symbol verdict
    channel can carry (½ for binary) -- a constant of the READOUT, not of the task. So `cap - R_hat` is a
    valid upper bound on OPT - R_hat but scientifically vacuous and must NOT be reported as distance-to-
    optimum. The two legitimate jobs:
      * `cap_violation` -- R_hat > cap flags an estimator / readout / LEAK bug (R can't exceed channel cap);
      * `readout_compressed` -- R_hat near cap means the binary readout is compressing top-end headroom;
        switch to a K-ary scale (lifts the cap to 1-1/K) to separate strong prompts.
    Deliberately returns NO `gap_to_optimum` field."""
    cap = (n_bins - 1) / n_bins
    return {"cap_tvd": float(cap), "R_hat": float(R_hat),
            "cap_violation": bool(np.isfinite(R_hat) and R_hat > cap + 1e-9),
            "readout_compressed": bool(np.isfinite(R_hat) and R_hat >= near_frac * cap),
            "note": "channel-capacity sanity check; cap - R_hat is NOT a proximity-to-optimum KPI"}


def iv_from_reconstruction(induced_verdicts: np.ndarray,
                           orig_verdicts: Optional[np.ndarray] = None) -> dict:
    """I_V of a metric recovered through a genuine reconstruction bottleneck.

    `induced_verdicts` is (N_held_items, n_passes) binary scores from the INDUCED rubric (a rule
    reconstructed from m's behavior on a disjoint split) re-applied by a fresh executor over passes.
    Optionally `orig_verdicts` (N_held_items,) = the original metric's verdict on the same held-out
    items, to report fidelity-to-m alongside the recovered information.
    """
    rep = cell_report(induced_verdicts, min_items=0)
    rep["channel"] = "reconstruction"
    if orig_verdicts is not None:
        o = np.asarray(orig_verdicts, dtype=float)
        p, _, _ = _per_item(induced_verdicts)
        m = np.isfinite(o) & np.isfinite(p)
        if m.sum() >= 4 and np.std(o[m]) > _EPS and np.std(p[m]) > _EPS:
            rep["agree_orig_pearson"] = float(np.corrcoef(o[m], p[m])[0, 1])
        else:
            rep["agree_orig_pearson"] = float("nan")
    return rep
