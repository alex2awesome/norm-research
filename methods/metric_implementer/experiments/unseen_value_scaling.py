"""Parametric scaling-law fits (Heaps/Herdan) for discovery AND value curves, plus an
adaptive-sampler de-bias layer.

WHY THIS EXISTS (what was missing before it):
  ``alpha_probe.rarefaction`` / ``heaps_alpha`` already give the de-noised discovery curve
  E[S(m)] and its LOCAL slope alpha(m) = d log S / d log m. ``value_census.value_rarefaction``
  gives the value-axis analog E[V(m)]. What did NOT exist:
    (1) an explicit PARAMETRIC fit of a Heaps/Herdan power law  y = c * m^alpha  (and a
        saturating alternative  y = y_inf * (1 - e^{-m/tau})) with bootstrap CIs and a
        model-selection verdict, so we can say "this curve follows a power law with
        alpha = 0.42 [0.38, 0.47]" or "no — it saturates";
    (2) the same fit applied to the VALUE curve, not just the count curve, to answer
        "does unseen VALUE follow its own scaling law?";
    (3) any de-bias for the fact that our mining is ADAPTIVE, not i.i.d.

ON THE ADAPTIVE DE-BIAS (the "just estimate the weight and normalize" question):
  The intuition is Horvitz-Thompson / inverse-propensity weighting. It is directionally right for
  expectations over the OBSERVED support, but it DOES NOT de-bias MISSING MASS, and we verified this
  numerically (tmp/audit_iw.py): under a known adaptive sampler with true missing mass 0.445, naive
  Good-Turing gives 0.010 (under-claims) and self-normalized IW gives 0.954 (worse, ESS = 0.3% of n).
  Two reasons, one theoretical, one now-empirical:
    * Missing mass is a sum over UNSEEN types, which carry ZERO observed weight. No reweighting of
      observed draws can reach the unseen tail. "Singleton status" is a sample-dependent event, not a
      fixed function of type, so it has no valid inverse-propensity correction.
    * The weight is 1/propensity; its variance explodes for small propensity — exactly the rare tail —
      so ESS collapses and the estimate is dominated by a few draws.
  Therefore this module treats the two branches ASYMMETRICALLY:
    - ``audit_anchored_missing_mass``: design-based; Good-Turing on an i.i.d. audit subset only, valid
      by construction, needs no weights. THIS is the de-bias. The mined-minus-audit gap MEASURES the
      adaptive bias (verified: it correctly reports the -0.35 over-coverage gap in the simulation).
    - ``importance_weighted_singleton_mass``: a self-flagging DIAGNOSTIC, NOT a missing-mass estimator.
      It returns the IW singleton mass together with ESS and a ``valid_missing_mass_estimator=False``
      marker; trust it for nothing beyond "how concentrated are the weights" (near-1 ESS_frac => the
      sampler was ~uniform; low => heavy adaptivity, and no point estimate here is usable).
  Bottom line for the user's question: no, estimating the weight and normalizing is NOT sufficient —
  the unseen tail needs an i.i.d. audit (or a parametric Chao model), not a weight.

UNCERTAINTY: bootstrapping CURVE POINTS (as ``fit_power_law`` does) measures FIT STABILITY of a
smooth, autocorrelated, near-deterministic curve — it is NOT a sampling CI and is ~70x too narrow on
cumulative curves (verified, tmp/audit_ci_chao.py). For a real sampling CI on a cumulative discovery
curve, resample the underlying DRAWS: use ``heaps_unit_bootstrap_ci``. The ``*_ci`` fields from
``fit_power_law`` are labelled ``ci_kind="fit_stability"`` accordingly.

Literature: Good (1953) / Gale-Sampson (1995) missing mass; Chao (1984) / Chao-Bunge (2002)
richness under unequal catchability; Colwell et al. (2012) & Chao et al. (2014) rarefaction /
extrapolation; Heaps'/Herdan's law for type accumulation; Horvitz-Thompson (1952) for the IW leg.

Pure numpy/scipy, zero-GPU. Consumes curves produced by ``alpha_probe`` / ``value_census``; it
does NOT re-implement their estimators.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit


# ============================================================================================
# A — parametric scaling-law fits  (Heaps power law  vs  exponential saturation)
# ============================================================================================

def _power(m: np.ndarray, c: float, alpha: float) -> np.ndarray:
    return c * np.power(m, alpha)


def _saturating(m: np.ndarray, y_inf: float, tau: float) -> np.ndarray:
    return y_inf * (1.0 - np.exp(-m / tau))


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")


def _aic(y: np.ndarray, yhat: np.ndarray, k_params: int) -> float:
    """Gaussian-residual AIC = n·log(RSS/n) + 2k. Lower is better. Comparable ACROSS the two
    forms because both fit the same (m, y) on the same points with the same noise model."""
    y = np.asarray(y, float)
    n = len(y)
    rss = float(np.sum((y - yhat) ** 2))
    if rss <= 0:
        rss = 1e-12
    return n * np.log(rss / n) + 2 * k_params


def fit_power_law(m: Sequence[float], y: Sequence[float], *, n_boot: int = 1000,
                  seed: int = 0) -> dict:
    """Fit the Heaps/Herdan power law  y = c * m^alpha  by OLS in log-log space (the standard,
    variance-stabilizing Heaps estimator). ``alpha`` is the GLOBAL exponent (contrast:
    ``alpha_probe.heaps_alpha`` returns the local slope trajectory). alpha ~ 1 => inexhaustible
    (linear growth); alpha << 1 => fast saturation / low effective dimension.

    ⚠ The point estimate (alpha, c) is sound. The bootstrap here resamples the (log m, log y) CURVE
    POINTS, which are autocorrelated nodes of a smooth, near-deterministic rarefaction/accumulation
    curve — so ``alpha_ci`` measures FIT STABILITY (sensitivity to which grid nodes are included), NOT
    the sampling variability of alpha, and it is severely TOO NARROW (~70x on cumulative curves,
    verified). It is returned with ``ci_kind="fit_stability"`` and MUST NOT be quoted as a confidence
    interval. For a real sampling CI on a cumulative discovery curve use ``heaps_unit_bootstrap_ci``,
    which resamples the underlying draws."""
    m = np.asarray(m, float)
    y = np.asarray(y, float)
    keep = (m > 0) & (y > 0)
    lm, ly = np.log(m[keep]), np.log(y[keep])
    if keep.sum() < 3:
        return {"form": "power", "ok": False, "reason": "need >=3 positive points"}
    alpha, logc = np.polyfit(lm, ly, 1)
    c = float(np.exp(logc))
    yhat = _power(m, c, alpha)
    boot_a, boot_c = [], []
    rng = np.random.default_rng(seed)
    n = len(lm)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            a_b, lc_b = np.polyfit(lm[idx], ly[idx], 1)
        except Exception:
            continue
        boot_a.append(a_b)
        boot_c.append(np.exp(lc_b))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [float("nan")] * 2
    return {"form": "power", "ok": True, "c": c, "alpha": float(alpha),
            "alpha_ci": ci(boot_a), "c_ci": ci(boot_c), "ci_kind": "fit_stability",
            "r2_loglog": _r2(ly, np.log(np.maximum(yhat[keep], 1e-12))),
            "r2_linear": _r2(y[keep], yhat[keep]),
            "aic": _aic(y[keep], yhat[keep], 2)}


def heaps_unit_bootstrap_ci(increments: Sequence[int], *, block: int = 5, n_boot: int = 1000,
                            seed: int = 0) -> dict:
    """A REAL sampling CI for the Heaps exponent of a cumulative discovery curve, by moving-block
    bootstrap of the underlying per-draw discovery increments (1 if that draw revealed a new type,
    else 0). Rebuilds the cumulative curve per replicate and refits, preserving short-range
    autocorrelation via blocks. This is the honest partner to ``fit_power_law``'s fit-stability CI —
    on real registry data it is ~70x wider. Returns the point alpha (full curve) and the unit-boot CI."""
    inc = np.asarray(increments, float)
    n = len(inc)
    if n < 8:
        return {"ok": False, "reason": "need >=8 increments"}
    m = np.arange(1, n + 1, dtype=float)
    full = fit_power_law(m, np.cumsum(inc), n_boot=0)
    rng = np.random.default_rng(seed)
    alphas = []
    for _ in range(n_boot):
        blocks = []
        total = 0
        while total < n:
            s = int(rng.integers(0, max(1, n - block)))
            blocks.append(inc[s:s + block]); total += min(block, n - s)
        bi = np.concatenate(blocks)[:n]
        c = np.cumsum(bi)
        c[c <= 0] = 1e-9
        f = fit_power_law(m, c, n_boot=0)
        if f.get("ok"):
            alphas.append(f["alpha"])
    if not alphas:
        return {"ok": False, "reason": "no successful refits"}
    return {"ok": True, "alpha": full.get("alpha"), "ci_kind": "sampling_moving_block",
            "alpha_ci": [float(np.percentile(alphas, 2.5)), float(np.percentile(alphas, 97.5))],
            "block": block, "n_increments": n}


def fit_saturating(m: Sequence[float], y: Sequence[float], *, n_boot: int = 1000,
                   seed: int = 0) -> dict:
    """Fit the exponential-saturation form  y = y_inf * (1 - e^{-m/tau})  (a FINITE ceiling, unlike
    the power law). ``y_inf`` is the asymptotic total value/richness; ``tau`` the draw scale to
    reach 1-1/e of it. Fit by nonlinear least squares with sensible seeds/bounds; residual-resample
    bootstrap for CIs."""
    m = np.asarray(m, float)
    y = np.asarray(y, float)
    if len(m) < 3:
        return {"form": "saturating", "ok": False, "reason": "need >=3 points"}
    p0 = [max(float(y.max()) * 1.1, 1e-6), max(float(m.mean()), 1.0)]
    bounds = ([1e-9, 1e-9], [np.inf, np.inf])
    try:
        popt, _ = curve_fit(_saturating, m, y, p0=p0, bounds=bounds, maxfev=20000)
    except Exception as e:
        return {"form": "saturating", "ok": False, "reason": f"curve_fit failed: {e}"}
    y_inf, tau = float(popt[0]), float(popt[1])
    yhat = _saturating(m, y_inf, tau)
    resid = y - yhat
    boot_inf, boot_tau = [], []
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yb = yhat + rng.permutation(resid)
        try:
            pb, _ = curve_fit(_saturating, m, yb, p0=popt, bounds=bounds, maxfev=20000)
            boot_inf.append(pb[0]); boot_tau.append(pb[1])
        except Exception:
            continue
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [float("nan")] * 2
    return {"form": "saturating", "ok": True, "y_inf": y_inf, "tau": tau,
            "y_inf_ci": ci(boot_inf), "tau_ci": ci(boot_tau), "ci_kind": "fit_stability",
            "r2_linear": _r2(y, yhat), "aic": _aic(y, yhat, 2)}


def compare_scaling_forms(m: Sequence[float], y: Sequence[float], *, holdout_frac: float = 0.2,
                          n_boot: int = 1000, seed: int = 0) -> dict:
    """Fit BOTH forms and pick a winner two ways that must agree to be trusted:
      * in-sample RSS (reported as ``aic``, but since BOTH forms have k=2 the AIC penalty CANCELS in
        every comparison — it is purely a lower-RSS test, and its magnitude scales with the arbitrary
        ``n_points`` grid, so only the WINNER's sign is meaningful, never the ΔAIC magnitude), and
      * HELD-OUT tail extrapolation error: fit on the first (1-holdout_frac) of draws, predict the
        last ``holdout_frac``, compare MSE.
    The tail test is the honest one for a scaling claim: a power law and a saturating curve agree in
    the body and disagree in the tail, which is exactly the regime a "value of more mining" question
    lives in. Returns both fits, both verdicts, and a combined ``verdict``/``verdict_note``. NOTE: in
    the linear (alpha~1, unsaturated) regime the two forms nearly coincide even in the tail, so a
    'verdict' there is weak — read it together with the alpha estimate, not alone."""
    m = np.asarray(m, float)
    y = np.asarray(y, float)
    full_pow = fit_power_law(m, y, n_boot=n_boot, seed=seed)
    full_sat = fit_saturating(m, y, n_boot=n_boot, seed=seed)

    order = np.argsort(m)
    ms, ys = m[order], y[order]
    cut = max(3, int(round(len(ms) * (1 - holdout_frac))))
    tail = {"ok": False}
    if cut < len(ms):
        m_tr, y_tr, m_te, y_te = ms[:cut], ys[:cut], ms[cut:], ys[cut:]
        p = fit_power_law(m_tr, y_tr, n_boot=0)
        s = fit_saturating(m_tr, y_tr, n_boot=0)
        if p.get("ok") and s.get("ok"):
            mse_p = float(np.mean((y_te - _power(m_te, p["c"], p["alpha"])) ** 2))
            mse_s = float(np.mean((y_te - _saturating(m_te, s["y_inf"], s["tau"])) ** 2))
            tail = {"ok": True, "mse_power": mse_p, "mse_saturating": mse_s,
                    "winner": "power" if mse_p < mse_s else "saturating"}

    aic_winner = None
    if full_pow.get("ok") and full_sat.get("ok"):
        aic_winner = "power" if full_pow["aic"] < full_sat["aic"] else "saturating"

    if aic_winner and tail.get("ok"):
        if aic_winner == tail["winner"]:
            verdict, note = aic_winner, "AIC and held-out tail agree"
        else:
            verdict, note = "ambiguous", f"AIC says {aic_winner}, tail says {tail['winner']} — no clean scaling law"
    else:
        verdict, note = (aic_winner or "undetermined"), "tail test unavailable; AIC only"
    return {"power": full_pow, "saturating": full_sat, "tail_test": tail,
            "aic_winner": aic_winner, "verdict": verdict, "verdict_note": note}


# ============================================================================================
# B — extrapolation / marginal value of additional draws  (the "gain(h)" curve)
# ============================================================================================

def extrapolate(fit: dict, m_observed: float, horizons: Sequence[float]) -> dict:
    """Project a fitted curve forward: cumulative value at m_observed+h and the marginal gain of the
    h extra draws, for each horizon h. Closed-form derivative gives the per-draw marginal at the
    horizon. Works for either fit ``form``. This is the "model cumulative value from additional
    draws" branch — the one the frozen mining process makes legitimate to extrapolate along."""
    horizons = np.asarray(horizons, float)
    m_future = m_observed + horizons
    if fit.get("form") == "power" and fit.get("ok"):
        c, a = fit["c"], fit["alpha"]
        y_now = _power(np.array([m_observed]), c, a)[0]
        y_fut = _power(m_future, c, a)
        marg = c * a * np.power(m_future, a - 1.0)          # dy/dm
    elif fit.get("form") == "saturating" and fit.get("ok"):
        yi, tau = fit["y_inf"], fit["tau"]
        y_now = _saturating(np.array([m_observed]), yi, tau)[0]
        y_fut = _saturating(m_future, yi, tau)
        marg = (yi / tau) * np.exp(-m_future / tau)
    else:
        return {"ok": False, "reason": "fit not usable"}
    return {"ok": True, "form": fit["form"], "horizons": horizons.tolist(),
            "cumulative_value": y_fut.tolist(),
            "marginal_gain_from_now": (y_fut - y_now).tolist(),
            "marginal_per_draw_at_horizon": marg.tolist()}


# ============================================================================================
# B2 — the value-coverage curve  f_{p_seen}(value): value as a function of COVERAGE, not draws
# ============================================================================================

def value_coverage_curve(m: Sequence[float], S: Sequence[float], V: Sequence[float],
                         *, richness: Optional[float] = None) -> dict:
    """The THIRD curve: cumulative value V plotted against coverage, parameterized by draws m — with
    the draw-rate eliminated. This is the sampler-RATE-INVARIANT object: an unbiased sampler that
    merely draws faster moves S(m) and V(m) together but leaves this curve fixed; a VALUE-BIASED
    sampler (skims high-value patterns first) bends it.

    ⚠ TWO caveats a reader must carry: (1) the x-axis. Coverage = S(m)/richness. If ``richness`` is a
    Chao1 estimate it is UNSTABLE when doubletons are scarce (f2->0 makes Chao1 explode, so the
    absolute '% covered' is unreliable — verified it swings 0.9%->19% across plausible f2). Prefer
    passing ``richness`` = OBSERVED richness S(N); then C is 'fraction of observed richness' and the
    SHAPE is what matters, not the absolute scale. (2) the y-axis. If V came from summing per-species
    ADDITIVE (standalone) value, it is redundancy-BLIND: near-duplicate later criteria each re-add
    their bits, so V can look linear even after the joint (redundancy-discounted) value saturated.
    dV/dS is then 'additive value per fresh discovery', an upper read; compare to a submodular
    (redundancy-discounted) value curve before concluding anything about real preference value.

    The local slope dV/dS = E[value | newly discovered type] at that coverage (Good-Turing identity:
    dS/dm = P(new), dV/dm = P(new)*E[value|new], so their ratio is the mean value of a fresh
    discovery). Flattening => head-concentrated value; rising => tail gems (subject to caveat 2)."""
    m = np.asarray(m, float)
    S = np.asarray(S, float)
    V = np.asarray(V, float)
    denom = richness if (richness and richness > 0) else (S[-1] if S[-1] > 0 else 1.0)
    C = S / denom
    order = np.argsort(S)
    So = S[order]
    dS = np.diff(So)
    dV_dS = (np.gradient(V[order], So) if np.all(dS > 0)   # guard equal-S (would give inf)
             else np.full_like(So, np.nan))
    return {"coverage": C.tolist(), "distinct": S.tolist(), "value": V.tolist(),
            "value_per_new_discovery_at_S": np.asarray(dV_dS).tolist(),
            "richness_used": float(denom),
            "value_axis_note": "additive/redundancy-blind unless V is a submodular curve"}


def value_coverage_exponent(alpha_count: float, alpha_value: float,
                            *, alpha_count_ci=None, alpha_value_ci=None) -> dict:
    """Derived exponent of value-per-fresh-discovery: dV/dS ~ m^{alpha_value - alpha_count}. The SIGN
    is the qualitative story of the value-coverage curve, free from the two Heaps fits.

    ⚠ A regime label is only meaningful if the exponent is DISTINGUISHABLE from 0. The +-0.02
    thresholds are heuristic and carry NO error control; if the two alphas' CIs overlap (or you only
    have the fit-stability CI, which is too narrow), a near-zero exponent means 'indistinguishable
    from neutral', NOT an established 'rarity-neutral' finding. We flag that explicitly."""
    d = float(alpha_value) - float(alpha_count)
    distinguishable = None
    if alpha_count_ci and alpha_value_ci:                   # overlap test on whatever CIs are supplied
        distinguishable = not (alpha_value_ci[0] <= alpha_count_ci[1] and
                               alpha_count_ci[0] <= alpha_value_ci[1])
    if abs(d) <= 0.02 or distinguishable is False:
        regime = "indistinguishable from rarity-neutral (exponent ~ 0 within available uncertainty)"
    elif d < 0:
        regime = "diminishing: value is head-concentrated, later discoveries worth less"
    else:
        regime = "tail-gems: later (rarer) discoveries are MORE valuable"
    return {"alpha_count": float(alpha_count), "alpha_value": float(alpha_value),
            "value_coverage_exponent": d, "distinguishable_from_zero": distinguishable,
            "regime": regime}


def value_frontloading_stat(m: Sequence[float], V: Sequence[float], S: Sequence[float]) -> dict:
    """Distribution-free front-loading statistic on PAIRED (same-subset) curves:

        D = mean_over_grid[ V(m)/V(m_max) − S(m)/S(m_max) ].

    D > 0 ⟺ the normalized value curve runs AHEAD of the normalized discovery curve — value
    concentrates early (saturation-type behavior) while discovery keeps going. It commits to NO
    parametric form (no power law, no exponential), so combined with a probe/subset bootstrap it is
    the error-controlled replacement for the regime call that ``value_coverage_exponent`` could only
    label heuristically. Requires curves computed on the SAME random subsets (see
    ``value_census.exchangeable_joint_value``) so the pairing is exact. D ∈ (−1, 1); its magnitude
    is the mean vertical gap between the two normalized curves."""
    V = np.asarray(V, float)
    S = np.asarray(S, float)
    if len(V) < 3 or V[-1] <= 1e-9 or S[-1] <= 1e-9:
        return {"ok": False, "reason": "degenerate terminal value (V or S ~ 0) or <3 points"}
    d = float(np.mean(V / V[-1] - S / S[-1]))
    return {"ok": True, "D": d,
            "read": ("value front-loaded vs discovery (saturation-type)" if d > 0 else
                     "value trails discovery (tail-gems-type)" if d < 0 else "matched shapes")}


# ============================================================================================
# C — adaptive-sampler de-bias  (IW Good-Turing  +  design-based audit anchor)
# ============================================================================================

def effective_sample_size(weights: Sequence[float]) -> float:
    """Kish ESS = (Σw)^2 / Σw^2. Collapses toward 1 as weights concentrate — the honest read on how
    much an inverse-propensity estimate actually 'saw'. ESS/n well below ~0.5 means the weighted
    number is dominated by a few draws and should be distrusted."""
    w = np.asarray(weights, float)
    s2 = float(np.sum(w ** 2))
    return float(np.sum(w) ** 2 / s2) if s2 > 0 else 0.0


def inverse_propensity_novelty_rate(type_ids: Sequence, propensity: Sequence[float]) -> dict:
    """A self-flagging DIAGNOSTIC — NOT a missing-mass estimator. (Formerly mislabeled
    ``importance_weighted_missing_mass``.)

    Computes the inverse-propensity-weighted singleton rate
        r_ip = ( Σ_{i: singleton} w_i ) / ( Σ_i w_i ),   w_i = 1/propensity_i,
    alongside the naive singleton rate and the effective sample size. It is documented as invalid for
    missing mass for two reasons: (a) missing mass sums over UNSEEN types, which carry zero observed
    weight — no reweighting reaches them; (b) 'singleton' is a sample-count-INDUCED event, so it is
    not a fixed outcome eligible for a Horvitz-Thompson correction against the very propensity that
    induced it. Verified numerically (tmp/audit_iw.py): under a known adaptive sampler with true
    missing mass 0.445, r_ip = 0.954 while ESS/n = 0.003. USE: only ``ess_frac`` as a weight-
    concentration read (near 1 => sampler ~uniform; low => heavy adaptivity). For actual de-bias use
    ``audit_anchored_missing_mass``. Note: when nearly every type is a singleton (M0->1, the frequent
    real-data regime) r_ip == raw_rate identically, so this branch is inert there."""
    ids = list(type_ids)
    q = np.asarray(propensity, float)
    n = len(ids)
    if n == 0 or len(q) != n:
        return {"ok": False, "reason": "empty or mismatched"}
    if np.any(q <= 0):
        return {"ok": False, "reason": "propensity must be strictly positive"}
    w = 1.0 / q
    counts: Dict[object, int] = {}
    for t in ids:
        counts[t] = counts.get(t, 0) + 1
    singleton = np.array([counts[t] == 1 for t in ids], bool)
    W = float(w.sum())
    r_ip = float(w[singleton].sum() / W) if W > 0 else float("nan")
    r_raw = float(singleton.sum() / n)
    ess = effective_sample_size(w)
    return {"ok": True, "valid_missing_mass_estimator": False,
            "raw_novelty_rate": r_raw, "ip_novelty_rate": r_ip, "n": n,
            "n_singletons": int(singleton.sum()), "ess": ess, "ess_frac": ess / n,
            "weights_stable": bool(ess / n >= 0.5),
            "warning": "diagnostic only; not missing mass. Use ess_frac; de-bias via audit anchor."}


def audit_anchored_missing_mass(type_ids: Sequence, is_audit: Sequence[bool]) -> dict:
    """Design-based de-bias: compute Good-Turing missing mass on the i.i.d. AUDIT subset only (where
    exchangeability holds by construction), and separately on the adaptively-MINED subset, and report
    the divergence. No weights, no propensity model — this is the trustworthy anchor. The mined-minus-
    audit gap is a direct estimate of how much the adaptive sampler distorts the missing-mass read
    (adaptive concentration typically DEPRESSES f1/N -> over-claims coverage -> mined M0 < audit M0)."""
    ids = list(type_ids)
    aud = np.asarray(is_audit, bool)
    if len(aud) != len(ids) or len(ids) == 0:
        return {"ok": False, "reason": "empty or mismatched"}

    def _gt(sub_ids):
        counts: Dict[object, int] = {}
        for t in sub_ids:
            counts[t] = counts.get(t, 0) + 1
        n = len(sub_ids)
        f1 = sum(1 for t in sub_ids if counts[t] == 1)
        return {"n": n, "f1": f1, "m0": (f1 / n if n else float("nan"))}

    audit = _gt([t for t, a in zip(ids, aud) if a])
    mined = _gt([t for t, a in zip(ids, aud) if not a])
    div = (mined["m0"] - audit["m0"]) if (audit["n"] and mined["n"]) else float("nan")
    return {"ok": True, "audit": audit, "mined": mined, "mined_minus_audit": div,
            "interpretation": ("audit is the valid missing-mass estimate; a negative gap means the "
                               "adaptive sampler over-claims coverage relative to i.i.d.")}


def debias_report(type_ids: Sequence, *, propensity: Optional[Sequence[float]] = None,
                  is_audit: Optional[Sequence[bool]] = None) -> dict:
    """Run whichever de-bias branches the available signals support and return them together. Give
    ``propensity`` for the IW branch, ``is_audit`` for the design-based branch; both is best (their
    agreement is the actual evidence). Neither -> only the naive Good-Turing point, flagged."""
    out: dict = {}
    ids = list(type_ids)
    counts: Dict[object, int] = {}
    for t in ids:
        counts[t] = counts.get(t, 0) + 1
    f1 = sum(1 for t in ids if counts[t] == 1)
    out["naive_good_turing_m0"] = f1 / len(ids) if ids else float("nan")
    if is_audit is not None:                                # the ACTUAL de-bias (design-based)
        out["audit_anchored"] = audit_anchored_missing_mass(ids, is_audit)
    if propensity is not None:                              # diagnostic only, not a de-bias
        out["ip_novelty_diagnostic"] = inverse_propensity_novelty_rate(ids, propensity)
    if is_audit is None:
        out["warning"] = ("no i.i.d. audit split provided — missing mass cannot be de-biased for "
                          "adaptivity; the IP diagnostic is NOT a substitute")
    return out
