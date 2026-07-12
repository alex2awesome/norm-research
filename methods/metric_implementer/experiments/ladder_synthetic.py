"""Synthetic E0 demonstration of the prompt-optimality LADDER (notes/2026-06-18) in TVD.

Plants thin/thick/noise metrics with KNOWN recovery R, transmission T and articulation gap A,
then shows what each rung of the ladder certifies — on ground truth, before any real number is
trusted (the plan's kill-switch gate). Zero GPU, no sk3.

Planted world: each item has K latent binary criteria. The ORIGINAL metric m fires when >= 1 of
all K criteria is present (its in-sample discrimination = T). An ARTICULATION captures only k of
the K criteria (m̂); the unrecovered K-k criteria are the tacit residual. So:
  * k = K  -> thin / fully articulable        -> R ~ T, A ~ 0
  * 0<k<K  -> thick / partly tacit            -> R < T, A > 0  (A grows as k falls)
  * k = 0  -> nothing recovered (noise rubric) -> R ~ 0, T unchanged, A ~ T

Run:  python -m methods.metric_implementer.experiments.ladder_synthetic
"""
from __future__ import annotations

import numpy as np

from methods.metric_implementer import vinfo as v

CAP_TVD = 0.5          # binary cap_TVD = 1 - 1/min(N,K). Rung 1 = a CHANNEL-CAPACITY SANITY CHECK
#                        (verify R̂ ≤ cap; detect binary-readout compression), NOT a proximity-to-optimum
#                        KPI: cap - R̂ must not be reported as distance-to-optimum (theory §3.1).


def plant(n_items: int, K: int, p_crit: float, rng: np.random.Generator):
    """Latent (n_items, K) binary criteria; original verdict m = OR over all K (>=1 present)."""
    crit = (rng.random((n_items, K)) < p_crit).astype(float)
    m = (crit.sum(axis=1) >= 1).astype(float)
    return crit, m


def recovered_passes(crit: np.ndarray, k: int, n_passes: int, exec_noise: float,
                     rng: np.random.Generator) -> np.ndarray:
    """m̂ = OR over the first k recovered criteria, re-executed with per-pass executor noise."""
    if k == 0:                                   # nothing articulated -> constant-ish guess
        base = float(rng.random())
        return (rng.random((crit.shape[0], n_passes)) < base).astype(float)
    mhat = (crit[:, :k].sum(axis=1) >= 1).astype(float)
    flip = rng.random((crit.shape[0], n_passes)) < exec_noise
    return np.where(flip, 1.0 - mhat[:, None], mhat[:, None]).astype(float)


def run(n_items: int = 400, K: int = 4, p_crit: float = 0.16, n_passes: int = 5,
        exec_noise: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    crit, m = plant(n_items, K, p_crit, rng)
    # original metric measured across passes (executor noise on m too) -> T = I_TVD(I; m)
    m_passes = np.where(rng.random((n_items, n_passes)) < exec_noise, 1.0 - m[:, None],
                        m[:, None]).astype(float)
    T = v.tvd_transmission(m_passes, seed=seed)["tvd_t"]
    pi = float(np.mean(m))

    print(f"planted world: N={n_items}  K={K}  p_crit={p_crit}  exec_noise={exec_noise}")
    print(f"original metric m: P(m=1)={pi:.3f}   T_tvd (in-sample consistency) = {T:.3f} "
          f"(cap_TVD={CAP_TVD})\n")

    # A = T(m) - R uses the ORIGINAL metric's transmission (fixed); T(m̂) is the local DPI partner.
    print(f"{'k/K':>4} {'regime':>10} {'R_tvd':>7} {'R_ci':>15} {'T(m̂)':>6} "
          f"{'A=Tm-R':>7} {'cap_gap':>8} {'dpi':>4}")
    rows = []
    for k in range(0, K + 1):
        mhat = recovered_passes(crit, k, n_passes, exec_noise, rng)
        band = v.tvd_guardrail(mhat, m, seed=seed)
        R, Tmhat = band["R_tvd"], band["T_tvd"]
        A = T - R                                    # theory's articulation gap of m
        ci = band["R_ci"]
        regime = "noise" if k == 0 else ("thin" if k == K else "thick")
        cap_gap = CAP_TVD - R
        dpi_ok = bool(R <= T + 1e-9 and R <= Tmhat + 1e-9)   # R <= min(T(m), T(m̂))
        rows.append({"k": k, "regime": regime, "R": R, "T_mhat": Tmhat, "A": A,
                     "ci_lo": ci[0], "ci_hi": ci[1], "cap_gap": cap_gap, "dpi": dpi_ok})
        print(f"{k}/{K:>2} {regime:>10} {R:7.3f} [{ci[0]:5.3f},{ci[1]:5.3f}] {Tmhat:6.3f} "
              f"{A:7.3f} {cap_gap:8.3f} {str(dpi_ok):>4}")

    # ---- Rung 3: best-in-set certification among the candidate articulations {k=0..K} ----
    print("\n--- Rung 3 (best-in-set with CIs): is the top candidate certified best? ---")
    best = max(rows, key=lambda r: r["R"])
    dominated = [r for r in rows if r is not best and r["ci_hi"] < best["ci_lo"]]
    ties = [r for r in rows if r is not best and not (r["ci_hi"] < best["ci_lo"])]
    print(f"argmax R: k={best['k']}/{K} (R={best['R']:.3f}); "
          f"CI-dominates {len(dominated)}/{len(rows)-1} rivals; "
          f"statistical ties (CI overlap): {[r['k'] for r in ties]}")
    if not ties:
        print(f"  => CERTIFIED best-in-set at the bootstrap CI level: k={best['k']} (full articulation).")
    else:
        print(f"  => top GROUP {{k={best['k']}}} ∪ {{k={[r['k'] for r in ties]}}} indistinguishable.")

    # ---- Rung 1: global gap to the universal cap ----
    print(f"\n--- Rung 1 (global cap): best R = {best['R']:.3f}, cap_TVD = {CAP_TVD}, "
          f"gap-to-cap = {CAP_TVD - best['R']:.3f} (loose: every prompt's R is <= cap_TVD). ---")

    # ---- the A monotonicity claim: A should fall as k rises (more criteria articulated) ----
    A_by_k = [r["A"] for r in rows]
    mono = all(A_by_k[i] >= A_by_k[i + 1] - 0.03 for i in range(len(A_by_k) - 1))
    print(f"\n--- articulation gap A vs k: {[round(a,3) for a in A_by_k]} "
          f"(decreasing as expected: {mono}) ---")
    return rows


if __name__ == "__main__":
    run()
