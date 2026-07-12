"""Aggregate gamma_runs/*.npz into a per-metric comparison table — RECOMPUTES the trustworthy scalars
from the saved behavioral signals (M, X), so nothing is taken on faith from the log. ZERO GPU.

Reports per (metric, granularity): K (behaviorally-distinct), R(OPT) recovery, N_eff (info-weighted
effective units), spectral γ (well-sampled), and the greedy saturation curve. Pairs criteria vs steps so
the granularity-floor experiment (does finer Ω raise R/N_eff, stay flat = resolution floor, or fall) is
directly readable.

  HOME=/lfs python -m methods.metric_implementer.experiments.aggregate_gamma
"""
from __future__ import annotations

import glob
import os

import numpy as np

from . import submod_conditional as sc


def _scalars(M, X):
    M = M.astype(int)
    X = X.astype(int)
    K = X.shape[1]
    if K == 0 or len(np.unique(M)) < 2:
        return dict(K=K, R_opt=float("nan"), N_eff=float("nan"), gamma_spec=float("nan"), curve=[])
    budget = max(2, K // 2)
    So, fo = sc.opt_brute(M, X, K, budget)
    deltas = [max(0.0, sc.f_recovery(M, X, So) - sc.f_recovery(M, X, [x for x in So if x != e])) for e in So]
    sden = sum(d * d for d in deltas)
    neff = (sum(deltas) ** 2 / sden) if sden > 1e-12 else float("nan")
    cm = np.nan_to_num(np.corrcoef(X.astype(float).T), nan=0.0) if K > 1 else np.array([[1.0]])
    gs = float(max(0.0, min(1.0, np.linalg.eigvalsh(cm)[0])))
    curve, Sacc = [], []
    for _ in range(K):
        gains = [(sc.f_recovery(M, X, Sacc + [e]) - sc.f_recovery(M, X, Sacc), e)
                 for e in range(K) if e not in Sacc]
        if not gains:
            break
        _, e = max(gains)
        Sacc.append(e)
        curve.append(round(sc.f_recovery(M, X, Sacc), 3))
    return dict(K=K, R_opt=float(fo), N_eff=float(neff), gamma_spec=gs, curve=curve)


def main():
    rows = {}
    for f in sorted(glob.glob("outputs/metric_implementer_scale/gamma_runs/gamma_*.npz")):
        base = os.path.basename(f)[:-4]
        tag = "steps" if base.startswith("gamma_steps_") else "criteria"
        metric = base.replace("gamma_steps_", "").replace("gamma_criteria_", "")
        try:
            z = np.load(f, allow_pickle=True)
            s = _scalars(z["M"], z["X"])
        except Exception as e:
            s = dict(K="ERR", R_opt=str(e)[:30], N_eff="", gamma_spec="", curve=[])
        rows.setdefault(metric, {})[tag] = s

    print(f"{'metric':46s}{'gran':9s}{'K':>4s}{'R(OPT)':>8s}{'N_eff':>7s}{'γ_spec':>8s}  saturation")
    for metric in sorted(rows):
        for tag in ("criteria", "steps"):
            s = rows[metric].get(tag)
            if not s:
                continue
            r = f"{s['R_opt']:.3f}" if isinstance(s["R_opt"], float) else str(s["R_opt"])
            ne = f"{s['N_eff']:.2f}" if isinstance(s["N_eff"], float) else str(s["N_eff"])
            gs = f"{s['gamma_spec']:.3f}" if isinstance(s["gamma_spec"], float) else str(s["gamma_spec"])
            print(f"{metric[:46]:46s}{tag:9s}{str(s['K']):>4s}{r:>8s}{ne:>7s}{gs:>8s}  {s['curve']}")
    # granularity-floor verdict where both exist
    print("\n--- granularity (criteria vs steps), where both ran ---")
    for metric in sorted(rows):
        c, st = rows[metric].get("criteria"), rows[metric].get("steps")
        if c and st and isinstance(c["R_opt"], float) and isinstance(st["R_opt"], float):
            dR = st["R_opt"] - c["R_opt"]
            verdict = ("steps RAISE R (criteria too coarse)" if dR > 0.02 else
                       "steps LOWER R (over-decomposition)" if dR < -0.02 else
                       "FLAT -> executor-resolution floor (validates §6.5)")
            print(f"  {metric[:40]:40s} R: {c['R_opt']:.3f}(crit,K={c['K']}) -> {st['R_opt']:.3f}"
                  f"(steps,K={st['K']})  ΔR={dR:+.3f}  N_eff {c['N_eff']:.2f}->{st['N_eff']:.2f}  | {verdict}")


if __name__ == "__main__":
    main()
