"""Observational scaling-law fit for metric articulability (Goal 2, 2026-06-16).

Fits the transmission T = I(m; m_recovered) (bits) as a saturating function of the two resources,
in the recovery-error framing we discussed (consensus/IRT flavor): T approaches a ceiling T_inf
(the articulable content; the L->inf / E->inf limit) as recovery error e -> 0.

  L-axis (articulation budget, the CLEAN axis -- one executor, controllable):
      T(L) = T_inf * (1 - exp(-L / Lc))            # recovery error e(L) = exp(-L/Lc)
      -> T_inf = articulability in the limit of L; Lc = knee (describability cost).

  E-axis (executor capability): logistic in a capability scalar S_E (Ruan-style observational):
      T(S) = T_inf * sigmoid(a * (S - b))
      NOTE: our 4 tiers are heterogeneous (size != capability for this task; math peaks at 8B not
      Mixtral) -> the E-fit is PROVISIONAL; report fit quality (R^2) and treat a poor fit as the
      honest finding that these models are not a capability ladder. The capability-axis definition
      is the deferred derivation discussion.

Reads the fixed-metric sweep JSONs in sk3:/lfs/.../tmp_vinfo/: fix_l<L>.json (L-sweep, one executor)
and fix_<tier>.json (E-sweep). Asymptote CI by point-resampling bootstrap.
"""
from __future__ import annotations

import glob
import json
import os
import re
import numpy as np

OUT = "/lfs/skampere3/0/alexspan/tmp_vinfo"
# PROVISIONAL capability scalars (0-1), to be replaced by the derivation discussion (Ruan PCA / IRT-theta).
CAP = {"llama3b": 0.30, "llama8b": 0.55, "qwen8b": 0.65, "mixtral": 0.60}


def _sat(L, tinf, lc):
    return tinf * (1.0 - np.exp(-np.asarray(L, float) / lc))


def _logistic(s, tinf, a, b):
    return tinf / (1.0 + np.exp(-a * (np.asarray(s, float) - b)))


def _r2(y, yhat):
    y = np.asarray(y, float); ss = np.sum((y - y.mean()) ** 2)
    return 1.0 - np.sum((y - yhat) ** 2) / ss if ss > 1e-12 else float("nan")


def fit_curve(x, y, kind="sat"):
    """Least-squares fit; returns (params, asymptote T_inf, R^2). Robust to tiny n."""
    from scipy.optimize import curve_fit
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return None, (float(np.max(y)) if len(y) else float("nan")), float("nan")
    try:
        if kind == "sat":
            p0 = [max(y.max(), 1e-3), max(np.median(x), 1.0)]
            bnds = ([0, 1e-3], [2.0, 1e5])
            popt, _ = curve_fit(_sat, x, y, p0=p0, bounds=bnds, maxfev=20000)
            return popt, float(popt[0]), _r2(y, _sat(x, *popt))
        else:
            p0 = [max(y.max(), 1e-3), 5.0, np.median(x)]
            bnds = ([0, -50, -1], [2.0, 50, 2])
            popt, _ = curve_fit(_logistic, x, y, p0=p0, bounds=bnds, maxfev=20000)
            return popt, float(popt[0]), _r2(y, _logistic(x, *popt))
    except Exception:
        return None, float(np.max(y)), float("nan")


def boot_asymptote(x, y, kind="sat", n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); y = np.asarray(y, float)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        _, tinf, _ = fit_curve(x[idx], y[idx], kind)
        if np.isfinite(tinf):
            vals.append(tinf)
    vals = np.array(vals)
    return (float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.95))) if vals.size else (np.nan, np.nan)


def _load(pat):
    out = {}
    for p in glob.glob(f"{OUT}/{pat}"):
        try:
            out[p] = json.load(open(p))
        except Exception:
            pass
    return out


def _task_curve(files_to_L):
    """files_to_L: dict path->L. Returns {task: ([L...],[meanT...])} of per-task mean transmission."""
    import collections
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for p, L in files_to_L.items():
        for r in json.load(open(p)):
            t = r.get("iv_transmission")
            if t is not None and np.isfinite(t):
                acc[r["task"]][L].append(t)
    curves = {}
    for task, byL in acc.items():
        Ls = sorted(byL)
        curves[task] = (Ls, [float(np.mean(byL[L])) for L in Ls])
    return curves


def main():
    print("=== L-AXIS scaling fit: T(L) = T_inf*(1 - exp(-L/Lc)) ===")
    lfiles = {}
    for p in glob.glob(f"{OUT}/fix_l*.json"):
        m = re.search(r"fix_l(\d+)\.json", p)
        if m:
            lfiles[p] = int(m.group(1))
    if lfiles:
        curves = _task_curve(lfiles)
        for task, (Ls, Ts) in sorted(curves.items()):
            print(f"\n  {task}: L={Ls}  T={[round(t,3) for t in Ts]}")
            popt, tinf, r2 = fit_curve(Ls, Ts, "sat")
            lo, hi = boot_asymptote(Ls, Ts, "sat")
            knee = popt[1] if popt is not None else float("nan")
            print(f"    -> T_inf(L) = {tinf:.3f} bits  ci=[{lo:.3f},{hi:.3f}]  knee Lc={knee:.0f} tok  R^2={r2:.2f}")
    else:
        print("  (no fix_l*.json yet)")

    print("\n=== E-AXIS scaling fit (PROVISIONAL capability axis): T(S)=T_inf*sigmoid(a(S-b)) ===")
    efiles = {f"{OUT}/fix_{tag}.json": CAP[tag] for tag in CAP if os.path.exists(f"{OUT}/fix_{tag}.json")}
    if efiles:
        import collections
        acc = collections.defaultdict(lambda: collections.defaultdict(list))
        for p, S in efiles.items():
            for r in json.load(open(p)):
                t = r.get("iv_transmission")
                if t is not None and np.isfinite(t):
                    acc[r["task"]][S].append(t)
        for task, byS in sorted(acc.items()):
            Ss = sorted(byS); Ts = [float(np.mean(byS[s])) for s in Ss]
            print(f"\n  {task}: S={Ss}  T={[round(t,3) for t in Ts]}")
            popt, tinf, r2 = fit_curve(Ss, Ts, "logistic")
            print(f"    -> T_inf(E) = {tinf:.3f} bits  R^2={r2:.2f}  "
                  f"{'(POOR FIT: not a clean capability ladder)' if not (np.isfinite(r2) and r2 > 0.5) else ''}")
    print("\nArticulability in the limit = T_inf (bits). ~0 => tacit; >0 (CI excl 0) => articulable.")


if __name__ == "__main__":
    main()
