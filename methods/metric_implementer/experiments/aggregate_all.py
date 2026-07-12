"""FINAL combined table — ideal recovery Ǐ (Shannon bits, from gamma_*.npz signals M,X) AND executor
recovery R (TVD, cap ½, from bfc_*.npz subset_R). Both RECOMPUTED from saved arrays (zero GPU, nothing
trusted from logs). The two are DIFFERENT f (bits vs TVD) — reported side by side, never differenced.

  HOME=/lfs python -m methods.metric_implementer.experiments.aggregate_all
"""
from __future__ import annotations

import glob
import itertools
import os

import numpy as np

from . import submod_conditional as sc

GR = "outputs/metric_implementer_scale/gamma_runs"


def ideal(M, X):
    M, X = M.astype(int), X.astype(int)
    K = X.shape[1]
    if K == 0 or len(np.unique(M)) < 2:
        return None
    budget = max(2, K // 2)
    So, fo = sc.opt_brute(M, X, K, budget)
    deltas = [max(0.0, sc.f_recovery(M, X, So) - sc.f_recovery(M, X, [x for x in So if x != e])) for e in So]
    sden = sum(d * d for d in deltas)
    neff = (sum(deltas) ** 2 / sden) if sden > 1e-12 else float("nan")
    cm = np.nan_to_num(np.corrcoef(X.astype(float).T), nan=0.0) if K > 1 else np.array([[1.0]])
    gs = float(max(0.0, min(1.0, np.linalg.eigvalsh(cm)[0])))
    return dict(K=K, Ihat=float(fo), N_eff=float(neff), gamma_spec=gs)


def execr(z, budget=4):
    keys = [tuple(int(x) for x in k.split(",")) if k else () for k in z["subset_keys"]]
    R = {k: float(r) for k, r in zip(keys, z["subset_R"])}
    K = len(z["crits"])
    optk = max((s for s in R if 0 < len(s) <= budget), key=lambda s: R[s])
    gb = max(R, key=lambda s: R[s])
    full = R.get(tuple(range(K)), float("nan"))
    prune = (len(gb) < K) and np.isfinite(full) and (R[gb] > full + 0.005)
    return dict(K=K, R_opt=float(R[optk]), best_size=len(gb), R_full=float(full), prune=bool(prune))


def _name(path):
    b = os.path.basename(path)[:-4]
    tag = "steps" if "_steps_" in b else "criteria"
    m = b.split("gamma_steps_")[-1].split("gamma_criteria_")[-1].split("bfc_steps_")[-1].split("bfc_criteria_")[-1]
    m = m.replace("bfc_", "")  # the lone bfc_aigner... single run
    return m, tag


def main():
    rows = {}
    for f in sorted(glob.glob(f"{GR}/gamma_*.npz")):
        m, tag = _name(f)
        try:
            z = np.load(f, allow_pickle=True)
            rows.setdefault((m, tag), {})["ideal"] = ideal(z["M"], z["X"])
        except Exception as e:
            rows.setdefault((m, tag), {})["ideal"] = {"err": str(e)[:24]}
    for f in sorted(glob.glob(f"{GR}/bfc_*.npz")):
        m, tag = _name(f)
        try:
            rows.setdefault((m, tag), {})["exec"] = execr(np.load(f, allow_pickle=True))
        except Exception as e:
            rows.setdefault((m, tag), {})["exec"] = {"err": str(e)[:24]}

    print(f"{'metric':40s}{'gran':9s}{'K':>3s} | {'Ǐ(OPT)bits':>10s}{'N_eff':>7s}{'γspec':>7s} | "
          f"{'R(OPT)TVD':>10s}{'best|S|':>8s}{'Rfull':>7s}{'PRUNE':>7s}")
    for (m, tag) in sorted(rows):
        r = rows[(m, tag)]
        i, e = r.get("ideal"), r.get("exec")
        Ks = (i or {}).get("K") or (e or {}).get("K") or "?"
        ih = f"{i['Ihat']:.3f}" if i and "Ihat" in i else "-"
        ne = f"{i['N_eff']:.2f}" if i and "N_eff" in i else "-"
        gs = f"{i['gamma_spec']:.3f}" if i and "gamma_spec" in i else "-"
        ro = f"{e['R_opt']:.3f}" if e and "R_opt" in e else "-"
        bs = str(e["best_size"]) if e and "best_size" in e else "-"
        rf = f"{e['R_full']:.3f}" if e and "R_full" in e else "-"
        pr = ("yes" if e["prune"] else "no") if e and "prune" in e else "-"
        print(f"{m[:40]:40s}{tag:9s}{str(Ks):>3s} | {ih:>10s}{ne:>7s}{gs:>7s} | {ro:>10s}{bs:>8s}{rf:>7s}{pr:>7s}")


if __name__ == "__main__":
    main()
