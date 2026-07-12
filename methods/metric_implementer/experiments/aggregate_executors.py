"""Cross-executor aggregation of per-model brute-force runs (limitation #5 / §6.7a multi-LLM).

Loads N per-model `small_omega_brute_force` npz files (SAME items, SAME Ω), builds a **consensus target**
`M = median-split(mean full-rubric verdict over models)` — i.e. consensus on the canonical rubric
C(Ω), a BEHAVIORAL standard (no strong-LLM "holistic quality" anchor; no-anchor pivot) — recomputes
each model's `R_i(S)=I_TVD(M_consensus; M̂_S^i)` against that shared target, and certifies:
  * per-model exact optimum (does the certified-optimal prompt MOVE across executors? -> single-LLM?)
  * R_avg(S)=(1/m)Σ R_i  -> optimal IN EXPECTATION over the family (submodular-friendly)
  * R_min(S)=min_i R_i    -> optimal for the WEAKEST model (worst-case robust; brute-force only)
  * substitution: optimal |S| per model vs model strength.

Zero-GPU.  $PY -m methods.metric_implementer.experiments.aggregate_executors a.npz b.npz c.npz --budget 6
"""
from __future__ import annotations

import argparse

import numpy as np

from .. import vinfo
from .small_omega_brute_force import greedy_on_R


def _key(s):
    return tuple(int(x) for x in s.split(",")) if s else ()


def _cert(R, K, subsets, budget, label):
    Sg, fg = greedy_on_R(R, K, budget)
    opt = max((s for s in subsets if 0 < len(s) <= budget), key=lambda s: R[s])
    fo = R[opt]
    glob = max(subsets, key=lambda s: R[s])
    print(f"  [{label}] greedy{Sg} R={fg:.3f}  OPT(k≤{budget}){list(opt)} R={fo:.3f}  "
          f"greedy/OPT={fg/fo if fo>1e-9 else float('nan'):.3f}  |  global-best |S|={len(glob)} R={R[glob]:.3f}")
    return opt, fo, glob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--budget", type=int, default=6)
    a = ap.parse_args()

    runs = [np.load(p, allow_pickle=True) for p in a.npz]
    models = [str(r["model"]) for r in runs]
    crits = list(runs[0]["crits"]); K = len(crits)
    order = [_key(s) for s in runs[0]["subset_order"]]
    subsets = order
    print(f"models: {[m.split('/')[-1] for m in models]}   K={K}   subsets={len(subsets)}")

    # consensus target M = median-split of the mean continuous FULL-RUBRIC verdict across models (same
    # items). M_cont is each model's verdict on the canonical rubric C(Ω) — a behavioral standard, so
    # consensus = agreement on that standard, NOT on a "holistic quality" anchor.
    H = np.stack([r["M_cont"] for r in runs])          # (m, N) full-rubric P(YES) per model
    Hmean = np.nanmean(H, axis=0)
    Mcons = (Hmean >= np.nanmedian(Hmean)).astype(float)
    # cross-model agreement of the raw full-rubric verdicts (sanity: do the models even agree on the standard?)
    Hbin = np.stack([(h >= np.nanmedian(h)).astype(float) for h in H])
    agree = float(np.mean([(Hbin[i] == Hbin[j]).mean()
                           for i in range(len(runs)) for j in range(i + 1, len(runs))]))
    print(f"full-rubric-M pairwise agreement across models: {agree:.2f}  (consensus M base-rate {Mcons.mean():.2f})")

    # recompute each model's R against the CONSENSUS target
    R_models = []
    for r in runs:
        MH = r["mhat"]
        Ri = {S: vinfo.tvd_recovery(MH[idx], Mcons)["tvd_recovery"] for idx, S in enumerate(order)}
        R_models.append(Ri)

    print("\n=== per-model exact optimum vs the CONSENSUS standard (is the optimum single-LLM?) ===")
    opts = []
    for m, Ri in zip(models, R_models):
        opt, fo, glob = _cert(Ri, K, subsets, a.budget, m.split("/")[-1])
        opts.append(set(opt))
    # do the models pick the SAME optimal criteria?
    inter = set.intersection(*opts); union = set.union(*opts)
    print(f"\noptimal-criterion overlap across models: |∩|={len(inter)} |∪|={len(union)} "
          f"(Jaccard {len(inter)/max(len(union),1):.2f}) -> "
          f"{'STABLE (model-robust optimum)' if len(inter)/max(len(union),1) > 0.6 else 'MODEL-SPECIFIC optimum (single-LLM)'}")

    print("\n=== cross-model robust objectives (shared consensus M) ===")
    Ravg = {S: float(np.mean([Ri[S] for Ri in R_models])) for S in subsets}
    Rmin = {S: float(min(Ri[S] for Ri in R_models)) for S in subsets}
    _cert(Ravg, K, subsets, a.budget, "R_avg (optimal in expectation over family)")
    _cert(Rmin, K, subsets, a.budget, "R_min (optimal for weakest model)")

    print("\n=== substitution: optimal |S| (unconstrained) per model vs strength ===")
    for m, Ri in zip(models, R_models):
        glob = max(subsets, key=lambda s: Ri[s])
        print(f"  {m.split('/')[-1]:<26} best |S|={len(glob)}  R={Ri[glob]:.3f}  picks={sorted(glob)}")


if __name__ == "__main__":
    main()
