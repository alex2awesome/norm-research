"""B_E variance decomposition + permutation-null species calibration (CPU, on saved checkpoints).

Resolves two questions about the conditional-species capture-recapture estimator:

1. **Is cmi_thresh=0.15 defensible?** The paraphrase-aware dedup (conditional_species) made B_E FINITE
   and gave estimator convergence (Chao1 ~= Lincoln-Petersen) at fixed thresh=0.15 — but that did NOT
   establish threshold-robustness. This reports D_obs / Chao1 / LP / singleton-fraction f1 / coverage
   across cmi_thresh so the "sweet spot" (where paraphrases collapse but distinct criteria don't:
   low f1, Chao1~=LP) is visible vs the high-thresh inflation regime (f1 large, Chao1 >> LP).

2. **Does permutation-calibrated measurement eliminate the explosion?** (the prior: correct measurement
   => no explosion.) The heuristic I >= 0.15*H(X_e) treats finite-sample MI upward bias as structure.
   We compute the label-shuffle null of pairwise MI and re-derive species at the data-driven null-95th
   threshold, then check whether the probe-subset sensitivity (ARI ~0.2 from dropping 50 probes)
   disappears under calibration.

Also decomposes the operating-point (th=0.15) B_E variance into Omega-order and probe-sample components.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from . import alpha_probe as ap


def _h(p: float) -> float:
    p = float(p)
    return 0.0 if p <= 0 or p >= 1 else float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def mi_matrix_and_h(B: np.ndarray):
    """Vectorized pairwise MI (bits) + per-criterion entropy for a binary matrix B (m x n)."""
    B = np.asarray(B, int)
    m, n = B.shape
    p = B.mean(axis=1)                                   # (m,) criterion marginals
    P11 = (B @ B.T) / n
    P10 = p[:, None] - P11
    P01 = p[None, :] - P11
    P00 = 1 - p[:, None] - p[None, :] + P11

    def term(P, pr, pc):
        with np.errstate(divide="ignore", invalid="ignore"):
            l = np.where(P > 1e-12,
                         np.log2(np.maximum(P, 1e-12)) - np.log2(np.maximum(pr, 1e-12))
                         - np.log2(np.maximum(pc, 1e-12)), 0.0)
        return P * l

    MI = (term(P11, p[:, None], p[None, :]) + term(P10, p[:, None], 1 - p[None, :])
          + term(P01, 1 - p[:, None], p[None, :]) + term(P00, 1 - p[:, None], 1 - p[None, :]))
    MI = np.clip(MI, 0, None)
    np.fill_diagonal(MI, 0.0)
    H = np.array([_h(pi) for pi in p])
    return MI, H, p


def perm_null(B: np.ndarray, n_perm: int = 30, seed: int = 0) -> np.ndarray:
    """Null pairwise-MI pool: independently shuffle each PROBE column (preserves criterion marginals/H,
    breaks criterion-criterion association). Returns pooled upper-triangle MI samples."""
    rng = np.random.default_rng(seed)
    B = np.asarray(B, int)
    m, n = B.shape
    pool = []
    for _ in range(n_perm):
        Bs = B.copy()
        for j in range(n):                              # shuffle within each probe column
            rng.shuffle(Bs[:, j])
        MI, _, _ = mi_matrix_and_h(Bs)
        iu = np.triu_indices(m, 1)
        pool.append(MI[iu])
    return np.concatenate(pool)


def species_abs(MI: np.ndarray, H: np.ndarray, abs_th: float, order) -> np.ndarray:
    """Greedy non-overlapping species with an ABSOLUTE MI threshold (candidate joins the core with max
    MI if that MI > abs_th; else founds a new species). Same greedy structure as conditional_species,
    only the threshold rule changes (absolute null-calibrated vs relative 0.15*H)."""
    m = len(H)
    core: list = []
    lab = np.full(m, -1, int)
    for e in order:
        if H[e] < 1e-9:
            continue
        if not core:
            core.append(e); lab[e] = 0; continue
        mis = MI[e, core]
        k = int(np.argmax(mis))
        if mis[k] > abs_th:
            lab[e] = lab[core[k]]
        else:
            lab[e] = len(core); core.append(e)
    return lab


def _dobs_chao_f1(lab, tags):
    v = lab >= 0
    sp = ap.spectrum(lab[v], [tags[i] for i in range(len(lab)) if v[i]])
    return sp["D"], ap.chao1(sp["f"], sp["D"]), sp["f"].get(1, 0), sp["N"]


def main(argv=None):
    p = argparse.ArgumentParser(prog="be_variance", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True)
    p.add_argument("--n-metrics", type=int, default=4)
    p.add_argument("--cmi-thresh", type=float, default=0.15)
    p.add_argument("--n-perm", type=int, default=30)
    a = p.parse_args(argv)

    ckpts = sorted(glob.glob(os.path.join(a.dir, "*_sigs.npz")))[: a.n_metrics]
    print(f"=== B_E variance decomposition + permutation-null calibration  (n={len(ckpts)} ckpts) ===\n")
    for f in ckpts:
        z = np.load(f, allow_pickle=True)
        S = np.asarray(z["sigs"], float); tags = list(z["tags"])
        name = str(z["name"]) if "name" in z.files else os.path.basename(f)
        iid = [i for i, t in enumerate(tags) if str(t) not in ("children", "gepa")]
        npr = S.shape[1]
        Bfull = (np.nan_to_num(S, nan=0.5) > 0.5).astype(int)
        MI, H, marg = mi_matrix_and_h(Bfull)
        order0 = sorted(range(len(H)), key=lambda e: abs(float(marg[e]) - 0.5))
        iu = np.triu_indices(len(H), 1)
        off = MI[iu]
        print(f"--- {name[:40]}  (n_crit={len(tags)}, iid={len(iid)}, probes={npr}) ---")

        # (1) cmi_thresh sweep: D_obs / Chao1 / LP / f1 / coverage  -> locate the sweet spot
        print("  cmi_thresh sweep (full probes):")
        print("    %6s %5s %6s %6s %5s %6s %5s"%("th","Dobs","Chao1","LP","f1","cov","f1/N"))
        for th in [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
            lab = ap.conditional_species(S, cmi_thresh=th)
            D, ch, f1, N = _dobs_chao_f1(lab, tags)
            h = len(iid) // 2
            lp = ap.two_list_crc(S, iid[:h], iid[h:], cmi_thresh=th)["B_E"]
            print("    %6.2f %5d %6.1f %6.1f %5d %6.2f %5.2f"%(
                th, D, ch, lp, f1, (D / lp if np.isfinite(lp) and lp > 0 else float("nan")), f1 / max(N, 1)))

        # (2) permutation-null calibration: is 0.15*H ~ null-95th?
        null = perm_null(Bfull, n_perm=a.n_perm, seed=0)
        null50, null95 = float(np.median(null)), float(np.percentile(null, 95))
        medH = float(np.median(H[H > 1e-9])) if (H > 1e-9).any() else float("nan")
        print("  permutation-null MI: median=%.4f  95th=%.4f  | median H=%.3f  | 0.15*H=%.4f  (heuristic vs null95)"%(
            null50, null95, medH, 0.15 * medH))
        print("    -> if 0.15*H >> null95 the heuristic is strict (real signal only); if ~ null95 it is calibrated.")
        # null-calibrated D_obs at full probes, and its stability across probe subsamples
        print("  probe-subset D_obs stability: heuristic(0.15*H) vs null-calibrated(null95):")
        print("    %6s %10s %10s"%("nprobes","heuristic","null-calg"))
        for nsub in [300, 240, 180, 120]:
            idx = list(range(npr)) if nsub == npr else sorted(
                np.random.default_rng(0).choice(npr, nsub, replace=False))
            Ss = S[:, idx]
            lab_h = ap.conditional_species(Ss, cmi_thresh=a.cmi_thresh)
            Dh, _, _, _ = _dobs_chao_f1(lab_h, tags)
            Bs = (np.nan_to_num(Ss, nan=0.5) > 0.5).astype(int)
            MIs, Hs, _ = mi_matrix_and_h(Bs)
            nulls = perm_null(Bs, n_perm=max(10, a.n_perm // 2), seed=0)
            n95s = float(np.percentile(nulls, 95))
            ords = sorted(range(len(Hs)), key=lambda e: abs(float(Bs[e].mean()) - 0.5))
            lab_n = species_abs(MIs, Hs, n95s, ords)
            Dn, _, _, _ = _dobs_chao_f1(lab_n, tags)
            print("    %6d %10d %10d"%(nsub, Dh, Dn))

        # (3) operating-point (th=0.15) B_E variance: Omega-order + probe-sample components
        ch_orders = []
        rng = np.random.default_rng(0)
        for _ in range(8):
            lab = ap.conditional_species(S, cmi_thresh=a.cmi_thresh,
                                         order=list(rng.permutation(len(tags))))
            ch_orders.append(_dobs_chao_f1(lab, tags)[1])
        ch_probes = []
        for seed in [0, 1, 2]:
            idx = sorted(rng.choice(npr, max(120, npr - 60), replace=False))
            lab = ap.conditional_species(S[:, idx], cmi_thresh=a.cmi_thresh)
            ch_probes.append(_dobs_chao_f1(lab, tags)[1])
        print("  operating-point Chao1 variance:  Omega-order(8) = %.1f +/- %.1f  |  probe-sample(3) = %.1f +/- %.1f"%(
            float(np.mean(ch_orders)), float(np.std(ch_orders)),
            float(np.mean(ch_probes)), float(np.std(ch_probes))))
        print()


if __name__ == "__main__":
    main()
