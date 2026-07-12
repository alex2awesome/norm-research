"""Matched-f executor gap (ZERO GPU). On the SAME best subset S:
  ideal   I_TVD(M; X_S)   — full signal vector, TVD          (from gamma_*.npz M,X)
  executor R = I_TVD(M; M̂_S) — one compressed verdict, TVD   (from bfc_*.npz subset_R)
gap = ideal − executor = the executor-compression loss, in CONSISTENT TVD units (cap ½).
Guards on crits-identity so we never assume the two runs share criterion ordering.
"""
from __future__ import annotations

import glob
import os

import numpy as np

GR = "outputs/metric_implementer_scale/gamma_runs"


def tvd_mi(M, Xsub):
    """I_TVD(M; X_sub) = ½ Σ_{m,x} |P(m,x) − P(m)P(x)|, M binary, X_sub a binary matrix (joint over 2^k)."""
    M = M.astype(int)
    N = len(M)
    code = np.zeros(N, int)
    for j in range(Xsub.shape[1]):
        code = code * 2 + Xsub[:, j].astype(int)
    tvd = 0.0
    for m in (0, 1):
        pm = (M == m).mean()
        for x in np.unique(code):
            tvd += abs(((M == m) & (code == x)).mean() - pm * (code == x).mean())
    return 0.5 * tvd


def main():
    print(f"{'metric':42s}{'|S|':>4s} | {'I_TVD_ideal':>12s}{'R_exec':>8s}{'gap(exec loss)':>16s}")
    bfs = sorted(glob.glob(f"{GR}/bfc_*.npz"))
    for bf in bfs:
        if "_steps_" in os.path.basename(bf):
            continue
        m = os.path.basename(bf)[:-4].replace("bfc_criteria_", "").replace("bfc_", "")
        gf = f"{GR}/gamma_criteria_{m}.npz"
        if not os.path.exists(gf):
            continue
        zb = np.load(bf, allow_pickle=True)
        zg = np.load(gf, allow_pickle=True)
        cb, cg = list(zb["crits"]), list(zg["crits"])
        if cb != cg:
            print(f"{m[:42]:42s}  SKIP — crits differ between bfc({len(cb)}) and gamma({len(cg)}) (no safe index map)")
            continue
        keys = [tuple(int(x) for x in k.split(",")) if k else () for k in zb["subset_keys"]]
        R = {k: float(r) for k, r in zip(keys, zb["subset_R"])}
        gb = max(R, key=lambda s: R[s])
        Re = R[gb]
        M, X = zg["M"], zg["X"]
        if not gb or max(gb) >= X.shape[1]:
            continue
        Ii = tvd_mi(M, X[:, list(gb)])
        print(f"{m[:42]:42s}{len(gb):>4d} | {Ii:>12.3f}{Re:>8.3f}{Ii - Re:>+16.3f}")


if __name__ == "__main__":
    main()
