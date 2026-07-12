"""Does combining per-criterion judgments OUTSIDE the LLM beat one holistic LLM verdict?

A SIDEWAYS test (not a deeper hierarchy): Ω is fixed at the criteria. We only change the AGGREGATOR.
Everything is paired inside each bfc_*.npz (same 200 items, same M, same LLM calls) — ZERO GPU.

The bfc lattice already scored the LLM on every subset, INCLUDING every singleton {i}. So:
  X_solo[:,i] = mhat[{i}]                 = LLM judging criterion i ALONE      (continuous P(YES))
  R_A(S)      = tvd_recovery(mhat[S], M)  = LLM judging all of S together, ONE holistic verdict
  R_B(S)      = tvd_recovery(combine(X_solo[:,S]), M) = solo verdicts combined OUTSIDE the LLM
  Iceil(S)    = I_TVD(M; X_S)             = full joint categorical content of the criteria (no single-
                                            verdict constraint) — the lossless-aggregation ceiling.

Three-level ladder on the SAME subset, increasingly compressed:  Iceil  ≥  R_B  vs  R_A.
  gap 1->2 (Iceil - R_B) = cost of collapsing K signals into ONE scalar verdict at all.
  gap 2->3 (R_B - R_A)   = the LLM being a worse aggregator than arithmetic (the 'plumbing' loss).
If R_B(full) >> R_A(full), the PRUNE collapse is the LLM aggregator's fault, NOT the criteria's: the
criteria DO carry the signal; one holistic pass can't combine them.

External combiners (both honest, no leakage):
  * LR-OOF : logistic regression, 5-fold out-of-fold predict_proba (best linear combiner; can downweight
             noisy criteria -> monotone-ish in features, so it should NOT PRUNE).
  * mean   : parameter-free mean of the solo P(YES) (the dumbest possible combiner).
R_A uses vinfo.tvd_recovery (== bfc's subset_R, verified termwise). All values are RAW I_TVD (cap 1/2).

Run (sk3, zero GPU):
  HOME=/lfs/skampere3/0/alexspan /lfs/.../miniconda3/bin/python -m \
      methods.metric_implementer.experiments.external_aggregator
"""
from __future__ import annotations

import glob
import itertools
import os

import numpy as np

from .. import vinfo

GR = "outputs/metric_implementer_scale/gamma_runs"


# ---- joint categorical I_TVD ceiling (binary matrix -> 2^|S| cells) ----------------------------
def tvd_mi_joint(M, Xbin):
    """I_TVD(M; X_S) = 1/2 Σ_{m,x} |P(m,x) - P(m)P(x)|, M binary, X_S a binary matrix (joint code)."""
    M = M.astype(int)
    N = len(M)
    code = np.zeros(N, int)
    for j in range(Xbin.shape[1]):
        code = code * 2 + Xbin[:, j].astype(int)
    tvd = 0.0
    for m in (0, 1):
        pm = (M == m).mean()
        for x in np.unique(code):
            tvd += abs(((M == m) & (code == x)).mean() - pm * (code == x).mean())
    return 0.5 * tvd


def _median_split(v):
    v = np.asarray(v, float)
    thr = np.median(v)
    out = (v > thr).astype(int)
    if out.sum() in (0, len(out)):              # ties pile on the median -> fall back to >=
        out = (v >= thr).astype(int)
    return out


def _oof_logreg(X, y, seed=0):
    """5-fold out-of-fold predict_proba of a logistic regression. Continuous verdict in [0,1]."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    if X.shape[1] == 0:
        return np.full(len(y), y.mean())
    oof = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        if np.unique(y[tr]).size < 2:
            oof[te] = y[tr].mean()
            continue
        lr = LogisticRegression(C=1.0, max_iter=2000)
        lr.fit(X[tr], y[tr])
        oof[te] = lr.predict_proba(X[te])[:, 1]
    return oof


def _R(mhat_col, M):
    return float(vinfo.tvd_recovery(np.asarray(mhat_col, float), M)["tvd_recovery"])


def _load(bf):
    z = np.load(bf, allow_pickle=True)
    M = z["M"].astype(int)
    mhat = z["mhat"]                                   # (n_subsets, N) continuous P(YES)
    crits = list(z["crits"])
    K = len(crits)
    keys = [tuple(int(x) for x in k.split(",")) if k else () for k in z["subset_keys"]]
    row = {k: i for i, k in enumerate(keys)}           # subset-tuple -> mhat row
    # per-criterion solo continuous verdicts (LLM judging each criterion ALONE)
    Xsolo = np.column_stack([mhat[row[(i,)]] for i in range(K)])
    return M, mhat, crits, K, row, Xsolo


def _name(bf):
    return os.path.basename(bf)[:-4].replace("bfc_criteria_", "").replace("bfc_steps_", "").replace("bfc_", "")


def main():
    bfs = sorted(b for b in glob.glob(f"{GR}/bfc_*.npz") if "_steps_" not in os.path.basename(b))
    print("=" * 100)
    print("EXTERNAL-AGGREGATOR test — combine per-criterion solo verdicts OUTSIDE the LLM vs one holistic verdict")
    print("All paired inside each bfc file (same items, same M, same LLM calls). RAW I_TVD, cap 1/2. ZERO GPU.\n")

    rows = []
    for bf in bfs:
        m = _name(bf)
        M, mhat, crits, K, row, Xsolo = _load(bf)
        Xbin = np.column_stack([_median_split(Xsolo[:, i]) for i in range(K)])
        all_keys = list(row.keys())
        # the LLM's own best subset (argmax R_A)
        RA_all = {S: _R(mhat[row[S]], M) for S in all_keys}
        bestA = max(RA_all, key=lambda S: RA_all[S])
        full = tuple(range(K))

        print(f"##### {m}   (K={K}, N={len(M)}, M base-rate={M.mean():.2f}) #####")
        print(f"{'subset':>14s} | {'|S|':>3s} | {'Iceil':>7s}{'R_B(LR)':>9s}{'R_B(mean)':>10s}{'R_A(LLM)':>9s}"
              f" | {'plumbing(R_B-R_A)':>17s}{'compress(Ice-R_B)':>18s}")

        def emit(S, tag):
            S = tuple(sorted(S))
            RA = RA_all[S]
            yhat_lr = _oof_logreg(Xsolo[:, list(S)], M)
            RB_lr = _R(yhat_lr, M)
            RB_mean = _R(Xsolo[:, list(S)].mean(axis=1), M)
            Ice = tvd_mi_joint(M, Xbin[:, list(S)])
            print(f"{tag:>14s} | {len(S):>3d} | {Ice:>7.3f}{RB_lr:>9.3f}{RB_mean:>10.3f}{RA:>9.3f}"
                  f" | {RB_lr - RA:>+17.3f}{Ice - RB_lr:>+18.3f}")
            return dict(metric=m, K=K, tag=tag, size=len(S), Iceil=Ice, RB_lr=RB_lr, RB_mean=RB_mean, RA=RA)

        # R_A's best subset PER SIZE k=1..K  -> shows R_A peaking then PRUNE-declining vs monotone R_B
        for k in range(1, K + 1):
            Sk = max((S for S in all_keys if len(S) == k), key=lambda S: RA_all[S])
            tag = f"k={k}" + ("*bestA" if Sk == bestA else "") + ("*FULL" if Sk == full else "")
            rows.append(emit(Sk, tag))
        if full not in [tuple(sorted(max((S for S in all_keys if len(S) == k), key=lambda S: RA_all[S]))) for k in range(1, K + 1)]:
            rows.append(emit(full, "FULL"))
        print()

    # ---- headline: FULL-set rows (no subset selection) — fairest single comparison ----
    print("=" * 100)
    print("HEADLINE — FULL criterion set (all K, no subset cherry-picking):")
    print(f"{'metric':40s}{'K':>3s} | {'Iceil':>7s}{'R_B(LR)':>9s}{'R_A(LLM)':>9s} | "
          f"{'plumbing':>9s}{'rescued%':>9s}")
    for r in rows:
        if r["size"] != r["K"]:
            continue
        resc = 100 * (r["RB_lr"] - r["RA"]) / r["RA"] if r["RA"] > 1e-6 else float("nan")
        print(f"{r['metric'][:40]:40s}{r['K']:>3d} | {r['Iceil']:>7.3f}{r['RB_lr']:>9.3f}{r['RA']:>9.3f} | "
              f"{r['RB_lr'] - r['RA']:>+9.3f}{resc:>8.0f}%")


if __name__ == "__main__":
    main()
