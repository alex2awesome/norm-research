#!/usr/bin/env python3
"""Layer-3 closure Stage 1.3/1.4 (peer VERDICT): VA_nl OOF inside FIT+MINE, then the
disagreement slice used by the round-1 proposers.

Prereg: the existing results/peer_verdict_va_nl_oof_seed0.npy was fit on the FULL
6,030-row population, which straddles the MONITOR split.  This recomputes the
nonlinear V+A stack with grouped OOF *within FIT+MINE only* (frozen Layer-1 spec,
seeds 0/1/2 mean per FREEZE CHANGE 1), so OOF predictions exist for the mining
slice M = FIT+MINE INTERSECT dense-held-out without any MONITOR leakage.

Disagreement slice = top-|dense_prob - VA_nl_OOF| rows within M, up to 60, balanced
across the two directions.  LABEL-BLIND: the emitted slice file carries text, dense
prob and VA_nl prediction only -- never `judgement`.

CPU only.  Usage: python stage1_disagreement.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import closure_lib as L

HERE = Path(__file__).resolve().parent
N_SLICE = 60


def main():
    pop = L.load_population()
    summary, split, dsplit, mining = L.load_splits()
    y, nt = pop["y"], pop["ntitle"]

    fit = split == "fit_mine"
    keep_v, med_v = L.clean_fit(pop["V"][fit])
    keep_a, med_a = L.clean_fit(pop["A"][fit])
    VA_fit = np.column_stack(
        [L.clean_apply(pop["V"][fit], keep_v, med_v), L.clean_apply(pop["A"][fit], keep_a, med_a)]
    )
    print(f"FIT+MINE n={fit.sum()} V={len(keep_v)}c A={len(keep_a)}c VA={VA_fit.shape[1]}c")

    gfit, yfit = nt[fit], y[fit]
    lin_oof = L.linear_oof(VA_fit, yfit, gfit)
    oofs, picks_all = [], []
    for s in L.SEEDS:
        o, picks = L.gbm_oof(VA_fit, yfit, gfit, seed=s)
        oofs.append(o)
        picks_all.append(picks)
        print(f"  VA_nl seed {s}: OOF AUC(fit+mine) {L.auc(yfit, o):.4f}  picks={picks}")
    oof_mean = np.mean(oofs, axis=0)

    internal = {
        "n_fit_mine": int(fit.sum()),
        "VA_lin_oof_fitmine": L.auc(yfit, lin_oof),
        "VA_nl_oof_fitmine_per_seed": [L.auc(yfit, o) for o in oofs],
        "VA_nl_oof_fitmine_mean_of_preds": L.auc(yfit, oof_mean),
        "grid_picks": picks_all,
        "n_features": {"V": int(len(keep_v)), "A": int(len(keep_a)), "VA": int(VA_fit.shape[1])},
    }
    print(json.dumps(internal, indent=2))

    np.savez(
        HERE / "peer_verdict_oof_fitmine.npz",
        idx=np.where(fit)[0],
        ntitle=nt[fit],
        va_nl_oof_mean=oof_mean,
        va_nl_oof_seeds=np.array(oofs),
        va_lin_oof=lin_oof,
        keep_v=keep_v,
        keep_a=keep_a,
        med_v=med_v,
        med_a=med_a,
    )
    (HERE / "peer_verdict_oof_fitmine.json").write_text(json.dumps(internal, indent=2))

    # ---------------------------------------------------------- disagreement --
    dp = pd.read_csv(HERE / "peer_verdict_dense_preds.csv").set_index("i")
    dense = dp.loc[np.arange(len(y)), "dense_prob"].values

    # rank dense probs to a comparable scale (VA_nl OOF is a probability on a
    # different calibration); both mapped to within-M percentile ranks.
    idx_fit = np.where(fit)[0]
    oof_full = np.full(len(y), np.nan)
    oof_full[idx_fit] = oof_mean

    M = mining.astype(bool)
    mi = np.where(M)[0]
    d_rank = pd.Series(dense[mi]).rank(pct=True).values
    v_rank = pd.Series(oof_full[mi]).rank(pct=True).values
    gap = d_rank - v_rank

    order_hi = mi[np.argsort(-gap)]  # dense-high / VA-low
    order_lo = mi[np.argsort(gap)]  # dense-low / VA-high
    half = N_SLICE // 2
    picked = list(order_hi[:half]) + list(order_lo[:half])

    slice_rows = []
    for i in picked:
        k = int(np.where(mi == i)[0][0])
        slice_rows.append(
            {
                "i": int(i),
                "direction": "dense_high_va_low" if gap[k] > 0 else "dense_low_va_high",
                "dense_prob": float(dense[i]),
                "dense_pct": float(d_rank[k]),
                "va_nl_oof": float(oof_full[i]),
                "va_nl_pct": float(v_rank[k]),
                "rank_gap": float(gap[k]),
                "text": pop["texts"][i],
            }
        )
    (HERE / "round1_disagreement_slice.json").write_text(json.dumps(slice_rows, indent=1))
    print(
        f"disagreement slice: {len(slice_rows)} rows "
        f"({sum(r['direction'] == 'dense_high_va_low' for r in slice_rows)} dense-high, "
        f"{sum(r['direction'] == 'dense_low_va_high' for r in slice_rows)} dense-low); "
        f"|M|={int(M.sum())}"
    )
    print("median |rank gap| in slice:", float(np.median([abs(r["rank_gap"]) for r in slice_rows])))


if __name__ == "__main__":
    main()
