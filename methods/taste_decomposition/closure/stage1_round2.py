#!/usr/bin/env python3
"""Layer-3 closure ROUND 2, Stage 1 (peer VERDICT): updated disagreement slice.

Round 2 mines against the ROUND-1 bank, i.e. V + the 154-criterion A bank + the 14
A-routed criteria that round 1 added.  VA_nl is refit with grouped OOF inside
FIT+MINE only (frozen Layer-1 spec, seeds 0/1/2 mean), so the disagreement being
read is the residual disagreement that round 1 did NOT close.

Split is UNCHANGED from round 1 so that the round-over-round MONITOR VA_nl gain --
the quantity the saturation rule is defined on -- stays apples-to-apples.  (The
prereg amendment's "MONITOR inside dense-held-out" rule governs confirmatory cells;
for the pilot the honest level is reported on the 1,244 dense-held-out rows instead.)

LABEL-BLIND: the emitted slice carries text, dense prob and round-1 VA_nl only.

CPU only.  Usage: python stage1_round2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import closure_lib as L
from stage4_readout import build_blocks

HERE = Path(__file__).resolve().parent
N_SLICE = 60


def main():
    pop, split, dsplit, XA1, XB1, a_ids, b_ids, summary = build_blocks()
    y, nt = pop["y"], pop["ntitle"]
    fit = split == "fit_mine"
    _, _, _, mining = L.load_splits()

    # round-1 bank = V + A154 + the 14 round-1 A criteria
    parts_fit = []
    for M in (pop["V"], pop["A"], XA1):
        keep, med = L.clean_fit(M[fit])
        parts_fit.append(L.clean_apply(M[fit], keep, med))
    Xfit = np.column_stack(parts_fit)
    gfit, yfit = nt[fit], y[fit]
    print(f"round-1 bank inside FIT+MINE: n={fit.sum()} features={Xfit.shape[1]}")

    oofs = []
    for s in L.SEEDS:
        o, picks = L.gbm_oof(Xfit, yfit, gfit, seed=s)
        oofs.append(o)
        print(f"  VA_nl(round1 bank) seed {s}: OOF AUC(fit+mine) {L.auc(yfit, o):.4f} picks={picks}")
    oof_mean = np.mean(oofs, axis=0)
    lin_oof = L.linear_oof(Xfit, yfit, gfit)
    internal = {
        "n_fit_mine": int(fit.sum()),
        "n_features_round1_bank": int(Xfit.shape[1]),
        "VA_lin_oof_fitmine_round1bank": L.auc(yfit, lin_oof),
        "VA_nl_oof_fitmine_round1bank_per_seed": [L.auc(yfit, o) for o in oofs],
        "VA_nl_oof_fitmine_round1bank_meanpred": L.auc(yfit, oof_mean),
    }
    print(json.dumps(internal, indent=2))
    (HERE / "round2_oof_fitmine.json").write_text(json.dumps(internal, indent=2))
    np.savez(HERE / "round2_oof_fitmine.npz", idx=np.where(fit)[0],
             va_nl_oof_mean=oof_mean, va_lin_oof=lin_oof)

    # ------------------------------------------------------- disagreement slice --
    dense = pd.read_csv(HERE / "peer_verdict_dense_preds.csv").set_index("i").loc[
        np.arange(len(y)), "dense_prob"
    ].values
    oof_full = np.full(len(y), np.nan)
    oof_full[np.where(fit)[0]] = oof_mean

    Mm = mining.astype(bool)
    mi = np.where(Mm)[0]
    d_rank = pd.Series(dense[mi]).rank(pct=True).values
    v_rank = pd.Series(oof_full[mi]).rank(pct=True).values
    gap = d_rank - v_rank

    # exclude rows already read in round 1 -- round 2 must look at NEW text
    prev = {r["i"] for r in json.loads((HERE / "round1_disagreement_slice.json").read_text())}
    order_hi = [i for i in mi[np.argsort(-gap)] if int(i) not in prev]
    order_lo = [i for i in mi[np.argsort(gap)] if int(i) not in prev]
    half = N_SLICE // 2
    picked = list(order_hi[:half]) + list(order_lo[:half])

    rows = []
    for i in picked:
        k = int(np.where(mi == i)[0][0])
        rows.append({
            "i": int(i),
            "direction": "dense_high_va_low" if gap[k] > 0 else "dense_low_va_high",
            "dense_prob": float(dense[i]),
            "dense_pct": float(d_rank[k]),
            "va_nl_round1": float(oof_full[i]),
            "va_nl_pct": float(v_rank[k]),
            "rank_gap": float(gap[k]),
            "text": pop["texts"][i],
        })
    (HERE / "round2_disagreement_slice.json").write_text(json.dumps(rows, indent=1))
    print(f"round-2 slice: {len(rows)} rows, "
          f"{sum(r['direction'] == 'dense_high_va_low' for r in rows)} dense-high / "
          f"{sum(r['direction'] == 'dense_low_va_high' for r in rows)} dense-low; "
          f"|M|={int(Mm.sum())}; excluded {len(prev)} round-1 rows")
    print("median |rank gap|:", float(np.median([abs(r["rank_gap"]) for r in rows])))


if __name__ == "__main__":
    main()
