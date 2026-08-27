#!/usr/bin/env python3
"""Layer-3 closure ROUND 4, Stage 1 (peer VERDICT): updated disagreement slice.

Round 4 mines against the ROUND-3 bank: V + the 154-criterion A bank + the round-1,
round-2 and round-3 A criteria (139 features after the screen).
VA_nl is refit with grouped OOF inside FIT+MINE only (frozen Layer-1 spec, seeds
0/1/2 mean), so the disagreement read is the residual that rounds 1 and 2 left.

Split unchanged from rounds 1-3 so the consecutive-round MONITOR VA_nl gains stay
apples-to-apples (Amendment 2).  Rows read in rounds 1, 2 AND 3 are excluded.

LABEL-BLIND: emitted slice carries text, dense prob and round-3 VA_nl only.

CPU only.  Usage: python stage1_round3.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import closure_lib as L
from stage4_readout import build_blocks
from stage4_round3 import load_round_blocks

HERE = Path(__file__).resolve().parent
N_SLICE = 60


def main():
    pop, split, dsplit, XA1, XB1, a1_ids, b1_ids, summary = build_blocks()
    XA2, XB2, a2_ids, b2_ids, _ = load_round_blocks(2)
    XA3, XB3, a3_ids, b3_ids, _ = load_round_blocks(3)
    y, nt = pop["y"], pop["ntitle"]
    fit = split == "fit_mine"
    _, _, _, mining = L.load_splits()

    parts = []
    for M in (pop["V"], pop["A"], XA1, XA2, XA3):
        keep, med = L.clean_fit(M[fit])
        parts.append(L.clean_apply(M[fit], keep, med))
    Xfit = np.column_stack(parts)
    gfit, yfit = nt[fit], y[fit]
    print(f"round-3 bank inside FIT+MINE: n={fit.sum()} features={Xfit.shape[1]}")

    oofs = []
    for s in L.SEEDS:
        o, picks = L.gbm_oof(Xfit, yfit, gfit, seed=s)
        oofs.append(o)
        print(f"  VA_nl(round3 bank) seed {s}: OOF AUC {L.auc(yfit, o):.4f} picks={picks}")
    oof_mean = np.mean(oofs, axis=0)
    lin_oof = L.linear_oof(Xfit, yfit, gfit)
    internal = {
        "n_fit_mine": int(fit.sum()),
        "n_features_round3_bank": int(Xfit.shape[1]),
        "VA_lin_oof_fitmine_round3bank": L.auc(yfit, lin_oof),
        "VA_nl_oof_fitmine_round3bank_per_seed": [L.auc(yfit, o) for o in oofs],
        "VA_nl_oof_fitmine_round3bank_meanpred": L.auc(yfit, oof_mean),
    }
    print(json.dumps(internal, indent=2))
    (HERE / "round4_oof_fitmine.json").write_text(json.dumps(internal, indent=2))

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

    prev = set()
    for f in ("round1_disagreement_slice.json", "round2_disagreement_slice.json",
              "round3_disagreement_slice.json"):
        prev |= {r["i"] for r in json.loads((HERE / f).read_text())}

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
            "va_nl_round3": float(oof_full[i]),
            "va_nl_pct": float(v_rank[k]),
            "rank_gap": float(gap[k]),
            "text": pop["texts"][i],
        })
    (HERE / "round4_disagreement_slice.json").write_text(json.dumps(rows, indent=1))
    print(f"round-4 slice: {len(rows)} rows "
          f"({sum(r['direction'] == 'dense_high_va_low' for r in rows)} dense-high / "
          f"{sum(r['direction'] == 'dense_low_va_high' for r in rows)} dense-low); "
          f"|M|={int(Mm.sum())}; excluded {len(prev)} previously-read rows")
    print("median |rank gap|:", float(np.median([abs(r["rank_gap"]) for r in rows])))


if __name__ == "__main__":
    main()
