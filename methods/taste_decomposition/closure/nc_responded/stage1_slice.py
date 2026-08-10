#!/usr/bin/env python3
"""Stage 1 of any round: the disagreement slice (N&C RESPONDED).

Mines against the CURRENT bank state (V + 198-rubric A bank + A-routed criteria of
all previous rounds).  VA_nl is refit with grouped OOF inside FIT+MINE only (frozen
Layer-1 spec, seeds 0/1/2 mean), so the disagreement read is the residual the
previous rounds left.

Slice = top |dense percentile rank - VA_nl percentile rank| inside the mining slice
M = FIT+MINE and dense-held-out, 30 per direction, EXCLUDING every row read in a
previous round (pilot rule).

LABEL-BLIND: the emitted slice carries text, dense prob and VA_nl only -- never y.

Usage: python stage1_slice.py --round 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import nc_closure_lib as L
from readout import load_round_scores, load_dense

HERE = Path(__file__).resolve().parent
N_SLICE = 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    a = ap.parse_args()
    r = a.round

    pop = L.load_population()
    summary, split, dsplit, mining, monitor_full = L.load_splits()
    y, docket = pop["y"], pop["docket"]
    fit = split == "fit_mine"

    gfit, yfit = docket[fit], y[fit]
    cached = HERE / f"state{r - 1}_preds.npz"
    if cached.exists():
        # stage4_curve.py already fit this exact state under the frozen spec and saved
        # its FIT+MINE OOF predictions; reusing them guarantees the slice is mined
        # against the SAME numbers the curve reports (and costs no refit).
        z = np.load(cached, allow_pickle=True)
        oof_mean = z["nl_oof_fm"]
        lin_oof = z["lin_oof_fm"]
        n_feat = int(len(z["kept_names"]))
        oofs = None
        print(f"round-{r-1} bank: reusing state{r-1}_preds.npz "
              f"(n={fit.sum()} features={n_feat})", flush=True)
    else:
        blocks = [pop["V"], pop["A"]]
        prev_rounds = list(range(1, r))
        if prev_rounds:
            Xr, _ = load_round_scores(prev_rounds)
            if Xr is not None:
                blocks.append(Xr)
        X = np.column_stack(blocks)
        keep, med = L.clean_fit(X[fit])
        Xfit = L.clean_apply(X[fit], keep, med)
        n_feat = int(Xfit.shape[1])
        print(f"round-{r-1} bank inside FIT+MINE: n={fit.sum()} features={n_feat}", flush=True)
        oofs = []
        for s in L.SEEDS:
            o, picks = L.gbm_oof(Xfit, yfit, gfit, seed=s)
            oofs.append(o)
            print(f"  VA_nl seed {s}: OOF AUC {L.auc(yfit, o):.4f} picks={picks}", flush=True)
        oof_mean = np.mean(oofs, axis=0)
        lin_oof = L.linear_oof(Xfit, yfit, gfit)

    internal = {
        "round": r,
        "n_fit_mine": int(fit.sum()),
        "n_features_prev_bank": n_feat,
        "source": "state_preds_cache" if cached.exists() else "refit",
        "VA_lin_oof_fitmine": L.auc(yfit, lin_oof),
        "VA_nl_oof_fitmine_per_seed": ([L.auc(yfit, o) for o in oofs] if oofs else None),
        "VA_nl_oof_fitmine_meanpred": L.auc(yfit, oof_mean),
    }
    print(json.dumps(internal, indent=2))
    (HERE / f"round{r}_oof_fitmine.json").write_text(json.dumps(internal, indent=2))

    dense = load_dense()
    oof_full = np.full(len(y), np.nan)
    oof_full[np.where(fit)[0]] = oof_mean

    Mm = mining.astype(bool)
    mi = np.where(Mm)[0]
    d_rank = pd.Series(dense[mi]).rank(pct=True).values
    v_rank = pd.Series(oof_full[mi]).rank(pct=True).values
    gap = d_rank - v_rank

    prev = set()
    for q in range(1, r):
        f = HERE / f"round{q}_disagreement_slice.json"
        if f.exists():
            prev |= {row["i"] for row in json.loads(f.read_text())}

    order_hi = [i for i in mi[np.argsort(-gap)] if int(i) not in prev]
    order_lo = [i for i in mi[np.argsort(gap)] if int(i) not in prev]
    half = N_SLICE // 2
    picked = list(order_hi[:half]) + list(order_lo[:half])

    rows = []
    for i in picked:
        k = int(np.where(mi == i)[0][0])
        rows.append({
            "i": int(i),
            "doc_id": str(pop["doc_id"][i]),
            "direction": "dense_high_va_low" if gap[k] > 0 else "dense_low_va_high",
            "dense_prob": float(dense[i]),
            "dense_pct": float(d_rank[k]),
            "va_nl": float(oof_full[i]),
            "va_nl_pct": float(v_rank[k]),
            "rank_gap": float(gap[k]),
            "text": pop["texts"][i],
        })
    (HERE / f"round{r}_disagreement_slice.json").write_text(json.dumps(rows, indent=1))
    print(f"round-{r} slice: {len(rows)} rows "
          f"({sum(x['direction'] == 'dense_high_va_low' for x in rows)} dense-high / "
          f"{sum(x['direction'] == 'dense_low_va_high' for x in rows)} dense-low); "
          f"|M|={int(Mm.sum())}; excluded {len(prev)} previously-read rows")
    print("median |rank gap|:", float(np.median([abs(x["rank_gap"]) for x in rows])))


if __name__ == "__main__":
    main()
