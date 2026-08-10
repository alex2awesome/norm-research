#!/usr/bin/env python3
"""Stage 1.3/1.4 for any map-focused cell: VA_nl grouped-OOF inside FIT+MINE, then
the round-r disagreement slice the proposer fleet reads.

Round r>=1 rebuilds the OOF on the CURRENT bank (base V+A plus every A-routed
criterion accepted in rounds 1..r), exactly as the pilot's stage1_round{2,3,4}.py
did, so the slice tracks what the scorecard still misses.

LABEL-BLIND: the emitted slice carries text + both models' percentile ranks only,
never `judgement`.

CPU only.  Usage: python stage1_slice.py --cell cap_crowd --round 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import cells as C
import closure_core as L

HERE = Path(__file__).resolve().parent
N_SLICE = 60


def current_blocks(d, rnd):
    """[V, A] plus the A-routed score columns accepted in rounds 1..rnd-1."""
    blocks = [d["V"], d["A"]]
    tags = ["V", "A_base"]
    seq = [] if str(rnd) == "d" else ["d"] + list(range(1, int(rnd)))
    for r in seq:
        f = HERE / f"{d['tag']}_r{r}_scores.npz"
        rt = HERE / f"{d['tag']}_r{r}_routing_final.json"
        if not (f.exists() and rt.exists()):
            continue
        z = np.load(f, allow_pickle=True)
        cids = [str(s) for s in z["crit_ids"]]
        routing = json.loads(rt.read_text())
        a_ids = [x["blind_id"] for x in routing["final"] if x["final_route"] == "A"]
        idx = [cids.index(i) for i in a_ids if i in cids]
        if idx:
            blocks.append(z["X"][:, idx])
            tags.append(f"A_round{r}")
    return blocks, tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--round", default="1")
    a = ap.parse_args()

    d = C.load(a.cell)
    sp = json.loads((HERE / f"{a.cell}_splits.json").read_text())
    rows = sp["rows"]
    split = np.array([r["split"] for r in rows])
    mining = np.array([r["in_mining_slice"] for r in rows])
    y, groups = d["y"], d["groups"]
    fit = split == "fit_mine"

    blocks, tags = current_blocks(d, a.round)
    parts = []
    for M in blocks:
        keep, med = L.clean_fit(M[fit])
        if len(keep):
            parts.append(L.clean_apply(M[fit], keep, med))
    Xfit = np.column_stack(parts)
    gfit, yfit = groups[fit], y[fit]
    print(f"[{a.cell} r{a.round}] FIT+MINE n={fit.sum()} blocks={tags} feats={Xfit.shape[1]}")

    lin_oof = L.linear_oof(Xfit, yfit, gfit)
    oofs = []
    for s in L.SEEDS:
        o, picks = L.gbm_oof(Xfit, yfit, gfit, seed=s)
        oofs.append(o)
        print(f"   VA_nl seed {s}: OOF AUC(fit+mine) {L.auc(yfit, o):.4f} picks={picks}")
    oof_mean = np.mean(oofs, axis=0)
    internal = {
        "cell": a.cell, "round": a.round, "blocks": tags,
        "n_fit_mine": int(fit.sum()), "n_features": int(Xfit.shape[1]),
        "VA_lin_oof_fitmine": L.auc(yfit, lin_oof),
        "VA_nl_oof_fitmine_per_seed": [L.auc(yfit, o) for o in oofs],
        "VA_nl_oof_fitmine_mean_of_preds": L.auc(yfit, oof_mean),
    }
    np.savez(HERE / f"{a.cell}_r{a.round}_oof_fitmine.npz",
             idx=np.where(fit)[0], va_nl_oof_mean=oof_mean,
             va_nl_oof_seeds=np.array(oofs), va_lin_oof=lin_oof)
    (HERE / f"{a.cell}_r{a.round}_oof_fitmine.json").write_text(json.dumps(internal, indent=1))
    print(json.dumps(internal, indent=1))

    # ------------------------------------------------------- disagreement --
    dense = d["dense"]
    assert dense is not None and np.isfinite(dense[mining]).all(), f"{a.cell}: dense missing on mining slice"
    oof_full = np.full(len(y), np.nan)
    oof_full[np.where(fit)[0]] = oof_mean

    mi = np.where(mining)[0]
    d_rank = pd.Series(dense[mi]).rank(pct=True).values
    v_rank = pd.Series(oof_full[mi]).rank(pct=True).values
    gap = d_rank - v_rank
    order_hi = np.argsort(-gap)
    order_lo = np.argsort(gap)
    half = N_SLICE // 2
    picked = list(order_hi[:half]) + list(order_lo[:half])

    trunc = d["meta"]["text_trunc"]
    slice_rows = []
    for k in picked:
        i = int(mi[k])
        slice_rows.append({
            "i": i, "id": str(d["ids"][i]),
            "direction": "dense_high_va_low" if gap[k] > 0 else "dense_low_va_high",
            "dense_prob": float(dense[i]), "dense_pct": float(d_rank[k]),
            "va_nl_oof": float(oof_full[i]), "va_nl_pct": float(v_rank[k]),
            "rank_gap": float(gap[k]),
            "text": (d["texts"][i] or "")[:trunc],
        })
    out = HERE / f"{a.cell}_r{a.round}_slice.json"
    out.write_text(json.dumps(slice_rows, indent=1))
    print(f"slice: {len(slice_rows)} rows |M|={int(mining.sum())} median|gap|="
          f"{np.median([abs(r['rank_gap']) for r in slice_rows]):.3f} -> {out.name}")


if __name__ == "__main__":
    main()
