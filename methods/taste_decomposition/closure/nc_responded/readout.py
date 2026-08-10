#!/usr/bin/env python3
"""Frozen per-round readout for the N&C RESPONDED closure campaign.

One function, `fit_state`, computes every quantity the prereg asks for at a given
bank state (round 0 = the incoming 198-rubric bank; round r = plus the A-routed
criteria of rounds 1..r).  It is called identically for every state so the curve
is estimator-identical by construction.

Readouts (pre-declared in build_splits_nc.py's docstring):
  * VA_lin / VA_nl on MONITOR_FULL (n=1,892)  -> SATURATION STATISTIC (VA-honest)
  * VA_lin / VA_nl on MONITOR      (n=377)    -> Delta_r level (T also honest)
  * VA_nl on the honest population (all dense-held-out rows, n=1,904): OOF inside
    FIT+MINE, refit-predict on the MONITOR side -> better-powered, mildly
    mining-contaminated, therefore CONSERVATIVE (understates Delta).

Every fit is inside FIT+MINE only; MONITOR rows never touch a fit, a degeneracy
screen, or an imputation median.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import nc_closure_lib as L

HERE = Path(__file__).resolve().parent


def load_round_scores(rounds):
    """Concatenate the A-routed score blocks of the given rounds (in order)."""
    mats, names = [], []
    for r in rounds:
        p = HERE / f"round{r}_scores.npz"
        if not p.exists():
            raise FileNotFoundError(p)
        z = np.load(p, allow_pickle=True)
        routed = json.loads((HERE / f"round{r}_routing_final.json").read_text())
        keep_ids = {c["id"] for c in routed["A"]}
        gate = json.loads((HERE / f"round{r}_score_report.json").read_text())
        cids = [str(s) for s in z["crit_ids"]]
        cnames = [str(s) for s in z["crit_names"]]
        cols, nms = [], []
        for k, cid in enumerate(cids):
            if cid not in keep_ids:
                continue
            if gate["per_criterion"][cid]["collapsed"]:
                continue
            cols.append(z["X"][:, k])
            nms.append(f"r{r}:{cid}:{cnames[k]}")
        if cols:
            mats.append(np.column_stack(cols))
            names += nms
    if not mats:
        return None, []
    return np.column_stack(mats), names


def fit_state(rounds, dense_prob=None, tag="", n_boot=2000, save_preds=None):
    pop = L.load_population()
    summary, split, dsplit, mining, monitor_full = L.load_splits()
    y, docket = pop["y"], pop["docket"]

    fm = split == "fit_mine"
    mon = split == "monitor"
    monf = monitor_full
    heldout = np.isin(dsplit, ["eval", "test"])

    blocks = [pop["V"], pop["A"]]
    names = [f"V:{n}" for n in pop["v_names"]] + [f"A:{n}" for n in pop["a_names"]]
    if rounds:
        Xr, nr = load_round_scores(rounds)
        if Xr is not None:
            blocks.append(Xr)
            names += nr
    X = np.column_stack(blocks)

    keep, meds = L.clean_fit(X[fm])
    Xc = L.clean_apply(X, keep, meds)
    kept_names = [names[j] for j in keep]

    Xfm, yfm, gfm = Xc[fm], y[fm], docket[fm]

    # OOF inside FIT+MINE (used for the mining slice + the honest-level readout)
    lin_oof = L.linear_oof(Xfm, yfm, gfm)
    nl_oof_seeds = []
    picks_oof = []
    for s in L.SEEDS:
        o, pk = L.gbm_oof(Xfm, yfm, gfm, seed=s)
        nl_oof_seeds.append(o)
        picks_oof.append(pk)
    nl_oof = np.mean(nl_oof_seeds, axis=0)

    # refit on ALL of FIT+MINE, predict the monitor side
    lin_all = np.full(len(y), np.nan)
    nl_all = np.full((len(L.SEEDS), len(y)), np.nan)
    lin_m, nl_m, picks_mon = L.fit_predict_monitor(Xfm, yfm, gfm, Xc[monf])
    lin_all[monf] = lin_m
    nl_all[:, monf] = nl_m
    lin_all[fm] = lin_oof
    for i, s in enumerate(L.SEEDS):
        nl_all[i, fm] = nl_oof_seeds[i]
    nl_mean = np.nanmean(nl_all, axis=0)

    res = {
        "tag": tag, "rounds": list(rounds), "n_features": int(Xc.shape[1]),
        "n_features_pre_screen": int(X.shape[1]),
        "n_new_cols": int(Xc.shape[1] - 0),
        "picks_monitor": picks_mon,
    }

    def readout(mask, label):
        yy, pl, pn = y[mask], lin_all[mask], nl_mean[mask]
        seeds = [L.auc(yy, nl_all[i][mask]) for i in range(len(L.SEEDS))]
        d = {
            "n": int(mask.sum()),
            "VA_lin": L.auc(yy, pl),
            "VA_nl": L.auc(yy, pn),
            "VA_nl_seed_aucs": seeds,
            "VA_nl_seed_spread": float(max(seeds) - min(seeds)),
            "Delta_interact": L.auc(yy, pn) - L.auc(yy, pl),
        }
        if dense_prob is not None:
            hm = mask & heldout
            if hm.sum() > 30:
                d["T"] = L.auc(y[hm], dense_prob[hm])
                d["VA_nl_on_T_rows"] = L.auc(y[hm], nl_mean[hm])
                d["Delta"] = d["T"] - d["VA_nl_on_T_rows"]
                d["n_T_rows"] = int(hm.sum())
                d["rho_VAnl_dense"] = float(
                    np.corrcoef(np.argsort(np.argsort(nl_mean[hm])),
                                np.argsort(np.argsort(dense_prob[hm])))[0, 1])
        res[label] = d
        return d

    readout(monf, "monitor_full")
    readout(mon, "monitor")
    readout(heldout, "honest")

    if save_preds:
        np.savez_compressed(HERE / save_preds, lin=lin_all, nl_seeds=nl_all,
                            nl_mean=nl_mean, lin_oof_fm=lin_oof, nl_oof_fm=nl_oof,
                            kept_names=np.array(kept_names, dtype=object))
    return res, {"lin": lin_all, "nl_mean": nl_mean, "nl_seeds": nl_all,
                 "kept_names": kept_names, "fm": fm, "mon": mon, "monf": monf,
                 "heldout": heldout}


def load_dense():
    """Dense probabilities ALIGNED to the local population row order (see
    build_splits_nc.py: the sk3 rescore row order is a permutation of the local
    loader's; the join is by unique doc_id, never positional)."""
    import pandas as pd
    df = pd.read_csv(HERE / "nc_responded_dense_preds_aligned.csv")
    return df["dense_prob"].values
