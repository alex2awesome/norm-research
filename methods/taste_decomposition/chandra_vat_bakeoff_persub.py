#!/usr/bin/env python3
"""PER-SUBREDDIT VAT bake-off + fused stack for the Chandrasekharan cells
(Task A, user order 2026-08-24). Versioned adaptation of vat_bakeoff.py /
unified_fused_stack.py with a --sub filter — the pooled scripts are untouched.

Design (mirrors the pooled discipline exactly):
  * VA  = the PER-SUB refit layer-1 OOF (chandra_layer1_persub.py npz).
  * T   = the POOLED dense model's seed-mean predictions RESTRICTED to the
          sub's held-out rows ("T = pooled-trained, sub-restricted readout").
  * Variants selected on the eval leg, reported on the test leg; VAT column =
    best FUSION variant by eval among {rank_mean, eval_weighted_rank}
    (parents = baselines; logistic_evalfit + fused_stack reported descriptively).

FRAME: v1 populations; era channel open; v2 rescore will supersede.

Usage: chandra_vat_bakeoff_persub.py --cell chandra_humor --sub funny
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path("/lfs/skampere3/0/alexspan/norm-research/methods/taste_decomposition")
sys.path.insert(0, str(HERE))
import unified_fused_stack as U

SEEDS = (42, 1, 2)


def auc_ci(y, p, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) == 2:
            vals.append(roc_auc_score(y[i], p[i]))
    return [round(float(np.percentile(vals, q)), 4) for q in (2.5, 97.5)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=["chandra_humor", "chandra_cw"])
    ap.add_argument("--sub", required=True)
    a = ap.parse_args()
    cfg = U.CELLS[a.cell]

    z = np.load(HERE / "results" / f"{a.cell}_persub_{a.sub}_va_oof.npz",
                allow_pickle=True)
    ids = [str(i) for i in z["ids"]]
    pos = {r: i for i, r in enumerate(ids)}
    assert len(pos) == len(ids), "duplicate ids in per-sub OOF"
    va_all, y_all = z["VA_nl"].astype(float), z["y"].astype(int)
    grp_all = np.array([str(g) for g in z["groups"]], dtype=object)  # pseudo-groups

    legs = {}
    for leg in ("eval", "test"):
        sp = pd.read_csv(cfg["dense"] / "split" / f"{leg}.csv")
        per = []
        for s in SEEDS:
            p = pd.read_csv(cfg["dense"] / f"rm_out_seed{s}" / f"preds_{leg}.csv")
            assert len(p) == len(sp) and \
                (p["judgement"].values == sp["judgement"].values).all(), \
                f"order-join fail {leg} seed{s}"
            per.append(p["prob"].values.astype(float))
        sp["prob"] = np.mean(per, axis=0)
        sp = sp[sp["group"].astype(str) == a.sub].reset_index(drop=True)
        idx = np.array([pos[r] for r in sp["row_id"].astype(str)])
        assert (y_all[idx] == sp["judgement"].astype(int).values).all(), "y mismatch on join"
        legs[leg] = dict(y=y_all[idx], va=va_all[idx], t=sp["prob"].values,
                         g=grp_all[idx])

    def rank01(x):
        return rankdata(x) / (len(x) + 1)

    ev, te = legs["eval"], legs["test"]
    res = {"cell": a.cell, "sub": a.sub,
           "frame": "v1 populations; era channel open; v2 rescore will supersede",
           "T_design": "pooled-trained dense, sub-restricted readout",
           "VA_design": "per-sub refit layer-1 OOF (pseudo-group folds)",
           "n_eval": int(len(ev["y"])), "n_test": int(len(te["y"]))}
    variants = {}
    variants["VA_alone"] = (ev["va"], te["va"])
    variants["T_alone"] = (ev["t"], te["t"])
    variants["rank_mean"] = (rank01(ev["va"]) + rank01(ev["t"]),
                             rank01(te["va"]) + rank01(te["t"]))
    wa = max(roc_auc_score(ev["y"], ev["va"]) - .5, 1e-3)
    wt = max(roc_auc_score(ev["y"], ev["t"]) - .5, 1e-3)
    variants["eval_weighted_rank"] = (wa * rank01(ev["va"]) + wt * rank01(ev["t"]),
                                      wa * rank01(te["va"]) + wt * rank01(te["t"]))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(np.column_stack([ev["va"], ev["t"]]), ev["y"])
    variants["logistic_evalfit"] = (
        clf.predict_proba(np.column_stack([ev["va"], ev["t"]]))[:, 1],
        clf.predict_proba(np.column_stack([te["va"], te["t"]]))[:, 1])

    table = {}
    for nm, (pe, pt) in variants.items():
        table[nm] = {"eval": round(float(roc_auc_score(ev["y"], pe)), 4),
                     "test": round(float(roc_auc_score(te["y"], pt)), 4)}

    # descriptive fused stack (unified_fused_stack pattern) on eval+test combined,
    # grouped-OOF logistic by pseudo-group
    y_c = np.concatenate([ev["y"], te["y"]])
    S = np.column_stack([np.concatenate([ev["va"], te["va"]]),
                         np.concatenate([ev["t"], te["t"]])])
    g_c = np.concatenate([ev["g"], te["g"]])
    leg_c = np.array(["eval"] * len(ev["y"]) + ["test"] * len(te["y"]))
    oof = np.zeros(len(y_c))
    for tr, th in GroupKFold(max(2, min(5, len(np.unique(g_c))))).split(S, groups=g_c):
        c2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        c2.fit(S[tr], y_c[tr])
        oof[th] = c2.predict_proba(S[th])[:, 1]
    table["fused_stack"] = {
        "eval": round(float(roc_auc_score(y_c[leg_c == "eval"], oof[leg_c == "eval"])), 4),
        "test": round(float(roc_auc_score(y_c[leg_c == "test"], oof[leg_c == "test"])), 4)}

    pool = {k: v for k, v in table.items() if k in ("rank_mean", "eval_weighted_rank")}
    winner = max(pool, key=lambda k: pool[k]["eval"])
    res["table"] = table
    res["winner_by_eval"] = winner
    res["VAT_bakeoff_test"] = table[winner]["test"]
    res["winner_test_ci95"] = auc_ci(te["y"], dict(variants)[winner][1])
    res["note"] = ("VAT = best FUSION variant by eval leg within the sub (parents = "
                   "baselines only); logistic_evalfit + fused_stack descriptive")
    out = HERE / "results" / f"{a.cell}_persub_{a.sub}_vat_bakeoff.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    print(f"{a.cell.upper()}_{a.sub}_PERSUB_VAT_DONE", flush=True)


if __name__ == "__main__":
    main()
