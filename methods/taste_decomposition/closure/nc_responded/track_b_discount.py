#!/usr/bin/env python3
"""Track-B discounting + stratification-free controls, N&C RESPONDED.

Readouts (all threshold-free; no residual regressions):

  1. SPURIOUS-ALONE AUC of the declared nuisance set (linear + HistGB), grouped-OOF
     inside FIT+MINE, evaluated on the honest population.
  2. DECILE-STRATIFIED AUC of T and of VA_nl, stratified by the JOINT B-model score
     (and by each individual channel).  Reported as T_adj / VA_adj / Delta_adj.
  3. MATCHED SAMPLING once spurious-alone > .65 (freeze).  Positives are matched to
     negatives on the joint B score by nearest-neighbour caliper within docket-free
     strata; AUC is then read on the matched set.  This replaces stratification when
     the stratifier approaches the label.
  4. STACKED-INCREMENT readout (FREEZE ADDENDUM, stratification-free): AUC(joint B)
     vs AUC(logistic stack of joint B + dense) and vs AUC(logistic stack of joint B
     + VA_nl) -- the increment of each instrument over ALL named channels in one
     scalar.  Does not degenerate as the nuisance set grows.
  5. MIXED-CHANNEL SENSITIVITY BAND (FREEZE ADDENDUM 2): every readout is computed
     twice -- once with ALL B channels in the nuisance set, once with the
     `mixed=true` channels EXCLUDED -- and both are reported as a band, never
     silently collapsed to one side.

Usage: python track_b_discount.py --round 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import nc_closure_lib as L
from readout import load_dense

HERE = Path(__file__).resolve().parent
MATCH_TRIGGER = 0.65


def load_b_blocks(upto):
    """All B-routed, non-collapsed score columns of rounds 1..upto, with tags."""
    cols, meta = [], []
    for r in range(1, upto + 1):
        p = HERE / f"round{r}_scores.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        routed = json.loads((HERE / f"round{r}_routing_final.json").read_text())
        gate = json.loads((HERE / f"round{r}_score_report.json").read_text())
        bmap = {c["id"]: c for c in routed["B"]}
        cids = [str(s) for s in z["crit_ids"]]
        cnames = [str(s) for s in z["crit_names"]]
        for k, cid in enumerate(cids):
            if cid not in bmap or gate["per_criterion"][cid]["collapsed"]:
                continue
            cols.append(z["X"][:, k])
            meta.append({"round": r, "id": cid, "name": cnames[k],
                         "upstream_parent": bmap[cid].get("upstream_parent"),
                         "mixed": bool(bmap[cid].get("mixed"))})
    if not cols:
        return None, []
    # FREEZE ADDENDUM 3: a MIXED parent is RETIRED from the readouts once its
    # decomposed components have been scored (recorded in retired_channels.json,
    # never deleted). Retirement is applied here so every downstream readout --
    # spurious-alone, stratified, matched, stacked -- sees the components instead
    # of the parent, and never both.
    ret = HERE / "retired_channels.json"
    if ret.exists():
        dead = {x["uid"] for x in json.loads(ret.read_text())["retired"]}
        keep = [i for i, m in enumerate(meta) if f"r{m['round']}:{m['id']}" not in dead]
        if len(keep) < len(meta):
            cols = [cols[i] for i in keep]
            meta = [meta[i] for i in keep]
    return np.column_stack(cols), meta


def oof_score(X, y, groups, model):
    folds = list(GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(np.zeros(len(y)), groups=groups))
    oof = np.zeros(len(y))
    for tr, te in folds:
        m = model()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def _lin():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def _gb():
    return HistGradientBoostingClassifier(max_leaf_nodes=31, learning_rate=0.06, max_iter=400,
                                          early_stopping=True, validation_fraction=0.1,
                                          n_iter_no_change=20, random_state=0)


def matched_auc(y, p, s, rng, caliper=0.02, n_rep=200):
    """Pair each positive with the nearest-|s| negative (caliper on the joint-B score),
    then read AUC on the matched pairs.  Threshold-free: the readout is the fraction
    of matched pairs the score orders correctly."""
    P, N = np.where(y == 1)[0], np.where(y == 0)[0]
    sN = s[N]
    order = np.argsort(sN)
    Ns = N[order]
    sNs = sN[order]
    conc, used = [], 0
    for i in P:
        j = np.searchsorted(sNs, s[i])
        best, bd = None, np.inf
        for jj in (j - 1, j, j + 1):
            if 0 <= jj < len(Ns):
                d = abs(sNs[jj] - s[i])
                if d < bd:
                    bd, best = d, Ns[jj]
        if best is None or bd > caliper:
            continue
        used += 1
        conc.append(1.0 if p[i] > p[best] else (0.5 if p[i] == p[best] else 0.0))
    if not conc:
        return float("nan"), 0
    return float(np.mean(conc)), used


def stacked_increment(y, b, other, groups):
    """AUC(joint B) vs AUC(logistic stack of B + other), grouped-OOF."""
    X1 = b.reshape(-1, 1)
    X2 = np.column_stack([b, other])
    a1 = roc_auc_score(y, oof_score(X1, y, groups, _lin))
    a2 = roc_auc_score(y, oof_score(X2, y, groups, _lin))
    return {"auc_B_only": a1, "auc_B_plus": a2, "increment": a2 - a1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    a = ap.parse_args()
    r = a.round

    pop = L.load_population()
    summary, split, dsplit, mining, monitor_full = L.load_splits()
    y, docket = pop["y"], pop["docket"]
    heldout = np.isin(dsplit, ["eval", "test"])
    dense = load_dense()

    XB, meta = load_b_blocks(r)
    if XB is None:
        print("no B blocks yet")
        return

    st = np.load(HERE / f"state{r}_preds.npz", allow_pickle=True)
    va = st["nl_mean"]

    out = {"round": r, "n_channels": XB.shape[1], "channels": meta, "variants": {}}

    for variant, mask_cols in (("all", np.ones(XB.shape[1], bool)),
                               ("ex_mixed", ~np.array([m["mixed"] for m in meta]))):
        if mask_cols.sum() == 0:
            continue
        X = XB[:, mask_cols]
        keep, meds = L.clean_fit(X[split == "fit_mine"])
        Xc = L.clean_apply(X, keep, meds)
        if Xc.shape[1] == 0:
            continue
        h = heldout
        bl = oof_score(Xc[h], y[h], docket[h], _lin)
        bg = oof_score(Xc[h], y[h], docket[h], _gb)
        alone_lin = float(roc_auc_score(y[h], bl))
        alone_gb = float(roc_auc_score(y[h], bg))
        joint = bg if alone_gb >= alone_lin else bl

        strat = L.decile_strata(joint)
        t_adj, t_info = L.stratified_auc(y[h], dense[h], strat)
        v_adj, v_info = L.stratified_auc(y[h], va[h], strat)
        pooled_T = float(roc_auc_score(y[h], dense[h]))
        pooled_V = float(roc_auc_score(y[h], va[h]))

        rec = {
            "n_channels_used": int(Xc.shape[1]),
            "spurious_alone_linear": alone_lin,
            "spurious_alone_histgb": alone_gb,
            "pooled_T": pooled_T, "pooled_VA_nl": pooled_V, "pooled_Delta": pooled_T - pooled_V,
            "q10_joint_B": {"T_adj": t_adj, "VA_adj": v_adj, "Delta_adj": t_adj - v_adj,
                            "strata_info": t_info},
            "stacked_increment_dense_over_B": stacked_increment(y[h], joint, dense[h], docket[h]),
            "stacked_increment_bank_over_B": stacked_increment(y[h], joint, va[h], docket[h]),
        }
        if max(alone_lin, alone_gb) > MATCH_TRIGGER:
            rng = np.random.default_rng(0)
            mt, nt = matched_auc(y[h], dense[h], joint, rng)
            mv, nv = matched_auc(y[h], va[h], joint, rng)
            rec["matched_sampling"] = {
                "triggered_at": MATCH_TRIGGER, "n_matched_pairs_T": nt, "n_matched_pairs_VA": nv,
                "T_matched": mt, "VA_matched": mv, "Delta_matched": mt - mv,
                "caliper": 0.02}
        # per-channel alone AUCs.  clean_fit drops degenerate columns, so column k of
        # Xc is column keep[k] of X, which is column np.where(mask_cols)[0][keep[k]]
        # of the full B matrix -- the double indirection is what names it correctly.
        sub_idx = np.where(mask_cols)[0]
        per = {}
        for k in range(Xc.shape[1]):
            name = meta[sub_idx[keep[k]]]["name"]
            try:
                per[name] = float(roc_auc_score(y[h], Xc[h][:, k]))
            except ValueError:
                pass
        rec["per_channel_alone_auc"] = per
        rec["dropped_by_screen"] = [meta[sub_idx[j]]["name"] for j in range(mask_cols.sum())
                                    if j not in set(keep.tolist())]
        out["variants"][variant] = rec

    # upstream-parent tally (FREEZE ADDENDUM 2 bookkeeping)
    parents = {}
    for m in meta:
        parents.setdefault(m["upstream_parent"] or "unspecified", []).append(m["name"])
    out["upstream_parent_tally"] = {k: len(v) for k, v in parents.items()}
    out["upstream_parents"] = parents
    out["n_mixed"] = int(sum(m["mixed"] for m in meta))

    (HERE / f"round{r}_track_b_discount.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({v: {k: x for k, x in d.items() if k != "per_channel_alone_auc"}
                      for v, d in out["variants"].items()}, indent=1))
    print("upstream parents:", json.dumps(out["upstream_parent_tally"], indent=1))


if __name__ == "__main__":
    main()
