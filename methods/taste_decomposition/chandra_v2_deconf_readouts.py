#!/usr/bin/env python3
"""Era-stratified + author-grouped deconfounding readouts for the chandra v2
cells (coordinator order 2026-08-24, leak-audit follow-up). READOUTS ONLY —
no refits; consumes the pooled layer-1 OOF npz + pooled dense seed-mean preds.

DECLARED LIMIT (carried in every output): the removal side is anonymized
(Chandrasekharan log = body+subreddit; no author, no timestamps). Therefore:
  * era strata exist for KEPT rows only -> readout = stability of the
    kept-vs-removed separation across kept-side eras (removed side pooled);
  * author-disjointness is enforceable on the KEPT side only (removed rows
    are treated as singleton authors by construction).

Readout 1 (era): per sub x kept-quarter AUC of VA_nl (all rows) and dense T
(eval+test legs) over {kept in quarter} vs {all removed in sub}; pair-weighted
aggregate per quarter. A flat profile across quarters = the separation is not
driven by which era the kept side was fetched from (the v1 failure mode).

Readout 2 (author): kept author duplication stats; dense-split author overlap
(share of test kept rows whose author_hash appears in train kept rows); dense
T test AUC restricted to author-DISJOINT kept test rows vs author-overlapping
kept test rows (removed test rows common to both sides of the comparison).

Usage: chandra_v2_deconf_readouts.py --cell chandra_humor_v2
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
sys.path.insert(0, str(REPO / "methods/taste_decomposition"))
import unified_fused_stack as U  # noqa: E402

RESULTS = REPO / "methods/taste_decomposition/results"
SEEDS = (42, 1, 2)
LIMIT = ("removal side anonymized (no author/timestamps): era strata and "
         "author-disjointness are kept-side constructs; removed rows pooled "
         "(era) / singleton-author (author) by construction")


def quarter(ts):
    d = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True,
                    choices=["chandra_humor_v2", "chandra_cw_v2"])
    a = ap.parse_args()
    cell = a.cell
    cfg = U.CELLS[cell]

    pop = pd.read_csv(REPO / f"datasets/prior_norms_cells/{cell}_population.csv.gz")
    z = np.load(RESULTS / f"{cell}_va_oof.npz", allow_pickle=True)
    ids = [str(i) for i in z["ids"]]
    pos = {r: i for i, r in enumerate(ids)}
    va = z["VA_nl"].astype(float)
    y_oof = z["y"].astype(int)
    pop = pop[pop.row_id.astype(str).isin(pos)].copy()
    pop["oof_i"] = pop.row_id.astype(str).map(pos)
    assert (y_oof[pop.oof_i.values] == pop.judgement.values).all(), "OOF join mismatch"
    pop["va"] = va[pop.oof_i.values]
    pop["quarter"] = [quarter(t) if pd.notna(t) else None for t in pop.ts]

    # dense seed-mean on eval+test legs
    dense = []
    for leg in ("eval", "test"):
        sp = pd.read_csv(cfg["dense"] / "split" / f"{leg}.csv")
        per = []
        for s in SEEDS:
            p = pd.read_csv(cfg["dense"] / f"rm_out_seed{s}" / f"preds_{leg}.csv")
            assert len(p) == len(sp) and \
                (p["judgement"].values == sp["judgement"].values).all(), \
                f"order-join fail {leg} seed{s}"
            per.append(p["prob"].values.astype(float))
        sp["t"] = np.mean(per, axis=0)
        sp["leg"] = leg
        dense.append(sp[["row_id", "judgement", "group", "t", "leg"]])
    dn = pd.concat(dense).reset_index(drop=True)
    dn = dn.merge(pop[["row_id", "ts", "author_hash", "quarter"]], on="row_id", how="left")

    out = {"cell": cell, "limit": LIMIT}

    # ---------------- Readout 1: era-stratified ----------------
    def era_profile(df, score_col):
        prof = {}
        quarters = sorted(q for q in df[df.judgement == 0].quarter.dropna().unique())
        for q in quarters:
            aucs, ws = [], []
            for sub, grp in df.groupby("group"):
                k = grp[(grp.judgement == 0) & (grp.quarter == q)]
                r = grp[grp.judgement == 1]
                if len(k) < 25 or len(r) < 25:
                    continue
                yy = np.r_[np.zeros(len(k)), np.ones(len(r))]
                ss = np.r_[k[score_col].values, r[score_col].values]
                aucs.append(roc_auc_score(yy, ss))
                ws.append(len(k) * len(r))
            if aucs:
                w = np.array(ws, float)
                prof[q] = {"auc_pairweighted": round(float((np.array(aucs) * w).sum() / w.sum()), 4),
                           "n_subs": len(aucs)}
        vals = [v["auc_pairweighted"] for v in prof.values()]
        return {"per_quarter": prof,
                "spread_max_minus_min": round(max(vals) - min(vals), 4) if vals else None}

    out["era_stratified_VA"] = era_profile(pop, "va")
    out["era_stratified_T_evaltest"] = era_profile(dn, "t")

    # ---------------- Readout 2: author-grouped ----------------
    kept = pop[pop.judgement == 0]
    ac = kept.author_hash.dropna().value_counts()
    out["author_stats"] = {
        "kept_rows_with_author": int(kept.author_hash.notna().sum()),
        "unique_authors": int(ac.size),
        "share_kept_rows_from_multirow_authors": round(float(ac[ac >= 2].sum() / ac.sum()), 4),
        "max_rows_one_author": int(ac.max()),
    }
    tr = pd.read_csv(cfg["dense"] / "split" / "train.csv")[["row_id"]]
    tr = tr.merge(pop[["row_id", "judgement", "author_hash"]], on="row_id")
    train_authors = set(tr[tr.judgement == 0].author_hash.dropna())
    te = dn[dn.leg == "test"].copy()
    te_k = te[te.judgement == 0]
    overlap = te_k.author_hash.map(lambda h: h in train_authors if pd.notna(h) else False)
    out["dense_split_author_overlap"] = {
        "share_test_kept_rows_with_author_in_train": round(float(overlap.mean()), 4)}
    te_r = te[te.judgement == 1]
    res = {}
    for tag, mask in (("author_disjoint", ~overlap.values), ("author_overlapping", overlap.values)):
        kk = te_k[mask]
        if len(kk) < 50:
            res[tag] = {"n_kept": int(len(kk)), "auc": None}
            continue
        yy = np.r_[np.zeros(len(kk)), np.ones(len(te_r))]
        ss = np.r_[kk.t.values, te_r.t.values]
        res[tag] = {"n_kept": int(len(kk)),
                    "auc": round(float(roc_auc_score(yy, ss)), 4)}
    out["dense_T_test_by_author_status"] = res
    # same comparison for VA (OOF is pseudo-group folded — declared weaker)
    va_res = {}
    kept_all = pop[pop.judgement == 0]
    rem_all = pop[pop.judgement == 1]
    ov_all = kept_all.author_hash.map(lambda h: pd.notna(h) and ac.get(h, 0) >= 2)
    for tag, mask in (("singleton_author", ~ov_all.values), ("multirow_author", ov_all.values)):
        kk = kept_all[mask]
        yy = np.r_[np.zeros(len(kk)), np.ones(len(rem_all))]
        ss = np.r_[kk.va.values, rem_all.va.values]
        va_res[tag] = {"n_kept": int(len(kk)),
                       "auc": round(float(roc_auc_score(yy, ss)), 4)}
    out["VA_oof_by_author_multiplicity"] = {
        **va_res,
        "note": "VA OOF folds are row-hash pseudo-groups (not author-grouped) — "
                "multirow-author kept rows CAN share folds; comparison is "
                "descriptive, the dense split readout above is the crisp one"}

    p = RESULTS / f"{cell}_deconf_readouts.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"{cell.upper()}_DECONF_DONE -> {p}", flush=True)


if __name__ == "__main__":
    main()
