#!/usr/bin/env python3
"""BBC most-read: Layer-1 ledger for journalism community cell #2 -- the
SAME-OUTLET READERSHIP counterpart to V9's cross-platform amplification.

REUSE: every estimator is IMPORTED, not reimplemented -- `layer1_gemma_cells`
(outer_folds, linear_oof_family1, gbm_oof_family1, GBM_SEEDS) and
`scaleupC_layer1` (load_scaleupC_bank, dense_T, group_bootstrap_*, run_cell), so
this cell is numerically comparable to journalism_tweets, so_votes,
mathse_vote_score, jokes_community, aops_curation and homepage_curation by
construction.

WHAT THIS FILE ADDS on top of SC.run_cell:

  1. DAY IDENTITY ALONE, measured FIRST -- and unlike V9 it is NOT at chance.
     V9's y is a within-group median split, which forces every group to a .500
     positive rate and makes group identity worthless by construction. This
     cell's y is natural (on the list or not), so the per-day positive rate
     varies with how many links the capture carried, and day identity alone
     scores ~.58. **Consequently the WITHIN-DAY readouts are this cell's honest
     primary numbers and the pooled ones are secondary** -- the exact reverse of
     V9's situation, and the single most important thing to keep straight when
     comparing the two cells.
  2. THE RANK LINE. Positives carry a most-read rank 1-10. Whether the
     articulated instruments order rank *within* the winners is a stronger test
     than binary membership, and it is free.
  3. ERA STABILITY. The population spans 2017-2024. A cell whose signal is
     carried by one era's news cycle would be a different claim from one that
     holds throughout, so AUC is reported per year.
  4. Same-rows Delta_beyond, so T is not differenced against a VA pooled over a
     different row set (the N&C _v2 landmine).
  5. The OOF alignment gate: SC.run_cell saves bare OOF arrays with no ids
     vector, so ids are shipped and the round-trip is proven to <1e-9 with a
     shuffled counterfactual for teeth.
  6. An ENFORCED collapse gate on the assembled A matrix.

  python3 methods/taste_decomposition/bbc_mostread_layer1.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
import layer1_gemma_cells as L  # noqa: E402
import scaleupC_layer1 as SC  # noqa: E402

REPO = SC.REPO
RESULTS = SC.RESULTS_DIR
BBC_OUT = Path("outputs/va_gemma_banks_bbc_mostread")
BBC_DIR = REPO / "datasets/bbc-mostread/va"
DENSE = BBC_DIR / "dense_standard_bbc_mostread"
SLUG = "bbc_mostread"


def cell_bbc_mostread():
    out = REPO / BBC_OUT
    meta, A, V, groups, shard, ids = SC.load_scaleupC_bank(SLUG, out=out)
    y = np.array(meta["ys"]["mostread"], dtype=float)
    keep = np.isfinite(y)
    T, Tinfo = SC.dense_T(DENSE)
    return dict(
        title="BBC News most-read (headline in the home page's ranked MOST READ "
              "top-10 vs elsewhere on the same capture)",
        A=A[keep], V=V[keep], y=y[keep].astype(int), groups=groups[keep],
        ids=ids[keep], meta=meta, shard_of=shard[keep],
        group_column="capture_day", T=T, T_info=Tinfo,
        matrix=f"{BBC_OUT}/{SLUG}_shard*.npz",
        dense_dir=str(DENSE), prior_published=None, keep_mask=keep)


def within_group_auc(y, groups, pred):
    tot = wsum = 0.0
    uw = []
    n = 0
    for q in np.unique(groups):
        m = groups == q
        yy = y[m]
        if yy.min() == yy.max():
            continue
        a = roc_auc_score(yy, pred[m])
        npair = int(yy.sum() * (len(yy) - yy.sum()))
        uw.append(a)
        tot += npair * a
        wsum += npair
        n += 1
    if not n:
        return None
    return {"pair_weighted": float(tot / wsum),
            "unweighted_mean": float(np.mean(uw)),
            "median": float(np.median(uw)),
            "n_mixed_groups": n, "n_pairs": int(wsum)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / f"{SLUG}_ledger.json"))
    a = ap.parse_args()

    SC.SCALEUPC_OUT = REPO / BBC_OUT
    SC.CELLS[SLUG] = cell_bbc_mostread
    res = SC.run_cell(SLUG)

    d = cell_bbc_mostread()
    y, groups, ids = d["y"], d["groups"], d["ids"]
    A, V = d["A"], d["V"]

    pop = pd.read_csv(BBC_DIR / "population.csv.gz")
    pop["row_id"] = pop.row_id.astype(str)
    pos = pop.set_index("row_id")
    order = [str(i) for i in ids]
    rank = pos.loc[order, "rank"].values.astype(float)
    split = pos.loc[order, "split"].values
    year = pos.loc[order, "day"].astype(str).str[:4].values
    hlen = pos.loc[order, "raw_headline"].astype(str).str.len().values.astype(float)

    folds = L.outer_folds(len(y), groups, n_splits=5)
    extra = {}

    # ---- 1. day identity alone -- NOT at chance here, unlike V9 -------------
    gmean = pd.Series(y).groupby(pd.Series(groups)).transform("mean").values
    extra["day_identity_alone_auc"] = float(roc_auc_score(y, gmean))
    extra["identity_note"] = (
        "Unlike V9 (whose within-group median-split y forces .500 exactly), this "
        "cell's y is natural membership, so the per-day positive rate varies and "
        "day identity alone carries real signal. The WITHIN-DAY readouts are "
        "therefore this cell's honest primary numbers; pooled ones are secondary.")
    extra["headline_charlen_alone_auc"] = float(roc_auc_score(y, hlen))

    # ---- 2. ENFORCED collapse gate on the assembled A matrix ---------------
    a_names = list(d["meta"]["a_names"])
    collapsed = []
    crit = {}
    for j, nm in enumerate(a_names):
        col = A[:, j]
        na = float(np.isnan(col).mean())
        fin = col[np.isfinite(col)]
        modal = (float(np.bincount(
            np.searchsorted(np.unique(fin), fin)).max() / len(col))
            if len(fin) else 0.0)
        crit[nm] = {"na_rate": round(na, 4), "modal_share_of_all_rows": round(modal, 4),
                    "mean": (float(np.nanmean(col)) if len(fin) else None)}
        if na >= 0.95 or modal >= 0.95:
            collapsed.append(nm)
    extra["criterion_health"] = crit
    extra["collapse_gate"] = {"collapsed": collapsed, "passes": not collapsed,
                              "rule": "fail if a criterion is >=95% NA or >=95% "
                                      "one value across all rows"}
    if collapsed:
        print(f"[collapse gate] FAIL: {collapsed}", flush=True)
    else:
        print("[collapse gate] PASS", flush=True)

    # ---- 3. pooled + within-day for every matrix ---------------------------
    mats = {"V": V, "A": A, "VA": np.column_stack([V, A])}
    preds, table = {}, {}
    for k, M in mats.items():
        auc, oof = L.linear_oof_family1(M, y, groups, folds)
        preds[k + "_lin"] = oof
        table[k + "_lin"] = {"pooled": auc,
                             "within_day": within_group_auc(y, groups, oof)}
        print(f"  {k+'_lin':8s} pooled {auc:.4f}  "
              f"within-day {table[k+'_lin']['within_day']['pair_weighted']:.4f}")
    for k in ["V", "VA"]:
        oofs = [L.gbm_oof_family1(mats[k], y, groups, folds, s)["oof"]
                for s in L.GBM_SEEDS]
        mo = np.mean(oofs, axis=0)
        preds[k + "_nl"] = mo
        table[k + "_nl"] = {"pooled_auc_of_seed_mean_oof": float(roc_auc_score(y, mo)),
                            "within_day": within_group_auc(y, groups, mo)}
        print(f"  {k+'_nl':8s} pooled {table[k+'_nl']['pooled_auc_of_seed_mean_oof']:.4f}"
              f"  within-day {table[k+'_nl']['within_day']['pair_weighted']:.4f}")
    extra["pooled_vs_within_day"] = table

    def conv_within(pred, min_n=20):
        num = den = 0.0
        ng = 0
        for q in np.unique(groups):
            m = groups == q
            if m.sum() < min_n or y[m].min() == y[m].max():
                continue
            num += m.sum() * roc_auc_score(y[m], pred[m])
            den += m.sum()
            ng += 1
        return ({"auc": float(num / den), "n_groups": ng, "n_rows": int(den)}
                if den else None)
    extra["program_convention_within_group_min20"] = {
        "VA_nl": conv_within(preds["VA_nl"]), "VA_lin": conv_within(preds["VA_lin"])}

    # ---- 4. the RANK line (ordering within the winners) --------------------
    ispos = y == 1
    r = rank[ispos]
    ok = np.isfinite(r)
    rl = {}
    for nm in ("VA_nl", "VA_lin", "A_lin", "V_nl"):
        p = preds[nm][ispos][ok]
        rr = r[ok]
        rl[nm] = {
            "spearman_vs_rank": float(pd.Series(p).corr(pd.Series(-rr), method="spearman")),
            "top3_vs_bottom3_auc": float(roc_auc_score(
                (rr <= 3).astype(int)[np.isin(rr, [1, 2, 3, 8, 9, 10])],
                p[np.isin(rr, [1, 2, 3, 8, 9, 10])])),
        }
    rl["n_ranked_positives"] = int(ok.sum())
    rl["note"] = ("Instruments are frozen on the binary membership y and merely "
                  "re-scored against rank; positive Spearman means the same "
                  "articulated signal that predicts membership also orders the "
                  "winners. Sign convention: correlated with -rank, so positive "
                  "= predicts a BETTER (numerically lower) rank.")
    extra["rank_line"] = rl

    # ---- 5. era stability ---------------------------------------------------
    per_year = {}
    for yr in np.unique(year):
        m = year == yr
        if m.sum() < 300 or len(np.unique(y[m])) < 2:
            continue
        per_year[str(yr)] = {
            "n": int(m.sum()),
            "VA_nl": float(roc_auc_score(y[m], preds["VA_nl"][m])),
            "A_lin": float(roc_auc_score(y[m], preds["A_lin"][m])),
            "within_day_VA_nl": within_group_auc(y[m], groups[m],
                                                 preds["VA_nl"][m])["pair_weighted"],
        }
    extra["per_year"] = per_year

    # ---- 6. same-rows Delta_beyond -----------------------------------------
    Tinfo = d["T_info"]
    same = {}
    for leg in ["eval", "test"]:
        m = split == leg
        if m.sum() < 50 or len(np.unique(y[m])) < 2:
            continue
        raw = (Tinfo or {}).get("raw", {})
        runs = raw.get("runs", raw) if isinstance(raw, dict) else {}
        aucs = [float(v[f"{leg}_auc"]) for v in runs.values()
                if isinstance(v, dict) and f"{leg}_auc" in v]
        legT = float(np.mean(aucs)) if aucs else None
        wg = within_group_auc(y[m], groups[m], preds["VA_nl"][m])
        same[leg] = {
            "n": int(m.sum()), "n_pos": int(y[m].sum()),
            "VA_lin": float(roc_auc_score(y[m], preds["VA_lin"][m])),
            "VA_nl": float(roc_auc_score(y[m], preds["VA_nl"][m])),
            "T_seed_mean": legT, "T_per_seed": aucs,
            "Delta_beyond": (legT - float(roc_auc_score(y[m], preds["VA_nl"][m]))
                             if legT is not None else None),
            "within_day_VA_nl": wg["pair_weighted"] if wg else None,
        }
    extra["same_rows"] = same

    # ---- 7. OOF alignment gate ---------------------------------------------
    np.save(RESULTS / f"{SLUG}_oof_ids.npy", np.array(ids, dtype=object),
            allow_pickle=True)
    gate = {}
    ids_disk = np.load(RESULTS / f"{SLUG}_oof_ids.npy", allow_pickle=True)
    pos_of = {str(x): i for i, x in enumerate(ids_disk)}
    perm = np.array([pos_of[str(x)] for x in ids])
    for tag, arr in (("seed0", preds["VA_nl"]), ("VA_lin", preds["VA_lin"])):
        published = float(roc_auc_score(y, arr))
        reassembled = float(roc_auc_score(y, arr[perm]))
        rng = np.random.default_rng(4242)
        shuf = arr.copy()
        rng.shuffle(shuf)
        gate[tag] = {"published_auc": published,
                     "reassembled_from_ids_auc": reassembled,
                     "abs_diff": abs(published - reassembled),
                     "passes_1e-9": bool(abs(published - reassembled) < 1e-9),
                     "shuffled_counterfactual_auc": float(roc_auc_score(y, shuf))}
        print(f"[align:{tag}] published {published:.6f} reassembled {reassembled:.6f} "
              f"diff {gate[tag]['abs_diff']:.2e} pass={gate[tag]['passes_1e-9']} "
              f"(shuffled {gate[tag]['shuffled_counterfactual_auc']:.4f})")
    gate["ids_path"] = str(RESULTS / f"{SLUG}_oof_ids.npy")
    gate["n_ids"] = int(len(ids))
    gate["ids_unique"] = bool(len(set(map(str, ids))) == len(ids))
    extra["oof_alignment_gate"] = gate

    # ---- 8. the cross-cell contrast this cell exists to make ---------------
    v9 = RESULTS / "journalism_tweets_ledger.json"
    if v9.exists():
        j = json.loads(v9.read_text())
        jl, je = j["ledger"], j["journalism_tweets_extras"]
        extra["contrast_with_v9_tweets"] = {
            "design": "SAME field, SAME item type (news headline), SAME V bank, "
                      "SAME A bank, SAME dense recipe, SAME judge. Differs only "
                      "in WHICH crowd and HOW it acts: BBC readers clicking BBC "
                      "(same-outlet readership) vs Twitter users amplifying "
                      "links (cross-platform).",
            "row_overlap": "ZERO -- V9 carries no BBC rows and the two corpora "
                           "share no headline (V9 is 2025-12..2026-04, this cell "
                           "is 2017-2024). The contrast is CELL-LEVEL, not "
                           "same-rows; no paired test is possible.",
            "v9": {k: jl.get(k) for k in ("V_lin", "V_nl_mean", "A_lin", "VA_lin",
                                          "VA_nl_mean", "T", "Delta_interact",
                                          "Delta_beyond")},
            "v9_group_identity_alone": je.get("outlet_day_identity_alone_auc"),
            "bbc": {k: res["ledger"].get(k) for k in
                    ("V_lin", "V_nl_mean", "A_lin", "VA_lin", "VA_nl_mean", "T",
                     "Delta_interact", "Delta_beyond")},
            "bbc_group_identity_alone": extra["day_identity_alone_auc"],
        }

    res[f"{SLUG}_extras"] = extra
    Path(a.out).write_text(json.dumps(res, indent=2, default=str))
    print("\nwrote", a.out)
    lg = res["ledger"]
    print("\n=== LEDGER ===")
    for k in ("V_lin", "V_nl_mean", "A_lin", "VA_lin", "VA_nl_mean", "T",
              "Delta_interact", "Delta_total", "Delta_beyond"):
        if lg.get(k) is not None:
            print(f"  {k:16s} {lg[k]:+.4f}")
    print(f"  day_identity_alone {extra['day_identity_alone_auc']:.4f}")


if __name__ == "__main__":
    main()
