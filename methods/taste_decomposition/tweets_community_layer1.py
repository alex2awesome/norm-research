#!/usr/bin/env python3
"""V9 journalism-tweets: Layer-1 ledger for the journalism/press field's
VOTE/REVEALED (crowd) column of the 3xN decomposition grid.

REUSE: every estimator is IMPORTED, not reimplemented --
  * `layer1_gemma_cells` (L): outer_folds, linear_oof_family1, gbm_oof_family1,
    GBM_SEEDS -- the frozen Layer-1 spec.
  * `scaleupC_layer1` (SC): load_scaleupC_bank (re-rooted at this cell's own OUT
    dir via the `out=` argument), dense_T, group_bootstrap_delta,
    group_bootstrap_auc, and run_cell itself.
so this cell is numerically comparable to so_votes, mathse_vote_score,
jokes_community, aops_curation and homepage_curation by construction.

WHAT THIS FILE ADDS on top of SC.run_cell (all cell-specific):

  1. OUTLET-DAY IDENTITY ALONE, measured FIRST. This is the V8 N&C discipline:
     a pooled AUC can be mostly group composition. Here it should come out at
     chance BY CONSTRUCTION -- y is a within-group median split, so every group
     has a ~.500 pos rate and group identity cannot predict y. Measuring it is
     how we PROVE the pooled number is honest rather than asserting it.
  2. THE CAP LINE. `capped` is a property of the LABEL CHANNEL (whether the
     scraper hit its 100-tweet retrieval limit), not of the headline. It alone
     scores AUC .611 on eval. It is therefore never a feature -- it appears
     here only as a diagnostic quantifying how much of y is raw tweet VOLUME
     as opposed to per-tweet intensity.
  3. CENSORING ROBUSTNESS: the same instruments scored against `y_maxlikes`
     (median split on max_likes, which is far less sensitive to the 100-tweet
     cap than a sum is) and against `y_quartile` (the harder-margin top-vs-
     bottom-quartile arm). If the decomposition holds under all three, the
     conclusion does not rest on the censored sum.
  4. Same-rows Delta_beyond: VA predictions restricted to EXACTLY the dense
     split's rows, so T is not differenced against a pooled VA over a
     different row set (the N&C _v2 cross-population landmine).

  NOT AVAILABLE, recorded as an inherited limitation: this cell's A bank is the
  homepage curation bank reused verbatim, and that bank carries no Track A /
  Track B field, so there is no A_real vs A_surface split here (V6 SO-votes
  has one). Adding tracks would mean re-authoring the bank, which would forfeit
  the zero-new-judging reuse.

  python3 methods/taste_decomposition/tweets_community_layer1.py
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
TW_OUT = Path("outputs/va_gemma_banks_journalism_tweets")
TW_DIR = REPO / "datasets/journalism-tweets/va"
DENSE = TW_DIR / "dense_standard_journalism_tweets"
SLUG = "journalism_tweets"


def cell_journalism_tweets():
    out = REPO / TW_OUT
    meta, A, V, groups, shard, ids = SC.load_scaleupC_bank(SLUG, out=out)
    y = np.array(meta["ys"]["engagement"], dtype=float)
    keep = np.isfinite(y)
    T, Tinfo = SC.dense_T(DENSE)
    return dict(
        title="Journalism tweet engagement (within outlet-day median split of "
              "summed Twitter likes on the article URL)",
        A=A[keep], V=V[keep], y=y[keep].astype(int), groups=groups[keep],
        ids=ids[keep], meta=meta, shard_of=shard[keep],
        group_column="outlet_day", T=T, T_info=Tinfo,
        matrix=f"{TW_OUT}/{SLUG}_shard*.npz (y = engagement; the maxlikes and "
               "quartile robustness y's share ONE scored matrix and are never "
               "merged into the primary)",
        dense_dir=str(DENSE),
        prior_published=None, keep_mask=keep)


def within_group_auc(y, groups, pred):
    """Pair-weighted within-outlet-day AUC over groups carrying both classes."""
    tot = wsum = 0.0
    per, uw = [], []
    for q in np.unique(groups):
        m = groups == q
        yy = y[m]
        if yy.min() == yy.max():
            continue
        a = roc_auc_score(yy, pred[m])
        npair = int(yy.sum() * (len(yy) - yy.sum()))
        per.append((q, npair, a))
        uw.append(a)
        tot += npair * a
        wsum += npair
    if not per:
        return None
    return {"pair_weighted": float(tot / wsum),
            "unweighted_mean": float(np.mean(uw)),
            "median": float(np.median(uw)),
            "n_mixed_groups": len(per),
            "n_pairs": int(wsum)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / f"{SLUG}_ledger.json"))
    a = ap.parse_args()

    SC.SCALEUPC_OUT = REPO / TW_OUT
    SC.CELLS[SLUG] = cell_journalism_tweets
    res = SC.run_cell(SLUG)

    d = cell_journalism_tweets()
    y, groups, ids = d["y"], d["groups"], d["ids"]
    A, V = d["A"], d["V"]

    pop = pd.read_csv(TW_DIR / "population.csv.gz")
    pop["row_id"] = pop.row_id.astype(str)
    pos = pop.set_index("row_id")
    order = [str(i) for i in ids]
    capped = pos.loc[order, "capped"].values.astype(bool)
    split = pos.loc[order, "split"].values
    outlet = pos.loc[order, "outlet"].values
    y_maxl = pos.loc[order, "y_maxlikes"].values.astype(float)
    y_quart = pos.loc[order, "y_quartile"].values.astype(float)
    head_len = pos.loc[order, "raw_headline"].astype(str).str.len().values.astype(float)

    folds = L.outer_folds(len(y), groups, n_splits=5)
    extra = {}

    # ---- 1. group identity alone, FIRST -------------------------------------
    gmean = pd.Series(y).groupby(pd.Series(groups)).transform("mean").values
    extra["outlet_day_identity_alone_auc"] = float(roc_auc_score(y, gmean))
    extra["identity_note"] = (
        "Expected at chance BY CONSTRUCTION: y is a within-(outlet,day) median "
        "split so every group has a ~.500 pos rate. A value near .500 is the "
        "proof that this cell's pooled AUCs are not group composition -- the "
        "failure mode the V8 N&C co-signing build documented.")

    # ---- 2. label-channel diagnostics (NEVER features) ----------------------
    extra["label_channel_lines"] = {
        "capped_flag_alone_auc": float(roc_auc_score(y, capped.astype(float))),
        "capped_rate_by_class": {
            "y0": float(capped[y == 0].mean()), "y1": float(capped[y == 1].mean())},
        "headline_charlen_alone_auc": float(roc_auc_score(y, head_len)),
        "headline_charlen_within_group": within_group_auc(y, groups, head_len),
        "note": "capped is a property of the retrieval limit on the LABEL, not "
                "of the headline, and is never in V, A or the dense input. It is "
                "reported to quantify how much of y is tweet VOLUME rather than "
                "per-tweet intensity. headline length is reported because a "
                "length-driven cell would be a Style-Invitational-style artifact; "
                "here it sits at chance.",
    }

    # ---- 3. pooled + within-group for every matrix --------------------------
    mats = {"V": V, "A": A, "VA": np.column_stack([V, A])}
    preds, table = {}, {}
    for k, M in mats.items():
        auc, oof = L.linear_oof_family1(M, y, groups, folds)
        preds[k + "_lin"] = oof
        table[k + "_lin"] = {"pooled": auc,
                             "within_group": within_group_auc(y, groups, oof)}
        print(f"  {k+'_lin':10s} pooled {auc:.4f}  "
              f"within-group {table[k+'_lin']['within_group']['pair_weighted']:.4f}")
    for k in ["V", "VA"]:
        oofs = [L.gbm_oof_family1(mats[k], y, groups, folds, s)["oof"]
                for s in L.GBM_SEEDS]
        mean_oof = np.mean(oofs, axis=0)
        preds[k + "_nl"] = mean_oof
        table[k + "_nl"] = {
            "pooled_auc_of_seed_mean_oof": float(roc_auc_score(y, mean_oof)),
            "within_group": within_group_auc(y, groups, mean_oof)}
        print(f"  {k+'_nl':10s} pooled {table[k+'_nl']['pooled_auc_of_seed_mean_oof']:.4f}"
              f"  within-group {table[k+'_nl']['within_group']['pair_weighted']:.4f}")
    extra["pooled_vs_within_group"] = table

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

    # ---- 4. same-rows Delta_beyond -----------------------------------------
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
        same[leg] = {
            "n": int(m.sum()), "n_pos": int(y[m].sum()),
            "VA_lin": float(roc_auc_score(y[m], preds["VA_lin"][m])),
            "VA_nl": float(roc_auc_score(y[m], preds["VA_nl"][m])),
            "T_seed_mean": legT,
            "T_per_seed": aucs,
            "Delta_beyond": (legT - float(roc_auc_score(y[m], preds["VA_nl"][m]))
                             if legT is not None else None),
            "within_group_VA_nl": within_group_auc(y[m], groups[m],
                                                   preds["VA_nl"][m]),
        }
    extra["same_rows"] = same

    # ---- 5. censoring / binarization robustness -----------------------------
    rob = {}
    for nm, yy in (("y_maxlikes", y_maxl), ("y_quartile", y_quart)):
        m = np.isfinite(yy) & (yy >= 0)
        if m.sum() < 100 or len(np.unique(yy[m])) < 2:
            continue
        rob[nm] = {
            "n": int(m.sum()),
            "pos_rate": float(yy[m].mean()),
            "agreement_with_primary_y": float((yy[m] == y[m]).mean()),
            "VA_nl_auc": float(roc_auc_score(yy[m], preds["VA_nl"][m])),
            "VA_lin_auc": float(roc_auc_score(yy[m], preds["VA_lin"][m])),
            "V_nl_auc": float(roc_auc_score(yy[m], preds["V_nl"][m])),
            "A_lin_auc": float(roc_auc_score(yy[m], preds["A_lin"][m])),
        }
    rob["note"] = ("Instruments are FROZEN on the primary y (OOF predictions "
                   "from the engagement fit) and merely re-scored against the "
                   "robustness labels -- no refit, so these are not independent "
                   "fits but transfer checks on the same predictor.")
    extra["censoring_robustness"] = rob

    # ---- 6. per-outlet decomposition ----------------------------------------
    per_out = {}
    for o in np.unique(outlet):
        m = outlet == o
        if m.sum() < 200 or len(np.unique(y[m])) < 2:
            continue
        per_out[str(o)] = {
            "n": int(m.sum()),
            "VA_nl": float(roc_auc_score(y[m], preds["VA_nl"][m])),
            "VA_lin": float(roc_auc_score(y[m], preds["VA_lin"][m])),
            "A_lin": float(roc_auc_score(y[m], preds["A_lin"][m])),
        }
    extra["per_outlet"] = per_out

    # ---- 7. OOF ALIGNMENT GATE ---------------------------------------------
    # SC.run_cell saves <slug>_va_nl_oof_{seed0,mean3}.npy as BARE arrays with
    # no ids vector -- the N&C _v2 landmine (notes/2026-07-27__vat-run-registry.md):
    # a bare OOF array's row order is implicit, so any later re-join can silently
    # permute it. Ship the ids vector next to the arrays, then PROVE the
    # round-trip by reassembling from disk in id order and reproducing the
    # published AUC to <1e-9. The shuffled counterfactual is what gives the gate
    # teeth: if a permuted array also reproduced the number, the check would be
    # vacuous.
    np.save(RESULTS / f"{SLUG}_oof_ids.npy", np.array(ids, dtype=object),
            allow_pickle=True)
    gate = {}
    for tag, arr in (("seed0", preds["VA_nl"]), ("VA_lin", preds["VA_lin"])):
        published = float(roc_auc_score(y, arr))
        ids_disk = np.load(RESULTS / f"{SLUG}_oof_ids.npy", allow_pickle=True)
        pos_of = {str(d): i for i, d in enumerate(ids_disk)}
        perm = np.array([pos_of[str(d)] for d in ids])
        reassembled = float(roc_auc_score(y, arr[perm]))
        rng = np.random.default_rng(4242)
        shuf = arr.copy()
        rng.shuffle(shuf)
        gate[tag] = {
            "published_auc": published,
            "reassembled_from_ids_auc": reassembled,
            "abs_diff": abs(published - reassembled),
            "passes_1e-9": bool(abs(published - reassembled) < 1e-9),
            "shuffled_counterfactual_auc": float(roc_auc_score(y, shuf)),
        }
    gate["ids_path"] = str(RESULTS / f"{SLUG}_oof_ids.npy")
    gate["n_ids"] = int(len(ids))
    gate["ids_unique"] = bool(len(set(map(str, ids))) == len(ids))
    extra["oof_alignment_gate"] = gate
    for tag in ("seed0", "VA_lin"):
        g = gate[tag]
        print(f"[align:{tag}] published {g['published_auc']:.6f} "
              f"reassembled {g['reassembled_from_ids_auc']:.6f} "
              f"diff {g['abs_diff']:.2e} pass={g['passes_1e-9']} "
              f"(shuffled {g['shuffled_counterfactual_auc']:.4f})")

    res[f"{SLUG}_extras"] = extra
    Path(a.out).write_text(json.dumps(res, indent=2, default=str))
    print("\nwrote", a.out)
    lg = res["ledger"]
    print("\n=== LEDGER ===")
    for k in ("V_lin", "V_nl_mean", "A_lin", "VA_lin", "VA_nl_mean", "T",
              "Delta_interact", "Delta_total", "Delta_beyond"):
        if lg.get(k) is not None:
            print(f"  {k:16s} {lg[k]:+.4f}")
    print(f"  outlet_day_identity_alone {extra['outlet_day_identity_alone_auc']:.4f}")


if __name__ == "__main__":
    main()
