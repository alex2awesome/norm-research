#!/usr/bin/env python3
"""V7 patents forward-citations: Layer-1 ledger for the patents COMMUNITY/VOTE
cell (y = the downstream inventor community's revealed judgment, measured as
5-year forward citations, within-cohort median split).

REUSE: every estimator is IMPORTED, not reimplemented --
  * `layer1_gemma_cells` (L): outer_folds, linear_oof_family1, gbm_oof_family1,
    GBM_SEEDS -- the frozen Layer-1 spec.
  * `scaleupC_layer1` (SC): load_scaleupC_bank (re-rooted at this cell's own OUT
    dir), dense_T, group_bootstrap_delta, group_bootstrap_auc, and run_cell.
so this cell is numerically comparable to so_votes, mathse_vote_score,
jokes_community, aops_curation and homepage_curation by construction.

WHAT THIS FILE ADDS (all cell-specific; none of it is a re-estimation):

 1. COHORT IDENTITY MEASURED FIRST. y is a within-(grant-year x CPC-class)
    median split, so cohort identity is this cell's docket-analog and must be
    ~.50. It is reported out-of-fold, beside its permutation null.

 2. THE STRUCT BLOCK -- mandated by the claim-fell round-0 post-mortem
    (closure/patents/RUNBOOK.md item 4: "bank the nuisance channel as a declared
    Track-B channel at round 0, and quote Delta over V + A + STRUCT, never over
    V + A"). claim-fell's killer was claim ordinal position (.754 alone). The
    analogous declared nuisance here is `num_claims` (.596 alone) together with
    raw document lengths. STRUCT is measured, banked, and Delta_beyond is quoted
    over V + A + STRUCT as well as over V + A.

 3. A_real (Track A) vs A_surface (Track B) decomposition, available because the
    bank carries a `track` field.

 4. Same-rows Delta_beyond: VA predictions restricted to EXACTLY the dense
    split's rows, so T is not differenced against a pooled VA over other rows.

 5. CROSS-Y CONTRASTS on identical rows -- examiner-added vs applicant-side
    (self-citation-exposed) vs the AGE-CONFOUNDED all-time count. The all-time
    contrast is the cell's own demonstration of the age trap: it is reported to
    show what the fixed window buys, and is never the headline.

  python3 methods/taste_decomposition/patents_fwdcites_layer1.py
"""
from __future__ import annotations

import argparse
import hashlib
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
V7_OUT = Path("outputs/va_gemma_banks_patents_fwdcites")
V7_DIR = REPO / "datasets/patents/v7_community"
DENSE = V7_DIR / "dense_standard"
SLUG = "patents_fwdcites"


def cell_patents_fwdcites():
    out = REPO / V7_OUT
    meta, A, V, groups, shard, ids = SC.load_scaleupC_bank(SLUG, out=out)
    y = np.array(meta["ys"]["fwd5_community"], dtype=float)
    keep = np.isfinite(y)
    T, Tinfo = SC.dense_T(DENSE)
    return dict(
        title="US granted patents, downstream inventor community (5-year forward "
              "citations, within grant-year x CPC-class median split)",
        A=A[keep], V=V[keep], y=y[keep].astype(int), groups=groups[keep],
        ids=ids[keep], meta=meta, shard_of=shard[keep],
        group_column="family_group (near-duplicate / continuation cluster)",
        T=T, T_info=Tinfo,
        matrix=f"{V7_OUT}/{SLUG}_shard*.npz (y = fwd5_community; the examiner / "
               "non-examiner / all-time y's share ONE scored matrix and are "
               "never merged)",
        dense_dir=str(DENSE),
        prior_published=None, keep_mask=keep)


def within_group_auc(y, groups, pred):
    """Pair-weighted within-cohort AUC over cohorts carrying both classes."""
    tot = wsum = 0.0
    per, uw = [], []
    for q in np.unique(groups):
        m = groups == q
        yy = y[m]
        if len(yy) < 2 or yy.min() == yy.max():
            continue
        a = roc_auc_score(yy, pred[m])
        npair = int(yy.sum() * (len(yy) - yy.sum()))
        per.append(a)
        uw.append(a)
        tot += npair * a
        wsum += npair
    if not per:
        return None
    return {"pair_weighted": float(tot / wsum), "unweighted_mean": float(np.mean(uw)),
            "median": float(np.median(uw)), "n_mixed_groups": len(per),
            "n_pairs": int(wsum)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / f"{SLUG}_ledger.json"))
    a = ap.parse_args()

    SC.SCALEUPC_OUT = REPO / V7_OUT
    SC.CELLS[SLUG] = cell_patents_fwdcites
    res = SC.run_cell(SLUG)

    d = cell_patents_fwdcites()
    y, groups, ids = d["y"], d["groups"], d["ids"]
    A, V = d["A"], d["V"]
    meta = d["meta"]
    a_names = list(meta["a_names"])
    rubrics = [json.loads(l) for l in open(V7_DIR / "rubrics.jsonl") if l.strip()]
    track = {r["name"]: r.get("track", "A") for r in rubrics}
    is_real = np.array([track.get(n, "A") == "A" for n in a_names])
    print(f"\n[bank split] A_real={int(is_real.sum())} "
          f"A_surface={int((~is_real).sum())}")

    pop = pd.read_csv(V7_DIR / "population.csv.gz")
    pop["row_id"] = pop.row_id.astype(str)
    pos = pop.set_index("row_id")
    order = [str(i) for i in ids]
    cohort = pos.loc[order, "cohort"].values.astype(str)
    split = pos.loc[order, "split"].values
    num_claims = pd.to_numeric(pos.loc[order, "num_claims"],
                               errors="coerce").fillna(0).values.astype(float)
    text_len = pos.loc[order, "text"].astype(str).str.len().values.astype(float)
    c1_len = pos.loc[order, "claim1"].astype(str).str.len().values.astype(float)
    abs_len = pos.loc[order, "abstract"].astype(str).str.len().values.astype(float)

    folds = L.outer_folds(len(y), groups, n_splits=5)
    extra = {}

    # ---- 1. cohort identity FIRST -------------------------------------------
    def oof_cohort_mean(k, n_folds=5):
        h = np.array([int(hashlib.blake2b(str(v).encode(), digest_size=4)
                          .hexdigest(), 16) % n_folds for v in ids])
        pred = np.full(len(y), np.nan)
        for f in range(n_folds):
            tr, te = h != f, h == f
            m = pd.Series(y[tr]).groupby(pd.Series(k[tr])).mean()
            pred[te] = pd.Series(k[te]).map(m).fillna(y[tr].mean()).values
        return pred
    rs = np.random.default_rng(7)
    null = [roc_auc_score(yp, pd.Series(yp).groupby(pd.Series(cohort))
                          .transform("mean").values)
            for yp in (rs.permutation(y) for _ in range(200))]
    extra["cohort_identity"] = {
        "oof_auc": float(roc_auc_score(y, oof_cohort_mean(cohort))),
        "insample_auc": float(roc_auc_score(
            y, pd.Series(y).groupby(pd.Series(cohort)).transform("mean").values)),
        "insample_perm_null_ci95": [float(np.percentile(null, 2.5)),
                                    float(np.percentile(null, 97.5))],
        "n_cohorts": int(len(np.unique(cohort))),
        "reading": "y is a WITHIN-cohort split, so cohort identity must sit near "
                   ".50; the in-sample figure is an overfit statistic on ~13 "
                   "sampled rows per cohort and is quotable only against its "
                   "permutation null."}

    # ---- 2. the STRUCT block (declared nuisance, RUNBOOK item 4) -------------
    STRUCT = np.column_stack([num_claims, text_len, c1_len, abs_len])
    struct_names = ["num_claims", "text_charlen", "claim1_charlen", "abstract_charlen"]
    extra["struct_block"] = {
        "names": struct_names,
        "alone_auc": {n: float(roc_auc_score(y, STRUCT[:, i]))
                      for i, n in enumerate(struct_names)},
        "why": "claim-fell closure RUNBOOK item 4: bank the declared nuisance "
               "channel at round 0 and quote Delta over V + A + STRUCT. "
               "num_claims is NOT recoverable from the text any instrument sees "
               "(claim 1 only), so it is an unexploited covariate here rather "
               "than a leak -- but it is banked and differenced anyway."}

    # ---- 3. pooled + within-cohort for every matrix --------------------------
    mats = {"V": V, "A": A, "VA": np.column_stack([V, A]),
            "A_real": A[:, is_real], "A_surface": A[:, ~is_real],
            "STRUCT": STRUCT,
            "VA_STRUCT": np.column_stack([V, A, STRUCT])}
    preds, table = {}, {}
    for k, M in mats.items():
        auc, oof = L.linear_oof_family1(M, y, groups, folds)
        preds[k + "_lin"] = oof
        table[k + "_lin"] = {"pooled": auc,
                             "within_cohort": within_group_auc(y, cohort, oof)}
        wc = table[k + "_lin"]["within_cohort"]
        print(f"  {k+'_lin':14s} pooled {auc:.4f}"
              + (f"  within-cohort {wc['pair_weighted']:.4f}" if wc else ""))
    for k in ["V", "VA", "A_real", "VA_STRUCT"]:
        oofs = [L.gbm_oof_family1(mats[k], y, groups, folds, s)["oof"]
                for s in L.GBM_SEEDS]
        mean_oof = np.mean(oofs, axis=0)
        preds[k + "_nl"] = mean_oof
        table[k + "_nl"] = {
            "pooled_auc_of_seed_mean_oof": float(roc_auc_score(y, mean_oof)),
            "within_cohort": within_group_auc(y, cohort, mean_oof)}
        print(f"  {k+'_nl':14s} pooled "
              f"{table[k+'_nl']['pooled_auc_of_seed_mean_oof']:.4f}")
    extra["pooled_vs_within_cohort"] = table

    # ---- 4. same-rows Delta_beyond -------------------------------------------
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
        va_nl = float(roc_auc_score(y[m], preds["VA_nl"][m]))
        vas_nl = float(roc_auc_score(y[m], preds["VA_STRUCT_nl"][m]))
        same[leg] = {
            "n": int(m.sum()), "n_pos": int(y[m].sum()),
            "VA_lin": float(roc_auc_score(y[m], preds["VA_lin"][m])),
            "VA_nl": va_nl, "VA_STRUCT_nl": vas_nl,
            "T_seed_mean": legT, "T_per_seed": aucs,
            "Delta_beyond": (legT - va_nl) if legT is not None else None,
            "Delta_beyond_over_VA_STRUCT": ((legT - vas_nl) if legT is not None
                                            else None),
        }
    extra["same_rows"] = same

    # ---- 5. cross-y contrasts on identical rows -------------------------------
    cross = {}
    for nm in ["y_fwd5_examiner", "y_fwd5_nonexaminer", "y_fwd_alltime",
               "y_fwd5_topquartile"]:
        yy = pd.to_numeric(pos.loc[order, nm], errors="coerce").values
        ok = np.isfinite(yy)
        if ok.sum() < 200 or len(np.unique(yy[ok])) < 2:
            continue
        cross[nm] = {
            "n": int(ok.sum()), "pos_rate": float(yy[ok].mean()),
            "phi_with_primary_y": float(np.corrcoef(y[ok], yy[ok])[0, 1]),
            "VA_nl_auc_against_this_y": float(
                roc_auc_score(yy[ok], preds["VA_nl"][ok])),
        }
    cross["_note"] = ("y_fwd_alltime is the AGE-CONFOUNDED comparator: it is the "
                      "count the task brief warns about, kept here only to show "
                      "what the fixed 5-year window buys. Never a headline.")
    extra["cross_y"] = cross

    # ---- 6. OOF reproduction check ------------------------------------------
    # Program rule: every OOF array ships ids, and the assembled-order AUC must
    # reproduce the published figure to < 1e-9. Reload from disk (not memory)
    # and re-derive, so a wrong write order cannot pass.
    np.save(RESULTS / f"{SLUG}_oof_ids.npy", np.array(ids, dtype=object))
    saved_ids = np.load(RESULTS / f"{SLUG}_oof_ids.npy", allow_pickle=True)
    saved_oof = np.load(RESULTS / f"{SLUG}_va_nl_oof_mean3.npy")
    assert list(saved_ids) == list(ids), "OOF id order does not match the matrix"
    pos_by_id = {d: i for i, d in enumerate(saved_ids)}
    reassembled = saved_oof[[pos_by_id[d] for d in ids]]
    repro = float(roc_auc_score(y, reassembled))
    published = float(res["ledger"]["VA_nl_mean"])
    # VA_nl_mean is the mean of per-seed AUCs; the AUC of the seed-mean OOF is a
    # different (also published) quantity -- check against the one this array IS.
    target = float(extra["pooled_vs_within_cohort"]["VA_nl"]
                   ["pooled_auc_of_seed_mean_oof"])
    extra["oof_reproduction"] = {
        "n_ids": int(len(saved_ids)),
        "auc_of_seed_mean_oof_from_disk": repro,
        "target_auc_of_seed_mean_oof": target,
        "abs_diff": abs(repro - target),
        "mean_of_per_seed_aucs": published,
        "passes_1e_9": bool(abs(repro - target) < 1e-9)}
    assert abs(repro - target) < 1e-9, \
        f"OOF reassembly {repro!r} != {target!r} (diff {abs(repro - target):.3g})"
    print(f"  [OOF repro] assembled-order AUC {repro:.12f} == published "
          f"{target:.12f} (diff {abs(repro - target):.2g}) PASS")

    res[f"{SLUG}_extras"] = extra
    Path(a.out).write_text(json.dumps(res, indent=2, default=str))
    print("\nwrote", a.out)
    lg = res["ledger"]
    print("\n=== LEDGER ===")
    for k in ("V_lin", "V_nl_mean", "A_lin", "VA_lin", "VA_nl_mean", "T",
              "Delta_interact", "Delta_total", "Delta_beyond"):
        if lg.get(k) is not None:
            print(f"  {k:16s} {lg[k]:+.4f}")
    print(f"  cohort_identity_oof {extra['cohort_identity']['oof_auc']:.4f}")


if __name__ == "__main__":
    main()
