#!/usr/bin/env python3
"""V6 SO-votes: Layer-1 ledger + Layer-2(a) grouped transfer for the software-code
field's VOTE/REVEALED column.

REUSE: every estimator is IMPORTED, not reimplemented --
  * `layer1_gemma_cells` (L): outer_folds, linear_oof_family1, gbm_oof_family1,
    GBM_SEEDS -- the frozen Layer-1 spec.
  * `scaleupC_layer1` (SC): load_scaleupC_bank (re-rooted at this cell's own OUT
    dir via the `out=` argument), dense_T, group_bootstrap_delta,
    group_bootstrap_auc, and run_cell itself.
so this cell is numerically comparable to mathse_vote_score, jokes_community,
aops_curation and homepage_curation by construction.

WHAT THIS FILE ADDS on top of SC.run_cell (all cell-specific, none of it a
re-estimation):
  1. WITHIN-QUESTION readouts beside every pooled number, plus
     question-identity-alone measured FIRST. Question identity is this cell's
     docket-analog; the V8 N&C build showed a pooled AUC can be ~92% group
     composition, so the pooled figure is never quotable on its own.
  2. The POSITION LINE -- answer order within the question (the first-answer
     advantage) -- as an observed covariate, pooled and within-question.
  3. A_real (36 Track-A criteria) vs A_surface (4 declared Track-B probes)
     decomposition, available because the bank carries a `track` field.
  4. Same-rows Delta_beyond: VA predictions restricted to EXACTLY the dense
     split's rows, so T is not differenced against a pooled VA over a different
     row set.

  python3 methods/taste_decomposition/so_votes_layer1.py
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
SO_OUT = Path("outputs/va_gemma_banks_so_votes")
SO_DIR = REPO / "datasets/stackoverflow-votes/va"
DENSE = SO_DIR / "dense_standard_so_votes"
SLUG = "so_votes"


COLLAPSE_MODAL_MAX = 0.98  # ENFORCED gate (ruling): drop, do not merely report


def modal_share(col):
    """Modal share computed the same way as readout_va_gemma.univariate:
    NA imputed to the column median first, so a criterion that is NA-heavy AND
    constant on its answered rows is caught."""
    v = col.copy()
    fin = np.isfinite(v)
    v[~fin] = float(np.nanmedian(v)) if fin.any() else 0.5
    _, cnts = np.unique(v, return_counts=True)
    return float(cnts.max() / len(v))


def cell_so_votes(apply_collapse_gate=True):
    out = REPO / SO_OUT
    meta, A, V, groups, shard, ids = SC.load_scaleupC_bank("so_votes", out=out)
    y = np.array(meta["ys"]["vote_score"], dtype=float)
    keep = np.isfinite(y)  # vote_score is undefined on median-tied rows
    A, V = A[keep], V[keep]
    y = y[keep].astype(int)
    groups, ids, shard = groups[keep], ids[keep], shard[keep]

    meta = dict(meta)
    a_names = list(meta["a_names"])

    # ---- ENFORCED COLLAPSE GATE ------------------------------------------
    # Ruling: a criterion whose modal response share exceeds .98 carries no
    # rank information and is DROPPED from the A matrix, not merely flagged.
    # Computed on this cell's own analysis rows (post tie-drop), never on y.
    gate = {"modal_max": COLLAPSE_MODAL_MAX, "dropped": [], "kept_n": None,
            "modal_by_criterion": {}}
    shares = np.array([modal_share(A[:, c]) for c in range(A.shape[1])])
    for nm, s in zip(a_names, shares):
        gate["modal_by_criterion"][nm] = round(float(s), 4)
    if apply_collapse_gate:
        keep_c = shares <= COLLAPSE_MODAL_MAX
        gate["dropped"] = [{"criterion": nm, "modal_share": round(float(s), 4)}
                           for nm, s in zip(a_names, shares) if s > COLLAPSE_MODAL_MAX]
        A = A[:, keep_c]
        a_names = [nm for nm, k in zip(a_names, keep_c) if k]
        meta["a_names"] = a_names
    gate["kept_n"] = int(A.shape[1])
    print(f"[collapse gate] modal>{COLLAPSE_MODAL_MAX}: dropped "
          f"{len(gate['dropped'])} of {len(shares)} criteria -> A has "
          f"{A.shape[1]} columns")
    for dd in gate["dropped"]:
        print(f"    DROPPED {dd['criterion']}  modal={dd['modal_share']}")

    T, Tinfo = SC.dense_T(DENSE)
    return dict(
        title="StackOverflow [python] answer votes "
              "(within-question median split of the raw net vote score)",
        A=A, V=V, y=y, groups=groups, ids=ids, meta=meta, shard_of=shard,
        group_column="question_id", T=T, T_info=Tinfo,
        matrix=f"{SO_OUT}/so_votes_shard*.npz (y = vote_score; the vote and "
               "accept y's share ONE scored matrix and are never merged)",
        dense_dir=str(DENSE),
        prior_published=None, keep_mask=keep, collapse_gate=gate)


def within_group_auc(y, groups, pred, weighting="pair"):
    """Pair-weighted within-question AUC over questions carrying both classes."""
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
    ap.add_argument("--out", default=str(RESULTS / "so_votes_ledger.json"))
    a = ap.parse_args()

    # ---- run the frozen Layer-1 stack via the shared runner -----------------
    # SC.run_cell reads the extended anchor battery from SC.SCALEUPC_OUT; point
    # it at this cell's own output dir so the K=50 battery is picked up. (The
    # scale-up-C matrices are never touched -- the cell's bank is loaded with an
    # explicit out= in cell_so_votes.)
    SC.SCALEUPC_OUT = REPO / SO_OUT
    SC.CELLS[SLUG] = cell_so_votes
    res = SC.run_cell(SLUG)

    # ---- reload the cell for the cell-specific extras -----------------------
    d = cell_so_votes()
    y, groups, ids = d["y"], d["groups"], d["ids"]
    A, V = d["A"], d["V"]
    meta = d["meta"]
    a_names = list(meta["a_names"])
    rubrics = [json.loads(l) for l in
               open(SO_DIR / "rubrics.jsonl") if l.strip()]
    track = {r["name"]: r["track"] for r in rubrics}
    is_real = np.array([track.get(n, "A") == "A" for n in a_names])
    print(f"\n[bank split] A_real={int(is_real.sum())} "
          f"A_surface={int((~is_real).sum())}")

    pop = pd.read_csv(SO_DIR / "population.csv.gz")
    pop["row_id"] = pop.row_id.astype(str)
    pos = pop.set_index("row_id")
    order = [str(i) for i in ids]
    position = pos.loc[order, "position"].values.astype(float)
    y_acc = pos.loc[order, "y_accepted"].values.astype(int)
    split = pos.loc[order, "split"].values
    body_len = pos.loc[order, "body"].astype(str).str.len().values.astype(float)

    folds = L.outer_folds(len(y), groups, n_splits=5)
    extra = {}

    # ---- 1. question identity alone, FIRST ---------------------------------
    qmean = pd.Series(y).groupby(pd.Series(groups)).transform("mean").values
    extra["question_identity_alone_auc"] = float(roc_auc_score(y, qmean))

    # ---- 2. the position line (observed covariate) -------------------------
    extra["position_line"] = {
        "pooled_auc_first_is_better": float(roc_auc_score(y, -position)),
        "within_question": within_group_auc(y, groups, -position),
        "by_position": {str(int(p)): {
            "n": int((np.minimum(position, 5) == p).sum()),
            "vote_pos_rate": float(y[np.minimum(position, 5) == p].mean())}
            for p in sorted(set(np.minimum(position, 5)))},
    }
    extra["charlen_alone_auc"] = float(roc_auc_score(y, body_len))
    extra["charlen_within_question"] = within_group_auc(y, groups, body_len)

    # ---- 3. pooled + within-question for every matrix ----------------------
    mats = {"V": V, "A": A, "VA": np.column_stack([V, A]),
            "A_real": A[:, is_real], "A_surface": A[:, ~is_real],
            "V_A_real": np.column_stack([V, A[:, is_real]])}
    preds, table = {}, {}
    for k, M in mats.items():
        auc, oof = L.linear_oof_family1(M, y, groups, folds)
        preds[k + "_lin"] = oof
        table[k + "_lin"] = {"pooled": auc,
                             "within_question": within_group_auc(y, groups, oof)}
        print(f"  {k+'_lin':14s} pooled {auc:.4f}  "
              f"within-q {table[k+'_lin']['within_question']['pair_weighted']:.4f}")
    for k in ["V", "VA", "A_real"]:
        oofs = [L.gbm_oof_family1(mats[k], y, groups, folds, s)["oof"]
                for s in L.GBM_SEEDS]
        mean_oof = np.mean(oofs, axis=0)
        preds[k + "_nl"] = mean_oof
        table[k + "_nl"] = {
            "pooled_auc_of_seed_mean_oof": float(roc_auc_score(y, mean_oof)),
            "within_question": within_group_auc(y, groups, mean_oof)}
        print(f"  {k+'_nl':14s} pooled {table[k+'_nl']['pooled_auc_of_seed_mean_oof']:.4f}"
              f"  within-q {table[k+'_nl']['within_question']['pair_weighted']:.4f}")
    extra["pooled_vs_within_question"] = table

    # program-convention within-group figure (min_n=20, n-weighted), for the
    # Layer-2 appendix table -- computed the same way as layer2_robustness
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
        legT = None
        raw = (Tinfo or {}).get("raw", {})
        runs = raw.get("runs", raw) if isinstance(raw, dict) else {}
        aucs = [float(v[f"{leg}_auc"]) for v in runs.values()
                if isinstance(v, dict) and f"{leg}_auc" in v]
        if aucs:
            legT = float(np.mean(aucs))
        same[leg] = {
            "n": int(m.sum()), "n_pos": int(y[m].sum()),
            "VA_lin": float(roc_auc_score(y[m], preds["VA_lin"][m])),
            "VA_nl": float(roc_auc_score(y[m], preds["VA_nl"][m])),
            "T_seed_mean": legT,
            "Delta_beyond": (legT - float(roc_auc_score(y[m], preds["VA_nl"][m]))
                             if legT is not None else None),
            "within_question_VA_nl": within_group_auc(y[m], groups[m],
                                                      preds["VA_nl"][m]),
        }
    extra["same_rows"] = same

    # ---- 5. cross-y contrast on identical rows ------------------------------
    extra["cross_y"] = {
        "n": int(len(y)),
        "accept_rate_when_vote_pos": float(y_acc[y == 1].mean()),
        "accept_rate_when_vote_neg": float(y_acc[y == 0].mean()),
        "phi": float(np.corrcoef(y, y_acc)[0, 1]),
        "VA_nl_auc_against_accept_y": float(roc_auc_score(y_acc, preds["VA_nl"])),
    }

    # ---- 6. ids-carried OOF + <1e-9 reproduction gate (ruling) --------------
    # Every OOF vector is persisted WITH its item ids in the exact row order it
    # was scored in, and each is required to reproduce its ledger AUC to <1e-9.
    # Without ids a later consumer cannot verify it is aligning the same rows;
    # the reproduction assert is what makes the saved vector quotable.
    repro = {"tol": 1e-9, "checks": {}, "all_pass": True}
    ledger_targets = {
        "V_lin": res["linear"]["V"], "A_lin": res["linear"]["A"],
        "VA_lin": res["linear"]["VA"],
    }
    for k, target in ledger_targets.items():
        got = float(roc_auc_score(y, preds[k]))
        # NOTE: preds[] here are refits over the SAME folds/estimators as
        # run_cell's, so they must agree bit-for-bit with the ledger legs.
        d_ = abs(got - target)
        repro["checks"][k] = {"ledger": target, "recomputed": got,
                              "abs_diff": d_, "pass": bool(d_ < 1e-9)}
        if d_ >= 1e-9:
            repro["all_pass"] = False
    # the nonlinear ledger quantity is the MEAN OF SEED AUCS, not the AUC of the
    # mean OOF -- both are recorded so neither is mistaken for the other
    for k in ["V", "VA"]:
        seeds = [res["nonlinear"][k][str(s)]["auc"] for s in L.GBM_SEEDS]
        repro["checks"][f"{k}_nl_mean_of_seed_aucs"] = {
            "ledger": float(np.mean(seeds)), "seeds": seeds,
            "auc_of_seed_mean_oof": float(roc_auc_score(y, preds[k + "_nl"])),
            "note": "these two are DIFFERENT quantities (ensembling gain); the "
                    "ledger row is the mean of seed AUCs"}
    extra["oof_reproduction"] = repro
    print(f"[repro <1e-9] all_pass={repro['all_pass']}")
    for k, v in repro["checks"].items():
        if "abs_diff" in v:
            print(f"    {k:8s} ledger {v['ledger']:.10f} "
                  f"recomputed {v['recomputed']:.10f} diff {v['abs_diff']:.2e} "
                  f"{'OK' if v['pass'] else 'FAIL'}")

    oof_path = RESULTS / "so_votes_oof_with_ids.npz"
    np.savez_compressed(
        oof_path,
        ids=np.array([str(i) for i in ids], dtype=object),
        groups=np.array([str(g) for g in groups], dtype=object),
        y=y, position=position, y_accepted=y_acc,
        split=np.array([str(s) for s in split], dtype=object),
        **{k: v for k, v in preds.items()})
    print("wrote", oof_path)
    extra["oof_artifact"] = str(oof_path)

    res["collapse_gate"] = d["collapse_gate"]
    res["so_votes_extras"] = extra
    Path(a.out).write_text(json.dumps(res, indent=2, default=str))
    print("\nwrote", a.out)
    lg = res["ledger"]
    print("\n=== LEDGER ===")
    for k in ("V_lin", "V_nl_mean", "A_lin", "VA_lin", "VA_nl_mean", "T",
              "Delta_interact", "Delta_total", "Delta_beyond"):
        if lg.get(k) is not None:
            print(f"  {k:16s} {lg[k]:+.4f}")
    print(f"  question_identity_alone {extra['question_identity_alone_auc']:.4f}")


if __name__ == "__main__":
    main()
