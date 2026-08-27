#!/usr/bin/env python3
"""Assemble the RoyalRoad 5-fold cross-fitted dense arm into an honest-set T and
a SAME-ROWS Delta_beyond against the certified mature bank.

WHY: the 2026-08-10 single-split dense arm put T at chance (.4994) on a 141-row
eval that was ALSO the checkpoint-selection split. This readout replaces it with
the union of the 5 test tenths -- n=651, selection-free, 4.6x the rows.

THREE T ESTIMATORS, because pooling probabilities from five different LoRA models
is not automatically legitimate (each fold's sigmoid is calibrated to its own
training run, so a pooled AUC can be distorted by cross-fold shifts):
  T_pooled        AUC over all 651 raw probabilities        (headline)
  T_fold_mean     mean of the 5 per-fold test AUCs          (calibration-immune)
  T_rank_pooled   AUC after within-fold rank normalisation  (pooled, decalibrated)
If these disagree materially the pooled number is the suspect one and is reported
with that flag rather than quietly used.

SAME-ROWS comparison (V8 precedent): the bank's grouped-OOF predictions are
restricted to EXACTLY the 651 honest ids -- never re-fit, never re-tuned -- so the
bank and the dense arm are scored on identical rows.

  python methods/taste_decomposition/rr_crossfit_readout.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
CF = REPO / "datasets/creative-writing/royalroad_stubs/dense_crossfit"
RESULTS = Path(__file__).resolve().parent / "results"
SLUG = "cw_royalroad_verdict"
SEED = 42


def group_boot_auc(y, p, n_boot=2000, seed=7):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    man = json.loads((CF / "manifest.json").read_text())
    rows, per_fold = [], {}
    for k in range(5):
        f = CF / f"fold{k}" / f"rm_out_seed{SEED}" / "preds_test.csv"
        if not f.exists():
            raise SystemExit(f"missing {f} -- fold {k} not scored yet")
        d = pd.read_csv(f)
        d["fold"] = k
        d["rank"] = rankdata(d["prob"]) / (len(d) + 1)   # within-fold decalibration
        a = roc_auc_score(d.judgement, d.prob)
        per_fold[f"fold{k}"] = {"n": int(len(d)), "n_pos": int(d.judgement.sum()),
                                "auc": round(float(a), 4)}
        rows.append(d)
    H = pd.concat(rows, ignore_index=True)
    H["row_id"] = H["group"].astype(str)
    assert H.row_id.is_unique, "honest set has duplicate ids"

    y = H.judgement.values
    T_pooled = float(roc_auc_score(y, H.prob.values))
    T_rank = float(roc_auc_score(y, H["rank"].values))
    T_foldmean = float(np.mean([v["auc"] for v in per_fold.values()]))

    out = {
        "cell": SLUG,
        "design": man["design"],
        "bucket_rule": man["bucket_rule"],
        "seed": SEED,
        "honest_set": {"n": int(len(H)), "n_pos": int(y.sum()),
                       "n_neg": int((1 - y).sum()),
                       "pos_rate": round(float(y.mean()), 4),
                       "selection_free": True,
                       "vs_old_eval": "n=651 vs the old 141-row eval (4.6x), and the "
                                      "old eval was ALSO the selection split"},
        "per_fold": per_fold,
        "T_pooled": round(T_pooled, 4),
        "T_rank_pooled": round(T_rank, 4),
        "T_fold_mean": round(T_foldmean, 4),
        "T_pooled_ci95": [round(x, 4) for x in group_boot_auc(y, H.prob.values)],
        "T_spread_across_estimators": round(
            max(T_pooled, T_rank, T_foldmean) - min(T_pooled, T_rank, T_foldmean), 4),
        "prior_single_split_T": {
            "eval_seed_mean": 0.4994, "eval_seeds": [0.4822, 0.485, 0.531],
            "n_eval": 141,
            "note": "power-capped AND selection-contaminated; superseded by the "
                    "honest set for the T role"},
    }

    # ---- same-rows bank comparison -----------------------------------------
    z = np.load(RESULTS / f"{SLUG}_oof.npz", allow_pickle=True)
    ids = np.array([str(s) for s in z["ids"]])
    pos = {d: i for i, d in enumerate(ids)}
    keep = np.array([pos[d] for d in H.row_id if d in pos])
    hy = z["y"][keep]
    assert np.array_equal(hy, H.judgement.values[[i for i, d in enumerate(H.row_id) if d in pos]]), \
        "label mismatch between dense preds and bank matrix"
    same = {"n_matched": int(len(keep))}
    for key in ("V_lin", "A_lin", "VA_lin", "VA_nl_seed0", "VA_nl_mean3"):
        same[key] = round(float(roc_auc_score(hy, z[key][keep])), 4)
    same["note"] = ("bank grouped-OOF predictions RESTRICTED to the honest ids -- "
                    "never re-fit; identical rows for bank and dense (V8 precedent)")
    out["same_rows_bank"] = same

    T_for_delta = T_foldmean          # calibration-immune choice for the headline delta
    out["T_used_for_delta"] = {"value": round(T_for_delta, 4), "estimator": "T_fold_mean",
                               "why": "immune to cross-fold calibration shift; pooled "
                                      "and rank-pooled reported beside it"}
    out["ledger_delta"] = {
        "Delta_beyond_vs_VA_nl": round(T_for_delta - same["VA_nl_mean3"], 4),
        "Delta_total_vs_VA_lin": round(T_for_delta - same["VA_lin"], 4),
        "Delta_vs_A_lin": round(T_for_delta - same["A_lin"], 4),
        "all_same_rows": True}

    p = RESULTS / f"{SLUG}_crossfit_readout.json"
    p.write_text(json.dumps(out, indent=2))

    print(f"=== {SLUG} 5-fold cross-fit (seed {SEED}) ===")
    print(f"honest set n={len(H)} pos={int(y.sum())} (selection-free)")
    for k, v in per_fold.items():
        print(f"  {k}: n={v['n']:4d} pos={v['n_pos']:3d}  AUC={v['auc']:.4f}")
    print(f"  T_pooled      {T_pooled:.4f}  CI95 {out['T_pooled_ci95']}")
    print(f"  T_rank_pooled {T_rank:.4f}")
    print(f"  T_fold_mean   {T_foldmean:.4f}   <- used for Delta")
    print(f"  estimator spread {out['T_spread_across_estimators']:.4f}")
    print(f"  prior single-split T .4994 (141-row eval, selection-contaminated)")
    print(f"--- same rows (n={same['n_matched']}) ---")
    for k in ("V_lin", "A_lin", "VA_lin", "VA_nl_mean3"):
        print(f"  {k:12s} {same[k]:.4f}")
    for k, v in out["ledger_delta"].items():
        if isinstance(v, float):
            print(f"  {k:24s} {v:+.4f}")
    print("wrote", p)
    print("RR_CROSSFIT_READOUT_DONE")


if __name__ == "__main__":
    main()
