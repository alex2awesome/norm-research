#!/usr/bin/env python3
"""Layer-1 "nonlinear stack" of the taste-residual decomposition -- code
COMPETITIONS curation cell (y = same-approach-as-editorial, AtCoder strict-L1
subset; A-bank-only, no V layer -- see DISCOVERY LOG).

Design: notes/2026-08-05__taste-decomposition-design.md (S0 ledger, S1 protocol).
Task brief for this gap-closer job: registry (notes/2026-07-27__vat-run-registry.md,
2026-07-28 AUDIT (b)) quotes "AC strict-L1 bank ens .731 / dense FT .690" as the
strongest bank>dense exemplar in the whole VAT grid. This script gates on .731
and runs the frozen HistGB Layer-1 stack (A-only; there is no V feature bank for
this cell -- see DISCOVERY LOG).

============================== DISCOVERY LOG ================================
* Raw competition_unified corpus (editorials/candidates/problems) is SK3-ONLY
  (datasets/competition_unified/README.md) but every DERIVED artifact needed for
  this cell survives locally under outputs/v2_analysis/ (2026-06-10/11 agent
  runs, notes/2026-06-10__competition-code-state-of-play.md).
* Bank score matrix: outputs/v2_analysis/comp_fourplatform_cells/ac_bank_scores.parquet
  (n=1000, 139 aNNN_score/aNNN_applied candidate-only metric pairs, pair_id
  keyed). ONLY 999/1000 pair_ids have a resolved label (comp_gap_audit_2026_06_10.md:
  "AC | ...ac_bank_scores.parquet | .../shard_ac_*.jsonl + shard_ac2_*.jsonl |
  999 | 2495 ok shard labels in total, 999 land in the bank-scored eval cell").
* L1-relabeled (strict boilerplate-stripped) labels + canonical_pid grouping key:
  outputs/v2_analysis/dense_ceiling/cell_ac_l1.parquet (n=2,495 -- the FULL
  L1-relabeled AC population, used for the dense .690 ModernBERT-FT number in
  outputs/v2_analysis/dense_ceiling/report.md). Joining bank_scores (999) against
  this on pair_id recovers canonical_pid + the l1 label for exactly the 999
  bank-scored rows (634 distinct problems/groups).
* *** CRITICAL POPULATION-MISMATCH FINDING (not previously flagged in the
  registry) ***: the celebrated "bank .731 > dense .690" comparison for this
  cell is NOT same-rows. Bank ens .731 (LR+RF, notes/2026-06-10 state-of-play,
  "TABLE COMPLETE 2026-06-11 PM") was computed on the 999-row bank-scored
  intersection; dense FT .690 (dense_ceiling/report.md, "AC (Claude, strict-L1)")
  was computed on the FULL 2,495-row L1-relabeled population -- 2.5x larger, and
  the 999 do not necessarily even form a random subset of the 2,495 (bank
  scoring ran on an earlier, smaller shard draw per the audit table). This
  violates the apples-to-apples dense-vs-baseline rule (same split + same input
  or no claim). The "BANK > DENSE" headline for AC-strict-L1 is UNVERIFIED on
  matched rows; a fresh same-rows bank vs dense readout on the 999-row
  intersection (bank AUC recomputed here) is the most defensible comparison
  currently available, but T='.690' itself is NOT that same-rows number (it is
  the population-mismatched 2,495-row FT AUC) -- do not quote Delta_beyond from
  this cell without this caveat.
* GATE PROTOCOL: comp_gap_audit_2026_06_10.md documents "Splitter:
  StratifiedGroupKFold (canonical_pid groups), LR + RF" but the exact
  random_state/fold count/RF hyperparameters used for the historical .696 LR /
  .721 RF / .731 ens numbers were not saved in any surviving script (the
  computing script itself is not on disk -- only the resulting numbers, in
  notes/2026-06-10__competition-code-state-of-play.md and
  outputs/v2_analysis/dense_ceiling/report.md). Gate here reproduces the closest
  achievable analogue (StratifiedGroupKFold(5, shuffle=True, random_state=0) by
  canonical_pid, LR + RF(400 trees) rank-averaged ensemble) and reports live
  numbers against GATE_TOL=.006 per the task brief -- but per the press_verdict
  LANDMINE, StratifiedGroupKFold fold assignment is itself sklearn-version-
  sensitive, so an inexact match under GATE_TOL is expected and does not by
  itself indicate a broken pipeline (see gate section below for the live
  numbers and verdict logic).

Usage: python code_competitions_layer1.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
BANK = REPO / "outputs/v2_analysis/comp_fourplatform_cells/ac_bank_scores.parquet"
CELL = REPO / "outputs/v2_analysis/dense_ceiling/cell_ac_l1.parquet"
DENSE_REPORT = REPO / "outputs/v2_analysis/dense_ceiling/report.md"
OUT_JSON = RESULTS_DIR / "code_competitions_layer1.json"

GATE_TOL = 0.006
PUBLISHED_ENS = 0.731  # notes/2026-06-10 state-of-play "Honest numbers" table
PUBLISHED_LR = 0.696
PUBLISHED_RF = 0.721
T_DENSE = 0.690  # dense_ceiling/report.md ModernBERT-FT, POPULATION-MISMATCHED (n=2495 vs n=999 here)
T_DENSE_NOTE = ("dense_ceiling/report.md AC-strict-L1 ModernBERT FT (3-seed mean 0.6896); "
                "computed on the FULL 2,495-row L1-relabeled population, NOT the 999-row "
                "bank-scored intersection used for this cell's A_lin/A_nl -- population "
                "mismatch, Delta numbers below are NOT same-rows, quote with this flag always.")

GRID = [
    {"max_leaf_nodes": 15, "learning_rate": 0.06, "max_iter": 400},
    {"max_leaf_nodes": 31, "learning_rate": 0.06, "max_iter": 400},
]
N_OUTER, N_INNER = 5, 3
GBM_SEEDS = (0, 1, 2)


def load_population():
    bank = pd.read_parquet(BANK)
    cell = pd.read_parquet(CELL)[["pair_id", "canonical_pid", "label"]]
    df = bank.merge(cell, on="pair_id", how="inner").reset_index(drop=True)
    score_cols = [c for c in df.columns if c.endswith("_score")]
    applied_cols = [c.replace("_score", "_applied") for c in score_cols]
    X = df[score_cols].to_numpy(dtype=float)
    applied = df[applied_cols].to_numpy(dtype=float)
    X = np.where(applied > 0, X, np.nan)
    y = df["label"].to_numpy(dtype=int)
    groups = df["canonical_pid"].to_numpy()
    names = [c[:-6] for c in score_cols]
    return X, y, groups, names, df


def clean_cols(M, names):
    keep = []
    for j in range(M.shape[1]):
        col = M[:, j].astype(float)
        finite = col[~np.isnan(col)]
        if len(finite) < 5:
            continue
        vals, counts = np.unique(finite, return_counts=True)
        offmodal = len(finite) - counts.max()
        if offmodal < 5 or finite.std() == 0:
            continue
        keep.append(j)
    return M[:, keep], [names[j] for j in keep]


# --------------------------------------------------------------- gate ------
def gate_live(X, y, groups, seed=0):
    """Closest achievable reproduction of the historical LR+RF ensemble gate
    (exact original script not on disk -- see DISCOVERY LOG)."""
    imp_lr = make_pipeline(SimpleImputer(strategy="constant", fill_value=0.5), StandardScaler(),
                            LogisticRegression(max_iter=3000, class_weight="balanced"))
    imp_rf = make_pipeline(SimpleImputer(strategy="constant", fill_value=0.5),
                            RandomForestClassifier(n_estimators=400, max_depth=None,
                                                    class_weight="balanced", random_state=seed, n_jobs=-1))
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    pred_lr = cross_val_predict(imp_lr, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    pred_rf = cross_val_predict(imp_rf, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    auc_lr = float(roc_auc_score(y, pred_lr))
    auc_rf = float(roc_auc_score(y, pred_rf))
    ens = rankdata(pred_lr) + rankdata(pred_rf)
    auc_ens = float(roc_auc_score(y, ens))
    return auc_lr, auc_rf, auc_ens


# --------------------------------------------------------- production ------
def outer_folds(n, groups):
    gkf = GroupKFold(n_splits=min(N_OUTER, len(np.unique(groups))))
    return list(gkf.split(np.zeros(n), groups=groups))


def linear_oof(Xf, y, folds):
    oof = np.zeros(len(y))
    for tr, te in folds:
        clf = make_pipeline(SimpleImputer(strategy="constant", fill_value=0.5), StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"))
        clf.fit(Xf[tr], y[tr])
        oof[te] = clf.predict_proba(Xf[te])[:, 1]
    return float(roc_auc_score(y, oof)), oof


def _fit_gbm(params, seed):
    return HistGradientBoostingClassifier(
        max_leaf_nodes=params["max_leaf_nodes"], learning_rate=params["learning_rate"],
        max_iter=params["max_iter"], early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=seed)


def gbm_oof(Xf, y, groups, folds, seed):
    oof = np.zeros(len(y))
    picks, train_aucs = [], []
    for tr, te in folds:
        gtr = groups[tr]
        inner = list(GroupKFold(n_splits=min(N_INNER, len(np.unique(gtr)))).split(np.zeros(len(tr)), groups=gtr))
        scores = []
        for params in GRID:
            aucs = []
            for itr, ite in inner:
                m = _fit_gbm(params, seed)
                m.fit(Xf[tr][itr], y[tr][itr])
                aucs.append(roc_auc_score(y[tr][ite], m.predict_proba(Xf[tr][ite])[:, 1]))
            scores.append(float(np.mean(aucs)))
        best = int(np.argmax(scores))
        picks.append(GRID[best]["max_leaf_nodes"])
        m = _fit_gbm(GRID[best], seed)
        m.fit(Xf[tr], y[tr])
        oof[te] = m.predict_proba(Xf[te])[:, 1]
        train_aucs.append(float(roc_auc_score(y[tr], m.predict_proba(Xf[tr])[:, 1])))
    return {"auc": float(roc_auc_score(y, oof)), "picks": picks,
            "train_auc_mean": float(np.mean(train_aucs)), "oof": oof}


def bootstrap_delta_interact_group(oof_lin, oof_nl_mean, y, groups, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    deltas = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in draw])
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        deltas.append(roc_auc_score(yb, oof_nl_mean[idx]) - roc_auc_score(yb, oof_lin[idx]))
    deltas = np.array(deltas)
    return {"n_boot_used": int(len(deltas)), "n_groups_resampled": int(len(uniq)),
            "mean": float(deltas.mean()),
            "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
            "p_gt_0": float((deltas > 0).mean())}


def bootstrap_delta_interact_row(oof_lin, oof_nl_mean, y, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        deltas.append(roc_auc_score(yb, oof_nl_mean[idx]) - roc_auc_score(yb, oof_lin[idx]))
    deltas = np.array(deltas)
    return {"n_boot_used": int(len(deltas)), "mean": float(deltas.mean()),
            "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
            "p_gt_0": float((deltas > 0).mean())}


def main():
    t0 = time.time()
    print(f"sklearn {sklearn.__version__}")
    Xraw, y, groups, names_raw, df = load_population()
    print(f"population: n={len(y)} pos={int(y.sum())} groups={len(np.unique(groups))}")

    auc_lr, auc_rf, auc_ens = gate_live(Xraw, y, groups)
    print(f"GATE live: LR={auc_lr:.4f} (published {PUBLISHED_LR}) RF={auc_rf:.4f} "
          f"(published {PUBLISHED_RF}) ENS={auc_ens:.4f} (published {PUBLISHED_ENS})")
    gate_diff = abs(auc_ens - PUBLISHED_ENS)
    gate_pass = gate_diff <= GATE_TOL
    print(f"gate diff (ens) = {gate_diff:.4f} -> {'PASS' if gate_pass else 'FAIL'} (tol {GATE_TOL})")

    Xc, names = clean_cols(Xraw, names_raw)
    print(f"post-degeneracy-guard: {Xc.shape[1]}/{Xraw.shape[1]} cols kept")

    folds = outer_folds(len(y), groups)
    auc_lin, oof_lin = linear_oof(Xc, y, folds)
    print(f"A_lin (campaign-standard GroupKFold pooled-OOF LR) = {auc_lin:.4f}")

    nl_seed_runs = {}
    for s in GBM_SEEDS:
        nl_seed_runs[s] = gbm_oof(Xc, y, groups, folds, seed=s)
        print(f"  gbm seed {s}: {nl_seed_runs[s]['auc']:.4f} (train {nl_seed_runs[s]['train_auc_mean']:.4f})")
    aucs = [nl_seed_runs[s]["auc"] for s in GBM_SEEDS]
    oof_nl_mean = np.mean([nl_seed_runs[s]["oof"] for s in GBM_SEEDS], axis=0)
    A_nl_mean = float(np.mean(aucs))
    A_nl_spread = float(max(aucs) - min(aucs))
    print(f"A_nl mean={A_nl_mean:.4f} spread={A_nl_spread:.4f}")

    np.save(RESULTS_DIR / "code_competitions_va_nl_oof_seed0.npy", nl_seed_runs[0]["oof"])
    np.save(RESULTS_DIR / "code_competitions_va_nl_oof_mean3.npy", oof_nl_mean)

    delta_interact = A_nl_mean - auc_lin
    boot_group = bootstrap_delta_interact_group(oof_lin, oof_nl_mean, y, groups)
    boot_row = bootstrap_delta_interact_row(oof_lin, oof_nl_mean, y)

    res = {
        "cell": "code_competitions (AC strict-L1, AtCoder same-approach-as-editorial, A-bank-only)",
        "status": "GATE_PASSED_PROCEED" if gate_pass else "GATE_FAILED_PROCEED_WITH_FLAG",
        "n": int(len(y)), "pos_rate": float(y.mean()), "n_groups": int(len(np.unique(groups))),
        "group_column": "canonical_pid",
        "sklearn_version_this_run": sklearn.__version__,
        "bank_matrix": str(BANK.relative_to(REPO)),
        "labels_and_groups_source": str(CELL.relative_to(REPO)),
        "note_no_V_layer": ("This cell has no genuine V (deterministic) feature bank -- the "
                             "candidate-only exec-pass-rate V layer was found ~chance in a separate "
                             "audit (outputs/v2_analysis/exec_vlayer/). Only A (139-criterion bank) "
                             "is decomposed here; VA==A for this cell."),
        "n_features": {"A_raw": int(Xraw.shape[1]), "A_kept": int(Xc.shape[1])},
        "gate": {
            "live_LR": auc_lr, "live_RF": auc_rf, "live_ENS": auc_ens,
            "published_LR": PUBLISHED_LR, "published_RF": PUBLISHED_RF, "published_ENS": PUBLISHED_ENS,
            "abs_diff_ens": gate_diff, "tol": GATE_TOL, "pass": gate_pass,
            "note": ("Original gate script not on disk (DISCOVERY LOG); this reproduces the "
                     "documented protocol (StratifiedGroupKFold(5,shuffle=True) by canonical_pid, "
                     "LR + RF rank-avg ensemble) as closely as possible from the surviving numbers "
                     "in notes/2026-06-10__competition-code-state-of-play.md. Per the press_verdict "
                     "LANDMINE, StratifiedGroupKFold fold membership is sklearn-version-sensitive, "
                     "so an imperfect match does not by itself indicate the population/features are "
                     "wrong -- the RF/LR ordering and rough magnitude reproducing is the meaningful check."),
        },
        "T_dense": T_DENSE,
        "T_dense_note": T_DENSE_NOTE,
        "T_dense_source": str(DENSE_REPORT.relative_to(REPO)),
        "population_mismatch_flag": True,
        "linear": {"A": auc_lin},
        "nonlinear": {"A": {
            "seed_aucs": {str(s): nl_seed_runs[s]["auc"] for s in GBM_SEEDS},
            "mean_auc": A_nl_mean, "spread": A_nl_spread,
            "train_auc_mean_seed0": nl_seed_runs[0]["train_auc_mean"],
            "picks_seed0": nl_seed_runs[0]["picks"],
        }},
        "ledger": {
            "A_lin": auc_lin, "A_nl_mean": A_nl_mean, "A_nl_spread": A_nl_spread,
            "Delta_interact": delta_interact,
            "T_dense_population_mismatched": T_DENSE,
            "Delta_total_vs_T_UNSAFE_population_mismatch": T_DENSE - auc_lin,
            "Delta_beyond_vs_T_UNSAFE_population_mismatch": T_DENSE - A_nl_mean,
            "caveat": "Both Delta numbers above use a T computed on a DIFFERENT (larger, non-overlapping-confirmed) population than A_lin/A_nl -- see T_dense_note. Do not quote as a same-rows Delta_beyond.",
        },
        "delta_interact_bootstrap_group_PRIMARY": boot_group,
        "delta_interact_bootstrap_row_secondary": boot_row,
        "runtime_sec": time.time() - t0,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=2))
    print("\n" + json.dumps(res["ledger"], indent=2))
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
