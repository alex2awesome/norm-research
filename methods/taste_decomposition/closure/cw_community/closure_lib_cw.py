#!/usr/bin/env python3
"""Shared machinery for the CW-community (writingprompts upvotes) confirmatory
Layer-3 articulation-closure campaign.

Mirrors methods/taste_decomposition/closure/closure_lib.py (the peer-verdict
pilot) but carries THIS cell's frozen Layer-1 spec, which is `family1` in
methods/taste_decomposition/layer1_gemma_cells.py:

  linear : SimpleImputer(median, add_indicator=True) + StandardScaler +
           LogisticRegression(C=1, solver='liblinear', max_iter=2000,
                              random_state=20260728)
  nonlin : HistGradientBoostingClassifier, max_leaf_nodes in {15,31},
           learning_rate .06, max_iter 400, early stopping (val .1, patience 20),
           fed the SAME per-fold imputed+indicator matrix as the linear model
  grid   : inner GroupKFold(3) inside train folds only
  outer  : GroupKFold(5) on prompt_id
  VA_nl  : mean over seeds {0,1,2}, spread reported (FREEZE CHANGE 1)

Everything is fit WITHIN FIT+MINE only; MONITOR is never read by any fit,
imputer, or column screen.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent

GRID = [
    {"max_leaf_nodes": 15, "learning_rate": 0.06, "max_iter": 400},
    {"max_leaf_nodes": 31, "learning_rate": 0.06, "max_iter": 400},
]
N_OUTER = 5
N_INNER = 3
SEEDS = (0, 1, 2)
LINEAR_SEED = 20260728
FIT_MINE_THRESH = 0.80


def hash_unit(key: str) -> float:
    """Stable sha256 -> [0,1). Never a seeded shuffle."""
    return int(hashlib.sha256(str(key).encode("utf-8")).hexdigest(), 16) / float(1 << 256)


# ------------------------------------------------------------------ models ---
def _fit_gbm(params, seed):
    return HistGradientBoostingClassifier(
        max_leaf_nodes=params["max_leaf_nodes"],
        learning_rate=params["learning_rate"],
        max_iter=params["max_iter"],
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=seed,
    )


def _linear():
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=1.0, solver="liblinear", max_iter=2000,
                           random_state=LINEAR_SEED))


def _pick_grid(Xtr, ytr, gtr, seed):
    inner = list(GroupKFold(n_splits=min(N_INNER, len(np.unique(gtr))))
                 .split(np.zeros(len(ytr)), groups=gtr))
    scores = []
    for params in GRID:
        aucs = []
        for itr, ite in inner:
            m = _fit_gbm(params, seed)
            m.fit(Xtr[itr], ytr[itr])
            aucs.append(roc_auc_score(ytr[ite], m.predict_proba(Xtr[ite])[:, 1]))
        scores.append(float(np.mean(aucs)))
    return GRID[int(np.argmax(scores))], scores


def oof_within(Xraw, y, groups, seed=0):
    """Grouped OOF inside the given rows (FIT+MINE), family1 preprocessing fit
    per train fold.  Returns (linear_oof, gbm_oof, picks)."""
    folds = list(GroupKFold(n_splits=min(N_OUTER, len(np.unique(groups))))
                 .split(np.zeros(len(y)), groups=groups))
    lin = np.full(len(y), np.nan)
    nl = np.full(len(y), np.nan)
    picks = []
    for tr, te in folds:
        imp = SimpleImputer(strategy="median", add_indicator=True)
        Xtr = imp.fit_transform(Xraw[tr])
        Xte = imp.transform(Xraw[te])

        pipe = _linear()
        pipe.fit(Xraw[tr], y[tr])
        lin[te] = pipe.predict_proba(Xraw[te])[:, 1]

        params, _ = _pick_grid(Xtr, y[tr], groups[tr], seed)
        picks.append(params["max_leaf_nodes"])
        m = _fit_gbm(params, seed)
        m.fit(Xtr, y[tr])
        nl[te] = m.predict_proba(Xte)[:, 1]
    return lin, nl, picks


def fit_predict_monitor(Xfit_raw, yfit, gfit, Xmon_raw, seeds=SEEDS):
    """Refit on ALL FIT+MINE rows (imputer learned on FIT+MINE only), predict
    MONITOR.  Returns (linear_preds, per-seed GBM preds array, picks)."""
    pipe = _linear()
    pipe.fit(Xfit_raw, yfit)
    lin_mon = pipe.predict_proba(Xmon_raw)[:, 1]

    imp = SimpleImputer(strategy="median", add_indicator=True)
    Xfit = imp.fit_transform(Xfit_raw)
    Xmon = imp.transform(Xmon_raw)
    nl_mon, picks = [], []
    for s in seeds:
        params, _ = _pick_grid(Xfit, yfit, gfit, s)
        picks.append(params["max_leaf_nodes"])
        m = _fit_gbm(params, s)
        m.fit(Xfit, yfit)
        nl_mon.append(m.predict_proba(Xmon)[:, 1])
    return lin_mon, np.array(nl_mon), picks


def fit_block(Xraw, y, groups, split, seeds=SEEDS):
    """One complete frozen-protocol pass on a bank state.

    Returns dict with:
      va_lin_mon / va_nl_mon (seed-mean) AUC on MONITOR,
      per-seed MONITOR AUCs, and the HONEST-POPULATION prediction vector
      (grouped OOF inside FIT+MINE, refit-on-FIT+MINE prediction on the held-out
      MONITOR and TEST rows) so a same-rows Delta can be read on every split from
      one fit.  TEST is predicted but its AUC is only quoted at the stopping round.
    """
    fm = split == "fit_mine"
    mo = split == "monitor"
    held = ~fm                      # monitor + test: one refit predicts both
    lin_oof, nl_oof_s0, picks_oof = oof_within(Xraw[fm], y[fm], groups[fm], seed=seeds[0])
    nl_oof_seeds = [nl_oof_s0]
    for s in seeds[1:]:
        _, nls, _ = oof_within(Xraw[fm], y[fm], groups[fm], seed=s)
        nl_oof_seeds.append(nls)
    nl_oof = np.mean(nl_oof_seeds, axis=0)

    lin_h, nl_h_seeds, picks_mon = fit_predict_monitor(
        Xraw[fm], y[fm], groups[fm], Xraw[held], seeds=seeds)
    nl_h = nl_h_seeds.mean(axis=0)

    pop_lin = np.full(len(y), np.nan)
    pop_nl = np.full(len(y), np.nan)
    pop_lin[fm], pop_nl[fm] = lin_oof, nl_oof
    pop_lin[held], pop_nl[held] = lin_h, nl_h

    mo_in_held = mo[held]           # index MONITOR inside the held-out block
    return {
        "va_lin_monitor": float(roc_auc_score(y[mo], lin_h[mo_in_held])),
        "va_nl_monitor": float(np.mean([roc_auc_score(y[mo], p[mo_in_held])
                                        for p in nl_h_seeds])),
        "va_nl_monitor_seeds": [float(roc_auc_score(y[mo], p[mo_in_held]))
                                for p in nl_h_seeds],
        "va_lin_fitmine_oof": float(roc_auc_score(y[fm], lin_oof)),
        "va_nl_fitmine_oof": float(roc_auc_score(y[fm], nl_oof)),
        "va_lin_population": float(roc_auc_score(y, pop_lin)),
        "va_nl_population": float(roc_auc_score(y, pop_nl)),
        "picks_oof": picks_oof, "picks_monitor": picks_mon,
        "n_features": int(Xraw.shape[1]),
        "_pop_lin": pop_lin, "_pop_nl": pop_nl,
        # per-seed predictions on the held-out block, so the NEXT round can
        # reconstruct this bank state's readouts exactly without refitting
        "_held_seed_preds": nl_h_seeds, "_held_mask": held,
    }


def monitor_stats_from_saved(y, split, held_seed_preds, held_mask):
    """Recover fit_block's MONITOR statistics from saved per-seed predictions."""
    mo = split == "monitor"
    mo_in_held = mo[held_mask]
    per_seed = [float(roc_auc_score(y[mo], p[mo_in_held])) for p in held_seed_preds]
    return {"va_nl_monitor": float(np.mean(per_seed)),
            "va_nl_monitor_seeds": per_seed}


def auc(y, p):
    return float(roc_auc_score(y, p))


# ---------------------------------------------------- stratified readouts ----
def stratified_auc(y, p, strata, min_n=25):
    y, p, strata = np.asarray(y), np.asarray(p), np.asarray(strata)
    tot, num, used, dropped = 0, 0.0, 0, 0
    for s in np.unique(strata):
        m = strata == s
        n = int(m.sum())
        if n < min_n or len(set(y[m])) < 2:
            dropped += n
            continue
        num += n * roc_auc_score(y[m], p[m])
        tot += n
        used += 1
    if tot == 0:
        return float("nan"), {"n_strata_used": 0, "n_rows_used": 0,
                              "n_rows_dropped": dropped}
    return float(num / tot), {"n_strata_used": used, "n_rows_used": int(tot),
                              "n_rows_dropped": int(dropped)}


def decile_strata(x, q=10):
    x = np.asarray(x, dtype=float)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, q + 1)))
    if len(edges) < 3:
        return np.zeros(len(x), dtype=int)
    return np.clip(np.digitize(x, edges[1:-1], right=True), 0, len(edges) - 2)


def group_bootstrap_delta(y, pa, pb, groups, n_boot=2000, seed=12345):
    """Group-level paired bootstrap of AUC(pa) - AUC(pb)."""
    y, pa, pb, groups = (np.asarray(y), np.asarray(pa), np.asarray(pb),
                         np.asarray(groups))
    uq = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in uq}
    rng = np.random.default_rng(seed)
    point = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    ds = []
    for _ in range(n_boot):
        gs = rng.choice(uq, size=len(uq), replace=True)
        idx = np.concatenate([idx_of[g] for g in gs])
        ys = y[idx]
        if ys.min() == ys.max():
            continue
        ds.append(roc_auc_score(ys, pa[idx]) - roc_auc_score(ys, pb[idx]))
    ds = np.array(ds)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return {"estimate": float(point), "ci95": [float(lo), float(hi)],
            "p_gt_0": float((ds > 0).mean()), "n_boot": int(len(ds))}
