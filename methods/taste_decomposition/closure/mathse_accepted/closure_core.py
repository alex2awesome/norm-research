#!/usr/bin/env python3
"""Generic (any-cell) version of closure_lib.py for the map-focused batch.

The fitting spec is the FROZEN Layer-1 protocol, byte-for-byte the same as
closure/closure_lib.py (which the pilot and the missing-mass battery both used):

  linear : StandardScaler + LogisticRegression(C=1, max_iter=2000)
  nonlin : HistGradientBoostingClassifier, max_leaf_nodes in {15,31},
           learning_rate .06, max_iter 400, early stopping (val .1, patience 20)
  grid   : inner GroupKFold(3) inside train folds only
  outer  : GroupKFold(5) on the cell's group key
  VA_nl  : mean over seeds {0,1,2} (FREEZE CHANGE 1)

Everything is fit inside FIT+MINE only: the degeneracy screen, the imputation
medians, the scaler and the grid selection never see a MONITOR row.

CPU only.
"""
from __future__ import annotations

import hashlib

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GRID = [
    {"max_leaf_nodes": 15, "learning_rate": 0.06, "max_iter": 400},
    {"max_leaf_nodes": 31, "learning_rate": 0.06, "max_iter": 400},
]
N_OUTER = 5
N_INNER = 3
SEEDS = (0, 1, 2)


def hash_unit(key: str) -> float:
    return int(hashlib.sha256(str(key).encode("utf-8")).hexdigest(), 16) / float(1 << 256)


# ------------------------------------------------------- column cleaning ----
def clean_fit(M):
    keep, meds = [], []
    for j in range(M.shape[1]):
        col = M[:, j].astype(float)
        nonna = col[~np.isnan(col)]
        if len(nonna) == 0:
            continue
        med = float(np.median(nonna))
        c = np.where(np.isnan(col), med, col)
        vals, counts = np.unique(c, return_counts=True)
        offmodal = len(c) - counts.max()
        if offmodal < 5 or c.std() == 0:
            continue
        keep.append(j)
        meds.append(med)
    return np.array(keep, dtype=int), np.array(meds, dtype=float)


def clean_apply(M, keep, meds):
    if len(keep) == 0:
        return np.zeros((M.shape[0], 0))
    sub = M[:, keep].astype(float).copy()
    for k in range(sub.shape[1]):
        col = sub[:, k]
        col[np.isnan(col)] = meds[k]
    return sub


# ---------------------------------------------------------------- models ----
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


def pick_grid(Xf, y, groups, seed):
    inner = list(GroupKFold(n_splits=min(N_INNER, len(np.unique(groups)))).split(
        np.zeros(len(y)), groups=groups))
    scores = []
    for params in GRID:
        aucs = []
        for itr, ite in inner:
            m = _fit_gbm(params, seed)
            m.fit(Xf[itr], y[itr])
            aucs.append(roc_auc_score(y[ite], m.predict_proba(Xf[ite])[:, 1]))
        scores.append(float(np.mean(aucs)))
    return GRID[int(np.argmax(scores))], scores


def gbm_oof(Xf, y, groups, seed=0):
    folds = list(GroupKFold(n_splits=min(N_OUTER, len(np.unique(groups)))).split(
        np.zeros(len(y)), groups=groups))
    oof = np.zeros(len(y))
    picks = []
    for tr, te in folds:
        params, _ = pick_grid(Xf[tr], y[tr], groups[tr], seed)
        picks.append(params["max_leaf_nodes"])
        m = _fit_gbm(params, seed)
        m.fit(Xf[tr], y[tr])
        oof[te] = m.predict_proba(Xf[te])[:, 1]
    return oof, picks


def linear_oof(Xf, y, groups):
    folds = list(GroupKFold(n_splits=min(N_OUTER, len(np.unique(groups)))).split(
        np.zeros(len(y)), groups=groups))
    oof = np.zeros(len(y))
    for tr, te in folds:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(Xf[tr], y[tr])
        oof[te] = clf.predict_proba(Xf[te])[:, 1]
    return oof


def fit_predict_monitor(Xfit, yfit, gfit, Xmon, seeds=SEEDS):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    clf.fit(Xfit, yfit)
    lin_mon = clf.predict_proba(Xmon)[:, 1]
    nl_mon, picks = [], []
    for s in seeds:
        params, _ = pick_grid(Xfit, yfit, gfit, s)
        picks.append(params["max_leaf_nodes"])
        m = _fit_gbm(params, s)
        m.fit(Xfit, yfit)
        nl_mon.append(m.predict_proba(Xmon)[:, 1])
    return lin_mon, np.array(nl_mon), picks


def fit_block(raw_blocks, fitmask, monmask, y, groups, seeds=SEEDS, want_oof=True):
    """clean_fit each raw block on FIT+MINE, concat, refit, predict MONITOR.
    Mirrors closure/stage4_readout.py::fit_block exactly."""
    fit_parts, mon_parts, kept = [], [], []
    for M in raw_blocks:
        if M.shape[1] == 0:
            continue
        keep, med = clean_fit(M[fitmask])
        if len(keep) == 0:
            continue
        kept.append((keep, med))
        fit_parts.append(clean_apply(M[fitmask], keep, med))
        mon_parts.append(clean_apply(M[monmask], keep, med))
    Xfit = np.column_stack(fit_parts)
    Xmon = np.column_stack(mon_parts)
    lin_mon, nl_mon, picks = fit_predict_monitor(Xfit, y[fitmask], groups[fitmask], Xmon, seeds)
    out = {"n_features": int(Xfit.shape[1]), "lin_mon": lin_mon,
           "nl_mon_seeds": nl_mon, "nl_mon": nl_mon.mean(axis=0), "picks": picks,
           "screens": kept}
    if want_oof:
        out["oof_lin_fitmine"] = linear_oof(Xfit, y[fitmask], groups[fitmask])
        out["oof_nl_fitmine"] = np.mean(
            [gbm_oof(Xfit, y[fitmask], groups[fitmask], seed=s)[0] for s in seeds], axis=0)
    return out


def auc(y, p):
    return float(roc_auc_score(y, p))


# ---------------------------------------------------- stratified readouts ---
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
        return float("nan"), {"n_strata_used": 0, "n_rows_used": 0, "n_rows_dropped": dropped}
    return float(num / tot), {"n_strata_used": used, "n_rows_used": int(tot),
                              "n_rows_dropped": int(dropped)}


def decile_strata(x, q=10):
    x = np.asarray(x, dtype=float)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, q + 1)))
    if len(edges) < 3:
        return np.zeros(len(x), dtype=int)
    return np.clip(np.digitize(x, edges[1:-1], right=True), 0, len(edges) - 2)


def group_boot_ci(y, pa, pb, groups, n=2000, seed=0):
    """Group-level paired bootstrap of AUC(pa) - AUC(pb) (FREEZE CHANGE 3)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    out = []
    for _ in range(n):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in gs])
        if len(set(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    out = np.array(out)
    if len(out) == 0:
        return {"lo": None, "hi": None, "p_gt0": None}
    return {"lo": float(np.percentile(out, 2.5)), "hi": float(np.percentile(out, 97.5)),
            "p_gt0": float((out > 0).mean()), "mean": float(out.mean())}
