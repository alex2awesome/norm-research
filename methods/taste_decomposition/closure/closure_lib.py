#!/usr/bin/env python3
"""Shared machinery for the Layer-3 articulation-closure pilot (peer VERDICT).

Reuses the frozen Layer-1 protocol from methods/taste_decomposition/layer1_stack.py
(same grid, same nested grouped CV, same linear gate), but everything is now fit
*within the FIT+MINE split only* -- MONITOR rows are never read by any fit, any
column-degeneracy screen, or any median imputation.

Frozen Layer-1 spec carried over verbatim:
  linear : StandardScaler + LogisticRegression(C=1, max_iter=2000)
  nonlin : HistGradientBoostingClassifier, max_leaf_nodes in {15,31},
           learning_rate .06, max_iter 400, early stopping (val .1, patience 20)
  grid   : inner GroupKFold(3) inside train folds only
  outer  : GroupKFold(5) on `ntitle`
  VA_nl  : mean over seeds {0,1,2} with spread reported (FREEZE CHANGE 1)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
VAT = REPO / "datasets" / "peer-review" / "vat_3y"

GRID = [
    {"max_leaf_nodes": 15, "learning_rate": 0.06, "max_iter": 400},
    {"max_leaf_nodes": 31, "learning_rate": 0.06, "max_iter": 400},
]
N_OUTER = 5
N_INNER = 3
SEEDS = (0, 1, 2)


# ------------------------------------------------------------------ data ----
def _valid_y(r) -> bool:
    j = r.get("judgement")
    try:
        return int(float(j)) in (0, 1)
    except (TypeError, ValueError):
        return False


def load_population():
    """Population rows in the SAME order as layer1_stack.load_cell('verdict')."""
    z = np.load(VAT / "union_scores.npz", allow_pickle=True)
    X, V, nt = z["X"], z["V"], z["ntitle"]
    a_names = [str(s) for s in z["a_names"]]
    v_names = [str(s) for s in z["v_names"]]
    X_by = {str(nt[i]): X[i] for i in range(len(nt))}
    V_by = {str(nt[i]): V[i] for i in range(len(nt))}

    rows = [json.loads(l) for l in open(VAT / "verdict.jsonl") if l.strip()]
    R = [r for r in rows if str(r.get("ntitle")) in X_by and _valid_y(r)]
    ntl = [str(r["ntitle"]) for r in R]
    y = np.array([int(float(r["judgement"])) for r in R])
    A = np.array([X_by[k] for k in ntl], dtype=float)
    Vm = np.array([V_by[k] for k in ntl], dtype=float)
    texts = [r["text"] for r in R]
    return {
        "rows": R,
        "texts": texts,
        "ntitle": np.array(ntl),
        "y": y,
        "A": A,
        "V": Vm,
        "a_names": a_names,
        "v_names": v_names,
    }


def load_splits():
    d = json.loads((HERE / "peer_verdict_splits.json").read_text())
    recs = d["rows"]
    return d["summary"], np.array([r["split"] for r in recs]), np.array(
        [r["dense_split"] for r in recs]
    ), np.array([r["in_mining_slice"] for r in recs])


def hash_unit(key: str) -> float:
    return int(hashlib.sha256(str(key).encode("utf-8")).hexdigest(), 16) / float(1 << 256)


# ------------------------------------------------------- column cleaning ----
def clean_fit(M):
    """clean_cols (layer1_stack) refactored into fit/apply so the screen and the
    imputation medians are learned on FIT+MINE only."""
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
    sub = M[:, keep].astype(float)
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
    """Inner GroupKFold(3) grid pick -- never touches held-out rows."""
    inner = list(
        GroupKFold(n_splits=min(N_INNER, len(np.unique(groups)))).split(
            np.zeros(len(y)), groups=groups
        )
    )
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
    """Grouped OOF within the given (FIT+MINE) rows, nested grid selection."""
    folds = list(
        GroupKFold(n_splits=min(N_OUTER, len(np.unique(groups)))).split(
            np.zeros(len(y)), groups=groups
        )
    )
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
    folds = list(
        GroupKFold(n_splits=min(N_OUTER, len(np.unique(groups)))).split(
            np.zeros(len(y)), groups=groups
        )
    )
    oof = np.zeros(len(y))
    for tr, te in folds:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(Xf[tr], y[tr])
        oof[te] = clf.predict_proba(Xf[te])[:, 1]
    return oof


def fit_predict_monitor(Xfit, yfit, gfit, Xmon, seeds=SEEDS):
    """Refit within FIT+MINE (grid chosen by inner grouped CV on FIT+MINE only),
    then predict MONITOR.  Returns linear preds and per-seed GBM preds."""
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


def auc(y, p):
    return float(roc_auc_score(y, p))


# ---------------------------------------------------- stratified readouts ---
def stratified_auc(y, p, strata, min_n=25):
    """n-weighted mean of within-stratum AUCs (threshold-free rule; no residual
    regressions).  Strata with <min_n rows or a single class are dropped and the
    dropped fraction is reported."""
    y = np.asarray(y)
    p = np.asarray(p)
    strata = np.asarray(strata)
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
    return float(num / tot), {
        "n_strata_used": used,
        "n_rows_used": int(tot),
        "n_rows_dropped": int(dropped),
    }


def decile_strata(x, q=10):
    x = np.asarray(x, dtype=float)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, q + 1)))
    if len(edges) < 3:  # constant / near-constant feature -> one stratum
        return np.zeros(len(x), dtype=int)
    return np.clip(np.digitize(x, edges[1:-1], right=True), 0, len(edges) - 2)
