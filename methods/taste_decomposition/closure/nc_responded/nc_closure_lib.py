#!/usr/bin/env python3
"""Shared machinery for the Layer-3 articulation-closure CONFIRMATORY campaign,
N&C RESPONDED cell (regulatory comments -> received a government response).

Protocol: notes/2026-08-05__layer3-closure-prereg.md (FREEZE DECLARATION 2026-08-06
+ FREEZE ADDENDUM).  Direct port of the peer-verdict pilot's
methods/taste_decomposition/closure/closure_lib.py, with the cell's own frozen
Layer-1 spec substituted:

  * grouping unit  : docket (NOT ntitle)
  * population     : nc_layer1_stack.load_cell("responded") -- matched (y=1) union
                     genuinely-unmatched (y=0), 9,521 rows, 1,814 dockets
  * A bank         : pre-GEPA 198-rubric Gemma bank, nc_scores_shard0..4.npz
                     (+ nc_scores_unmatched.npz), FROZEN GKF design
  * linear         : StandardScaler + LogisticRegression(C=1, max_iter=2000)
  * nonlinear      : HistGradientBoostingClassifier, max_leaf_nodes in {15,31},
                     learning_rate .06, max_iter 400, early stopping (val .1, pat 20)
  * grid           : inner GroupKFold(3) inside train folds only
  * outer          : GroupKFold(5) on docket
  * VA_nl          : mean over seeds {0,1,2}, spread reported (FREEZE CHANGE 1)

Everything is fit WITHIN the FIT+MINE split only -- MONITOR rows are never read by
any fit, any column-degeneracy screen, or any median imputation.

The V-feature matrix is expensive to recompute (v_features over 9,521 long comment
texts), so it is cached to `nc_responded_base.npz` on first use.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
NC = REPO / "datasets" / "notice-and-comment" / "v4"
TD = REPO / "methods" / "taste_decomposition"
CACHE = HERE / "nc_responded_base.npz"

GRID = [
    {"max_leaf_nodes": 15, "learning_rate": 0.06, "max_iter": 400},
    {"max_leaf_nodes": 31, "learning_rate": 0.06, "max_iter": 400},
]
N_OUTER = 5
N_INNER = 3
SEEDS = (0, 1, 2)
THRESH = 0.80  # frozen FIT+MINE / MONITOR stable-hash threshold


# ------------------------------------------------------------------ data ----
def build_base_cache():
    """Population in the EXACT order of nc_layer1_stack.load_cell('responded').

    That loader iterates `self.X_m` then `self.X_u` (both dicts, insertion-ordered
    from sorted shard files) -- deterministic, unlike the set-based `valid_out`/
    `valid_agr` universes used by the other two N&C cells.
    """
    sys.path.insert(0, str(TD))
    sys.path.insert(0, str(NC))
    from nc_layer1_stack import NCData  # noqa: E402
    from aggregate_vat_nc import v_features, V_NAMES  # noqa: E402

    d = NCData()
    doc_ids, y, dockets, agencies, texts = [], [], [], [], []
    A = []
    for did in d.X_m:
        doc_ids.append(did)
        y.append(1)
        dockets.append(d.docket_m[did])
        agencies.append(d.agency_m[did])
        texts.append(d.text_m.get(did, ""))
        A.append(d.X_m[did].astype(float))
    for did in d.X_u:
        doc_ids.append(did)
        y.append(0)
        dockets.append(d.docket_u[did])
        agencies.append(d.agency_u[did])
        texts.append(d.text_u.get(did, ""))
        A.append(d.X_u[did].astype(float))

    V = np.array([[v_features(t)[n] for n in V_NAMES] for t in texts], dtype=float)
    np.savez_compressed(
        CACHE,
        doc_id=np.array(doc_ids, dtype=object),
        y=np.array(y, dtype=int),
        docket=np.array(dockets, dtype=object),
        agency=np.array(agencies, dtype=object),
        A=np.array(A, dtype=float),
        V=V,
        a_names=np.array(d.a_names, dtype=object),
        v_names=np.array(list(V_NAMES), dtype=object),
        texts=np.array(texts, dtype=object),
    )
    return CACHE


def load_population():
    if not CACHE.exists():
        build_base_cache()
    z = np.load(CACHE, allow_pickle=True)
    return {
        "doc_id": np.array([str(s) for s in z["doc_id"]]),
        "y": z["y"].astype(int),
        "docket": np.array([str(s) for s in z["docket"]]),
        "agency": np.array([str(s) for s in z["agency"]]),
        "A": z["A"].astype(float),
        "V": z["V"].astype(float),
        "a_names": [str(s) for s in z["a_names"]],
        "v_names": [str(s) for s in z["v_names"]],
        "texts": [str(s) for s in z["texts"]],
    }


def load_splits():
    d = json.loads((HERE / "nc_responded_splits.json").read_text())
    recs = d["rows"]
    return (
        d["summary"],
        np.array([r["split"] for r in recs]),
        np.array([r["dense_split"] for r in recs]),
        np.array([r["in_mining_slice"] for r in recs]),
        np.array([r["monitor_full"] for r in recs]),
    )


def hash_unit(key: str) -> float:
    return int(hashlib.sha256(str(key).encode("utf-8")).hexdigest(), 16) / float(1 << 256)


# ------------------------------------------------------- column cleaning ----
def clean_fit(M):
    """clean_cols (nc_layer1_stack) refactored into fit/apply so the degeneracy
    screen and the imputation medians are learned on FIT+MINE only."""
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
    """Refit within FIT+MINE (grid by inner grouped CV on FIT+MINE only), predict
    the held-out rows.  Returns linear preds and per-seed GBM preds."""
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
    """n-weighted mean of within-stratum AUCs (threshold-free; no residual
    regressions).  Strata with <min_n rows or a single class are dropped."""
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
    edges = np.unique(np.quantile(x[~np.isnan(x)], np.linspace(0, 1, q + 1)))
    if len(edges) < 3:
        return np.zeros(len(x), dtype=int)
    return np.clip(np.digitize(x, edges[1:-1], right=True), 0, len(edges) - 2)


def group_bootstrap_delta(y, p_a, p_b, groups, n_boot=2000, seed=0):
    """Docket-level cluster bootstrap of AUC(p_b) - AUC(p_a)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    out = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in draw])
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        out.append(roc_auc_score(yb, p_b[idx]) - roc_auc_score(yb, p_a[idx]))
    out = np.array(out)
    return {
        "n_boot_used": int(len(out)),
        "mean": float(out.mean()),
        "ci95": [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))],
        "p_gt_0": float((out > 0).mean()),
    }
