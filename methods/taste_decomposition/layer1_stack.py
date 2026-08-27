#!/usr/bin/env python3
"""Layer-1 "nonlinear stack" of the taste-residual decomposition.

Design: notes/2026-08-05__taste-decomposition-design.md (§0 ledger, §1 protocol).

Pilot cell: peer-review VERDICT (accept/reject), matrix
datasets/peer-review/vat_3y/union_scores.npz (17 V features + 154 A rubric
criteria per abstract, union n=14,307), labels from verdict.jsonl,
grouping unit = `ntitle` (normalised paper title) -- identical to
datasets/peer-review/vat_3y/aggregate_3y.py.

Runs, on the SAME rows / SAME cleaned matrices / SAME grouped OOF folds:
  * linear gate   : StandardScaler + LogisticRegression(C=1) -> V_lin, A_lin, VA_lin
  * nonlinear     : HistGradientBoostingClassifier, frozen grid
                    max_leaf_nodes in {15,31}, learning_rate .06, max_iter 400
                    + early stopping; grid picked by inner GroupKFold(3) inside
                    each train fold ONLY -> V_nl, A_nl, VA_nl
  * seed spread   : VA_nl re-run with seeds 1, 2
  * overfit check : train-fold AUC vs OOF AUC at the selected grid point
  * SHAP          : top-10 interaction pairs (descriptive only)

CPU only. No new judging. Usage:  python layer1_stack.py [--cell verdict]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
VAT = REPO / "datasets" / "peer-review" / "vat_3y"
NPZ = VAT / "union_scores.npz"

# frozen grid (design section 1)
GRID = [
    {"max_leaf_nodes": 15, "learning_rate": 0.06, "max_iter": 400},
    {"max_leaf_nodes": 31, "learning_rate": 0.06, "max_iter": 400},
]
N_OUTER = 5
N_INNER = 3


# ---------------------------------------------------------------- data -----
def clean_cols(M):
    """EXACT copy of aggregate_3y.clean_cols: drop degenerate cols, median-impute NA."""
    keep, out = [], []
    for j in range(M.shape[1]):
        col = M[:, j].astype(float)
        nonna = col[~np.isnan(col)]
        if len(nonna) == 0:
            continue
        med = np.median(nonna)
        c = np.where(np.isnan(col), med, col)
        vals, counts = np.unique(c, return_counts=True)
        offmodal = len(c) - counts.max()
        if offmodal < 5 or c.std() == 0:
            continue
        keep.append(j)
        out.append(c)
    if not out:
        return np.zeros((M.shape[0], 0)), keep
    return np.column_stack(out), keep


def _valid_y(r):
    j = r.get("judgement")
    try:
        return int(float(j)) in (0, 1)
    except (TypeError, ValueError):
        return False


def load_cell(cell: str):
    z = np.load(NPZ, allow_pickle=True)
    X, V, nt = z["X"], z["V"], z["ntitle"]
    a_names = [str(s) for s in z["a_names"]]
    v_names = [str(s) for s in z["v_names"]]
    X_by_nt = {nt[i]: X[i] for i in range(len(nt))}
    V_by_nt = {nt[i]: V[i] for i in range(len(nt))}

    rows = [json.loads(l) for l in open(VAT / f"{cell}.jsonl") if l.strip()]
    R = [r for r in rows if r.get("ntitle") in X_by_nt and _valid_y(r)]
    ntl = [r["ntitle"] for r in R]
    y = np.array([int(float(r["judgement"])) for r in R])
    A = np.array([X_by_nt[k] for k in ntl], dtype=float)
    Vm = np.array([V_by_nt[k] for k in ntl], dtype=float)
    groups = np.array(ntl)

    Ac, a_keep = clean_cols(A)
    Vc, v_keep = clean_cols(Vm)
    VA = np.column_stack([Vc, Ac]) if Vc.shape[1] and Ac.shape[1] else (Vc if Vc.shape[1] else Ac)
    names = {
        "V": [v_names[j] for j in v_keep],
        "A": [a_names[j] for j in a_keep],
    }
    names["VA"] = names["V"] + names["A"]
    mats = {"V": Vc, "A": Ac, "VA": VA}
    return mats, names, y, groups


# --------------------------------------------------------------- models ----
def outer_folds(n, groups):
    gkf = GroupKFold(n_splits=min(N_OUTER, len(np.unique(groups))))
    return list(gkf.split(np.zeros(n), groups=groups))


def linear_oof(Xf, y, groups, folds):
    if Xf.shape[1] == 0:
        return float("nan"), None
    oof = np.zeros(len(y))
    for tr, te in folds:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(Xf[tr], y[tr])
        oof[te] = clf.predict_proba(Xf[te])[:, 1]
    return float(roc_auc_score(y, oof)), oof


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


def gbm_oof(Xf, y, groups, folds, seed=0, verbose=False):
    """Nested: inner GroupKFold(3) grid pick inside each train fold only."""
    oof = np.zeros(len(y))
    picks, train_aucs, inner_log = [], [], []
    for fi, (tr, te) in enumerate(folds):
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
        inner_log.append({"fold": fi, "inner_auc": dict(zip([str(g["max_leaf_nodes"]) for g in GRID], scores))})
        m = _fit_gbm(GRID[best], seed)
        m.fit(Xf[tr], y[tr])
        oof[te] = m.predict_proba(Xf[te])[:, 1]
        train_aucs.append(float(roc_auc_score(y[tr], m.predict_proba(Xf[tr])[:, 1])))
        if verbose:
            print(f"    fold {fi}: pick leaves={picks[-1]} inner={scores} train_auc={train_aucs[-1]:.3f}")
    return {
        "auc": float(roc_auc_score(y, oof)),
        "picks": picks,
        "train_auc_mean": float(np.mean(train_aucs)),
        "train_aucs": train_aucs,
        "inner": inner_log,
        "oof": oof,
    }


# ----------------------------------------------------------------- shap ----
def shap_interactions(Xf, y, names, seed=0, n_sub=300, top_k=15):
    """Top-10 SHAP interaction pairs (descriptive).

    Full-model interaction values over ~150 features are O(F^2) and slow, so we
    screen: fit the frozen-grid model on all rows, rank features by mean|SHAP|,
    refit on the top_k features, and compute exact TreeSHAP interaction values
    on that reduced model over a subsample. Reported as a screening result.
    """
    import shap

    m = _fit_gbm(GRID[1], seed)
    m.fit(Xf, y)
    ex = shap.TreeExplainer(m)
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(y), size=min(n_sub, len(y)), replace=False)
    sv = ex.shap_values(Xf[sub])
    if isinstance(sv, list):
        sv = sv[-1]
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    imp = np.abs(sv).mean(0)
    top = np.argsort(-imp)[:top_k]
    top_names = [names[j] for j in top]

    m2 = _fit_gbm(GRID[1], seed)
    m2.fit(Xf[:, top], y)
    ex2 = shap.TreeExplainer(m2)
    iv = ex2.shap_interaction_values(Xf[sub][:, top])
    if isinstance(iv, list):
        iv = iv[-1]
    if iv.ndim == 4:
        iv = iv[:, :, :, -1]
    M = np.abs(iv).mean(0)
    pairs = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            pairs.append((top_names[i], top_names[j], float(M[i, j] + M[j, i])))
    pairs.sort(key=lambda t: -t[2])
    diag = [(top_names[i], float(M[i, i])) for i in range(len(top))]
    off_frac = float((M.sum() - np.trace(M)) / M.sum())
    return {
        "method": "TreeSHAP exact interaction values on top-15-feature refit (screened by mean|SHAP| on full model)",
        "top_features": [{"name": n, "mean_abs_shap": float(imp[j])} for n, j in zip(top_names, top)],
        "top_pairs": [{"a": a, "b": b, "mean_abs_interaction": v} for a, b, v in pairs[:10]],
        "main_effects": [{"name": n, "mean_abs_main": v} for n, v in diag],
        "offdiagonal_mass_fraction": off_frac,
        "n_subsample": int(len(sub)),
    }


# ----------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="verdict")
    ap.add_argument("--T", type=float, default=0.753, help="dense clean-eval AUC")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "peer_verdict_layer1.json"))
    args = ap.parse_args()

    t0 = time.time()
    mats, names, y, groups = load_cell(args.cell)
    folds = outer_folds(len(y), groups)
    print(f"cell={args.cell} n={len(y)} pos={y.mean():.4f} groups={len(np.unique(groups))} "
          f"V={mats['V'].shape[1]}c A={mats['A'].shape[1]}c VA={mats['VA'].shape[1]}c")

    res = {
        "cell": f"peer-review {args.cell}",
        "n": int(len(y)),
        "pos_rate": float(y.mean()),
        "n_groups": int(len(np.unique(groups))),
        "group_column": "ntitle",
        "matrix": str(NPZ),
        "n_features": {k: int(v.shape[1]) for k, v in mats.items()},
        "T_dense": args.T,
        "linear": {},
        "nonlinear": {},
    }

    for k in ["V", "A", "VA"]:
        auc, _ = linear_oof(mats[k], y, groups, folds)
        res["linear"][k] = auc
        print(f"  linear  {k:2s}: {auc:.4f}")

    for k in ["V", "A", "VA"]:
        print(f"  gbm {k} ...")
        r = gbm_oof(mats[k], y, groups, folds, seed=0, verbose=True)
        oof = r.pop("oof")
        if k == "VA":
            np.save(Path(args.out).with_name("peer_verdict_va_nl_oof_seed0.npy"), oof)
        res["nonlinear"][k] = r
        print(f"  gbm     {k:2s}: {r['auc']:.4f}  (train {r['train_auc_mean']:.4f})")

    # seed sensitivity on VA
    res["seed_spread"] = {"0": res["nonlinear"]["VA"]["auc"]}
    for s in (1, 2):
        r = gbm_oof(mats["VA"], y, groups, folds, seed=s)
        res["seed_spread"][str(s)] = r["auc"]
        print(f"  gbm VA seed {s}: {r['auc']:.4f}")
    vals = list(res["seed_spread"].values())
    res["seed_spread_range"] = float(max(vals) - min(vals))

    L, N = res["linear"], res["nonlinear"]
    res["ledger"] = {
        "V_lin": L["V"], "V_nl": N["V"]["auc"],
        "A_lin": L["A"], "A_nl": N["A"]["auc"],
        "VA_lin": L["VA"], "VA_nl": N["VA"]["auc"],
        "T": args.T,
        "Delta_total": args.T - L["VA"],
        "Delta_interact": N["VA"]["auc"] - L["VA"],
        "Delta_beyond": args.T - N["VA"]["auc"],
        "V_interact": N["V"]["auc"] - L["V"],
    }
    res["overfit_gap"] = {k: N[k]["train_auc_mean"] - N[k]["auc"] for k in ["V", "A", "VA"]}

    published = {"V": 0.6128041549359611, "A": 0.6834696830751332, "VA": 0.6896181275813786}
    res["gate"] = {
        k: {"published": published[k], "reproduced": L[k], "abs_diff": abs(L[k] - published[k]),
            "pass": abs(L[k] - published[k]) <= 0.005}
        for k in published
    }

    print("  shap ...")
    try:
        res["shap"] = shap_interactions(mats["VA"], y, names["VA"], seed=0)
    except Exception as e:  # pragma: no cover
        res["shap"] = {"error": repr(e)}
        print("  shap FAILED:", e)

    res["runtime_sec"] = time.time() - t0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res["ledger"], indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
