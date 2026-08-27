#!/usr/bin/env python3
"""CONDITIONAL BANK TREE v1 — metric-tree revival, stage 1 (Addendum F-d,
frozen 2026-08-25). Zero new LLM calls: the tree is built over the CERTIFIED
enriched-bank matrix, testing the architecture question alone — does
SELECTIVELY APPLYING the existing criteria to recursively smaller
subpopulations beat the flat bank that applies all criteria everywhere?

Node = subpopulation. Split = the single criterion whose value partition
(binarized ternary: >=0.5 vs <0.5/NA) maximizes weighted child label-AUC
gain, min-size guarded. Leaf = frozen-recipe GBM fit on the node's rows and
its LOCALLY top-m criteria (m=24). Honest evaluation: the ENTIRE tree
(splits + leaf models) is rebuilt inside each grouped-OOF fold. Same rows,
same folds as the flat-bank baseline computed alongside (apples-to-apples).

Subcommunity readout: per-leaf top-criteria lists + their divergence from
the global ranking (Spearman footrule), + per-leaf gap-vs-dense.

ONE CELL PER PROCESS. CPU.
Usage: python3 conditional_bank_tree.py --cell cw_community
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
TD = HERE.parent / "taste_decomposition"
RESULTS = TD / "results"


def _mod(path, alias):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m
    spec.loader.exec_module(m)
    return m


F2 = _mod(TD / "fusion" / "f2_deconf.py", "f2_deconf_tree")

MIN_LEAF = 250          # rows; below this, stop splitting
MAX_DEPTH = 3
TOP_M = 24              # criteria per leaf model
SEEDS = (0, 1, 2)


def leaf_model(X, y, seed):
    imp = SimpleImputer(strategy="median", add_indicator=True)
    Xi = imp.fit_transform(X)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=.06,
                                       max_leaf_nodes=15, random_state=seed)
    m.fit(Xi, y)
    return imp, m


def local_top(X, y, m=TOP_M):
    """Rank criteria by |univariate AUC - .5| on THIS node's rows."""
    scores = []
    for j in range(X.shape[1]):
        col = np.nan_to_num(X[:, j], nan=-1)
        if len(np.unique(col)) < 2 or y.min() == y.max():
            scores.append(0.0)
            continue
        scores.append(abs(roc_auc_score(y, col) - .5))
    return np.argsort(scores)[::-1][:m], np.array(scores)


def split_gain(X, y, j):
    mask = np.nan_to_num(X[:, j], nan=0) >= 0.5
    n1, n0 = mask.sum(), (~mask).sum()
    if min(n1, n0) < MIN_LEAF:
        return -1
    g = 0.0
    for mk in (mask, ~mask):
        yy = y[mk]
        if yy.min() == yy.max():
            g += mk.sum() / len(y) * 0.5   # pure node: neutral credit
            continue
        cols, sc = local_top(X[mk], yy, m=1)
        g += mk.sum() / len(y) * (0.5 + sc[cols[0]])
    return g


def build(X, y, depth=0, path="R"):
    node = {"path": path, "n": int(len(y)), "pos_rate": float(y.mean())}
    if depth >= MAX_DEPTH or len(y) < 2 * MIN_LEAF or y.min() == y.max():
        node["leaf"] = True
        return node
    cand, _ = local_top(X, y, m=12)         # split candidates = local top-12
    best_j, best_g = -1, -1
    for j in cand:
        g = split_gain(X, y, j)
        if g > best_g:
            best_j, best_g = int(j), g
    if best_j < 0:
        node["leaf"] = True
        return node
    mask = np.nan_to_num(X[:, best_j], nan=0) >= 0.5
    node.update(leaf=False, split_j=best_j)
    node["hi"] = build(X[mask], y[mask], depth + 1, path + "1")
    node["lo"] = build(X[~mask], y[~mask], depth + 1, path + "0")
    node["_mask"] = mask
    return node


def fit_leaves(node, X, y, seed):
    if node["leaf"]:
        cols, _ = local_top(X, y)
        node["cols"] = cols
        if y.min() == y.max():
            node["model"] = ("const", float(y.mean()))
        else:
            node["model"] = ("gbm", leaf_model(X[:, cols], y, seed))
        return
    m = node["_mask"]
    fit_leaves(node["hi"], X[m], y[m], seed)
    fit_leaves(node["lo"], X[~m], y[~m], seed)


def predict(node, X):
    out = np.zeros(len(X))
    if node["leaf"]:
        kind, mdl = node["model"]
        if kind == "const":
            return np.full(len(X), mdl)
        imp, m = mdl
        return m.predict_proba(imp.transform(X[:, node["cols"]]))[:, 1]
    mask = np.nan_to_num(X[:, node["split_j"]], nan=0) >= 0.5
    if mask.any():
        out[mask] = predict(node["hi"], X[mask])
    if (~mask).any():
        out[~mask] = predict(node["lo"], X[~mask])
    return out


def strip(node, names):
    d = {k: node[k] for k in ("path", "n", "pos_rate", "leaf")}
    if node["leaf"]:
        d["top_criteria"] = [names[j] for j in node["cols"][:8]]
    else:
        d["split"] = names[node["split_j"]]
        d["hi"], d["lo"] = strip(node["hi"], names), strip(node["lo"], names)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    args = ap.parse_args()
    cell = args.cell
    t0 = time.time()

    meta, ids_E, y, groups, dense, t0col = F2.load_E(cell)
    a = F2.F2C.ADAPTERS[cell]()
    bank, nuis, join = F2.align(cell, a, ids_E, y, groups)
    y = np.asarray(y)
    g = np.asarray(groups)
    names = [f"c{j}" for j in range(bank.shape[1])]
    if cell == "cw_community":      # per-column names live in the closure state
        z7 = np.load(TD / "closure/cw_community/round7_state.npz", allow_pickle=True)
        names = [str(x) for x in z7["bank_names"]]

    oof_tree = np.full(len(y), np.nan)
    oof_flat = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(bank, y, g):
        preds_t, preds_f = [], []
        for seed in SEEDS:
            tree = build(bank[tr], y[tr])
            fit_leaves(tree, bank[tr], y[tr], seed)
            preds_t.append(predict(tree, bank[te]))
            impf, mf = leaf_model(bank[tr], y[tr], seed)
            preds_f.append(mf.predict_proba(impf.transform(bank[te]))[:, 1])
        oof_tree[te] = np.mean(preds_t, axis=0)
        oof_flat[te] = np.mean(preds_f, axis=0)

    auc_tree = float(roc_auc_score(y, oof_tree))
    auc_flat = float(roc_auc_score(y, oof_flat))
    auc_dense = float(roc_auc_score(y, np.asarray(dense, float)))

    # full-data tree for the SUBCOMMUNITY readout (descriptive)
    tree = build(bank, y)
    fit_leaves(tree, bank, y, 0)
    glob_cols, glob_sc = local_top(bank, y, m=bank.shape[1])

    out = {"cell": cell, "design": "ADDENDUM F-d stage 1 (conditional bank tree, no new LLM calls)",
           "n_E": int(len(y)), "max_depth": MAX_DEPTH, "min_leaf": MIN_LEAF,
           "auc_tree_oof": auc_tree, "auc_flat_same_folds": auc_flat,
           "auc_dense": auc_dense,
           "delta_tree_minus_flat": auc_tree - auc_flat,
           "tree_structure": strip(tree, names),
           "runtime_sec": time.time() - t0}
    fp = RESULTS / f"cbtree_{cell}.json"
    fp.write_text(json.dumps(out, indent=2))
    print(f"CBTREE_DONE {cell} tree={auc_tree:.4f} flat={auc_flat:.4f} "
          f"dense={auc_dense:.4f} delta={auc_tree-auc_flat:+.4f} | "
          f"{out['runtime_sec']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
