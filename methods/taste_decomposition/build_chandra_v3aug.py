#!/usr/bin/env python3
"""V3 criteria-in-prompt dense build for the Chandrasekharan cells (Task B,
user order 2026-08-24). Reuses the G3 code-competitions precedent
(build_code_competitions_v3aug.py, arm a): prompt = criteria block FIRST
("<name>: <score>" lines, top-10), then the row's original dense text.

Differences from G3, DECLARED:
  * Splits are the FROZEN dense_standard_chandra_* row-hash splits (train/eval/
    test reused verbatim, same rows same order) — so T_v3 is directly
    comparable to the existing raw-text dense T.
  * Importance = ONE ranking per cell computed on the TRAIN split rows only
    (GroupKFold(3) by subreddit inside train, frozen HistGB leaves=31 lr=.06
    400 iter, permutation_importance roc_auc n_repeats=5, mean over folds) —
    the frozen eval/test rows never touch the ranking.
  * Features ranked = the pooled collapse-kept Gemma A-criteria only (user ask:
    "top-10 criteria ... with their Gemma scores").

FRAME: v1 populations; era channel open; v2 rescore will supersede.

Usage: build_chandra_v3aug.py --cell chandra_humor
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
sys.path.insert(0, str(REPO / "methods/taste_decomposition"))
import so_votes_layer1 as SV
import scaleupC_layer1 as SC

TOP_K = 10
LABEL_WORDS = ("remov", "moderat", "delet", "banned", "flair")

ap = argparse.ArgumentParser()
ap.add_argument("--cell", required=True, choices=["chandra_humor", "chandra_cw"])
a = ap.parse_args()
cell = a.cell
DENSE = REPO / f"datasets/prior_norms_cells/dense_standard_{cell}"
OUT = REPO / f"datasets/prior_norms_cells/dense_v3aug_{cell}"
(OUT / "split").mkdir(parents=True, exist_ok=True)


def fmt(v):
    if v != v:
        return "NA"
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return f"{v:.3f}".rstrip("0").rstrip(".")


meta, A, V, groups, shard, ids = SC.load_scaleupC_bank(
    cell, out=REPO / "outputs/va_gemma_banks_scaleupC")
a_names = list(meta["a_names"])
# pooled collapse gate (same gate as pooled layer-1)
shares = np.array([SV.modal_share(A[:, c]) for c in range(A.shape[1])])
keep_c = shares <= SV.COLLAPSE_MODAL_MAX
A = A[:, keep_c]
a_names = [nm for nm, k in zip(a_names, keep_c) if k]
pos = {str(r): i for i, r in enumerate(ids)}

splits = {leg: pd.read_csv(DENSE / "split" / f"{leg}.csv") for leg in ("train", "eval", "test")}
tr = splits["train"]
idx_tr = np.array([pos[str(r)] for r in tr["row_id"]])
Xtr = A[idx_tr]
ytr = tr["judgement"].astype(int).values
gtr = tr["group"].astype(str).values
Xi = np.where(np.isnan(Xtr), np.nanmedian(Xtr, axis=0), Xtr)

imps, nf = np.zeros(A.shape[1]), 0
for itr, ite in GroupKFold(n_splits=3).split(Xi, ytr, gtr):
    m = HistGradientBoostingClassifier(max_leaf_nodes=31, learning_rate=0.06,
                                       max_iter=400, early_stopping=True,
                                       validation_fraction=0.1, n_iter_no_change=20,
                                       random_state=0)
    m.fit(Xi[itr], ytr[itr])
    r = permutation_importance(m, Xi[ite], ytr[ite], scoring="roc_auc",
                               n_repeats=5, random_state=0, n_jobs=-1)
    imps += r.importances_mean
    nf += 1
imps /= nf
order = np.argsort(-imps)[:TOP_K]
top = [{"name": a_names[j], "col": int(j), "importance": float(imps[j])} for j in order]
for t in top:
    flagged = [w for w in LABEL_WORDS if w in t["name"].lower()]
    if flagged:
        print(f"!! LEAK-WORD FLAG on '{t['name']}': {flagged}", flush=True)
print(f"[{cell}] top-{TOP_K}: " + " | ".join(t["name"] for t in top), flush=True)


def block(i):
    out = ["VA metrics:"]
    for t in top:
        out.append(f"    {t['name']}: {fmt(float(A[i, t['col']]))}")
    return "\n".join(out)


parts = []
for leg, sp in splits.items():
    idx = np.array([pos[str(r)] for r in sp["row_id"]])
    sp = sp.copy()
    sp["text"] = [block(i) + "\n\nPOST:\n" + t for i, t in zip(idx, sp["text"].astype(str))]
    sp.to_csv(OUT / "split" / f"{leg}.csv", index=False)
    parts.append(sp)
    print(f"[{cell}] {leg} n={len(sp)} rendered", flush=True)
pd.concat(parts).to_csv(OUT / "data.csv", index=False)

man = {"design_id": f"{cell}_v3aug", "precedent": "G3 build_code_competitions_v3aug.py arm a",
       "frame": "v1 populations; era channel open; v2 rescore will supersede",
       "estimand": "T_v3 vs raw-text dense T on the SAME frozen splits (criteria-"
                   "conditioning question); fused/max-of-variants column only per V3 ruling",
       "splits": "frozen dense_standard row-hash splits reused verbatim",
       "importance_protocol": "train-rows-only, GroupKFold(3) by subreddit, frozen "
                              "HistGB (31/.06/400), permutation_importance roc_auc "
                              "n_repeats=5, mean over folds; pooled collapse-kept "
                              "Gemma A-criteria only",
       "prompt_order": "criteria block FIRST ('VA metrics:'), then 'POST:' + original text",
       "top_criteria": top,
       "collapse_kept": int(A.shape[1])}
(OUT / "manifest.json").write_text(json.dumps(man, indent=1))

# 20-row sample dump for manual leakage spot-check
samp = pd.read_csv(OUT / "split" / "train.csv").sample(20, random_state=0)
(OUT / "render_samples.txt").write_text(
    "\n\n================\n\n".join(
        f"row_id={r.row_id} group={r.group} judgement={r.judgement}\n{r.text[:1500]}"
        for r in samp.itertuples()))
print(f"BUILD_DONE {OUT}", flush=True)
