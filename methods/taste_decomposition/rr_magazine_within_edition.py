#!/usr/bin/env python3
"""Within-edition readout for rr_magazine (PRIMARY frame for the PILOT curated
cell: editions are judged-together pools, so rank-within-edition is the
construct; pooled AUC mixes edition base rates). Recomputes the frozen linear
OOF (deterministic machinery, same folds) and reads per-edition AUCs of
A_lin / VA_lin / VA_nl(seed0); mean is unweighted over editions, plus a
pos-weighted mean. 26 positives — PILOT flag on everything."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import layer1_gemma_cells as L  # noqa: E402
import scaleupC_layer1 as SC  # noqa: E402

d = SC.CELLS["rr_magazine"]()
A, V, y, groups = d["A"], d["V"], d["y"], d["groups"]
VA = np.column_stack([V, A])
folds = L.outer_folds(len(y), groups, n_splits=5)
oof = {}
for k, M in (("A", A), ("VA", VA)):
    _, oof[k] = L.linear_oof_family1(M, y, groups, folds)
r = L.gbm_oof_family1(VA, y, groups, folds, 0)
oof["VA_nl_seed0"] = r["oof"]

res = {"n": int(len(y)), "pos": int(y.sum()), "frame": "within-edition (PRIMARY, PILOT)"}
for k, p in oof.items():
    per = {}
    for ed in sorted(set(groups)):
        m = groups == ed
        if len(set(y[m])) < 2:
            continue
        per[str(ed)] = round(float(roc_auc_score(y[m], p[m])), 4)
    aucs = np.array(list(per.values()))
    pos_w = np.array([y[groups == ed].sum() for ed in per])
    res[k] = {"per_edition": per,
              "mean_unweighted": round(float(aucs.mean()), 4),
              "mean_pos_weighted": round(float((aucs * pos_w).sum() / pos_w.sum()), 4),
              "pooled": round(float(roc_auc_score(y, p)), 4)}
    print(k, res[k])
out = HERE / "results" / "rr_magazine_within_edition.json"
out.write_text(json.dumps(res, indent=1))
print("wrote", out)
