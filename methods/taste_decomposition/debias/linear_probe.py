#!/usr/bin/env python3
"""V2 failure-mode discrimination (CPU only, no GPU touched).

For each finished run, probe the FROZEN representation for the plant with a plain
LINEAR (logistic) probe, and -- decisively -- WITHIN EACH LABEL STRATUM.

Why the strata matter: the plant is correlated with y by construction (.65/.35),
and h necessarily encodes y (that is the task).  So a pooled plant probe could in
principle be reading the y direction rather than a plant direction, which would be
failure mode (c) "the plant leaks through a feature the nuisance head cannot
separate".  Within y=1 rows only (and y=0 rows only) the plant is INDEPENDENT of
the label by construction, so any recovery there is a genuine plant direction.
"""
import json, glob, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

nz = np.load("build/nuisance.npz", allow_pickle=True)
nz_ids = np.array([str(s) for s in nz["doc_id"]])
order = {d: i for i, d in enumerate(nz_ids)}
plant_all = nz["plant"].astype(int)

def probe(Xtr, ttr, Xte, tte):
    if len(set(ttr)) < 2 or len(set(tte)) < 2 or len(ttr) < 50:
        return None
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.01))
    m.fit(Xtr, ttr)
    return float(roc_auc_score(tte, m.predict_proba(Xte)[:, 1]))

out = {}
for f in sorted(glob.glob("runs/*/reps.npz")):
    tag = os.path.basename(os.path.dirname(f))
    z = np.load(f, allow_pickle=True)
    ids = np.array([str(s) for s in z["doc_id"]])
    X = z["rep"].astype(np.float32)
    split = np.array([str(s) for s in z["split"]])
    y = z["y"].astype(int)
    t = plant_all[np.array([order[d] for d in ids])]
    tr, ev = split == "train", split == "eval"
    r = {"linear_plant_probe_pooled": probe(X[tr], t[tr], X[ev], t[ev])}
    for lab in (0, 1):
        a, b = tr & (y == lab), ev & (y == lab)
        r[f"linear_plant_probe_within_y{lab}"] = probe(X[a], t[a], X[b], t[b])
        r[f"n_eval_y{lab}"] = int(b.sum())
    out[tag] = r
    print(tag, json.dumps(r), flush=True)
json.dump(out, open("results_linear_probe.json", "w"), indent=2)
