#!/usr/bin/env python3
"""Homepage (journalism curation) FUSED bar (2026-08-13): §11-style grouped-OOF
logistic stack of [VA_nl OOF, dense T] on the dense-held-out rows — fills the one
empty ladder column for a cell whose closure is already terminal (r0, ε-resolvability
fails). Runs ON sk3 (dense preds live there). Instrument: VA_nl OOF from
homepage_curation_storygrouped_oof.npz (seed-mean), dense = seed-mean preds from
dense_standard_storygrouped rm_out_seed{42,1,2}, order-join asserted per leg.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NR = Path("/lfs/skampere3/0/alexspan/norm-research")
D = NR / "datasets/news-homepages/va/dense_standard_storygrouped"
OOF = NR / "methods/taste_decomposition/results/homepage_curation_storygrouped_oof.npz"

z = np.load(OOF, allow_pickle=True)
ids = [str(i) for i in z["ids"]]
pos = {r: i for i, r in enumerate(ids)}
va = z["VA_nl_mean3"]
yv = z["y"].astype(int)
grp = np.array([str(g) for g in z["groups"]])

rows = []
for leg in ("eval", "test"):
    sp = pd.read_csv(D / "split" / f"{leg}.csv")
    per_seed = []
    for s in (42, 1, 2):
        p = pd.read_csv(D / f"rm_out_seed{s}" / f"preds_{leg}.csv")
        assert len(p) == len(sp) and (p["judgement"].values == sp["judgement"].values).all(), \
            f"order-join fail {leg} seed{s}"
        per_seed.append(p["prob"].values.astype(float))
    dm = np.mean(per_seed, axis=0)
    key = "row_id" if "row_id" in sp.columns else "id"
    for rid, dp, yy in zip(sp[key].astype(str), dm, sp["judgement"].astype(int)):
        rows.append((rid, dp, yy, leg))

hit = [(r, d_, y_, l_) for r, d_, y_, l_ in rows if r in pos]
print(f"dense-held-out rows joined to OOF matrix: {len(hit)}/{len(rows)}")
idx = np.array([pos[r] for r, _, _, _ in hit])
dense = np.array([d_ for _, d_, _, _ in hit])
y = yv[idx]
assert (y == np.array([y_ for _, _, y_, _ in hit])).all(), "y mismatch on join"
g = grp[idx]
leg = np.array([l_ for _, _, _, l_ in hit])

S = np.column_stack([va[idx], dense])
oof = np.zeros(len(y))
for tr, te in GroupKFold(5).split(S, groups=g):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(S[tr], y[tr])
    oof[te] = clf.predict_proba(S[te])[:, 1]

out = {
    "cell": "homepage_curation_storygrouped",
    "n_heldout": int(len(y)),
    "fused_stack_VA_T": float(roc_auc_score(y, oof)),
    "fused_eval": float(roc_auc_score(y[leg == "eval"], oof[leg == "eval"])),
    "fused_test": float(roc_auc_score(y[leg == "test"], oof[leg == "test"])),
    "VA_nl_same_rows": float(roc_auc_score(y, va[idx])),
    "dense_same_rows": float(roc_auc_score(y, dense)),
    "note": "grouped-OOF logistic stack [VA_nl OOF, dense seed-mean] on dense-held-out; "
            "fills the fused ladder column; closure remains terminal-r0 (unchanged)",
}
(NR / "methods/taste_decomposition/results/homepage_fused_stack.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
print("HOMEPAGE_FUSED_DONE")
