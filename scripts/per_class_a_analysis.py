#!/usr/bin/env python3
"""Per-rejection-class A (articulated-metric) analysis for patents.

*** SS102/103 "A-BLIND" CLAIM WITHDRAWN (2026-07-10, Codex audit + WS3) ***
The judge cells here are DOC-ONLY: no prior-art record is attached, so on
retrieval-dependent grounds (SS102 anticipation / SS103 obviousness) the judge
is structurally blind — I(judge(X);Z|X)=0, the exact false-null design WS3
identified (datasets/patents/2026-07-10__evidence_aware_judge_ws3.md). The
~0.50 AUC on SS102/103 is an instrument artifact, NOT evidence those grounds
are non-articulable; re-scoring needs an evidence-aware judge M(x,Z).
The SS101 result stands (eligibility is text-intrinsic, doc-only is level-matched).

Question: do judged rubric scores predict rejection, sliced by statutory
ground?  Original reading (2026-06-11, now scoped as above): SS101 (eligibility)
is articulable (pooled A-AUC ~0.76); SS102/103 at chance — valid only as a
statement about DOC-ONLY articulation.

Inputs (all laptop-local):
  outputs/v2_db/cells_v1/task=patents/**.parquet   judge cells (canonical DB)
  runs/validity_full/v2/patents/dp_rejection_flags.json  dp -> OARD class flags
  runs/validity_full/v2/patents/datapoints.json          dp -> accept/reject label

Output:
  runs/validity_full/v2/patents/per_class_a_results.json

Cohort per class: accepted dp (label=1) vs rejected dp (label=0) that carry
that class's OARD flag.  Two estimates:
  pooled  - L2 logistic over the full aspect matrix (mean-imputed), 5-fold CV
            AUC + bootstrap CI
  per-aspect - raw AUC of each single aspect's mean score
"""
import glob
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

RNG = np.random.default_rng(0)
BASE = "runs/validity_full/v2/patents"
CLASSES = ["rejected_101", "rejected_102", "rejected_103",
           "rejected_112a", "rejected_112b", "rejected_112d",
           "rejected_112f", "rejected_dp"]
MIN_REJ = 5


def load_matrix(judge):
    fs = glob.glob("outputs/v2_db/cells_v1/task=patents/**/*.parquet",
                   recursive=True)
    df = pd.concat([pd.read_parquet(f) for f in fs])
    c = df[(df.judge_model == judge) & df.applicable & df.score.notna()]
    return c.groupby(["aspect_id", "datapoint_id"]).score.mean().unstack().T


def pooled_cv_auc(X, y, n_boot=500):
    """Mean-impute, L2 logistic, 5-fold CV AUC + bootstrap CI on CV preds."""
    Xi = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    p = cross_val_predict(clf, Xi, y, cv=skf, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, p)
    boots = []
    idx = np.arange(len(y))
    for _ in range(n_boot):
        b = RNG.choice(idx, len(idx))
        if len(set(y[b])) == 2:
            boots.append(roc_auc_score(y[b], p[b]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # top-weight aspects from a full fit
    clf.fit(Xi, y)
    return auc, (lo, hi), clf.coef_[0]


def run(judge):
    mat = load_matrix(judge)
    flags = json.load(open(f"{BASE}/dp_rejection_flags.json"))
    dps = json.load(open(f"{BASE}/datapoints.json"))
    lab = {d["datapoint_id"]: d.get("judgement", d.get("label"))
           for d in (dps if isinstance(dps, list) else dps.values())}
    acc = [d for d in mat.index if lab.get(d) == 1]
    out = {"judge": judge, "matrix_shape": list(mat.shape),
           "n_accepted": len(acc), "classes": {}}
    for cls in CLASSES:
        rej = [d for d, f in flags.items()
               if isinstance(f, dict) and str(f.get(cls)) in ("1", "True")
               and lab.get(d) == 0 and d in mat.index]
        entry = {"n_rejected": len(rej)}
        if len(rej) >= MIN_REJ:
            dp_ids = acc + rej
            sub = mat.loc[dp_ids]
            keep = sub.columns[sub.notna().mean() >= 0.25]
            X = sub[keep].values
            y = np.array([1 if d in set(rej) else 0 for d in dp_ids])
            auc, (lo, hi), coef = pooled_cv_auc(X, y)
            top = sorted(zip(keep, coef), key=lambda t: -abs(t[1]))[:5]
            entry.update(
                pooled_auc=round(auc, 4), ci=[round(lo, 3), round(hi, 3)],
                n_aspects=len(keep),
                top_weights={a: round(float(w), 3) for a, w in top})
            per = {}
            for a in mat.columns:
                s = mat.loc[dp_ids, a].dropna()
                ya = np.array([1 if d in set(rej) else 0 for d in s.index])
                if ya.sum() >= MIN_REJ and (1 - ya).sum() >= MIN_REJ:
                    per[a] = round(roc_auc_score(ya, s.values), 4)
            entry["per_aspect_auc"] = dict(
                sorted(per.items(), key=lambda t: -t[1]))
        out["classes"][cls] = entry
        ci = entry.get("ci", "")
        print(f"{judge:>32} {cls:<14} rej={len(rej):>4} "
              f"AUC={entry.get('pooled_auc', '--')} {ci}")
    return out


if __name__ == "__main__":
    results = [run(j) for j in
               ["qwen_thinking_fp8_20x1_r2post", "claude"]]
    with open(f"{BASE}/per_class_a_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {BASE}/per_class_a_results.json")
