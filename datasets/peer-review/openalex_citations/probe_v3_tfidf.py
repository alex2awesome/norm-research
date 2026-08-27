#!/usr/bin/env python3
"""TF-IDF/LogReg floor AUC per (venue x year), v2 vs v3, for acad-C.

Within each cell: text -> judgement (1=top quartile, 0=bottom; middle/blank
excluded), group-stratified 5-fold CV AUC. Reports the per-cell floor for v2
and v3 side by side, the mean per-cell floor for each, and the NeurIPS-2022
before/after. No GPU.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
HERE = "/lfs/skampere3/0/alexspan/norm-research/datasets/peer-review/openalex_citations"
V2 = f"{HERE}/openalex_citations_v2.csv.gz"
V3 = f"{HERE}/openalex_citations_v3.csv.gz"


def cell_auc(sub):
    sub = sub[sub.judgement.isin([0, 1, "0", "1"])].copy()
    if len(sub) == 0:
        return None, 0
    sub["y"] = sub.judgement.astype(float).astype(int)
    if sub.y.nunique() < 2 or len(sub) < 30:
        return None, len(sub)
    X = sub.text.fillna("").values
    y = sub.y.values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        vec = TfidfVectorizer(min_df=2, max_features=40000,
                              ngram_range=(1, 2), sublinear_tf=True)
        Xtr = vec.fit_transform(X[tr])
        Xte = vec.transform(X[te])
        clf = LogisticRegression(max_iter=2000, C=4.0)
        clf.fit(Xtr, y[tr])
        oof[te] = clf.predict_proba(Xte)[:, 1]
    return roc_auc_score(y, oof), int((y == 1).sum() + (y == 0).sum())


def per_cell(path):
    df = pd.read_csv(path, dtype={"judgement": "object"})
    res = {}
    for (v, yr), sub in df.groupby(["venue", "year"]):
        auc, n = cell_auc(sub)
        res[(v, yr)] = (auc, n)
    return res


def main():
    r2 = per_cell(V2)
    r3 = per_cell(V3)
    keys = sorted(set(r2) | set(r3))
    print(f"{'cell':22s}  {'v2_AUC':>7s} {'v2_n':>6s}   {'v3_AUC':>7s} {'v3_n':>6s}")
    a2, a3 = [], []
    for k in keys:
        v2a, v2n = r2.get(k, (None, 0))
        v3a, v3n = r3.get(k, (None, 0))
        s2 = f"{v2a:.3f}" if v2a is not None else "  -  "
        s3 = f"{v3a:.3f}" if v3a is not None else "  -  "
        tag = "  <== NeurIPS-2022" if k == ("NeurIPS", 2022) else ""
        print(f"{k[0]:8s} {k[1]}        {s2:>7s} {v2n:>6d}   {s3:>7s} {v3n:>6d}{tag}")
        if v2a is not None:
            a2.append(v2a)
        if v3a is not None:
            a3.append(v3a)
    print(f"\nmean per-cell floor  v2 = {np.mean(a2):.3f} over {len(a2)} cells")
    print(f"mean per-cell floor  v3 = {np.mean(a3):.3f} over {len(a3)} cells")
    print(f"\nNeurIPS-2022 floor:  v2 = {r2.get(('NeurIPS',2022),(None,))[0]}  "
          f"-> v3 = {r3.get(('NeurIPS',2022),(None,))[0]}")


if __name__ == "__main__":
    main()
