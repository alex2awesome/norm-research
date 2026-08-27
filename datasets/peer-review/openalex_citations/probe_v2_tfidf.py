#!/usr/bin/env python3
"""TF-IDF/LogReg floor AUC for acad-C v2, within venue x year.

Reports a per-cell floor (group-stratified 5-fold CV AUC) and specifically the
NeurIPS 2023 cell, on text -> judgement (1=top quartile, 0=bottom quartile;
middle/blank rows excluded). No GPU.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
PATH = "openalex_citations_v2.csv.gz"


def cell_auc(sub):
    sub = sub[sub.judgement.isin([0, 1, "0", "1"])].copy()
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


def main():
    df = pd.read_csv(PATH, dtype={"judgement": "object"})
    print(f"v2 rows: {len(df)}")
    print("cite_source:", df.cite_source.value_counts().to_dict())

    print("\n=== per-cell TF-IDF floor AUC (within venue x year) ===")
    aucs = []
    for (v, yr), sub in df.groupby(["venue", "year"]):
        auc, n = cell_auc(sub)
        tag = "  <-- S2" if (v == "NeurIPS" and yr == 2023) else ""
        if auc is None:
            print(f"{v:8s} {yr}  n_labeled={n:5d}  AUC=  (skipped){tag}")
        else:
            print(f"{v:8s} {yr}  n_labeled={n:5d}  AUC={auc:.3f}{tag}")
            aucs.append(auc)
    print(f"\nmean per-cell floor AUC (labeled cells): {np.mean(aucs):.3f} "
          f"over {len(aucs)} cells")

    # pooled floor with venue x year fixed effect removed via per-cell standard:
    # report a single number = mean over cells (weights equal) already above.
    print("\n=== NeurIPS 2023 sanity (S2 cell) ===")
    n23 = df[(df.venue == "NeurIPS") & (df.year == 2023)]
    c = n23.cited_by_count
    print(f"rows={len(n23)} median_cites={c.median()} zero-frac={(c==0).mean():.3f}")
    topq = n23[n23.judgement.isin([1, "1"])].cited_by_count
    print(f"top-q median cites = {topq.median()} (should be > 0)")


if __name__ == "__main__":
    main()
