#!/usr/bin/env python3
"""Artifact-probe gate for the RoyalRoad COMMUNITY cell (mandatory before any
GPU scoring — matched-pipeline rule). Char-ngram + word-ngram TF-IDF logistic
probes, grouped OOF (GroupKFold by stratum so fold assignment never splits a
stratum; every prediction is out-of-fold). Gate: pooled AUC < .58 = CLEAN
(rr expansion precedent), .58-.60 boundary, >= .60 DIRTY.
"""
import gzip
import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
POP = os.path.join(HERE, "rr_community_population.jsonl.gz")

rows = [json.loads(l) for l in gzip.open(POP, "rt")]
texts = [r["text"] for r in rows]
y = np.array([r["judgement"] for r in rows])
groups = np.array([r["stratum"] for r in rows])
print(f"n={len(rows)} pos={y.mean():.4f} strata={len(set(groups))}")

res = {}
for name, vec in [
    ("char_3_5", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=200000)),
    ("word_1_2", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=200000)),
]:
    X = vec.fit_transform(texts)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, oof)
    res[name] = round(float(auc), 4)
    print(f"{name}: grouped-OOF AUC {auc:.4f}")

verdict = ("CLEAN" if max(res.values()) < .58 else
           "BOUNDARY" if max(res.values()) < .60 else "DIRTY")
res["gate"] = verdict
json.dump(res, open(os.path.join(HERE, "probe_results.json"), "w"), indent=1)
print("GATE:", verdict)
