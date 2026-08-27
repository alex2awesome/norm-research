#!/usr/bin/env python3
"""PILOT (user proposal 2026-08-18): LR word-probe -> LLM-crafted deconfounding
channels, on the best-closed cell (bbc_mostread, terminal 5-round campaign).

Step 1 (this script, CPU): grouped-OOF logistic regression on TF-IDF unigrams+
bigrams of the headline vs y. Output: top +/-40 features by coefficient, with
per-feature frequency and alone-AUC — the DISCOVERY list. The LR is a probe,
never an instrument (LLM-judges-do-all-measurement rule: any channel that enters
a nuisance block must be an LLM-judged criterion crafted FROM these words).

Step 2 (separate, LLM): hand the word lists to a frontier model to induce
INTERPRETABLE channels ("weekday-morning service-news vocabulary", "royal-family
running-story lexicon"), which are then Gemma-scored and added to the nuisance
block beside the fleet-mined B channels — measuring how much residual they
absorb that the fleet's upstream-reasoning missed.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import round0_bbc as R0  # noqa: E402

pop = pd.read_csv(R0.VA_DIR / "population.csv.gz")
texts = pop.text.astype(str).values if "text" in pop.columns else pop.headline.astype(str).values
y = pop.judgement.astype(int).values
g = pop.group.astype(str).values
print(f"rows {len(y)} pos {y.mean():.3f}")

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=20, max_features=30000,
                      sublinear_tf=True)
X = vec.fit_transform(texts)
names = np.array(vec.get_feature_names_out())

oof = np.zeros(len(y))
coefs = []
for tr, te in GroupKFold(5).split(X, groups=g):
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X[tr], y[tr])
    oof[te] = clf.predict_proba(X[te])[:, 1]
    coefs.append(clf.coef_[0])
print(f"LR word-probe grouped-OOF AUC: {roc_auc_score(y, oof):.4f}")
coef = np.mean(coefs, axis=0)

rows = []
for sign, idx in (("+", np.argsort(-coef)[:40]), ("-", np.argsort(coef)[:40])):
    for i in idx:
        col = (X[:, i] > 0).toarray().ravel().astype(float)
        rows.append({"feature": str(names[i]), "sign": sign,
                     "coef": float(coef[i]), "doc_freq": int(col.sum()),
                     "alone_auc": float(roc_auc_score(y, col))})
out = {"cell": "bbc_mostread", "lr_oof_auc": float(roc_auc_score(y, oof)),
       "note": "LR = discovery probe only; channels crafted from these words are "
               "LLM-judged before entering any nuisance block",
       "features": rows}
(HERE / "lr_word_probe.json").write_text(json.dumps(out, indent=1))
for r in rows[:20]:
    print(f"  {r['sign']} {r['feature']:28s} coef={r['coef']:+.2f} "
          f"df={r['doc_freq']:6d} alone={r['alone_auc']:.3f}")
print("LR_WORD_PROBE_DONE")
