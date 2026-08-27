#!/usr/bin/env python3
"""Quick test: do applicant cites add real signal, or are they noise?

Splits the with_applicant_cites_balanced file into three pools:
  A. Rows WITHOUT cites (cite lookup failed): ~70% of rows. Baseline.
  B. Rows WITH cites, cites STRIPPED: claims only. Same apps as C.
  C. Rows WITH cites, FULL text: claims + cites.

Compare AUC of tf-idf + LR on each.
  - (C - B): does cite text help, holding the application set fixed?
  - (A vs B): are with-cites applications systematically different?

If (C - B) is meaningful (>0.01-0.02 AUC), cites carry signal. If not,
dropping to cites-only sacrifices 70% of data for nothing.
"""
import csv
import gzip
import re
import time

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

csv.field_size_limit(2**31 - 1)

INPUT = ("/lfs/skampere3/0/alexspan/norm-research/datasets/patents/"
         "patents_first_draft_with_applicant_cites_balanced.csv.gz")

CITE_MARKER = re.compile(r"\n\nCITED PRIOR ART:", re.IGNORECASE)


def strip_cites(t):
    """Returns text with CITED PRIOR ART section removed."""
    m = CITE_MARKER.search(t)
    if m:
        return t[:m.start()]
    return t


def main():
    print("Loading data...")
    pool_a = []  # no cites, claims only
    pool_b = []  # has cites, cites stripped
    pool_c = []  # has cites, full text
    labels_a, labels_b, labels_c = [], [], []
    with gzip.open(INPUT, "rt") as f:
        for r in csv.DictReader(f):
            t = r["text"]
            label = int(r["judgement"])
            if CITE_MARKER.search(t):
                stripped = strip_cites(t)
                pool_b.append(stripped)
                pool_c.append(t)
                labels_b.append(label)
                labels_c.append(label)
            else:
                pool_a.append(t)
                labels_a.append(label)

    print(f"  Pool A (no cites): {len(pool_a):,} rows, pos rate {sum(labels_a)/len(labels_a):.2%}")
    print(f"  Pool B/C (with cites, same apps): {len(pool_b):,} rows, "
          f"pos rate {sum(labels_b)/len(labels_b):.2%}")
    print()

    def evaluate(name, texts, labels, max_features=2 ** 18):
        print(f"--- {name} ---")
        t0 = time.time()
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels)
        print(f"  train={len(X_train):,} test={len(X_test):,}")
        pipe = Pipeline([
            ("h", HashingVectorizer(n_features=max_features, ngram_range=(1, 2),
                                    alternate_sign=False, lowercase=True)),
            ("t", TfidfTransformer()),
            ("lr", LogisticRegression(C=1.0, solver="liblinear", max_iter=200)),
        ])
        pipe.fit(X_train, y_train)
        scores = pipe.decision_function(X_test)
        auc = roc_auc_score(y_test, scores)
        print(f"  AUC: {auc:.4f}  (took {time.time() - t0:.0f}s)")
        return auc

    auc_a = evaluate("A: NO cites (claims only)", pool_a, labels_a)
    auc_b = evaluate("B: WITH cites, STRIPPED (claims only, same apps as C)",
                     pool_b, labels_b)
    auc_c = evaluate("C: WITH cites, FULL (claims + cites)", pool_c, labels_c)

    print()
    print("=== Summary ===")
    print(f"  A (no-cites pool, claims):      {auc_a:.4f}")
    print(f"  B (cites pool, stripped):       {auc_b:.4f}")
    print(f"  C (cites pool, full):           {auc_c:.4f}")
    print(f"  C - B (cites contribution):     {auc_c - auc_b:+.4f}")
    print(f"  B - A (population difference):  {auc_b - auc_a:+.4f}")
    print()
    if auc_c - auc_b >= 0.015:
        print("→ Cites add meaningful signal. Cites-only filter worth considering.")
    elif auc_c - auc_b >= 0.005:
        print("→ Cites add marginal signal. Not worth losing 70% of data.")
    else:
        print("→ Cites add ~no signal. Don't bother with cites-only filter.")


if __name__ == "__main__":
    main()
