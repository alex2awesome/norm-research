#!/usr/bin/env python3
"""Quick AUC test on the CPC-balanced patents file vs the old baseline.

Uses tf-idf + LR on a 100K stratified sample of each. Compares directly so
we can see whether CPC + year + length balancing lowers AUC (= deconfounded).
"""
import csv
import gzip
import random
import time

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

csv.field_size_limit(2**31 - 1)

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
OLD = f"{BASE}/patents_first_draft_balanced.csv.gz"
NEW = f"{BASE}/patents_first_draft_cpc_balanced.csv.gz"

SAMPLE = 100_000  # per file
SEED = 42


def load_sample(path, n_target):
    """Reservoir-balanced sample of n_target rows."""
    print(f"Loading {path} ...")
    texts_pos, texts_neg = [], []
    rng = random.Random(SEED)
    with gzip.open(path, "rt") as f:
        for r in csv.DictReader(f):
            t = r["text"]
            y = int(r["judgement"])
            if y == 1:
                if len(texts_pos) < n_target:
                    texts_pos.append(t)
                else:
                    j = rng.randrange(len(texts_pos) + 1)
                    if j < n_target:
                        texts_pos[j] = t
            else:
                if len(texts_neg) < n_target:
                    texts_neg.append(t)
                else:
                    j = rng.randrange(len(texts_neg) + 1)
                    if j < n_target:
                        texts_neg[j] = t
    print(f"  pos={len(texts_pos):,}  neg={len(texts_neg):,}")
    n_per = min(len(texts_pos), len(texts_neg), n_target // 2)
    rng.shuffle(texts_pos)
    rng.shuffle(texts_neg)
    texts = texts_pos[:n_per] + texts_neg[:n_per]
    labels = [1] * n_per + [0] * n_per
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    return [texts[i] for i in idx], [labels[i] for i in idx]


def evaluate(name, texts, labels):
    print(f"\n=== {name} ===")
    t0 = time.time()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels)
    print(f"  train={len(X_train):,}  test={len(X_test):,}")
    pipe = Pipeline([
        ("h", HashingVectorizer(n_features=2 ** 18, ngram_range=(1, 2),
                                alternate_sign=False, lowercase=True)),
        ("t", TfidfTransformer()),
        ("lr", LogisticRegression(C=1.0, solver="liblinear", max_iter=200)),
    ])
    pipe.fit(X_train, y_train)
    scores = pipe.decision_function(X_test)
    auc = roc_auc_score(y_test, scores)
    print(f"  AUC: {auc:.4f}  (took {time.time() - t0:.0f}s)")
    return auc


def main():
    print(f"Sample size per file: {SAMPLE:,}")
    print()
    old_texts, old_labels = load_sample(OLD, SAMPLE)
    new_texts, new_labels = load_sample(NEW, SAMPLE)
    auc_old = evaluate("OLD: patents_first_draft_balanced (500K, length-balanced, all years)",
                       old_texts, old_labels)
    auc_new = evaluate("NEW: patents_first_draft_cpc_balanced (928K, CPC+length+year<=2021)",
                       new_texts, new_labels)
    print()
    print("=== Summary ===")
    print(f"  OLD: {auc_old:.4f}")
    print(f"  NEW: {auc_new:.4f}")
    print(f"  Delta (NEW - OLD): {auc_new - auc_old:+.4f}")
    print()
    if auc_new < auc_old - 0.01:
        print("→ Balancing meaningfully lowered AUC. CPC/length/year confounds were real.")
    elif auc_new < auc_old:
        print("→ Marginal drop. Balancing helped a little.")
    else:
        print("→ AUC did NOT drop. Confounds may not have been the main signal driver, "
              "OR the LR baseline isn't sensitive to the CPC shortcut "
              "(the dense Llama model might be).")


if __name__ == "__main__":
    main()
