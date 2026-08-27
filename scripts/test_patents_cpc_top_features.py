#!/usr/bin/env python3
"""Inspect what the LR learned on the CPC-balanced file:
   1. Per-CPC-section pos rate (verify balance).
   2. Per-section length stats.
   3. Top LR features (using TfidfVectorizer so we can recover token names).
   4. Compare top features OLD vs NEW.
"""
import csv
import gzip
import random
import statistics
import time
from collections import Counter, defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

csv.field_size_limit(2**31 - 1)

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
OLD = f"{BASE}/patents_first_draft_balanced.csv.gz"
NEW = f"{BASE}/patents_first_draft_cpc_balanced.csv.gz"

SAMPLE = 80_000  # per file
SEED = 42


def load_sample(path, n_target, with_meta=False):
    print(f"Loading {path} ...")
    rng = random.Random(SEED)
    bucket = {0: [], 1: []}
    n_each = n_target // 2
    with gzip.open(path, "rt") as f:
        for r in csv.DictReader(f):
            y = int(r["judgement"])
            row = {"text": r["text"], "label": y}
            if with_meta:
                row["cpc_section"] = r.get("cpc_section", "")
                row["year"] = r.get("year", "")
                row["length_bucket"] = r.get("length_bucket", "")
            cur = bucket[y]
            if len(cur) < n_each:
                cur.append(row)
            else:
                j = rng.randrange(len(cur) + 1)
                if j < n_each:
                    cur[j] = row
    rows = bucket[0] + bucket[1]
    rng.shuffle(rows)
    print(f"  loaded n={len(rows):,} (pos={sum(r['label'] for r in rows):,})")
    return rows


def per_section(rows):
    by_sec = defaultdict(lambda: {0: 0, 1: 0})
    by_sec_len = defaultdict(list)
    for r in rows:
        sec = r["cpc_section"]
        by_sec[sec][r["label"]] += 1
        by_sec_len[sec].append(len(r["text"]))
    print("\n--- Per-CPC-section label balance ---")
    print(f"  {'sec':4s} | {'pos':>8s} {'neg':>8s} {'tot':>8s} {'pos_rate':>9s} {'len_med':>9s}")
    for sec in sorted(by_sec):
        d = by_sec[sec]
        tot = d[0] + d[1]
        rate = d[1] / tot if tot else 0
        lens = by_sec_len[sec]
        med_len = statistics.median(lens) if lens else 0
        print(f"  {sec:4s} | {d[1]:>8d} {d[0]:>8d} {tot:>8d} {rate:>8.1%} {med_len:>9.0f}")


def top_features(rows, name, top_k=25):
    print(f"\n=== {name}: top LR features ===")
    t0 = time.time()
    texts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels)
    vec = TfidfVectorizer(
        ngram_range=(1, 2), max_features=200_000,
        lowercase=True, min_df=20, sublinear_tf=True)
    Xt = vec.fit_transform(X_train)
    Xv = vec.transform(X_test)
    lr = LogisticRegression(C=1.0, solver="liblinear", max_iter=200)
    lr.fit(Xt, y_train)
    auc = roc_auc_score(y_test, lr.decision_function(Xv))
    print(f"  AUC: {auc:.4f}  (took {time.time() - t0:.0f}s)")
    print(f"  Vocab size: {len(vec.vocabulary_):,}")

    vocab = vec.get_feature_names_out()
    coefs = lr.coef_[0]
    pos_idx = coefs.argsort()[::-1][:top_k]
    neg_idx = coefs.argsort()[:top_k]
    print(f"\n  Top {top_k} POS-leaning features (label=1, first-draft approved):")
    for i in pos_idx:
        print(f"    {coefs[i]:+.3f}  {vocab[i]!r}")
    print(f"\n  Top {top_k} NEG-leaning features (label=0, NOT approved):")
    for i in neg_idx:
        print(f"    {coefs[i]:+.3f}  {vocab[i]!r}")
    return auc, vocab, coefs


def main():
    print(f"Sample per file: {SAMPLE:,}")
    new_rows = load_sample(NEW, SAMPLE, with_meta=True)
    per_section(new_rows)

    new_auc, new_vocab, new_coefs = top_features(new_rows, "NEW (CPC-balanced)")

    old_rows = load_sample(OLD, SAMPLE, with_meta=False)
    old_auc, old_vocab, old_coefs = top_features(old_rows, "OLD (length-balanced only)")

    print()
    print("=== Summary ===")
    print(f"  OLD AUC: {old_auc:.4f}")
    print(f"  NEW AUC: {new_auc:.4f}")


if __name__ == "__main__":
    main()
