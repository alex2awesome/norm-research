#!/usr/bin/env python
"""
Deduplicate the creative writing (LitBench) dataset using:
  1. Exact dedup (drop rows with identical text, keep first occurrence)
  2. MinHash LSH on character 9-grams (threshold=0.5, num_perm=128)
Then run LDA topic modeling (k=100) on the deduped dataset.
"""

import csv
import sys
import time

csv.field_size_limit(2**31 - 1)

import pandas as pd
import numpy as np
from datasketch import MinHash, MinHashLSH

# ── paths ──────────────────────────────────────────────────────────────
BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/creative-writing"
INPUT  = f"{BASE}/litbench-to-train.csv.gz"
OUTPUT_DEDUP  = f"{BASE}/litbench-to-train-dedup.csv.gz"
OUTPUT_TOPICS = f"{BASE}/litbench-to-train-dedup_with_topics.csv.gz"

NUM_PERM = 128
LSH_THRESHOLD = 0.5
CHAR_NGRAM = 9

# ── 0. Load ────────────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(INPUT, compression="gzip")
# Drop the unnamed index column if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
df = df.reset_index(drop=True)
original_count = len(df)
print(f"  Original count: {original_count:,}")

# ── 1. Exact dedup ────────────────────────────────────────────────────
print("\nStep 1: Exact dedup (keeping first occurrence) …")
df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
after_exact = len(df)
exact_removed = original_count - after_exact
print(f"  After exact dedup: {after_exact:,}  (removed {exact_removed:,}, "
      f"{100*exact_removed/original_count:.1f}%)")

# ── 2. MinHash LSH ────────────────────────────────────────────────────
print(f"\nStep 2: MinHash LSH  (char {CHAR_NGRAM}-grams, num_perm={NUM_PERM}, "
      f"threshold={LSH_THRESHOLD}) …")

def char_ngram_bytes(text, n=CHAR_NGRAM):
    """Return list of encoded character n-grams from text."""
    t = str(text)
    return [t[i:i+n].encode("utf-8") for i in range(len(t) - n + 1)]

# Build MinHash for every row using update_batch for speed
minhashes = []
t0 = time.time()
for idx, text in enumerate(df["text"].values):
    m = MinHash(num_perm=NUM_PERM)
    grams = char_ngram_bytes(text)
    if grams:
        m.update_batch(grams)
    minhashes.append(m)
    if (idx + 1) % 10_000 == 0:
        elapsed = time.time() - t0
        print(f"    MinHash computed for {idx+1:,}/{after_exact:,} rows  "
              f"({elapsed:.1f}s elapsed)")

elapsed = time.time() - t0
print(f"    MinHash done for all {after_exact:,} rows  ({elapsed:.1f}s)")

# Insert into LSH index
print("  Building LSH index …")
lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
for idx, m in enumerate(minhashes):
    lsh.insert(str(idx), m)
print("  LSH index built.")

# Find clusters via connected-component traversal
print("  Querying LSH for near-duplicate clusters …")
visited = set()
clusters = []  # list of lists of indices
for idx in range(after_exact):
    if idx in visited:
        continue
    # query returns list of keys (strings) that are near-duplicates
    result = lsh.query(minhashes[idx])
    group = [int(r) for r in result]
    if len(group) > 1:
        clusters.append(group)
    for g in group:
        visited.add(g)

print(f"  Found {len(clusters):,} near-duplicate clusters "
      f"(covering {sum(len(c) for c in clusters):,} rows)")

# For each cluster, keep the row with the highest upvotes
indices_to_drop = set()
for cluster in clusters:
    # Find the index with highest upvotes
    upvotes_in_cluster = df.iloc[cluster]["upvotes"].values
    best_local = np.argmax(upvotes_in_cluster)
    best_idx = cluster[best_local]
    for idx in cluster:
        if idx != best_idx:
            indices_to_drop.add(idx)

print(f"  Dropping {len(indices_to_drop):,} near-duplicate rows")

df_dedup = df.drop(index=list(indices_to_drop)).reset_index(drop=True)
after_minhash = len(df_dedup)
minhash_removed = after_exact - after_minhash
total_removed = original_count - after_minhash

print(f"\n{'='*60}")
print(f"  Original count:       {original_count:>10,}")
print(f"  After exact dedup:    {after_exact:>10,}  (−{exact_removed:,})")
print(f"  After MinHash dedup:  {after_minhash:>10,}  (−{minhash_removed:,})")
print(f"  Total removed:        {total_removed:>10,}  "
      f"({100*total_removed/original_count:.1f}%)")
print(f"{'='*60}")

# ── 3. Save deduped dataset ───────────────────────────────────────────
print(f"\nSaving deduped dataset to {OUTPUT_DEDUP} …")
df_dedup.to_csv(OUTPUT_DEDUP, index=False, compression="gzip")
print("  Saved.")

# ── 4. LDA topic modeling ─────────────────────────────────────────────
print("\nStep 3: LDA topic modeling (k=100) on deduped dataset …")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

print("  Fitting TF-IDF …")
tfidf = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    max_df=0.95,
    min_df=5,
)
X = tfidf.fit_transform(df_dedup["text"].fillna(""))
print(f"  TF-IDF matrix shape: {X.shape}")

print("  Fitting LDA (n_components=100, max_iter=20) …")
lda = LatentDirichletAllocation(
    n_components=100,
    random_state=42,
    max_iter=20,
    n_jobs=-1,
)
doc_topics = lda.fit_transform(X)
print("  LDA done.")

# Assign dominant topic
df_dedup["topic"] = doc_topics.argmax(axis=1)

# Print top words per topic (first 10 topics as sample)
feature_names = tfidf.get_feature_names_out()
print("\n  Top-10 words for first 10 topics:")
for t in range(min(10, lda.n_components)):
    top_idx = lda.components_[t].argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_idx]
    print(f"    Topic {t:>3d}: {', '.join(top_words)}")

# Save with topics
print(f"\nSaving deduped+topics dataset to {OUTPUT_TOPICS} …")
df_dedup.to_csv(OUTPUT_TOPICS, index=False, compression="gzip")
print("  Saved.")

print("\nDone!")
