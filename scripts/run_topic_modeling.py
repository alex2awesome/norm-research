#!/usr/bin/env python
"""
Run LDA topic modeling (k=100) on each of 12 modeling datasets.
Hard-assign each document to its argmax topic and save alongside the original.
"""

import csv
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

csv.field_size_limit(2**31 - 1)

DATASETS = [
    "/lfs/skampere3/0/alexspan/norm-research/datasets/code-review/code_review_dense_4096tok.csv.gz",
    "/lfs/skampere3/0/alexspan/norm-research/datasets/creative-writing/litbench-to-train.csv.gz",
    "/lfs/skampere3/0/alexspan/norm-research/datasets/notice-and-comment/notice_and_comment_len_balanced.csv.gz",
    "/lfs/skampere3/0/alexspan/norm-research/datasets/humor/reddit_humor_modeling.csv.gz",
    "/lfs/skampere3/0/alexspan/norm-research/datasets/patents/patents_first_draft_balanced.csv.gz",
    "/lfs/skampere3/0/alexspan/norm-research/datasets/patents/patents_first_draft_with_applicant_cites_balanced.csv.gz",
    "/lfs/skampere3/0/alexspan/norm-research/datasets/patents/patents_final_outcome_balanced.csv.gz",
    "/lfs/skampere3/0/alexspan/norm-research/datasets/patents/patents_final_outcome_with_examiner_cites_balanced.csv.gz",
    "/lfs/skampere3/0/alexspan/norm-research/datasets/legal-outcome-prediction/legal_outcome_facts_statutes.csv.gz",
    "/lfs/skampere3/0/alexspan/norm-research/datasets/press-releases/press_release_modeling_dataset_clean.csv.gz",
    "/lfs/skampere3/0/alexspan/norm-research/datasets/news-homepages/homepage_newsworthiness_topic_balanced.csv.gz",
    "/lfs/skampere3/0/alexspan/norm-research/datasets/peer-review/peer_review_modeling_dataset.csv.gz",
]


def get_output_path(input_path):
    """Generate output path: {original_name}_with_topics.csv.gz"""
    directory = os.path.dirname(input_path)
    basename = os.path.basename(input_path)
    # Remove .csv.gz suffix
    if basename.endswith(".csv.gz"):
        stem = basename[:-len(".csv.gz")]
    else:
        stem = os.path.splitext(basename)[0]
    return os.path.join(directory, f"{stem}_with_topics.csv.gz")


def print_top_words(model, feature_names, n_topics=10, n_words=5):
    """Print top words for the first n_topics topics."""
    print(f"\n  Top {n_words} words for topics 0-{n_topics-1}:")
    for topic_idx in range(min(n_topics, model.n_components)):
        top_indices = model.components_[topic_idx].argsort()[:-n_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        print(f"    Topic {topic_idx:3d}: {', '.join(top_words)}")


def print_topic_stats(topics):
    """Print topic distribution statistics."""
    unique, counts = np.unique(topics, return_counts=True)
    print(f"\n  Topic distribution stats:")
    print(f"    Number of unique topics assigned: {len(unique)} / 100")
    print(f"    Most common topic: {unique[counts.argmax()]} ({counts.max()} docs)")
    print(f"    Least common topic: {unique[counts.argmin()]} ({counts.min()} docs)")
    print(f"    Mean docs/topic: {counts.mean():.1f}")
    print(f"    Std docs/topic: {counts.std():.1f}")
    print(f"    Median docs/topic: {np.median(counts):.1f}")


def process_dataset(path):
    """Process a single dataset: TF-IDF -> LDA -> hard-assign topics."""
    name = os.path.basename(path)
    print(f"\n{'='*70}")
    print(f"Processing: {name}")
    print(f"  Path: {path}")

    # Check if file exists
    if not os.path.exists(path):
        print(f"  ERROR: File not found, skipping.")
        return False

    # Load data
    t0 = time.time()
    print(f"  Loading data...", end=" ", flush=True)
    df = pd.read_csv(path, compression="gzip")
    print(f"done. {len(df)} rows, {df.columns.tolist()} in {time.time()-t0:.1f}s")

    # Verify required columns
    if "text" not in df.columns:
        print(f"  ERROR: 'text' column not found. Columns: {df.columns.tolist()}")
        return False
    if "judgement" not in df.columns:
        print(f"  ERROR: 'judgement' column not found. Columns: {df.columns.tolist()}")
        return False

    # Fill NaN texts with empty string
    n_nan = df["text"].isna().sum()
    if n_nan > 0:
        print(f"  Warning: {n_nan} NaN texts, replacing with empty string.")
        df["text"] = df["text"].fillna("")

    # TF-IDF
    t1 = time.time()
    print(f"  Running TF-IDF (max_features=10000)...", end=" ", flush=True)
    tfidf = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        max_df=0.95,
        min_df=5,
    )
    tfidf_matrix = tfidf.fit_transform(df["text"])
    feature_names = tfidf.get_feature_names_out()
    print(f"done. Matrix shape: {tfidf_matrix.shape} in {time.time()-t1:.1f}s")

    # LDA
    t2 = time.time()
    print(f"  Fitting LDA (k=100, max_iter=20)...", end=" ", flush=True)
    lda = LatentDirichletAllocation(
        n_components=100,
        random_state=42,
        max_iter=20,
        n_jobs=-1,
    )
    doc_topic_dist = lda.fit_transform(tfidf_matrix)
    print(f"done in {time.time()-t2:.1f}s")

    # Hard-assign topics
    topics = doc_topic_dist.argmax(axis=1)
    df["topic"] = topics

    # Print diagnostics
    print_top_words(lda, feature_names, n_topics=10, n_words=5)
    print_topic_stats(topics)

    # Save output
    output_path = get_output_path(path)
    t3 = time.time()
    print(f"\n  Saving to: {output_path}...", end=" ", flush=True)
    df[["text", "judgement", "topic"]].to_csv(
        output_path, index=False, compression="gzip"
    )
    print(f"done in {time.time()-t3:.1f}s")

    return True


def main():
    print("LDA Topic Modeling Pipeline")
    print(f"Datasets to process: {len(DATASETS)}")
    print(f"Settings: k=100, max_features=10000, max_iter=20, random_state=42")

    total_t0 = time.time()
    results = []

    for i, path in enumerate(DATASETS, 1):
        print(f"\n[{i}/{len(DATASETS)}]", end="")
        success = process_dataset(path)
        results.append((os.path.basename(path), success))

    # Summary
    print(f"\n\n{'='*70}")
    print(f"SUMMARY ({time.time()-total_t0:.1f}s total)")
    print(f"{'='*70}")
    for name, success in results:
        status = "OK" if success else "FAILED"
        print(f"  [{status}] {name}")

    n_ok = sum(1 for _, s in results if s)
    print(f"\n{n_ok}/{len(results)} datasets processed successfully.")


if __name__ == "__main__":
    main()
