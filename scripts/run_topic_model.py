#!/usr/bin/env python3
"""
Run LDA topic modeling on a text column in a CSV file.

Outputs two files alongside the input CSV:
  - {filestem}__doc_to_topic.csv   — columns: id, topic
  - {filestem}__topic_to_words.csv — columns: topic, top_20_words

Usage:
    python scripts/run_topic_model.py data.csv.gz --id press_release_id --text text --k 25
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def main():
    parser = argparse.ArgumentParser(description="LDA topic modeling on a CSV text column")
    parser.add_argument("input_csv", type=str, help="Path to input CSV (supports .gz)")
    parser.add_argument("--id", type=str, default="id", help="Name of the ID column (default: id)")
    parser.add_argument("--text", type=str, default="text", help="Name of the text column (default: text)")
    parser.add_argument("--k", type=int, default=25, help="Number of topics (default: 25)")
    parser.add_argument("--max-features", type=int, default=20000, help="Max vocabulary size (default: 20000)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    # Strip all suffixes for the filestem (e.g. .csv.gz -> base name)
    stem = input_path.name
    for s in input_path.suffixes:
        stem = stem.removesuffix(s)
    out_dir = input_path.parent

    # Load data
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  {len(df):,} rows, columns: {list(df.columns)}")

    if args.id not in df.columns:
        raise ValueError(f"ID column '{args.id}' not found. Available: {list(df.columns)}")
    if args.text not in df.columns:
        raise ValueError(f"Text column '{args.text}' not found. Available: {list(df.columns)}")

    # Drop rows with missing text
    df = df.dropna(subset=[args.text]).reset_index(drop=True)
    texts = df[args.text].astype(str).tolist()
    ids = df[args.id].tolist()
    print(f"  {len(texts):,} rows after dropping missing text")

    # Vectorize
    print(f"Vectorizing (max_features={args.max_features})...")
    vectorizer = CountVectorizer(
        max_features=args.max_features,
        stop_words="english",
        min_df=5,
        max_df=0.95,
    )
    dtm = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    print(f"  Vocabulary size: {len(vocab):,}, document-term matrix: {dtm.shape}")

    # Fit LDA
    print(f"Fitting LDA with k={args.k}...")
    lda = LatentDirichletAllocation(
        n_components=args.k,
        random_state=args.random_state,
        n_jobs=-1,
        verbose=1,
    )
    doc_topic = lda.fit_transform(dtm)  # (n_docs, k)
    print("  Done.")

    # Hard-assign topics
    assigned_topics = doc_topic.argmax(axis=1)

    # Write doc_to_topic
    doc_topic_path = out_dir / f"{stem}__doc_to_topic.csv"
    doc_topic_df = pd.DataFrame({"id": ids, "topic": assigned_topics})
    doc_topic_df.to_csv(doc_topic_path, index=False)
    print(f"Wrote {doc_topic_path} ({len(doc_topic_df):,} rows)")

    # Write topic_to_words
    topic_words_path = out_dir / f"{stem}__topic_to_words.csv"
    rows = []
    for topic_idx in range(args.k):
        top_word_indices = lda.components_[topic_idx].argsort()[::-1][:20]
        top_words = ", ".join(vocab[i] for i in top_word_indices)
        rows.append({"topic": topic_idx, "top_20_words": top_words})
    topic_words_df = pd.DataFrame(rows)
    topic_words_df.to_csv(topic_words_path, index=False)
    print(f"Wrote {topic_words_path} ({len(topic_words_df)} topics)")

    # Print summary
    print(f"\n--- Topic distribution ---")
    counts = doc_topic_df["topic"].value_counts().sort_index()
    for topic_idx, count in counts.items():
        words = rows[topic_idx]["top_20_words"]
        # Truncate for display
        words_short = ", ".join(words.split(", ")[:8]) + "..."
        print(f"  Topic {topic_idx:>2}: {count:>6,} docs — {words_short}")


if __name__ == "__main__":
    main()
