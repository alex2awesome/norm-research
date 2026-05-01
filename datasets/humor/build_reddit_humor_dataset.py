#!/usr/bin/env python3
"""
Build a stratified humor preference dataset from r/jokes Reddit data.

Strategy:
1. Combine title + body text
2. Drop NSFW, removed/deleted, very short
3. Stratify by (length_bin, format, topic_cluster)
4. Within each cell, label top 25% by score as 1, bottom 25% as 0
5. Drop middle 50% to enforce clean signal
6. Output text/judgement CSV ready for training

Confounders controlled:
- Length (100-char bins)
- Format: narrative vs one-liner vs riddle
- Topic (TF-IDF + KMeans clustering)
- Implicitly: syllable count, sentence count (correlated with length)
"""
import argparse
import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

csv.field_size_limit(2**31 - 1)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

LENGTH_BIN_SIZE = 100  # characters
N_TOPIC_CLUSTERS = 30
MIN_CELL_SIZE = 20  # cells with fewer than this are dropped
PERCENTILE_LOW = 0.25
PERCENTILE_HIGH = 0.75
MIN_TEXT_LENGTH = 30
MAX_TEXT_LENGTH = 5000  # drop extremely long jokes (likely copypasta walls)

# Dedup
MINHASH_NUM_PERM = 128
MINHASH_JACCARD_THRESHOLD = 0.8
SHINGLE_SIZE = 5  # word-level k-shingles


# ─────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────

# Strip edit-markers that are strong leakage signals — they only appear on
# posts that received enough engagement for the author to come back and edit.
EDIT_MARKER_CLEANERS = [
    # "EDIT:", "Edit 2 -", etc., from the marker to end of text (almost always trailing)
    (re.compile(r"(?is)\n?\s*edit\s*\d*\s*[:\-].*$"), ""),
    (re.compile(r"(?is)\n?\s*update\s*\d*\s*[:\-].*$"), ""),
    (re.compile(r"(?is)\n?\s*tl\s*[;:]?\s*dr\s*[:\-]?.*$"), ""),
    # Engagement markers
    (re.compile(r"(?i)\[oc\]\s*"), ""),
    (re.compile(r"(?is)\n?\s*(?:thanks|thank you)[^\n]*?(?:gold|silver|platinum|award|upvotes?|front[- ]?page)[^\n]*"), ""),
    # Reddit sarcasm tag
    (re.compile(r"(?i)\s/s\b"), ""),
]


def strip_edit_markers(text):
    """Remove edit/update/TL;DR blocks and award-thanks text."""
    out = text
    for pat, repl in EDIT_MARKER_CLEANERS:
        out = pat.sub(repl, out)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


# ─────────────────────────────────────────────
# FORMAT CLASSIFICATION
# ─────────────────────────────────────────────

NARRATIVE_STARTS = re.compile(
    r"^(a |an |the |two |three |four |so |once |yesterday |last (night|week)|"
    r"this (morning|guy|man|lady)|my (wife|husband|friend|dad|mom|grandma)|"
    r"there (was|were|is|are)|i was|when i)",
    re.IGNORECASE,
)
HAS_DIALOGUE = re.compile(r'\b(said|asked|replied|told|shouted|whispered|exclaimed)\b', re.IGNORECASE)
HAS_QUOTE = re.compile(r'["\u201c\u201d]')


def classify_format(text):
    """Return one of: 'narrative', 'one_liner', 'riddle'."""
    text = text.strip()
    if not text:
        return "one_liner"

    # Riddle: ends with ? and is short (typically title-only)
    if "?" in text[:200] and len(text) < 200 and text.rstrip().endswith("?"):
        return "riddle"

    # Narrative: has dialogue, quotes, or storytelling opener
    if HAS_DIALOGUE.search(text) or HAS_QUOTE.search(text) or NARRATIVE_STARTS.match(text):
        return "narrative"

    return "one_liner"


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

NSFW_PATTERNS = re.compile(
    r"\b(nsfw|sex|fuck|dick|pussy|cock|cum|porn|boob|tit)\b", re.IGNORECASE
)


def load_jokes(path, max_rows=None):
    """Load and clean reddit jokes. Returns DataFrame."""
    print(f"Loading {path}...")
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break

            title = (row.get("title") or "").strip()
            body = (row.get("selftext") or "").strip()

            # Drop removed/deleted bodies
            if body in ("[removed]", "[deleted]", "[deleted by user]"):
                body = ""

            text = title + (" " + body if body else "")
            text = strip_edit_markers(text)

            if len(text) < MIN_TEXT_LENGTH or len(text) > MAX_TEXT_LENGTH:
                continue

            # Drop NSFW (small population, large confound)
            if row.get("subreddit.nsfw", "").lower() == "true":
                continue
            if NSFW_PATTERNS.search(title):
                continue

            try:
                score = int(row.get("score", 0))
            except (ValueError, TypeError):
                continue

            if score < 0:
                continue

            rows.append({
                "text": text,
                "score": score,
                "char_count": len(text),
                "title": title,
                "has_body": bool(body),
            })

    df = pd.DataFrame(rows)
    print(f"  Loaded {len(df):,} valid jokes")
    return df


# ─────────────────────────────────────────────
# NEAR-DUPLICATE DEDUPLICATION
# ─────────────────────────────────────────────

def _text_to_shingles(text, k=SHINGLE_SIZE):
    # Lowercase, split on whitespace, word-level k-shingles.
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < k:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)]


def _build_minhash(text, num_perm=MINHASH_NUM_PERM):
    mh = MinHash(num_perm=num_perm)
    for sh in _text_to_shingles(text):
        mh.update(sh.encode("utf-8"))
    return mh


def dedupe_minhash(df, threshold=MINHASH_JACCARD_THRESHOLD, num_perm=MINHASH_NUM_PERM):
    """Collapse near-duplicate jokes (reposts) into one row with averaged score.

    Uses MinHash LSH on word-level 5-shingles, then union-find over candidate
    neighbors to get transitive clusters. One row per cluster with mean score.
    """
    print(f"Deduping near-duplicates (Jaccard≥{threshold}, num_perm={num_perm})...")

    texts = df["text"].values
    scores = df["score"].values
    char_counts = df["char_count"].values
    n = len(df)

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    hashes = [None] * n
    for i, text in enumerate(texts):
        mh = _build_minhash(text, num_perm=num_perm)
        lsh.insert(str(i), mh)
        hashes[i] = mh
        if (i + 1) % 100000 == 0:
            print(f"  hashed {i + 1:,} / {n:,}")

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for neighbor in lsh.query(hashes[i]):
            j = int(neighbor)
            if j != i:
                union(i, j)
        if (i + 1) % 100000 == 0:
            print(f"  queried {i + 1:,} / {n:,}")

    cluster_ids = np.array([find(i) for i in range(n)])
    df = df.copy()
    df["_cluster"] = cluster_ids

    # Within each cluster: canonical text = longest (most complete version,
    # reposts tend to drop bodies); score = mean across the cluster.
    agg = df.groupby("_cluster").agg(
        text=("text", lambda g: g.iloc[g.str.len().argmax()]),
        score=("score", "mean"),
        score_max=("score", "max"),
        score_count=("score", "size"),
        char_count=("char_count", "max"),
        has_body=("has_body", "any"),
    ).reset_index(drop=True)

    n_merged = n - len(agg)
    print(f"  Original: {n:,} rows | {len(agg):,} unique clusters | merged {n_merged:,} ({n_merged / n * 100:.1f}%)")
    size_dist = agg["score_count"].value_counts().sort_index()
    print("  Cluster-size distribution (first 10):")
    print(size_dist.head(10).to_string())
    big = (agg["score_count"] >= 10).sum()
    huge = (agg["score_count"] >= 50).sum()
    print(f"  Clusters with ≥10 reposts: {big:,} | ≥50 reposts: {huge:,}")
    return agg


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────

def add_format_and_length(df):
    """Add length_bin and format columns."""
    print("Computing format and length bins...")
    df["length_bin"] = (df["char_count"] // LENGTH_BIN_SIZE).astype(int)
    df["format"] = df["text"].apply(classify_format)

    print("  Length bin distribution:")
    print(df["length_bin"].value_counts().sort_index().head(15).to_string())
    print("  Format distribution:")
    print(df["format"].value_counts().to_string())
    return df


def add_topic_clusters(df, n_clusters=N_TOPIC_CLUSTERS):
    """Cluster jokes by topic using LDA with hard topic assignment.

    LDA is more robust than KMeans on TF-IDF for short text — it can find
    actual topical structure even when vocabulary overlap is high.
    """
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

    print(f"Clustering into {n_clusters} topics with LDA...")

    extra_stopwords = {
        "said", "asked", "told", "replied", "got", "went", "just", "like",
        "know", "says", "say", "get", "one", "guy", "man", "men", "lady",
        "woman", "women", "people", "day", "time", "way", "right", "good",
        "back", "yeah", "really", "well", "ok", "okay", "ve", "ll", "don",
        "didn", "doesn", "wasn", "couldn", "wouldn", "isn", "wife", "husband",
        "told", "tells", "tell", "make", "makes", "made", "look", "looks",
        "looked", "thing", "things", "didn", "going", "want", "wanted",
    }
    custom_stop = list(ENGLISH_STOP_WORDS.union(extra_stopwords))

    # LDA needs raw counts, not TF-IDF
    vectorizer = CountVectorizer(
        max_features=10000,
        ngram_range=(1, 1),
        stop_words=custom_stop,
        min_df=20,
        max_df=0.10,
    )
    X = vectorizer.fit_transform(df["text"].values)
    print(f"  Count matrix: {X.shape}")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")

    lda = LatentDirichletAllocation(
        n_components=n_clusters,
        random_state=42,
        learning_method="online",
        batch_size=4096,
        max_iter=10,
        n_jobs=-1,
    )
    print(f"  Fitting LDA on {X.shape[0]:,} documents...")
    doc_topic_dist = lda.fit_transform(X)

    # Hard assignment: pick the most-represented topic per document
    df["topic"] = doc_topic_dist.argmax(axis=1)

    print("  Topic distribution (sorted by size):")
    topic_counts = df["topic"].value_counts().sort_values(ascending=False)
    print(topic_counts.head(15).to_string())

    # Show top words per topic for sanity (top words = highest LDA component weight)
    feature_names = vectorizer.get_feature_names_out()
    print("  Top words per topic (sorted by size):")
    for topic_id in topic_counts.index[:15]:
        top_idx = lda.components_[topic_id].argsort()[-7:][::-1]
        words = [feature_names[j] for j in top_idx]
        n = topic_counts[topic_id]
        print(f"    Topic {topic_id:2d} (n={n:6d}): {', '.join(words)}")

    return df


# ─────────────────────────────────────────────
# STRATIFIED LABELING
# ─────────────────────────────────────────────

def stratified_label(df):
    """
    Within each (length_bin, format, topic) cell:
    - Label top 25% by score as 1
    - Label bottom 25% as 0
    - Drop middle 50%
    """
    print("Applying stratified labeling...")
    cell_groups = df.groupby(["length_bin", "format", "topic"])

    labeled_rows = []
    cell_stats = []
    dropped_small = 0
    dropped_no_variance = 0

    for (lbin, fmt, topic), cell in cell_groups:
        n = len(cell)
        if n < MIN_CELL_SIZE:
            dropped_small += n
            continue

        scores = cell["score"].values
        if scores.max() == scores.min():
            dropped_no_variance += n
            continue

        # Sort by score with a random tiebreaker, then take top/bottom 25% by rank
        # This ensures clean 25/25 splits even when many jokes share the same score
        cell_sorted = cell.sample(frac=1, random_state=42).sort_values(
            "score", kind="mergesort"
        )
        n_quartile = max(1, int(n * PERCENTILE_LOW))
        cell_low = cell_sorted.iloc[:n_quartile]
        cell_high = cell_sorted.iloc[-n_quartile:]

        # Sanity: verify high scores are actually higher than low scores
        if cell_low["score"].max() >= cell_high["score"].min():
            # Heavy ties — skip this cell entirely
            dropped_no_variance += n
            continue

        low_thresh = cell_low["score"].max()
        high_thresh = cell_high["score"].min()

        for _, row in cell_low.iterrows():
            labeled_rows.append({
                "text": row["text"],
                "judgement": 0,
                "_score": row["score"],
                "_length_bin": lbin,
                "_format": fmt,
                "_topic": topic,
            })
        for _, row in cell_high.iterrows():
            labeled_rows.append({
                "text": row["text"],
                "judgement": 1,
                "_score": row["score"],
                "_length_bin": lbin,
                "_format": fmt,
                "_topic": topic,
            })

        cell_stats.append({
            "length_bin": lbin,
            "format": fmt,
            "topic": topic,
            "n_total": n,
            "n_low": len(cell_low),
            "n_high": len(cell_high),
            "low_thresh": low_thresh,
            "high_thresh": high_thresh,
        })

    print(f"  Dropped {dropped_small:,} jokes from small cells (< {MIN_CELL_SIZE})")
    print(f"  Dropped {dropped_no_variance:,} jokes from zero-variance cells")
    print(f"  Kept {len(labeled_rows):,} labeled examples from {len(cell_stats):,} cells")

    out_df = pd.DataFrame(labeled_rows)
    label_counts = out_df["judgement"].value_counts()
    print(f"  Label 1 (high score): {label_counts.get(1, 0):,}")
    print(f"  Label 0 (low score):  {label_counts.get(0, 0):,}")

    return out_df, pd.DataFrame(cell_stats)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reddit_jokes_1m.csv.gz")
    parser.add_argument("--output", default="reddit_humor_modeling.csv.gz")
    parser.add_argument("--cell-stats", default="reddit_humor_cell_stats.csv")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--n-clusters", type=int, default=N_TOPIC_CLUSTERS)
    parser.add_argument("--no-dedup", action="store_true", help="Skip MinHash dedup")
    parser.add_argument("--dedup-threshold", type=float, default=MINHASH_JACCARD_THRESHOLD)
    args = parser.parse_args()

    df = load_jokes(args.input, max_rows=args.max_rows)
    if not args.no_dedup:
        df = dedupe_minhash(df, threshold=args.dedup_threshold)
    df = add_format_and_length(df)
    df = add_topic_clusters(df, n_clusters=args.n_clusters)
    labeled, stats = stratified_label(df)

    # Write output (only text + judgement for training)
    print(f"\nWriting {len(labeled):,} examples to {args.output}")
    out_cols = ["text", "judgement"]
    with gzip.open(args.output, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        writer.writeheader()
        for _, row in labeled.iterrows():
            writer.writerow({"text": row["text"], "judgement": int(row["judgement"])})

    # Save cell stats for inspection
    stats.to_csv(args.cell_stats, index=False)
    print(f"Saved cell stats to {args.cell_stats}")
    print("Done!")


if __name__ == "__main__":
    main()
