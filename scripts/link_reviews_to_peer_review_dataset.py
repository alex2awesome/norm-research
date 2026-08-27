"""
Link explicit review text (reasoning) to the peer review modeling dataset.

For each paper in the modeling dataset, finds all corresponding non-meta reviews
from unified_reviews.csv.gz and concatenates them as "reasoning". Outputs a new
file peer_review_modeling_dataset_with_reasoning.csv.gz alongside the original.

Usage:
    python link_reviews_to_peer_review_dataset.py \
        --data-dir /lfs/skampere3/0/alexspan/norm-research/datasets/peer-review
"""

import argparse
import csv
import gzip
import os
import sys
from collections import defaultdict

csv.field_size_limit(2**31 - 1)

REVIEW_SEPARATOR = "\n\n" + "=" * 60 + "\n\n"


def load_reviews(reviews_path: str, paper_ids: set) -> dict:
    """Load reviews for the given paper_ids, skipping meta-reviews.

    Returns dict mapping paper_id -> list of review texts (sorted by review_id).
    """
    reviews = defaultdict(list)
    total = 0
    skipped_meta = 0
    skipped_irrelevant = 0

    with gzip.open(reviews_path, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            pid = row["paper_id"]
            if pid not in paper_ids:
                skipped_irrelevant += 1
                continue
            if row.get("is_meta_review", "False") == "True":
                skipped_meta += 1
                continue
            review_text = row.get("review_text", "")
            if review_text and review_text.strip():
                reviews[pid].append((row.get("review_id", ""), review_text.strip()))

    # Sort each paper's reviews by review_id for deterministic ordering
    for pid in reviews:
        reviews[pid].sort(key=lambda x: x[0])

    print(f"Reviews file: {total:,} total rows")
    print(f"  Skipped (not in modeling dataset): {skipped_irrelevant:,}")
    print(f"  Skipped (meta-review): {skipped_meta:,}")
    print(f"  Kept: {sum(len(v) for v in reviews.values()):,} reviews for {len(reviews):,} papers")

    return reviews


def main():
    parser = argparse.ArgumentParser(description="Link reviews to peer review modeling dataset")
    parser.add_argument(
        "--data-dir",
        default="/lfs/skampere3/0/alexspan/norm-research/datasets/peer-review",
        help="Directory containing the peer review data files",
    )
    args = parser.parse_args()

    modeling_path = os.path.join(args.data_dir, "peer_review_modeling_dataset.csv.gz")
    reviews_path = os.path.join(args.data_dir, "unified_reviews.csv.gz")
    output_path = os.path.join(args.data_dir, "peer_review_modeling_dataset_with_reasoning.csv.gz")

    # Step 1: Load modeling dataset to get all paper_ids
    print("Loading modeling dataset...")
    paper_ids = set()
    modeling_rows = []
    with gzip.open(modeling_path, "rt") as f:
        reader = csv.DictReader(f)
        modeling_fields = reader.fieldnames
        for row in reader:
            paper_ids.add(row["paper_id"])
            modeling_rows.append(row)

    print(f"Modeling dataset: {len(modeling_rows):,} rows, {len(paper_ids):,} unique paper_ids")

    # Step 2: Load reviews for those paper_ids
    print("\nLoading reviews...")
    reviews = load_reviews(reviews_path, paper_ids)

    # Step 3: Write output with reasoning column
    print(f"\nWriting output to {output_path}...")
    output_fields = list(modeling_fields) + ["reasoning", "num_reviews"]

    rows_with_reasoning = 0
    rows_without_reasoning = 0
    total_reviews_linked = 0
    reasoning_lengths = []

    with gzip.open(output_path, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()

        for row in modeling_rows:
            pid = row["paper_id"]
            paper_reviews = reviews.get(pid, [])

            if paper_reviews:
                # Concatenate review texts with separator
                reasoning = REVIEW_SEPARATOR.join(text for _, text in paper_reviews)
                row["reasoning"] = reasoning
                row["num_reviews"] = len(paper_reviews)
                rows_with_reasoning += 1
                total_reviews_linked += len(paper_reviews)
                reasoning_lengths.append(len(reasoning))
            else:
                row["reasoning"] = ""
                row["num_reviews"] = 0
                rows_without_reasoning += 1

            writer.writerow(row)

    # Step 4: Report stats
    print("\n" + "=" * 60)
    print("STATS")
    print("=" * 60)
    print(f"Total rows in output: {len(modeling_rows):,}")
    print(f"Rows WITH reasoning:  {rows_with_reasoning:,} ({100*rows_with_reasoning/len(modeling_rows):.1f}%)")
    print(f"Rows WITHOUT reasoning: {rows_without_reasoning:,} ({100*rows_without_reasoning/len(modeling_rows):.1f}%)")
    print(f"Total reviews linked: {total_reviews_linked:,}")
    if rows_with_reasoning > 0:
        avg_reviews = total_reviews_linked / rows_with_reasoning
        avg_len = sum(reasoning_lengths) / len(reasoning_lengths)
        min_len = min(reasoning_lengths)
        max_len = max(reasoning_lengths)
        median_len = sorted(reasoning_lengths)[len(reasoning_lengths) // 2]
        print(f"Avg reviews per paper (when present): {avg_reviews:.2f}")
        print(f"Reasoning length (chars) - avg: {avg_len:,.0f}, median: {median_len:,}, min: {min_len:,}, max: {max_len:,}")
    print(f"\nOutput file: {output_path}")


if __name__ == "__main__":
    main()
