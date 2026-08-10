"""
Link explicit PR review comments (reasoning) to the code review modeling dataset.
Creates code_review_dense_4096tok_with_reasoning.csv.gz alongside the original.

Strategy:
  1. Load the modeling dataset to get the set of paper_ids we care about
  2. Parse each paper_id into (owner, repo, pr_number)
  3. Stream the 2.9GB parsed_comments.csv.gz, collecting comments only for matching PRs
  4. Concatenate all comments per PR into a single "reasoning" string
  5. Join back to the modeling dataset and save
"""

import csv
import gzip
import sys
import os
import time
from collections import defaultdict

csv.field_size_limit(2**31 - 1)

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/code-review"
MODELING_PATH = os.path.join(BASE, "code_review_dense_4096tok.csv.gz")
COMMENTS_PATH = os.path.join(BASE, "parsed_comments.csv.gz")
OUTPUT_PATH = os.path.join(BASE, "code_review_dense_4096tok_with_reasoning.csv.gz")


def parse_paper_id(paper_id):
    """Parse 'owner/repo#pr_number' into (owner, repo, pr_number_str)."""
    try:
        repo_part, pr_num = paper_id.rsplit("#", 1)
        owner, repo = repo_part.split("/", 1)
        return owner, repo, pr_num
    except (ValueError, AttributeError):
        return None, None, None


def main():
    t0 = time.time()

    # ------------------------------------------------------------------
    # Step 1: Load modeling dataset, build lookup of paper_ids we need
    # ------------------------------------------------------------------
    print("Step 1: Loading modeling dataset...")
    modeling_rows = []
    paper_id_to_key = {}  # paper_id -> (owner, repo, pr_number)
    needed_keys = set()   # set of (owner, repo, pr_number)

    with gzip.open(MODELING_PATH, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            modeling_rows.append(row)
            pid = row["paper_id"]
            if pid not in paper_id_to_key:
                owner, repo, pr_num = parse_paper_id(pid)
                if owner is not None:
                    key = (owner, repo, pr_num)
                    paper_id_to_key[pid] = key
                    needed_keys.add(key)

    print(f"  Loaded {len(modeling_rows):,} modeling rows")
    print(f"  Unique PRs: {len(needed_keys):,}")
    print(f"  Time: {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 2: Stream comments file, collecting only matching PR comments
    # ------------------------------------------------------------------
    print("\nStep 2: Streaming parsed_comments.csv.gz...")
    t1 = time.time()

    # key -> list of comment body strings
    comments_by_pr = defaultdict(list)
    total_comments = 0
    matched_comments = 0
    skipped_empty = 0

    with gzip.open(COMMENTS_PATH, "rt") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            total_comments += 1
            if (i + 1) % 5_000_000 == 0:
                elapsed = time.time() - t1
                print(f"  Processed {i+1:,} comments ({elapsed:.1f}s), matched {matched_comments:,}...")

            key = (row["owner"], row["repo"], row["pr_number"])
            if key in needed_keys:
                body = row.get("body", "").strip()
                if body:
                    comments_by_pr[key].append(body)
                    matched_comments += 1
                else:
                    skipped_empty += 1

    elapsed2 = time.time() - t1
    print(f"  Total comments scanned: {total_comments:,}")
    print(f"  Matched comments: {matched_comments:,}")
    print(f"  Skipped empty: {skipped_empty:,}")
    print(f"  PRs with comments: {len(comments_by_pr):,}")
    print(f"  Time: {elapsed2:.1f}s")

    # ------------------------------------------------------------------
    # Step 3: Build reasoning strings and join to modeling dataset
    # ------------------------------------------------------------------
    print("\nStep 3: Building reasoning and saving output...")
    t2 = time.time()

    # Pre-build reasoning lookup: paper_id -> concatenated reasoning
    reasoning_lookup = {}
    for pid, key in paper_id_to_key.items():
        if key in comments_by_pr:
            bodies = comments_by_pr[key]
            # Concatenate with separator
            reasoning = "\n---\n".join(bodies)
            reasoning_lookup[pid] = reasoning

    # Free memory
    del comments_by_pr

    # Stats
    rows_with_reasoning = 0
    rows_without_reasoning = 0
    total_reasoning_len = 0
    total_comments_per_pr = 0
    comment_counts = []

    output_columns = ["text", "judgement", "reasoning"]

    with gzip.open(OUTPUT_PATH, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_columns)
        writer.writeheader()

        for row in modeling_rows:
            pid = row["paper_id"]
            reasoning = reasoning_lookup.get(pid, "")

            writer.writerow({
                "text": row["text"],
                "judgement": row["judgement"],
                "reasoning": reasoning,
            })

            if reasoning:
                rows_with_reasoning += 1
                total_reasoning_len += len(reasoning)
                # Count comments (number of separators + 1)
                n_comments = reasoning.count("\n---\n") + 1
                comment_counts.append(n_comments)
            else:
                rows_without_reasoning += 1

    elapsed3 = time.time() - t2
    print(f"  Time: {elapsed3:.1f}s")

    # ------------------------------------------------------------------
    # Step 4: Report stats
    # ------------------------------------------------------------------
    total_rows = len(modeling_rows)
    avg_reasoning_len = total_reasoning_len / rows_with_reasoning if rows_with_reasoning else 0
    avg_comments = sum(comment_counts) / len(comment_counts) if comment_counts else 0

    print("\n" + "=" * 60)
    print("FINAL STATS")
    print("=" * 60)
    print(f"Total modeling rows:          {total_rows:,}")
    print(f"Rows WITH reasoning:          {rows_with_reasoning:,} ({100*rows_with_reasoning/total_rows:.1f}%)")
    print(f"Rows WITHOUT reasoning:       {rows_without_reasoning:,} ({100*rows_without_reasoning/total_rows:.1f}%)")
    print(f"Avg comments per PR (w/ any): {avg_comments:.1f}")
    print(f"Avg reasoning length (chars): {avg_reasoning_len:.0f}")
    if comment_counts:
        comment_counts.sort()
        print(f"Median comments per PR:       {comment_counts[len(comment_counts)//2]}")
        print(f"Max comments per PR:          {max(comment_counts)}")
        print(f"Min comments per PR:          {min(comment_counts)}")
    print(f"\nOutput saved to: {OUTPUT_PATH}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
