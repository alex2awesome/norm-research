#!/usr/bin/env python
"""Inspect all modeling datasets for data cleanliness issues."""

import csv
import gzip
import os
import random
import re
import sys
from collections import Counter

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

# Patterns to detect issues
HTML_PATTERN = re.compile(r'<[a-zA-Z][^>]*>|&[a-z]+;|&#\d+;')
ENCODING_PATTERN = re.compile(r'[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]')
REPEATED_PATTERN = re.compile(r'(.{20,}?)\1{2,}')  # same 20+ char block repeated 3+ times

def detect_issues(text):
    issues = []
    if HTML_PATTERN.search(text[:2000]):
        issues.append("HTML_ARTIFACTS")
    if ENCODING_PATTERN.search(text[:2000]):
        issues.append("ENCODING_ISSUES")
    if REPEATED_PATTERN.search(text[:2000]):
        issues.append("REPEATED_TEXT")
    # Check for non-ASCII heavy text (potential non-English)
    if len(text) > 50:
        ascii_ratio = sum(1 for c in text[:1000] if ord(c) < 128) / min(len(text), 1000)
        if ascii_ratio < 0.7:
            issues.append("POSSIBLY_NON_ENGLISH")
    return issues

def inspect_dataset(path):
    short_name = os.path.basename(path)
    print(f"\n{'='*80}")
    print(f"DATASET: {short_name}")
    print(f"PATH: {path}")
    print(f"{'='*80}")

    if not os.path.exists(path):
        print("  *** FILE NOT FOUND ***")
        return

    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Compressed size: {file_size_mb:.1f} MB")

    texts = []
    labels = []
    all_columns = None
    parse_errors = 0

    try:
        with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            all_columns = reader.fieldnames
            for i, row in enumerate(reader):
                try:
                    text = row.get('text', '')
                    label = row.get('judgement', '')
                    texts.append(text)
                    labels.append(str(label).strip())
                except Exception as e:
                    parse_errors += 1
                    if parse_errors <= 3:
                        print(f"  Parse error row {i}: {e}")
    except Exception as e:
        print(f"  *** ERROR reading file: {e} ***")
        return

    total = len(texts)
    if total == 0:
        print("  *** EMPTY DATASET ***")
        return

    print(f"  Columns: {all_columns}")
    print(f"  Total rows: {total:,}")
    if parse_errors:
        print(f"  Parse errors: {parse_errors}")

    # Label distribution
    label_counts = Counter(labels)
    print(f"  Label distribution:")
    for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"    judgement={lbl}: {cnt:,} ({100*cnt/total:.1f}%)")

    # Text length stats
    lengths = [len(t) for t in texts]
    avg_len = sum(lengths) / total
    min_len = min(lengths)
    max_len = max(lengths)
    empty_count = sum(1 for l in lengths if l == 0)
    very_short = sum(1 for l in lengths if 0 < l < 50)
    print(f"  Text length: avg={avg_len:.0f}, min={min_len}, max={max_len}")
    print(f"  Empty texts: {empty_count} ({100*empty_count/total:.2f}%)")
    print(f"  Very short (<50 chars): {very_short} ({100*very_short/total:.2f}%)")

    # Detect issues across all texts (sample for speed on large datasets)
    issue_counts = Counter()
    check_indices = range(total) if total <= 10000 else random.sample(range(total), 10000)
    for idx in check_indices:
        for issue in detect_issues(texts[idx]):
            issue_counts[issue] += 1
    checked = len(list(check_indices)) if total <= 10000 else 10000
    if issue_counts:
        print(f"  Issues detected (checked {checked:,} rows):")
        for issue, cnt in issue_counts.most_common():
            print(f"    {issue}: {cnt} ({100*cnt/checked:.1f}%)")
    else:
        print(f"  No issues detected (checked {checked:,} rows)")

    # Check for duplicate texts
    if total <= 200000:
        text_hashes = [hash(t) for t in texts]
        unique_hashes = len(set(text_hashes))
        dup_count = total - unique_hashes
        if dup_count > 0:
            print(f"  Duplicate texts (by hash): {dup_count} ({100*dup_count/total:.2f}%)")
    else:
        # Sample-based duplicate check
        sample_size = 50000
        sample_idx = random.sample(range(total), sample_size)
        sample_hashes = [hash(texts[i]) for i in sample_idx]
        unique_sample = len(set(sample_hashes))
        dup_est = sample_size - unique_sample
        if dup_est > 0:
            print(f"  Estimated duplicates (sampled {sample_size:,}): ~{dup_est} ({100*dup_est/sample_size:.2f}%)")

    # Sample examples: 2-3 from each label
    print(f"\n  --- SAMPLE EXAMPLES ---")
    unique_labels = sorted(set(labels))
    for lbl in unique_labels[:5]:  # cap at 5 unique labels
        lbl_indices = [i for i in range(total) if labels[i] == lbl]
        n_sample = min(3 if lbl in unique_labels[:2] else 2, len(lbl_indices))
        sampled = random.sample(lbl_indices, n_sample)
        for idx in sampled:
            text_preview = texts[idx][:500].replace('\n', '\\n').replace('\r', '\\r')
            print(f"  [judgement={lbl}] (len={lengths[idx]:,}) {text_preview}")
            issues = detect_issues(texts[idx])
            if issues:
                print(f"    ^ Issues: {', '.join(issues)}")
            print()

    # Flag summary
    flags = []
    if empty_count > 0:
        flags.append(f"{empty_count} empty texts")
    if very_short > total * 0.01:
        flags.append(f"{100*very_short/total:.1f}% very short")
    if parse_errors > 0:
        flags.append(f"{parse_errors} parse errors")
    if len(label_counts) < 2:
        flags.append("ONLY ONE LABEL VALUE")
    if len(label_counts) > 10:
        flags.append(f"{len(label_counts)} unique labels (expected 2)")
    for issue, cnt in issue_counts.most_common():
        pct = 100 * cnt / checked
        if pct > 5:
            flags.append(f"{pct:.0f}% have {issue}")
    if total - unique_hashes if total <= 200000 else 0 > total * 0.05:
        flags.append("high duplicate rate")

    if flags:
        print(f"  *** FLAGS: {'; '.join(flags)} ***")
    else:
        print(f"  No flags.")

if __name__ == "__main__":
    random.seed(42)
    print("=" * 80)
    print("DATASET INSPECTION REPORT")
    print("=" * 80)
    for ds in DATASETS:
        try:
            inspect_dataset(ds)
        except Exception as e:
            print(f"\n*** FATAL ERROR on {ds}: {e} ***")
    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)
