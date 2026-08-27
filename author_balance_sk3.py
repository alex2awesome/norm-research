#!/usr/bin/env python3
"""
Author-balance analysis on StackExchange datasets (sk3 version).
Tests whether acceptance signal is author-identity-driven.
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import sys

def load_posts_xml_mapping(posts_xml_path):
    """
    Build answer_id -> OwnerUserId mapping from Posts.xml.
    StackExchange Posts.xml contains both questions and answers.
    """
    print(f"Parsing {posts_xml_path}...")

    answer_to_owner = {}
    count = 0

    for event, elem in ET.iterparse(posts_xml_path, events=['start']):
        if elem.tag == 'row':
            attrs = elem.attrib
            post_type_id = attrs.get('PostTypeId')

            # PostTypeId=1 is question, =2 is answer
            if post_type_id == '2':
                answer_id = attrs.get('Id')
                owner_user_id = attrs.get('OwnerUserId')

                if answer_id and owner_user_id:
                    try:
                        answer_to_owner[int(answer_id)] = int(owner_user_id)
                        count += 1
                    except (ValueError, TypeError):
                        pass

            # Clear element to save memory
            elem.clear()

    print(f"Mapped {count} answers to owners")
    return answer_to_owner

def analyze_with_author_id(df, answer_to_owner, dataset_name):
    """Analyze dataset with author ID mapping."""

    # Map answer_id to OwnerUserId
    df['author_id'] = df['answer_id'].map(answer_to_owner)

    n_with_author = df['author_id'].notna().sum()
    print(f"\n{dataset_name}: {n_with_author}/{len(df)} rows matched to authors")

    if n_with_author == 0:
        print("No author matches - skipping author analysis")
        return None

    # Filter to rows with authors
    df_author = df[df['author_id'].notna()].copy()

    # Basic acceptance stats
    accept_rate = df_author['judgement'].mean()
    print(f"Overall acceptance rate (with authors): {accept_rate:.3f}")

    # Score correlation
    score_corr = df_author[['score', 'judgement']].corr()['score']['judgement']
    print(f"Score ↔ Acceptance correlation: {score_corr:.3f}")

    # Per-author analysis
    author_stats = df_author.groupby('author_id').agg({
        'answer_id': 'count',
        'judgement': 'mean'
    }).rename(columns={'answer_id': 'n_answers', 'judgement': 'accept_rate'})

    print(f"\nTotal authors: {len(author_stats)}")

    # Volume vs acceptance correlation (all authors)
    vol_accept_corr_all = author_stats[['n_answers', 'accept_rate']].corr()['n_answers']['accept_rate']
    print(f"Author volume ↔ acceptance rate correlation (all): {vol_accept_corr_all:.3f}")

    # Filter to authors with >= 5 answers for more stable estimates
    author_stats_filtered = author_stats[author_stats['n_answers'] >= 5]
    print(f"Authors with >= 5 answers: {len(author_stats_filtered)}")

    vol_accept_corr = author_stats_filtered[['n_answers', 'accept_rate']].corr()['n_answers']['accept_rate']
    print(f"Author volume ↔ acceptance rate correlation (>=5 answers): {vol_accept_corr:.3f}")

    # Top vs bottom decile by volume
    if len(author_stats_filtered) >= 20:
        author_stats_filtered['volume_decile'] = pd.qcut(author_stats_filtered['n_answers'], 10, duplicates='drop')
        top_decile = author_stats_filtered[author_stats_filtered['volume_decile'] == author_stats_filtered['volume_decile'].max()]
        bottom_decile = author_stats_filtered[author_stats_filtered['volume_decile'] == author_stats_filtered['volume_decile'].min()]

        print(f"\nTop volume decile acceptance rate: {top_decile['accept_rate'].mean():.3f} (n={len(top_decile)} authors, avg {top_decile['n_answers'].mean():.1f} answers)")
        print(f"Bottom volume decile acceptance rate: {bottom_decile['accept_rate'].mean():.3f} (n={len(bottom_decile)} authors, avg {bottom_decile['n_answers'].mean():.1f} answers)")

        # Top vs bottom quartile
        author_stats_filtered['volume_quartile'] = pd.qcut(author_stats_filtered['n_answers'], 4, duplicates='drop')
        top_quartile = author_stats_filtered[author_stats_filtered['volume_quartile'] == author_stats_filtered['volume_quartile'].max()]
        bottom_quartile = author_stats_filtered[author_stats_filtered['volume_quartile'] == author_stats_filtered['volume_quartile'].min()]

        print(f"Top volume quartile acceptance rate: {top_quartile['accept_rate'].mean():.3f}")
        print(f"Bottom volume quartile acceptance rate: {bottom_quartile['accept_rate'].mean():.3f}")

    # How much does author identity predict acceptance?
    # Add author mean acceptance as a feature
    author_mean_accept = df_author.groupby('author_id')['judgement'].transform('mean')
    df_author['author_mean_accept'] = author_mean_accept

    # Correlation between author mean and actual acceptance
    author_accept_corr = df_author[['author_mean_accept', 'judgement']].corr()['author_mean_accept']['judgement']
    print(f"\nAuthor mean acceptance ↔ actual acceptance correlation: {author_accept_corr:.3f}")

    # Compare to score prediction
    print(f"\nFor comparison - Score predicts acceptance with r={score_corr:.3f}")
    print(f"Author identity predicts acceptance with r={author_accept_corr:.3f}")

    results = {
        'n_rows': len(df_author),
        'n_authors': len(author_stats),
        'n_authors_filtered': len(author_stats_filtered),
        'accept_rate': accept_rate,
        'score_corr': score_corr,
        'vol_accept_corr': vol_accept_corr,
        'author_accept_corr': author_accept_corr,
        'top_decile_accept': top_decile['accept_rate'].mean() if len(top_decile) > 0 else None,
        'bottom_decile_accept': bottom_decile['accept_rate'].mean() if len(bottom_decile) > 0 else None,
    }

    return results

def analyze_crse():
    """Analyze Code Review StackExchange."""

    print("\n" + "="*60)
    print("CODE REVIEW STACKEXCHANGE")
    print("="*60)

    # Load balanced dataset
    csv_path = "/lfs/skampere3/0/alexspan/norm-research/datasets/code-review/crse_balanced_v2/crse_v2_propensity_balanced.csv.gz"
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    # Load Posts.xml mapping
    posts_xml = "/lfs/skampere3/0/alexspan/norm-research/datasets/codereview_se/raw_dump/Posts.xml"
    answer_to_owner = load_posts_xml_mapping(posts_xml)

    return analyze_with_author_id(df, answer_to_owner, "CR.SE")

def analyze_so_python():
    """Analyze StackOverflow Python."""

    print("\n" + "="*60)
    print("STACKOVERFLOW PYTHON")
    print("="*60)

    # Load balanced dataset
    csv_path = "/lfs/skampere3/0/alexspan/norm-research/datasets/stackoverflow_python/balanced/so_python_v2_propensity_balanced.csv.gz"
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    # Load answers parquet - use pyarrow
    answers_path = "/lfs/skampere3/0/alexspan/norm-research/datasets/stackoverflow_python/so_python_answers.parquet"

    try:
        import pyarrow.parquet as pq
        answers = pq.read_table(answers_path).to_pandas()
        print(f"Loaded {len(answers)} answer rows")

        if 'OwnerUserId' in answers.columns:
            # Build mapping - column is called 'Id' not 'answer_id'
            # Filter out NaN OwnerUserIds
            valid_answers = answers[answers['OwnerUserId'].notna()]
            answer_to_owner = dict(zip(valid_answers['Id'], valid_answers['OwnerUserId']))
            print(f"Mapped {len(answer_to_owner)} answers to owners")

            return analyze_with_author_id(df, answer_to_owner, "SO Python")
        else:
            print(f"OwnerUserId not found in {answers_path}")
            print(f"Available columns: {answers.columns.tolist()}")
            return None
    except Exception as e:
        print(f"Error loading answers: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = {}

    # Analyze CR.SE
    try:
        results['crse'] = analyze_crse()
    except Exception as e:
        print(f"CR.SE analysis failed: {e}")
        import traceback
        traceback.print_exc()

    # Analyze SO Python
    try:
        results['so_python'] = analyze_so_python()
    except Exception as e:
        print(f"SO Python analysis failed: {e}")
        import traceback
        traceback.print_exc()

    # Print summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    for dataset, res in results.items():
        if res:
            print(f"\n{dataset.upper()}:")
            print(f"  Rows with authors: {res['n_rows']:,}")
            print(f"  Total authors: {res['n_authors']:,}")
            print(f"  Authors (>=5 answers): {res['n_authors_filtered']:,}")
            print(f"  Acceptance rate: {res['accept_rate']:.3f}")
            print(f"  Score predicts acceptance: r={res['score_corr']:.3f}")
            print(f"  Volume ↔ acceptance: r={res['vol_accept_corr']:.3f}")
            print(f"  Author identity predicts acceptance: r={res['author_accept_corr']:.3f}")

            if res['top_decile_accept'] and res['bottom_decile_accept']:
                print(f"  Top decile acceptance: {res['top_decile_accept']:.3f}")
                print(f"  Bottom decile acceptance: {res['bottom_decile_accept']:.3f}")
