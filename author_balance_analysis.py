#!/usr/bin/env python3
"""
Author-balance analysis on StackExchange datasets.
Tests whether acceptance signal is author-identity-driven.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def load_math_se_local():
    """Load Math SE v3.3 dataset locally."""
    path = "/Users/spangher/Projects/stanford-research/norm-research/datasets/math/stackexchange/math_se_v3_3_propensity_balanced.csv.gz"
    df = pd.read_csv(path)
    print(f"Math SE: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    return df

def analyze_math_se(df):
    """Analyze Math SE for author effects (no author field available)."""
    print("\n=== Math SE Analysis (no author field) ===")

    # Basic acceptance stats
    accept_rate = df['judgement'].mean()
    print(f"Overall acceptance rate: {accept_rate:.3f}")

    # Check if score correlates with acceptance
    corr = df[['score', 'judgement']].corr()['score']['judgement']
    print(f"Score ↔ Acceptance correlation: {corr:.3f}")

    # Acceptance by score deciles
    df['score_decile'] = pd.qcut(df['score'], 10, duplicates='drop')
    score_accept = df.groupby('score_decile', observed=True)['judgement'].agg(['mean', 'count'])
    print("\nAcceptance rate by score decile:")
    print(score_accept)

    # Check if position correlates with acceptance
    pos_corr = df[['answer_position', 'judgement']].corr()['answer_position']['judgement']
    print(f"\nPosition ↔ Acceptance correlation: {pos_corr:.3f}")

    return {
        'accept_rate': accept_rate,
        'score_corr': corr,
        'position_corr': pos_corr
    }

def analyze_sk3_dataset(dataset_name, csv_path, posts_xml_path=None, answers_parquet_path=None):
    """Analyze CR.SE or SO Python on sk3."""
    print(f"\n=== {dataset_name} ===")

    # Load balanced dataset
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Basic stats
    accept_rate = df['judgement'].mean()
    print(f"Overall acceptance rate: {accept_rate:.3f}")

    # Score correlation
    score_corr = df[['score', 'judgement']].corr()['score']['judgement']
    print(f"Score ↔ Acceptance correlation: {score_corr:.3f}")

    # Position correlation
    pos_corr = df[['answer_position', 'judgement']].corr()['answer_position']['judgement']
    print(f"Position ↔ Acceptance correlation: {pos_corr:.3f}")

    # Answer count correlation
    count_corr = df[['n_answers_on_question', 'judgement']].corr()['n_answers_on_question']['judgement']
    print(f"Num answers on question ↔ Acceptance correlation: {count_corr:.3f}")

    results = {
        'accept_rate': accept_rate,
        'score_corr': score_corr,
        'position_corr': pos_corr,
        'count_corr': count_corr
    }

    # Try to join author data
    if answers_parquet_path:
        print(f"\nAttempting to join author data from {answers_parquet_path}")
        try:
            # For SO Python, try using pyarrow directly
            import pyarrow.parquet as pq
            answers = pq.read_table(answers_parquet_path).to_pandas()
            print(f"Loaded {len(answers)} answer rows")
            print(f"Answer columns: {answers.columns.tolist()}")

            if 'OwnerUserId' in answers.columns:
                df_with_author = df.merge(answers[['answer_id', 'OwnerUserId']], on='answer_id', how='left')
                print(f"Merged author data: {df_with_author['OwnerUserId'].notna().sum()} non-null authors")

                # Per-author analysis
                author_stats = df_with_author.groupby('OwnerUserId').agg({
                    'answer_id': 'count',
                    'judgement': 'mean'
                }).rename(columns={'answer_id': 'n_answers', 'judgement': 'accept_rate'})

                # Filter to authors with >= 5 answers
                author_stats = author_stats[author_stats['n_answers'] >= 5]
                print(f"\nAuthors with >= 5 answers: {len(author_stats)}")

                # Volume vs acceptance correlation
                vol_accept_corr = author_stats[['n_answers', 'accept_rate']].corr()['n_answers']['accept_rate']
                print(f"Author volume ↔ acceptance rate correlation: {vol_accept_corr:.3f}")

                # Top vs bottom decile by volume
                author_stats['volume_decile'] = pd.qcut(author_stats['n_answers'], 10, duplicates='drop')
                top_decile = author_stats[author_stats['volume_decile'] == author_stats['volume_decile'].max()]
                bottom_decile = author_stats[author_stats['volume_decile'] == author_stats['volume_decile'].min()]

                print(f"\nTop volume decile acceptance rate: {top_decile['accept_rate'].mean():.3f} (n={len(top_decile)} authors)")
                print(f"Bottom volume decile acceptance rate: {bottom_decile['accept_rate'].mean():.3f} (n={len(bottom_decile)} authors)")

                results.update({
                    'n_authors': len(author_stats),
                    'vol_accept_corr': vol_accept_corr,
                    'top_decile_accept': top_decile['accept_rate'].mean(),
                    'bottom_decile_accept': bottom_decile['accept_rate'].mean()
                })

                # How much does author identity predict acceptance?
                # Add author mean acceptance as a feature
                author_mean_accept = df_with_author.groupby('OwnerUserId')['judgement'].transform('mean')
                df_with_author['author_mean_accept'] = author_mean_accept

                # Correlation between author mean and actual acceptance
                author_accept_corr = df_with_author[['author_mean_accept', 'judgement']].corr()['author_mean_accept']['judgement']
                print(f"\nAuthor mean acceptance ↔ actual acceptance correlation: {author_accept_corr:.3f}")

                results['author_accept_corr'] = author_accept_corr

        except Exception as e:
            print(f"Could not load author data: {e}")

    return results

if __name__ == "__main__":
    # Analyze Math SE locally
    math_df = load_math_se_local()
    math_results = analyze_math_se(math_df)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nMath SE (no author field):")
    print(f"  Acceptance rate: {math_results['accept_rate']:.3f}")
    print(f"  Score ↔ acceptance: {math_results['score_corr']:.3f}")
    print(f"  Position ↔ acceptance: {math_results['position_corr']:.3f}")
