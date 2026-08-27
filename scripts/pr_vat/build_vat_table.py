#!/usr/bin/env python3
"""Build the unified V/A/y table for the PR V/A/T ladder.
Joins consolidated verdicts (V features + y) with A metrics from the bank.

Usage:
  python3 scripts/pr_vat/build_vat_table.py
"""
import pandas as pd
import numpy as np

VAT_CSV = "outputs/consolidated_verdicts_ALL_final.csv"
A_METRICS = "outputs/pr_a_metrics.parquet"
OUTPUT = "outputs/pr_vat_table.parquet"

def main():
    # Load V features (verdicts)
    v = pd.read_csv(VAT_CSV)
    print(f"V (verdicts): {len(v)} rows, {v['repo'].nunique()} repos")

    # Build V feature columns
    v['v_p2f'] = v['verdict'].isin(['regression','new_failing']).astype(int)
    v['v_f2p'] = v['verdict'].isin(['fix','new_passing']).astype(int)
    v['v_has_signal'] = (v['v_p2f'] | v['v_f2p']).astype(int)
    v['v_smoke_rc'] = pd.to_numeric(v.get('smoke_rc', 0), errors='coerce').fillna(-1)
    v['v_baseline_failed'] = pd.to_numeric(v.get('baseline_failed', 0), errors='coerce').fillna(0)
    v['v_baseline_passed'] = pd.to_numeric(v.get('baseline_passed', 0), errors='coerce').fillna(0)
    v['v_post_failed'] = pd.to_numeric(v.get('post_failed', 0), errors='coerce').fillna(0)
    v['v_post_passed'] = pd.to_numeric(v.get('post_passed', 0), errors='coerce').fillna(0)

    # Normalize join key
    v['join_key'] = v['repo'] + '/' + v['pr_number'].astype(str).str.replace('.0','').str.strip()

    # Load A features (metrics)
    a = pd.read_parquet(A_METRICS)
    print(f"A (metrics): {len(a)} rows")
    a['join_key'] = a['repo'] + '/' + a['pr_number'].astype(str).str.strip()

    # Join
    merged = v.merge(a, on='join_key', how='inner', suffixes=('','_a'))
    print(f"Joined V+A: {len(merged)} rows (overlap: {len(merged)/len(v)*100:.1f}% of V)")

    # Deduplicate (keep first per join_key)
    merged = merged.drop_duplicates(subset='join_key', keep='first')

    # Clean up
    merged['judgement'] = merged['judgement'].fillna('unknown')
    merged.to_parquet(OUTPUT)
    print(f"\nV/A/T table: {len(merged)} rows, {merged.shape[1]} columns → {OUTPUT}")
    print(f"  repos: {merged['repo'].nunique()}")
    print(f"  clean merge-status: {merged['judgement'].isin(['accepted','rejected']).sum()}")

    # Quick A coverage check
    a_cols = [c for c in merged.columns if c.endswith('_score') and not c.startswith('v_')]
    coverage = {c: merged[c].notna().sum() for c in a_cols}
    top = sorted(coverage.items(), key=lambda x: -x[1])[:5]
    print(f"\nTop A metrics by coverage:")
    for mid, n in top:
        print(f"  {mid}: {n} ({n/len(merged)*100:.1f}%)")

if __name__ == "__main__":
    main()
