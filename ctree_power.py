#!/usr/bin/env python3
"""
Recompute V AUCs with and without identity for SE datasets.

This script:
1. Loads balanced CSV files for CR.SE and SO Python
2. Joins with source data to get OwnerUserId (identity)
3. Computes V-WITH-identity AUC
4. Computes V-WITHOUT-identity AUC (only artifact-intrinsic)
5. Computes V-without-identity + A combined AUC
6. Reports the deltas

Author identity is NOT artifact-intrinsic - it's a reputation-bias confound.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import sys
from pathlib import Path

# sk3 paths - update these to actual paths
SK3_BASE = "/lfs/skampere3/0/alexspan/norm-research"

# Datasets
DATASETS = {
    "crse": {
        "balanced_csv": f"{SK3_BASE}/datasets/code-review/crse_balanced_v2/crse_v2_propensity_balanced.csv.gz",
        "posts_xml": f"{SK3_BASE}/datasets/codereview_se/raw_dump/Posts.xml",
        "input_parquet": f"{SK3_BASE}/outputs/v2_analysis/se_ladder/crse_input.parquet",
        "shards_dir": f"{SK3_BASE}/outputs/v2_analysis/se_ladder/shards/crse",
        "n_metrics": 166,
    },
    "so_python": {
        "balanced_csv": f"{SK3_BASE}/datasets/stackoverflow_python/balanced/so_python_v2_propensity_balanced.csv.gz",
        "answers_parquet": f"{SK3_BASE}/datasets/stackoverflow_python/so_python_answers.parquet",
        "input_parquet": f"{SK3_BASE}/outputs/v2_analysis/se_ladder/so_python_input.parquet",
        "shards_dir": f"{SK3_BASE}/outputs/v2_analysis/se_ladder/shards/so_python",
        "n_metrics": 128,
    },
}

# V features available in balanced CSV
V_FEATURES = [
    "answer_position",
    "n_answers_on_question",
    "answer_age_gap_days",
    "answer_year",
    "score",
]

# Identity feature
IDENTITY_FEATURE = "OwnerUserId"


def load_crse_data():
    """Load CR.SE data and join OwnerUserId from Posts.xml."""
    print("Loading CR.SE dataset...")

    # Load balanced CSV
    balanced = pd.read_csv(DATASETS["crse"]["balanced_csv"])
    print(f"  Balanced: {len(balanced)} rows")

    # Load Posts.xml to get OwnerUserId
    # Posts.xml is huge - we'll load only Answer posts and filter by answer_id
    print("  Loading Posts.xml for OwnerUserId join...")
    # For now, skip this - we'll need to parse the XML
    # Placeholder: balanced["OwnerUserId"] = np.random.randint(0, 1000, len(balanced))

    return balanced


def load_so_python_data():
    """Load SO Python data and join OwnerUserId from answers parquet."""
    print("Loading SO Python dataset...")

    # Load balanced CSV
    balanced = pd.read_csv(DATASETS["so_python"]["balanced_csv"])
    print(f"  Balanced: {len(balanced)} rows")

    # Load answers parquet to get OwnerUserId
    print("  Loading answers parquet for OwnerUserId join...")
    answers = pd.read_parquet(DATASETS["so_python"]["answers_parquet"])

    # Join on answer_id
    balanced = balanced.merge(
        answers[["answer_id", "OwnerUserId"]], on="answer_id", how="left"
    )
    print(f"  After join: {len(balanced)} rows, {balanced['OwnerUserId'].notna().sum()} with OwnerUserId")

    return balanced


def compute_v_aucs(df, dataset_name):
    """Compute V AUCs with and without identity."""
    print(f"\n{'='*60}")
    print(f"Computing V AUCs for {dataset_name}")
    print(f"{'='*60}")

    # Prepare features
    X = df.copy()

    # Check if OwnerUserId is available
    has_identity = IDENTITY_FEATURE in X.columns and X[IDENTITY_FEATURE].notna().any()

    # V features (without identity)
    v_features = [f for f in V_FEATURES if f in X.columns]

    # Check which V features are actually available
    available_v_features = [f for f in v_features if f in X.columns and X[f].notna().any()]
    print(f"Available V features: {available_v_features}")

    # Label
    y = X["judgement"].values

    results = {}

    # 1. V-WITH-identity (if identity available)
    if has_identity:
        print("\n1. V-WITH-identity")
        v_with_id_features = available_v_features + [IDENTITY_FEATURE]

        # One-hot encode OwnerUserId
        X_with_id = pd.get_dummies(X[v_with_id_features], columns=[IDENTITY_FEATURE], dummy_na=True)

        # Drop NaN rows
        mask = X_with_id.notna().all(axis=1)
        X_with_id_clean = X_with_id[mask]
        y_clean = y[mask]

        print(f"  Features: {X_with_id_clean.shape[1]}")
        print(f"  Samples: {len(y_clean)}")

        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(X_with_id_clean, y_clean)

        if len(y_clean) > 0:
            auc = roc_auc_score(y_clean, model.predict_proba(X_with_id_clean)[:, 1])
            results["v_with_identity"] = auc
            print(f"  AUC: {auc:.4f}")
        else:
            print("  WARNING: No valid samples after dropping NaNs")
            results["v_with_identity"] = None
    else:
        print("\n1. V-WITH-identity - SKIPPED (identity not available)")
        results["v_with_identity"] = None

    # 2. V-WITHOUT-identity (artifact-intrinsic only)
    print("\n2. V-WITHOUT-identity")
    v_without_id_features = available_v_features

    X_without_id = X[v_without_id_features]

    # Drop NaN rows
    mask = X_without_id.notna().all(axis=1)
    X_without_id_clean = X_without_id[mask]
    y_clean = y[mask]

    print(f"  Features: {X_without_id_clean.shape[1]}")
    print(f"  Samples: {len(y_clean)}")

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_without_id_clean, y_clean)

    if len(y_clean) > 0:
        auc = roc_auc_score(y_clean, model.predict_proba(X_without_id_clean)[:, 1])
        results["v_without_identity"] = auc
        print(f"  AUC: {auc:.4f}")
    else:
        print("  WARNING: No valid samples after dropping NaNs")
        results["v_without_identity"] = None

    # 3. V-without-identity + A (bank metrics)
    print("\n3. V-without-identity + A (bank metrics)")
    print("  NOTE: A features not yet implemented - need to load from shards")
    results["v_without_id_plus_a"] = None

    # 4. Delta
    if results["v_with_identity"] is not None and results["v_without_identity"] is not None:
        delta = results["v_with_identity"] - results["v_without_identity"]
        results["delta"] = delta
        print(f"\n4. Delta (with - without identity): {delta:.4f}")
    else:
        results["delta"] = None

    return results


def main():
    """Main entry point."""
    print("REMOVE-IDENTITY V recompute on SE data")
    print("=" * 60)

    all_results = {}

    # CR.SE
    try:
        crse_df = load_crse_data()
        all_results["crse"] = compute_v_aucs(crse_df, "CR.SE")
    except Exception as e:
        print(f"\nERROR loading CR.SE: {e}")
        import traceback
        traceback.print_exc()
        all_results["crse"] = None

    # SO Python
    try:
        so_df = load_so_python_data()
        all_results["so_python"] = compute_v_aucs(so_df, "SO Python")
    except Exception as e:
        print(f"\nERROR loading SO Python: {e}")
        import traceback
        traceback.print_exc()
        all_results["so_python"] = None

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for dataset, results in all_results.items():
        if results is None:
            print(f"\n{dataset.upper()}: ERROR")
            continue

        print(f"\n{dataset.upper()}:")
        if results["v_with_identity"] is not None:
            print(f"  V-WITH-identity AUC:    {results['v_with_identity']:.4f}")
        if results["v_without_identity"] is not None:
            print(f"  V-WITHOUT-identity AUC: {results['v_without_identity']:.4f}")
        if results["delta"] is not None:
            print(f"  Delta (with - without):  {results['delta']:.4f}")

    print("\n" + "=" * 60)
    print("NOTE: V+A (bank metrics) not yet implemented")
    print("=" * 60)


if __name__ == "__main__":
    main()
