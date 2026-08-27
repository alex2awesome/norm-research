#!/usr/bin/env python3
"""
Recompute V AUCs with and without identity for SE datasets.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import sys

# V features available in balanced CSV
# Note: "score" is a SOCIAL feature (votes from community), not artifact-intrinsic
V_SOCIAL_FEATURES = [
    "answer_position",
    "n_answers_on_question",
    "answer_age_gap_days",
    "answer_year",
    "score",  # Social feature
]

# Artifact-intrinsic V features (no social signals)
V_INTRINSIC_FEATURES = [
    "answer_position",
    "n_answers_on_question",
    "answer_age_gap_days",
    "answer_year",
]

IDENTITY_FEATURE = "OwnerUserId"


def load_so_python_data():
    """Load SO Python data."""
    print("Loading SO Python dataset...")

    balanced = pd.read_csv("/lfs/skampere3/0/alexspan/norm-research/datasets/stackoverflow_python/balanced/so_python_v2_propensity_balanced.csv.gz")
    print("  Balanced: {} rows".format(len(balanced)))

    # Load answers parquet to get OwnerUserId
    answers = pd.read_parquet("/lfs/skampere3/0/alexspan/norm-research/datasets/stackoverflow_python/so_python_answers.parquet")

    # Join on answer_id (column is 'Id' in answers, 'answer_id' in balanced)
    answers_renamed = answers.rename(columns={"Id": "answer_id"})
    balanced = balanced.merge(
        answers_renamed[["answer_id", "OwnerUserId"]], on="answer_id", how="left"
    )
    print("  After join: {} rows, {} with OwnerUserId".format(
        len(balanced), balanced["OwnerUserId"].notna().sum()))

    return balanced


def compute_v_aucs(df, dataset_name):
    """Compute V AUCs with and without identity."""
    print("\n" + "="*60)
    print("Computing V AUCs for {}".format(dataset_name))
    print("="*60)

    # Prepare features
    X = df.copy()

    # Check which V features are actually available
    available_v_social = [f for f in V_SOCIAL_FEATURES if f in X.columns]
    available_v_intrinsic = [f for f in V_INTRINSIC_FEATURES if f in X.columns]
    print("Available V-social features: {}".format(available_v_social))
    print("Available V-intrinsic features: {}".format(available_v_intrinsic))

    # Label
    y = X["judgement"].values

    results = {}

    # 1. V-social WITH identity
    print("\n1. V-social WITH identity (current baseline)")

    # One-hot encode OwnerUserId (top 1000 users to reduce dimensionality)
    top_users = X[IDENTITY_FEATURE].value_counts().head(1000).index
    X_identity = X[IDENTITY_FEATURE].apply(lambda x: x if x in top_users else np.nan)

    X_social_id = X[available_v_social].copy()
    X_social_id["OwnerUserId_encoded"] = X_identity.astype(str)

    # Drop NaN rows (including those with rare users)
    mask = X_social_id.notna().all(axis=1)
    X_social_id_clean = X_social_id[mask]
    y_clean = y[mask]

    # One-hot encode
    X_social_id_encoded = pd.get_dummies(X_social_id_clean, columns=["OwnerUserId_encoded"], dummy_na=True)

    print("  Features: {}".format(X_social_id_encoded.shape[1]))
    print("  Samples: {}".format(len(y_clean)))

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_social_id_encoded, y_clean)

    auc_social_id = roc_auc_score(y_clean, model.predict_proba(X_social_id_encoded)[:, 1])
    results["v_social_with_identity"] = auc_social_id
    print("  AUC: {:.4f}".format(auc_social_id))

    # 2. V-social WITHOUT identity (removes owner, keeps score)
    print("\n2. V-social WITHOUT identity (removes owner)")
    X_social_no_id = X[available_v_social].copy()

    # Drop NaN rows
    mask = X_social_no_id.notna().all(axis=1)
    X_social_no_id_clean = X_social_no_id[mask]
    y_clean = y[mask]

    print("  Features: {}".format(X_social_no_id_clean.shape[1]))
    print("  Samples: {}".format(len(y_clean)))

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_social_no_id_clean, y_clean)

    auc_social_no_id = roc_auc_score(y_clean, model.predict_proba(X_social_no_id_clean)[:, 1])
    results["v_social_without_identity"] = auc_social_no_id
    print("  AUC: {:.4f}".format(auc_social_no_id))

    # 3. V-intrinsic WITHOUT identity (removes owner AND score)
    print("\n3. V-intrinsic WITHOUT identity (removes owner + score)")
    X_intrinsic = X[available_v_intrinsic].copy()

    # Drop NaN rows
    mask = X_intrinsic.notna().all(axis=1)
    X_intrinsic_clean = X_intrinsic[mask]
    y_clean = y[mask]

    print("  Features: {}".format(X_intrinsic_clean.shape[1]))
    print("  Samples: {}".format(len(y_clean)))

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_intrinsic_clean, y_clean)

    auc_intrinsic = roc_auc_score(y_clean, model.predict_proba(X_intrinsic_clean)[:, 1])
    results["v_intrinsic_no_identity"] = auc_intrinsic
    print("  AUC: {:.4f}".format(auc_intrinsic))

    # 4. Deltas
    delta_identity = auc_social_id - auc_social_no_id
    delta_score = auc_social_no_id - auc_intrinsic
    delta_both = auc_social_id - auc_intrinsic

    results["delta_identity"] = delta_identity
    results["delta_score"] = delta_score
    results["delta_both"] = delta_both

    print("\n4. Deltas:")
    print("  Identity contribution: {:.4f}".format(delta_identity))
    print("  Score contribution: {:.4f}".format(delta_score))
    print("  Both (identity + score): {:.4f}".format(delta_both))

    return results


def main():
    """Main entry point."""
    print("REMOVE-IDENTITY V recompute on SE data")
    print("=" * 60)

    all_results = {}

    # SO Python
    try:
        so_df = load_so_python_data()
        all_results["so_python"] = compute_v_aucs(so_df, "SO Python")
    except Exception as e:
        print("\nERROR loading SO Python: {}".format(e))
        import traceback
        traceback.print_exc()
        all_results["so_python"] = None

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for dataset, results in all_results.items():
        if results is None:
            print("\n{}: ERROR".format(dataset.upper()))
            continue

        print("\n{}:".format(dataset.upper()))
        print("  V-social WITH identity:      {:.4f}".format(results['v_social_with_identity']))
        print("  V-social WITHOUT identity:   {:.4f}".format(results['v_social_without_identity']))
        print("  V-intrinsic NO identity:      {:.4f}".format(results['v_intrinsic_no_identity']))
        print("  Identity contribution:       {:.4f}".format(results['delta_identity']))
        print("  Score contribution:           {:.4f}".format(results['delta_score']))
        print("  Both (identity + score):      {:.4f}".format(results['delta_both']))

    print("\n" + "=" * 60)
    print("NOTE: CR.SE skipped (requires XML parsing)")
    print("V-social includes: position, n_answers, age_gap, year, score")
    print("V-intrinsic includes: position, n_answers, age_gap, year (NO score)")
    print("=" * 60)


if __name__ == "__main__":
    main()
