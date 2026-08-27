#!/usr/bin/env python3
"""V/A/T ladder for GitHub PRs. Tests whether articulable metrics (A) add
predictive power on top of test-signal features (V), and compares to the
dense ceiling (T). Runs 3 y-definitions.

Usage:
  python3 scripts/pr_vat/run_vat_ladder.py --vat_table outputs/pr_vat_table.parquet
  # optionally: --dense_preds outputs/pr_dense_test_predictions.csv
"""
import argparse, warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
warnings.filterwarnings('ignore')

def get_feature_groups(df):
    """Identify V, A feature columns."""
    v_cols = [c for c in df.columns if c.startswith('v_') or c in
              ['p2f','f2p','smoke_rc','baseline_failed','baseline_passed',
               'baseline_n_tests','post_failed','post_passed','n_fail_genuine',
               'v2_fix','v2_regression']]
    a_cols = [c for c in df.columns if c.endswith('_score') and c not in v_cols]
    return v_cols, a_cols

def auc_cv(X, y, groups=None, n_splits=5, group_kfold=False):
    """Cross-validated AUC via LogisticRegression."""
    if X.shape[1] == 0 or len(np.unique(y)) < 2:
        return float('nan'), 0
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=500, C=0.1, solver='lbfgs'))
    ])
    # impute NaN
    X = np.nan_to_num(X, nan=0.0)
    if group_kfold and groups is not None:
        cv = GroupKFold(n_splits=min(n_splits, len(np.unique(groups)))
                        if len(np.unique(groups)) >= n_splits else 2)
        scores = cross_val_score(pipe, X, y, groups=groups, cv=cv, scoring='roc_auc')
    else:
        cv = StratifiedKFold(n_splits=min(n_splits, min(np.bincount(y))),
                             shuffle=True, random_state=42)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')
    return scores.mean(), scores.std()

def run_ladder(df, v_cols, a_cols, y_col, groups=None, label=""):
    """Run the V/A/V+A/T ladder for one y-definition."""
    print(f"\n{'='*60}")
    print(f"Y = {label} (n={len(df)}, positive={df[y_col].sum()} ({df[y_col].mean()*100:.1f}%))")
    print(f"{'='*60}")

    # filter to applied A metrics (coverage > 5%)
    a_applied = [c for c in a_cols if df[c].notna().sum() > len(df) * 0.05]
    print(f"V features: {len(v_cols)}, A features (≥5% coverage): {len(a_applied)}")

    results = {}

    # V only
    X_v = df[v_cols].values
    auc_v, std_v = auc_cv(X_v, df[y_col].values, groups, group_kfold=bool(groups is not None))
    print(f"  V only:       AUC = {auc_v:.3f} ± {std_v:.3f}")
    results['V'] = (auc_v, std_v)

    # A only
    X_a = df[a_applied].values
    auc_a, std_a = auc_cv(X_a, df[y_col].values, groups, group_kfold=bool(groups is not None))
    print(f"  A only:       AUC = {auc_a:.3f} ± {std_a:.3f}")
    results['A'] = (auc_a, std_a)

    # V + A
    X_va = np.hstack([X_v, X_a])
    auc_va, std_va = auc_cv(X_va, df[y_col].values, groups, group_kfold=bool(groups is not None))
    print(f"  V + A:        AUC = {auc_va:.3f} ± {std_va:.3f}")
    results['V+A'] = (auc_va, std_va)

    # Delta (does A add?)
    delta = auc_va - auc_v
    print(f"  Δ (V+A - V):  {delta:+.3f} {'*** A ADDS' if delta > 0.01 else ('(no gain)' if abs(delta) < 0.01 else '(A hurts)')}")

    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vat_table', required=True, help='Path to V/A/T table parquet')
    ap.add_argument('--dense_preds', default=None, help='Optional: dense model test predictions CSV')
    args = ap.parse_args()

    df = pd.read_parquet(args.vat_table)
    v_cols, a_cols = get_feature_groups(df)
    print(f"Loaded {len(df)} rows, {len(v_cols)} V features, {len(a_cols)} A features")
    print(f"Repos: {df['repo'].nunique()}")

    # Prepare y variables
    # y1: accept/reject (pooled)
    df['y_accept_reject'] = (df['judgement'] == 'rejected').astype(int)

    # y3: P2F (craft test — does A predict regression-introduction?)
    df['y_p2f'] = df['verdict'].isin(['regression', 'new_failing']).astype(int)

    # ===== Y1: Accept/Reject (pooled) =====
    clean = df[df['judgement'].isin(['accepted', 'rejected'])].copy()
    run_ladder(clean, v_cols, a_cols, 'y_accept_reject', label="Accept/Reject (pooled)")

    # ===== Y2: Accept/Reject (within-repo, GroupKFold) =====
    run_ladder(clean, v_cols, a_cols, 'y_accept_reject',
               groups=clean['repo'].values, label="Accept/Reject (GroupKFold by repo)")

    # ===== Y3: P2F as Y (craft test) =====
    signal = df[df['verdict'].isin(['regression','new_failing','fix','new_passing',
                                    'no_change_all_pass','no_change_still_broken'])].copy()
    run_ladder(signal, v_cols, a_cols, 'y_p2f', label="P2F (does A predict regression?)")

    # ===== Dense ceiling (if provided) =====
    if args.dense_preds and pd.io.common.os.path.exists(args.dense_preds):
        preds = pd.read_csv(args.dense_preds)
        print(f"\n{'='*60}")
        print(f"DENSE CEILING")
        print(f"{'='*60}")
        print(f"Dense test AUC: {preds['test_auc'].iloc[-1]:.3f}" if 'test_auc' in preds.columns else
              f"(check dense model log for test AUC)")

    print("\n" + "="*60)
    print("LADDER SUMMARY")
    print("="*60)
    print("Compare V-only vs V+A AUCs across y-definitions.")
    print("If V+A > V: A (articulable metrics) add signal on top of V (test signal).")
    print("If T > V+A: there's a taste residual the bank doesn't capture.")

if __name__ == "__main__":
    main()
