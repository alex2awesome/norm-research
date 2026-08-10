"""Follow-up to cr_run_metric_implementer.py:
  (a) Univariate filter MI metrics by full-dataset |AUC-0.5|, refit
  (b) L1 LR on the full MI matrix
Both compared against T1+T2+T3 baseline (0.627).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
SEED = 42


def fit_rf(Xc, y, label):
    Xtr, Xte, ytr, yte = train_test_split(
        Xc, y, test_size=0.20, stratify=y, random_state=SEED)
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("rf", RandomForestClassifier(
                         n_estimators=500, min_samples_leaf=2,
                         class_weight="balanced", n_jobs=-1,
                         random_state=SEED))])
    pipe.fit(Xtr, ytr)
    auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])
    print(f"  RF  {label:<55} feats={Xc.shape[1]:4d}  AUC={auc:.3f}")
    return pipe


def fit_lr_l1(Xc, y, label, C=0.1):
    Xtr, Xte, ytr, yte = train_test_split(
        Xc, y, test_size=0.20, stratify=y, random_state=SEED)
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler(with_mean=False)),
        ("lr", LogisticRegression(
            penalty="l1", C=C, class_weight="balanced",
            max_iter=3000, solver="saga"))])
    pipe.fit(Xtr, ytr)
    auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])
    coef = pipe.named_steps["lr"].coef_[0]
    n_nz = int((coef != 0).sum())
    print(f"  L1-LR {label:<53} feats={Xc.shape[1]:4d} nz={n_nz:3d} AUC={auc:.3f}")
    return pipe


def main():
    t12 = pd.read_parquet(REPO / "outputs/v2_analysis/cr_tier12_features.parquet")
    t34 = pd.read_parquet(REPO / "outputs/v2_analysis/cr_tier34_features.parquet"
                          ).drop(columns=["y"], errors="ignore")
    mi = pd.read_parquet(REPO / "outputs/v2_analysis/cr_metric_implementer_scores.parquet"
                         ).drop(columns=["y"], errors="ignore")

    base = t12.merge(t34, on="datapoint_id")
    combo = base.merge(mi, on="datapoint_id", how="inner")
    y = combo["y"].astype(int).values
    base_cols = [c for c in t12.columns if c not in ("datapoint_id", "y")]
    t3_cols = [c for c in t34.columns if c.startswith("tier3_")]
    mi_score = [c for c in mi.columns if c.endswith("_score")]
    mi_applied = [c for c in mi.columns if c.endswith("_applied")]
    print(f"rows={len(combo)}, base+t3={len(base_cols + t3_cols)}, "
          f"mi_score={len(mi_score)}, mi_applied={len(mi_applied)}")

    # === (a) Univariate filter on MI scores ===
    print("\n=== Univariate filter on MI score columns (full-dataset AUC) ===")
    keep_strong = []
    keep_weak = []
    aucs = []
    for c in mi_score:
        col = combo[c].values
        mask = ~np.isnan(col)
        if mask.sum() < 50:
            continue
        try:
            a = roc_auc_score(y[mask], col[mask])
            aucs.append((c, a, mask.sum()))
            if abs(a - 0.5) >= 0.05:
                keep_strong.append(c)
            if abs(a - 0.5) >= 0.02:
                keep_weak.append(c)
        except Exception:
            pass
    aucs.sort(key=lambda x: -abs(x[1] - 0.5))
    print(f"\nTop 20 MI metrics by |AUC-0.5|:")
    for c, a, n in aucs[:20]:
        print(f"  {c:<25} AUC={a:.3f}  n_scored={n}")
    print(f"\n|AUC-0.5|>=0.05 → {len(keep_strong)} score features")
    print(f"|AUC-0.5|>=0.02 → {len(keep_weak)} score features")

    print("\n=== Ladder comparisons ===")
    btt = base_cols + t3_cols
    fit_rf(combo[btt].values, y, "T1+T2+T3 (baseline)")
    fit_rf(combo[btt + keep_strong].values, y,
           f"T1+T2+T3 + MI univariate>=0.05 ({len(keep_strong)})")
    fit_rf(combo[btt + keep_weak].values, y,
           f"T1+T2+T3 + MI univariate>=0.02 ({len(keep_weak)})")
    fit_rf(combo[btt + keep_strong + [c.replace("_score", "_applied")
                                       for c in keep_strong
                                       if c.replace("_score", "_applied")
                                       in mi_applied]].values,
           y, f"T1+T2+T3 + MI>=0.05 + matching applied flags")

    # === (b) L1 LR sparse selection ===
    print("\n=== L1 LR (saga, C=0.1) ===")
    fit_lr_l1(combo[btt].values, y, "T1+T2+T3 baseline")
    fit_lr_l1(combo[btt + mi_score].values, y, "T1+T2+T3 + all MI scores")
    fit_lr_l1(combo[btt + mi_score + mi_applied].values, y,
              "T1+T2+T3 + MI scores + applied")
    fit_lr_l1(combo[mi_score + mi_applied].values, y, "MI scores + applied ALONE")

    # === L1 with C sweep on full matrix ===
    print("\n=== L1 LR C-sweep on T1+T2+T3 + MI all ===")
    X_full = combo[btt + mi_score + mi_applied].values
    for C in [0.01, 0.03, 0.1, 0.3, 1.0]:
        fit_lr_l1(X_full, y, f"C={C:.2f}", C=C)


if __name__ == "__main__":
    main()
