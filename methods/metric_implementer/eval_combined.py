"""Combined feature-matrix evaluation on local fixtures.

For each fixture × each metric, compute (score, applied). Train RF on the
matrix vs the merged/rejected label, report AUC. n=50 is too small to be
statistically meaningful — this is a directional sanity check that the
architecture composes. The real test is on sk3's 4978-row dataset.
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import load_all

warnings.filterwarnings("ignore")

FIXTURES = Path(__file__).parent / "fixtures" / "sample_prs.json"


def main():
    metrics = load_all()
    fixtures = json.loads(FIXTURES.read_text())
    print(f"loaded {len(metrics)} metrics over {len(fixtures)} fixtures")

    # Build (N × 2M) matrix: per metric (score, applied)
    rows = []
    for fx in fixtures:
        row = {}
        for m in metrics:
            try:
                a = m.applies(fx["text"])
            except Exception:
                a = False
            applied = int(bool(a))
            score = float("nan")
            if applied:
                try:
                    s = m.score(fx["text"])
                    if s is not None and not (isinstance(s, float)
                                              and math.isnan(s)):
                        score = float(s)
                except Exception:
                    pass
            row[f"{m.ASPECT_ID}_score"] = score
            row[f"{m.ASPECT_ID}_applied"] = applied
        row["y"] = int(fx["label"])
        rows.append(row)

    df = pd.DataFrame(rows)
    X = df.drop(columns=["y"]).values
    y = df["y"].values
    feat_cols = [c for c in df.columns if c != "y"]
    n_score = sum(1 for c in feat_cols if c.endswith("_score"))
    n_applied = sum(1 for c in feat_cols if c.endswith("_applied"))
    print(f"feature matrix: {X.shape} ({n_score} score + {n_applied} applied)")
    print(f"label balance: y=1 count = {y.sum()}/{len(y)}")

    # Median-impute NaN scores (an abstained metric → median over applied)
    imp = SimpleImputer(strategy="median")
    Xi = imp.fit_transform(X)

    # 5-fold stratified CV (n=50, so each fold has 10)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=42)
    pipe_rf = Pipeline([("imp", SimpleImputer(strategy="median")),
                        ("rf", rf)])
    pipe_lr = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler(with_mean=False)),
        ("lr", LogisticRegression(penalty="l2", C=0.5,
                                  class_weight="balanced",
                                  max_iter=3000, solver="lbfgs"))])

    p_rf = cross_val_predict(pipe_rf, X, y, cv=skf, method="predict_proba")[:, 1]
    p_lr = cross_val_predict(pipe_lr, X, y, cv=skf, method="predict_proba")[:, 1]

    print()
    print("=" * 60)
    print(f"COMBINED FEATURE MATRIX (n={len(y)}, "
          f"{X.shape[1]} features) — 5-fold CV")
    print("=" * 60)
    print(f"  RF  AUC = {roc_auc_score(y, p_rf):.3f}")
    print(f"  LR  AUC = {roc_auc_score(y, p_lr):.3f}")
    print()
    # Score-only matrix (no applied flags) for comparison
    score_cols = [i for i, c in enumerate(feat_cols) if c.endswith("_score")]
    p_rf_s = cross_val_predict(pipe_rf, X[:, score_cols], y, cv=skf,
                               method="predict_proba")[:, 1]
    print(f"  RF (scores only, {len(score_cols)} feats) AUC = "
          f"{roc_auc_score(y, p_rf_s):.3f}")

    # Per-metric univariate AUC (signed direction)
    print("\nTop 12 metrics by |AUC - 0.5| (univariate, fixture-level):")
    aucs = []
    for c in feat_cols:
        col = df[c].values
        mask = ~np.isnan(col)
        if mask.sum() < 5 or len(set(y[mask])) < 2:
            continue
        try:
            a = roc_auc_score(y[mask], col[mask])
            aucs.append((c, a, mask.sum()))
        except Exception:
            pass
    aucs.sort(key=lambda x: -abs(x[1] - 0.5))
    for c, a, n in aucs[:12]:
        print(f"  {c:<25} AUC={a:.3f}  n={n}")


if __name__ == "__main__":
    main()
