"""4-cell MI ladder on the reconstructed single-file view.

Cells:
  - TF-IDF char_wb 3-5gram + LogisticRegression
  - TF-IDF char_wb 3-5gram + RandomForest
  - Bank scores + LogisticRegression
  - Bank scores + RandomForest

GroupKFold by (owner, repo). Reports AUC mean +/- std (5 folds).

Inputs:
  outputs/v2_analysis/dense_4096tok_single_file_reconstructed.parquet
  outputs/v2_analysis/dense_4096tok_single_file_bank_scores.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
RECON = REPO / "outputs/v2_analysis/dense_4096tok_single_file_reconstructed.parquet"
SCORES = REPO / "outputs/v2_analysis/dense_4096tok_single_file_bank_scores.parquet"

SEED = 42
N_FOLDS = 5


def load_data():
    recon = pd.read_parquet(RECON)
    bank = pd.read_parquet(SCORES)
    df = recon.merge(bank, on="paper_id", how="inner",
                     suffixes=("", "_dup"))
    # drop duplicated dup cols
    for c in list(df.columns):
        if c.endswith("_dup"):
            df.drop(columns=[c], inplace=True)
    # require reconstruction
    df = df[df["file_text"].notna() & (df["file_text"].str.len() > 0)].copy()
    df["group"] = df["owner"].astype(str) + "/" + df["repo"].astype(str)
    df["y"] = df["judgement"].astype(int)
    return df


def run_cell(X, y, groups, model_fn, label):
    cv = GroupKFold(n_splits=N_FOLDS)
    aucs = []
    for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
        m = model_fn()
        m.fit(X[tr], y[tr])
        if hasattr(m, "predict_proba"):
            p = m.predict_proba(X[te])[:, 1]
        else:
            p = m.decision_function(X[te])
        a = roc_auc_score(y[te], p)
        aucs.append(a)
    aucs = np.array(aucs)
    print(f"  {label:<40} AUC={aucs.mean():.3f} +/- {aucs.std():.3f}  "
          f"(folds: {', '.join(f'{a:.3f}' for a in aucs)})")
    return aucs


def main():
    df = load_data()
    print(f"loaded {len(df):,} rows; positives={int(df['y'].sum()):,} "
          f"({df['y'].mean():.1%}); groups={df['group'].nunique():,}")

    y = df["y"].values
    groups = df["group"].values

    # === TF-IDF char_wb 3-5gram ===
    # HashingVectorizer + TfidfTransformer to avoid building a 1M-token vocab
    # in memory; the feature space is fixed at 2**20.
    text = df["file_text"].fillna("").values
    print("\nbuilding tfidf char_wb 3-5gram features (hashing, 2**20)...")
    hv = HashingVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        n_features=2 ** 20, alternate_sign=False, norm=None,
    )
    counts = hv.transform(text)
    tfidf = TfidfTransformer(sublinear_tf=True).fit_transform(counts)
    print(f"  tfidf shape={tfidf.shape} nnz={tfidf.nnz:,}")

    print("\n=== MI ladder (GroupKFold by owner/repo, n=5) ===\n")

    def lr():
        return LogisticRegression(
            penalty="l2", C=1.0, solver="liblinear",
            class_weight="balanced", max_iter=2000, random_state=SEED)

    def rf():
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=SEED)

    print("TF-IDF char_wb 3-5gram:")
    run_cell(tfidf, y, groups, lr, "TFIDF x LR")
    # RF on sparse 2**20 is feasible with subsampling — but tractable enough
    # to just call. We pass csr directly; sklearn RF supports it as of >=1.4.
    run_cell(tfidf, y, groups, rf, "TFIDF x RF")

    # === Bank features ===
    score_cols = [c for c in df.columns if c.endswith("_score")]
    applied_cols = [c for c in df.columns if c.endswith("_applied")]
    feat_cols = score_cols + applied_cols
    Xb = df[feat_cols].values.astype(float)
    # Median-impute NaN scores so LR/RF can handle them.
    imp = SimpleImputer(strategy="median")
    Xb_i = imp.fit_transform(Xb)
    print(f"\nBank features: {len(score_cols)} scores + "
          f"{len(applied_cols)} applied = {Xb.shape[1]} cols")

    def lr_pipe():
        return Pipeline([
            ("sc", StandardScaler(with_mean=False)),
            ("lr", LogisticRegression(
                penalty="l2", C=0.5, solver="lbfgs",
                class_weight="balanced", max_iter=3000, random_state=SEED)),
        ])

    print("\nBank scores + applied flags:")
    run_cell(Xb_i, y, groups, lr_pipe, "BANK x LR")
    run_cell(Xb_i, y, groups, rf, "BANK x RF")


if __name__ == "__main__":
    main()
