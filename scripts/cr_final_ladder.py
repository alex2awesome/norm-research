"""Final code_review verifiability ladder.

Merges:
  - Tier 1+2 (5 metadata + 18 diff-parsing features)            -> 34 feats
  - Tier 3   (lizard CCN / function length / NLOC)              -> 6 feats
  - Tier 4a  (test code patterns added in diff)                 -> 3 feats
  - Tier 4b  (reviewer test-discussion patterns)                -> 3 feats
  - Codegen  (~1182 per-aspect Python score programs)           -> ~1182 feats

Trains RF on each cumulative cut + LR for the codegen-bearing tiers
(since RF can swamp deterministic features with high-D codegen).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
SEED = 42
OUT_DIR = REPO / "outputs/v2_analysis"
T12 = OUT_DIR / "cr_tier12_features.parquet"
T34 = OUT_DIR / "cr_tier34_features.parquet"
CG = OUT_DIR / "cr_codegen_scores.parquet"


def fit(X, y, label):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    rf = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=SEED)
    rf.fit(Xtr, ytr)
    p = rf.predict_proba(Xte)[:, 1]
    auc_rf = roc_auc_score(yte, p)

    sc = StandardScaler(with_mean=False)
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)
    lr = LogisticRegression(penalty="l2", C=0.5, class_weight="balanced",
                            max_iter=2000, solver="lbfgs")
    lr.fit(Xtr_s, ytr)
    p2 = lr.predict_proba(Xte_s)[:, 1]
    auc_lr = roc_auc_score(yte, p2)
    print(f"  {label:<35} feats={X.shape[1]:5d}  RF={auc_rf:.3f}  LR={auc_lr:.3f}")
    return rf, auc_rf, auc_lr


def main():
    t12 = pd.read_parquet(T12)
    t34 = pd.read_parquet(T34).drop(columns=["y"], errors="ignore")
    cg = pd.read_parquet(CG).drop(columns=["y"], errors="ignore")

    print(f"tier1+2: {t12.shape}, tier3+4: {t34.shape}, codegen: {cg.shape}")

    # Inner-join all three on datapoint_id
    df = t12.merge(t34, on="datapoint_id").merge(cg, on="datapoint_id")
    print(f"joined: {df.shape}")

    y = df["y"].astype(int).values
    base_cols = [c for c in t12.columns if c not in ("datapoint_id", "y")]
    t3_cols = [c for c in t34.columns if c.startswith("tier3_")]
    t4a_cols = [c for c in t34.columns if c.startswith("tier4a_")]
    t4b_cols = [c for c in t34.columns if c.startswith("tier4b_")]
    cg_cols = [c for c in cg.columns if c != "datapoint_id"]

    print(f"\nfeature group sizes: t12={len(base_cols)} t3={len(t3_cols)} "
          f"t4a={len(t4a_cols)} t4b={len(t4b_cols)} cg={len(cg_cols)}")

    print("\n" + "=" * 78)
    print("CODE_REVIEW VERIFIABILITY LADDER (final)")
    print("=" * 78)

    fit(df[base_cols].values, y, "T1+T2 (metadata+diff parse)")
    fit(df[base_cols + t3_cols].values, y, "+T3 (lizard CCN/NLOC)")
    fit(df[base_cols + t3_cols + t4a_cols].values, y, "+T4a (test-code patterns)")
    fit(df[base_cols + t3_cols + t4a_cols + t4b_cols].values, y,
        "+T4b (reviewer test-talk)")
    rf_cg, _, _ = fit(df[cg_cols].values, y, "Codegen ALONE")
    fit(df[base_cols + t3_cols + t4a_cols + t4b_cols + cg_cols].values, y,
        "ALL (deterministic + codegen)")

    print("\nTop 15 codegen features by RF importance:")
    for n, i in sorted(zip(cg_cols, rf_cg.feature_importances_),
                       key=lambda x: -x[1])[:15]:
        print(f"  {n:<40} {i:.4f}")

    # Save final feature matrix
    out_p = OUT_DIR / "cr_all_tiers_features.parquet"
    df.to_parquet(out_p)
    print(f"\nwrote {out_p}")


if __name__ == "__main__":
    main()
