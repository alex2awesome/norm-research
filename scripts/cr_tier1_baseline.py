"""Code_review Tier 1 baseline: 5 free metadata features → RF AUC.

Joins v2 task datapoints to dense source by PR title, pulls num_files,
num_comments, pr_additions, pr_deletions, language. Trains RF on the
same fixed 80/20 split (random_state=42) and reports AUC + permutation
importances.

This is the floor of the verifiability ladder.
"""
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
SEED = 42


def main():
    print("Loading v2 task datapoints + labels...")
    dps = json.loads((REPO / "runs/validity_full/v2/code_review/datapoints.json").read_text())
    v2 = pd.DataFrame([{
        "datapoint_id": d["datapoint_id"],
        "y": d.get("judgement"),
        "title": (re.match(r"PR TITLE: ([^\n]+)", d.get("text", "")) or [None, None])[1],
    } for d in dps if d.get("judgement") is not None])
    v2 = v2.dropna(subset=["title"])
    print(f"v2 datapoints with title: {len(v2)}")

    print("Loading dense_4096tok source (metadata cols)...")
    dense = pd.read_csv(
        REPO / "datasets/code-review/code_review_dense_4096tok.csv.gz",
        usecols=["paper_id", "text", "judgement", "language",
                 "num_files", "num_comments", "pr_additions", "pr_deletions"],
    )
    dense["title"] = dense["text"].str.extract(r"## PR Title\s*(.+?)(?:\n|$)", expand=False)
    print(f"dense rows: {len(dense)}, with title: {dense['title'].notna().sum()}")

    # Join by title — multiple dense rows can share title; take first match
    print("Joining v2 → dense by title...")
    j = v2.merge(dense.drop_duplicates("title", keep="first"),
                 on="title", how="left", suffixes=("", "_d"))
    print(f"joined: {len(j)}, with metadata: {j['num_files'].notna().sum()}")

    # Encode language as one-hot top categories
    lang = j["language"].fillna("unknown").str.lower()
    top_langs = lang.value_counts().head(8).index.tolist()
    print(f"top languages: {top_langs}")
    for L in top_langs:
        j[f"lang_{L}"] = (lang == L).astype(int)

    feat_cols = ["num_files", "num_comments", "pr_additions",
                 "pr_deletions"] + [f"lang_{L}" for L in top_langs]
    j_clean = j.dropna(subset=feat_cols)
    print(f"rows usable for modeling: {len(j_clean)}")
    print(f"label distribution: {j_clean['y'].value_counts().to_dict()}")

    X = j_clean[feat_cols].values
    y = j_clean["y"].astype(int).values
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)

    # RF
    rf = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=SEED)
    rf.fit(Xtr, ytr)
    p_rf = rf.predict_proba(Xte)[:, 1]
    auc_rf = roc_auc_score(yte, p_rf)
    acc_rf = accuracy_score(yte, (p_rf > 0.5).astype(int))

    # L2 LogReg
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)
    lr = LogisticRegression(
        penalty="l2", C=0.5, class_weight="balanced",
        max_iter=2000, solver="lbfgs")
    lr.fit(Xtr_s, ytr)
    p_lr = lr.predict_proba(Xte_s)[:, 1]
    auc_lr = roc_auc_score(yte, p_lr)
    acc_lr = accuracy_score(yte, (p_lr > 0.5).astype(int))

    print("\n" + "=" * 60)
    print("TIER 1 BASELINE (5 free metadata features)")
    print("=" * 60)
    print(f"  n_train={len(ytr)}  n_test={len(yte)}")
    print(f"  RF  AUC={auc_rf:.3f}  acc={acc_rf:.1%}")
    print(f"  LR  AUC={auc_lr:.3f}  acc={acc_lr:.1%}")
    maj = max(yte.mean(), 1 - yte.mean())
    print(f"  majority baseline = {maj:.3f}")
    print()
    print("RF feature importances:")
    imps = sorted(zip(feat_cols, rf.feature_importances_),
                  key=lambda x: -x[1])
    for n, i in imps:
        print(f"  {n:<22} {i:.4f}")

    out_p = REPO / "outputs/v2_analysis/cr_tier1_features.parquet"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    j_clean[["datapoint_id"] + feat_cols + ["y"]].to_parquet(out_p)
    print(f"\nwrote {out_p}")


if __name__ == "__main__":
    main()
