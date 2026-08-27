"""Deep dive task 3 prep: re-fit the bank LR/RF/ENS on so_python exactly as
se_ladder_eval.py and SAVE test predictions.
Writes {slice}_bank_preds.parquet (row_id, p_lr, p_rf, p_en).

Usage: python dd_bank_preds.py so_python
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
OUT_DIR = REPO / "outputs/v2_analysis/se_ladder"


def main():
    slice_name = sys.argv[1]
    shard_dir = OUT_DIR / "shards" / slice_name
    scored = pd.concat([pd.read_parquet(p) for p in
                        sorted(shard_dir.glob("shard_*.parquet"))],
                       ignore_index=True)
    meta = pd.read_parquet(OUT_DIR / f"{slice_name}_input.parquet",
                           columns=["row_id", "question_id", "label",
                                    "split", "stratum"])
    df = meta.merge(scored, on="row_id", how="inner")
    assert len(df) == len(meta)
    tr, te = df[df.split == "train"], df[df.split == "test"]
    cand = [c for c in df.columns
            if c.endswith("_score") or c.endswith("_applied")]
    feats = []
    for c in cand:
        v = tr[c]
        if v.notna().mean() < 0.05:
            continue
        nz = v[v.notna()]
        if c.endswith("_applied"):
            if nz.nunique() < 2:
                continue
        elif nz.nunique() < 3:
            continue
        feats.append(c)
    Xtr, ytr = tr[feats].values.astype(float), tr.label.values
    Xte, yte = te[feats].values.astype(float), te.label.values
    lr = Pipeline([("imp", SimpleImputer(strategy="median")),
                   ("sc", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=1000, C=1.0,
                                              solver="liblinear"))])
    rf = Pipeline([("imp", SimpleImputer(strategy="median")),
                   ("clf", RandomForestClassifier(n_estimators=500,
                                                  min_samples_leaf=3,
                                                  n_jobs=16,
                                                  random_state=0))])
    lr.fit(Xtr, ytr)
    rf.fit(Xtr, ytr)
    p_lr = lr.predict_proba(Xte)[:, 1]
    p_rf = rf.predict_proba(Xte)[:, 1]
    p_en = (rankdata(p_lr) + rankdata(p_rf)) / (2 * len(p_lr))
    print(f"AUC LR={roc_auc_score(yte, p_lr):.4f} "
          f"RF={roc_auc_score(yte, p_rf):.4f} "
          f"ENS={roc_auc_score(yte, p_en):.4f}", flush=True)
    pd.DataFrame({"row_id": te.row_id.values, "p_lr": p_lr,
                  "p_rf": p_rf, "p_en": p_en}).to_parquet(
        OUT_DIR / f"{slice_name}_bank_preds.parquet", index=False)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
