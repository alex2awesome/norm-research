"""Deep dive task 3: re-run bank LR/RF/ENS with the EXPANDED bank
(existing shard features + new h-metrics parquet), so_python.

Usage: python dd_task3_eval_expanded.py so_python
Reads outputs/v2_analysis/se_ladder/{slice}_hmetrics.parquet and reports
old-vs-new AUC overall + per stratum.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
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


def select_feats(tr, cand):
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
    return feats


def fit_eval(df, feats, tag):
    tr, te = df[df.split == "train"], df[df.split == "test"]
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
    res = {"n_features": len(feats),
           "LR": float(roc_auc_score(yte, p_lr)),
           "RF": float(roc_auc_score(yte, p_rf)),
           "ENS": float(roc_auc_score(yte, p_en))}
    te2 = te.copy()
    te2["p"] = p_en
    res["per_stratum_ENS"] = {
        s: {"n": int(len(g)), "ENS": float(roc_auc_score(g.label, g.p))}
        for s, g in te2.groupby("stratum")
        if len(g) >= 200 and g.label.nunique() == 2}
    # top new-feature coefficients
    coefs = lr.named_steps["clf"].coef_[0]
    order = np.argsort(-np.abs(coefs))[:25]
    res["top_lr_features"] = [
        {"feature": feats[i], "coef": float(coefs[i])} for i in order]
    print(tag, json.dumps({k: v for k, v in res.items()
                           if k != "top_lr_features"}, indent=2), flush=True)
    return res


def main():
    slice_name = sys.argv[1]
    shard_dir = OUT_DIR / "shards" / slice_name
    scored = pd.concat([pd.read_parquet(p) for p in
                        sorted(shard_dir.glob("shard_*.parquet"))],
                       ignore_index=True)
    meta = pd.read_parquet(OUT_DIR / f"{slice_name}_input.parquet",
                           columns=["row_id", "question_id", "label",
                                    "split", "stratum"])
    h = pd.read_parquet(OUT_DIR / f"{slice_name}_hmetrics.parquet")
    df = meta.merge(scored, on="row_id").merge(h, on="row_id")
    assert len(df) == len(meta), (len(df), len(meta))

    old_cand = [c for c in scored.columns
                if c.endswith("_score") or c.endswith("_applied")]
    new_cand = [c for c in h.columns
                if c.endswith("_score") or c.endswith("_applied")]
    tr = df[df.split == "train"]
    feats_old = select_feats(tr, old_cand)
    feats_new = select_feats(tr, new_cand)
    print(f"old feats: {len(feats_old)}  new feats kept: {len(feats_new)} "
          f"of {len(new_cand)}", flush=True)

    res = {"slice": slice_name,
           "old_bank": fit_eval(df, feats_old, "[old]"),
           "new_only": fit_eval(df, feats_new, "[new-only]"),
           "expanded": fit_eval(df, feats_old + feats_new, "[expanded]")}
    out = OUT_DIR / f"{slice_name}_bank_eval_expanded.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
