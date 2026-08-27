"""SE ladder step 2: bank-feature models (2026-06-11).

Per slice: assemble shard scores, train LR + RF(n=500) on split=='train'
(splits come from the shared splits.py baked into the balanced files —
NEVER re-split), report AUC on split=='test' (eval split as sanity).
Ensemble = rank-average of LR and RF test scores. Per-stratum AUCs are the
test-split AUCs of the SAME full-train models restricted to stratum rows.

Features: all {id}_score and {id}_applied columns with >=5% coverage and
>=3 unique values on TRAIN. Median impute + standardize for LR.

Usage: python se_ladder_eval.py <slice>
Writes outputs/v2_analysis/se_ladder/{slice}_bank_eval.json
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


def load_scored(slice_name: str) -> pd.DataFrame:
    shard_dir = OUT_DIR / "shards" / slice_name
    parts = sorted(shard_dir.glob("shard_*.parquet"))
    scored = pd.concat([pd.read_parquet(p) for p in parts],
                       ignore_index=True)
    meta = pd.read_parquet(OUT_DIR / f"{slice_name}_input.parquet",
                           columns=["row_id", "question_id", "label",
                                    "split", "stratum", "lang"])
    df = meta.merge(scored, on="row_id", how="inner")
    assert len(df) == len(meta), (len(df), len(meta))
    return df


def main():
    slice_name = sys.argv[1]
    df = load_scored(slice_name)
    tr = df[df.split == "train"]
    te = df[df.split == "test"]
    ev = df[df.split == "eval"]
    print(f"[{slice_name}] n={len(df)} train={len(tr)} test={len(te)}",
          flush=True)

    cand = [c for c in df.columns
            if c.endswith("_score") or c.endswith("_applied")]
    feats = []
    for c in cand:
        v = tr[c]
        if v.notna().mean() < 0.05:
            continue
        nz = v[v.notna()]
        if nz.nunique() < 3 and not c.endswith("_applied"):
            continue
        if c.endswith("_applied") and nz.nunique() < 2:
            continue
        feats.append(c)
    n_score = sum(1 for c in feats if c.endswith("_score"))
    n_appl = len(feats) - n_score
    print(f"[{slice_name}] features: {n_score} score + {n_appl} applied "
          f"of {len(cand)}", flush=True)

    Xtr, ytr = tr[feats].values.astype(float), tr.label.values
    Xte, yte = te[feats].values.astype(float), te.label.values
    Xev, yev = ev[feats].values.astype(float), ev.label.values

    lr = Pipeline([("imp", SimpleImputer(strategy="median")),
                   ("sc", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=1000, C=1.0,
                                              solver="liblinear"))])
    rf = Pipeline([("imp", SimpleImputer(strategy="median")),
                   ("clf", RandomForestClassifier(
                       n_estimators=500, min_samples_leaf=3,
                       n_jobs=16, random_state=0))])
    lr.fit(Xtr, ytr)
    rf.fit(Xtr, ytr)
    p_lr_te = lr.predict_proba(Xte)[:, 1]
    p_rf_te = rf.predict_proba(Xte)[:, 1]
    p_en_te = (rankdata(p_lr_te) + rankdata(p_rf_te)) / (2 * len(p_lr_te))
    p_lr_ev = lr.predict_proba(Xev)[:, 1]
    p_rf_ev = rf.predict_proba(Xev)[:, 1]
    p_en_ev = (rankdata(p_lr_ev) + rankdata(p_rf_ev)) / (2 * len(p_lr_ev))

    res = {
        "slice": slice_name, "n": int(len(df)),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "n_features": len(feats),
        "n_score_features": n_score, "n_applied_features": n_appl,
        "test": {
            "LR": float(roc_auc_score(yte, p_lr_te)),
            "RF": float(roc_auc_score(yte, p_rf_te)),
            "ENS": float(roc_auc_score(yte, p_en_te)),
        },
        "eval_sanity": {
            "LR": float(roc_auc_score(yev, p_lr_ev)),
            "RF": float(roc_auc_score(yev, p_rf_ev)),
            "ENS": float(roc_auc_score(yev, p_en_ev)),
        },
    }

    # per-stratum on test (same models)
    strata = {}
    te2 = te.copy()
    te2["p_lr"], te2["p_rf"] = p_lr_te, p_rf_te
    te2["p_en"] = p_en_te
    for s, sub in te2.groupby("stratum"):
        if len(sub) < 200 or sub.label.nunique() < 2:
            continue
        strata[s] = {
            "n_test": int(len(sub)),
            "LR": float(roc_auc_score(sub.label, sub.p_lr)),
            "RF": float(roc_auc_score(sub.label, sub.p_rf)),
            "ENS": float(roc_auc_score(sub.label, sub.p_en)),
        }
    res["per_stratum_test"] = strata

    # top LR coefficients for interpretability
    coefs = lr.named_steps["clf"].coef_[0]
    order = np.argsort(-np.abs(coefs))[:20]
    res["top_lr_features"] = [
        {"feature": feats[i], "coef": float(coefs[i])} for i in order]

    out = OUT_DIR / f"{slice_name}_bank_eval.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("top_lr_features",)}, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
