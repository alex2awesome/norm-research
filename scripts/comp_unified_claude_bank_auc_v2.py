"""AUC v2: pooled + per-platform LR over (bank + 13 new metrics), and
per-new-metric single AUC table. NO COSINE / NO SIM in features.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path("/Users/spangher/Projects/stanford-research/norm-research")
IN_PATH = REPO / "outputs/v2_analysis/comp_unified_claude_bank_scores_v2.parquet"
OUT_JSON = REPO / "outputs/v2_analysis/cpp_new_metrics_auc.json"
RNG = 17

NEW_IDS = ["a500", "a501", "a502", "a503", "a504",
           "a510", "a511", "a512", "a513", "a514",
           "a515", "a516", "a517"]


def safe_auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def joint_cv_auc(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RNG)
    oof = np.zeros(len(y))
    folds = []
    for tr, te in skf.split(X, y):
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, C=1.0,
                                      solver="liblinear")),
        ])
        pipe.fit(X[tr], y[tr])
        p = pipe.predict_proba(X[te])[:, 1]
        oof[te] = p
        folds.append(safe_auc(y[te], p))
    return safe_auc(y, oof), folds


def main():
    df = pd.read_parquet(IN_PATH)
    print(f"shape: {df.shape}")

    # Leakage guard
    bad = [c for c in df.columns
           if any(k in c.lower() for k in ("cos", "sim", "embed", "max_sim"))]
    assert not bad, f"LEAKAGE: {bad}"

    y = df["claude_label"].values.astype(int)

    # Original bank score cols only (exclude pair_id, claude_label, source,
    # language, split, and all *_applied)
    score_cols_all = sorted(c for c in df.columns if c.endswith("_score"))
    applied_cols_all = sorted(c for c in df.columns if c.endswith("_applied"))
    new_score = [f"{a}_score" for a in NEW_IDS]
    new_appl = [f"{a}_applied" for a in NEW_IDS]
    old_score = [c for c in score_cols_all if c not in new_score]
    old_appl = [c for c in applied_cols_all if c not in new_appl]
    print(f"old score cols: {len(old_score)} | new score cols: {len(new_score)}")

    out = {"n_total": int(len(df)),
           "pos_rate": float(y.mean()),
           "leakage_guard_passed": True}

    # --- (1) Pooled LR: old vs old+new ---
    X_old = df[old_score].values.astype(float)
    X_all = df[old_score + new_score].values.astype(float)
    X_old_full = df[old_score + old_appl].values.astype(float)
    X_all_full = df[old_score + new_score
                    + old_appl + new_appl].values.astype(float)

    a_old, _ = joint_cv_auc(X_old, y)
    a_all, _ = joint_cv_auc(X_all, y)
    a_old_full, _ = joint_cv_auc(X_old_full, y)
    a_all_full, _ = joint_cv_auc(X_all_full, y)
    print(f"\nPOOLED LR")
    print(f"  bank (score only)        : {a_old:.4f}")
    print(f"  bank+new (score only)    : {a_all:.4f}  delta={a_all-a_old:+.4f}")
    print(f"  bank (score+appl)        : {a_old_full:.4f}")
    print(f"  bank+new (score+appl)    : {a_all_full:.4f}  "
          f"delta={a_all_full-a_old_full:+.4f}")
    out["pooled"] = {
        "bank_score_only": a_old, "bank_plus_new_score_only": a_all,
        "delta_score_only": a_all - a_old,
        "bank_score_plus_applied": a_old_full,
        "bank_plus_new_score_plus_applied": a_all_full,
        "delta_score_plus_applied": a_all_full - a_old_full,
    }

    # --- (2) Per-platform ---
    out["per_platform"] = {}
    for plat, sub in df.groupby("source"):
        if len(sub) < 50 or sub["claude_label"].nunique() < 2:
            continue
        ys = sub["claude_label"].values.astype(int)
        X_o = sub[old_score].values.astype(float)
        X_a = sub[old_score + new_score].values.astype(float)
        a_o, _ = joint_cv_auc(X_o, ys)
        a_a, _ = joint_cv_auc(X_a, ys)
        # also do single-new only
        X_n = sub[new_score].values.astype(float)
        a_n, _ = joint_cv_auc(X_n, ys)
        out["per_platform"][plat] = {
            "n": int(len(sub)),
            "bank_only_auc": a_o,
            "bank_plus_new_auc": a_a,
            "delta": a_a - a_o,
            "new_only_auc": a_n,
        }
        print(f"\n{plat} (n={len(sub)})")
        print(f"  bank only       : {a_o:.4f}")
        print(f"  bank+new        : {a_a:.4f}  delta={a_a-a_o:+.4f}")
        print(f"  new only        : {a_n:.4f}")

    # --- (3) Per-new-metric single-AUC ---
    print("\nPer-new-metric single AUC (pooled, NaN imputed to median)")
    out["per_new_metric_single_auc"] = {}
    for aid in NEW_IDS:
        col = f"{aid}_score"
        v = df[col].values.astype(float)
        med = np.nanmedian(v) if np.any(~np.isnan(v)) else 0.0
        v_imp = np.where(np.isnan(v), med, v)
        if np.unique(v_imp).size < 2:
            auc = float("nan")
        else:
            auc = safe_auc(y, v_imp)
        cov_app = float(df[f"{aid}_applied"].mean())
        cov_sc = float(np.mean(~np.isnan(v)))
        # also per-platform single AUC
        per_p = {}
        for plat, sub in df.groupby("source"):
            if len(sub) < 50 or sub["claude_label"].nunique() < 2:
                continue
            vp = sub[col].values.astype(float)
            medp = np.nanmedian(vp) if np.any(~np.isnan(vp)) else 0.0
            vp_imp = np.where(np.isnan(vp), medp, vp)
            yp = sub["claude_label"].values.astype(int)
            if np.unique(vp_imp).size < 2:
                per_p[plat] = float("nan")
            else:
                per_p[plat] = safe_auc(yp, vp_imp)
        out["per_new_metric_single_auc"][aid] = {
            "applied_pct": cov_app,
            "scored_pct": cov_sc,
            "auc_pooled": auc,
            "auc_per_platform": per_p,
        }
        per_p_s = ", ".join(f"{k}={v:.3f}" for k, v in per_p.items())
        print(f"  {aid}  appl={cov_app:.2f} scored={cov_sc:.2f} "
              f"AUC={auc:.4f}  [{per_p_s}]")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
