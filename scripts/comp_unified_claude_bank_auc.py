"""Step 2: AUC analysis of the 121-metric bank vs Claude binary label.

Reads outputs/v2_analysis/comp_unified_claude_bank_scores.parquet (no cosine!).
Writes outputs/v2_analysis/comp_unified_claude_bank_auc_summary.json
and prints a Markdown-ish report.

Constraints:
 - NO cosine, NO embedding similarity in feature matrix
 - 0.5 (chance) is the baseline; this is not a "beat cosine" eval
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path("/Users/spangher/Projects/stanford-research/norm-research")
IN_PATH = REPO / "outputs/v2_analysis/comp_unified_claude_bank_scores.parquet"
SUMMARY_PATH = (REPO / "outputs/v2_analysis"
                       "/comp_unified_claude_bank_auc_summary.json")
RNG = 17


def safe_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def single_metric_aucs(df, score_cols, y):
    rows = []
    for c in score_cols:
        v = df[c].values.astype(float)
        applied = df[c.replace("_score", "_applied")].values.astype(float)
        cov = float(np.mean(~np.isnan(v)))
        applied_cov = float(np.mean(applied))
        # impute NaN with median (over non-NaN)
        med = np.nanmedian(v) if np.any(~np.isnan(v)) else 0.0
        v_imp = np.where(np.isnan(v), med, v)
        # If column is constant, AUC undefined
        if np.unique(v_imp).size < 2:
            auc = float("nan")
        else:
            auc = safe_auc(y, v_imp)
        rows.append({
            "metric": c[:-6],
            "coverage_scored": cov,
            "coverage_applied": applied_cov,
            "auc": auc,
            "abs_dev_chance": (abs(auc - 0.5) if not math.isnan(auc)
                               else float("nan")),
        })
    out = pd.DataFrame(rows)
    return out


def joint_cv_auc(X, y, n_splits=5, model="lr"):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RNG)
    oof = np.zeros(len(y), dtype=float)
    folds = []
    for tr, te in skf.split(X, y):
        if model == "lr":
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler(with_mean=True, with_std=True)),
                ("lr", LogisticRegression(max_iter=2000, C=1.0,
                                          solver="liblinear")),
            ])
        elif model == "rf":
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("rf", RandomForestClassifier(
                    n_estimators=200, random_state=RNG, n_jobs=-1)),
            ])
        else:
            raise ValueError(model)
        pipe.fit(X[tr], y[tr])
        p = pipe.predict_proba(X[te])[:, 1]
        oof[te] = p
        folds.append(safe_auc(y[te], p))
    return safe_auc(y, oof), folds, oof


def main():
    print(f"reading {IN_PATH}", flush=True)
    df = pd.read_parquet(IN_PATH)
    print(f"shape: {df.shape}")
    score_cols = sorted(c for c in df.columns if c.endswith("_score"))
    applied_cols = sorted(c for c in df.columns if c.endswith("_applied"))
    assert len(score_cols) == 121
    print(f"metrics: {len(score_cols)} score cols, "
          f"{len(applied_cols)} applied cols")
    print(f"label dist: {df['claude_label'].value_counts().to_dict()}")
    print(f"source dist: {df['source'].value_counts().to_dict()}")
    print(f"language dist: {df['language'].value_counts().to_dict()}")

    # ---- LEAKAGE GUARD ----
    bad = [c for c in df.columns
           if any(k in c.lower()
                  for k in ("cos", "sim", "embed", "max_sim"))]
    assert not bad, f"LEAKAGE: cosine/sim cols present: {bad}"

    y = df["claude_label"].values.astype(int)

    # A. Single-metric AUCs
    print("\n[A] Single-metric AUCs ...", flush=True)
    sm = single_metric_aucs(df, score_cols, y)
    sm_valid = sm.dropna(subset=["auc"]).copy()
    sm_valid_top = sm_valid.sort_values("abs_dev_chance",
                                        ascending=False).head(20)
    print(sm_valid_top.to_string(index=False))

    # B. Joint LR / RF
    print("\n[B] Joint LR / RF on bank scores ...", flush=True)
    X_score = df[score_cols].values.astype(float)
    X_full = df[score_cols + applied_cols].values.astype(float)
    auc_lr_score, folds_lr_score, _ = joint_cv_auc(X_score, y, model="lr")
    auc_lr_full, folds_lr_full, _ = joint_cv_auc(X_full, y, model="lr")
    auc_rf_score, folds_rf_score, _ = joint_cv_auc(X_score, y, model="rf")
    auc_rf_full, folds_rf_full, _ = joint_cv_auc(X_full, y, model="rf")
    print(f"LR score-only  AUC={auc_lr_score:.4f}  "
          f"folds={[round(x,3) for x in folds_lr_score]}")
    print(f"LR score+appl AUC={auc_lr_full:.4f}  "
          f"folds={[round(x,3) for x in folds_lr_full]}")
    print(f"RF score-only  AUC={auc_rf_score:.4f}  "
          f"folds={[round(x,3) for x in folds_rf_score]}")
    print(f"RF score+appl AUC={auc_rf_full:.4f}  "
          f"folds={[round(x,3) for x in folds_rf_full]}")

    # C. Per-platform
    print("\n[C] Per-platform breakdown ...", flush=True)
    per_platform = {}
    for plat, sub in df.groupby("source"):
        if len(sub) < 50 or sub["claude_label"].nunique() < 2:
            per_platform[plat] = {
                "n": int(len(sub)), "auc_lr_score": None,
                "note": "skipped n<50 or single-class",
            }
            continue
        Xs = sub[score_cols].values.astype(float)
        ys = sub["claude_label"].values.astype(int)
        try:
            auc_p, folds_p, _ = joint_cv_auc(Xs, ys, model="lr")
        except Exception as e:
            auc_p, folds_p = float("nan"), []
            per_platform[plat] = {
                "n": int(len(sub)), "auc_lr_score": auc_p,
                "error": str(e),
            }
            continue
        per_platform[plat] = {
            "n": int(len(sub)),
            "pos_rate": float(ys.mean()),
            "auc_lr_score": auc_p,
            "folds": [float(x) for x in folds_p],
        }
        print(f"  {plat}: n={len(sub)} pos={ys.mean():.2f} "
              f"AUC={auc_p:.4f} folds={[round(x,3) for x in folds_p]}")
    # Single-shot 5K-only and 300-only
    splits = {}
    for sp, sub in df.groupby("split"):
        Xs = sub[score_cols].values.astype(float)
        ys = sub["claude_label"].values.astype(int)
        if ys.size < 50 or len(np.unique(ys)) < 2:
            continue
        try:
            auc_s, folds_s, _ = joint_cv_auc(Xs, ys, model="lr")
        except Exception:
            auc_s, folds_s = float("nan"), []
        splits[sp] = {
            "n": int(len(sub)),
            "pos_rate": float(ys.mean()),
            "auc_lr_score": auc_s,
        }
        print(f"  split={sp}: n={len(sub)} pos={ys.mean():.2f} "
              f"AUC={auc_s:.4f}")

    summary = {
        "n_total": int(len(df)),
        "pos_rate": float(y.mean()),
        "leakage_guard_passed": True,
        "joint": {
            "lr_score_only_auc": auc_lr_score,
            "lr_score_only_folds": folds_lr_score,
            "lr_score_plus_applied_auc": auc_lr_full,
            "lr_score_plus_applied_folds": folds_lr_full,
            "rf_score_only_auc": auc_rf_score,
            "rf_score_plus_applied_auc": auc_rf_full,
        },
        "per_platform": per_platform,
        "per_split": splits,
        "top20_single_metric": (sm_valid.sort_values(
            "abs_dev_chance", ascending=False)
            .head(20)
            .to_dict(orient="records")),
        "single_metric_all_n": int(len(sm_valid)),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {SUMMARY_PATH}")

    # Markdown report
    print("\n" + "=" * 70)
    print("MARKDOWN REPORT")
    print("=" * 70)
    print(f"\n| Cell | n | AUC vs Claude label | vs chance (0.5) |")
    print(f"|---|---:|---:|---:|")
    print(f"| Bank LR (121 score cols) | {len(df)} | "
          f"{auc_lr_score:.4f} | {auc_lr_score - 0.5:+.4f} |")
    print(f"| Bank LR (242 score+applied cols) | {len(df)} | "
          f"{auc_lr_full:.4f} | {auc_lr_full - 0.5:+.4f} |")
    print(f"| Bank RF (121 score cols) | {len(df)} | "
          f"{auc_rf_score:.4f} | {auc_rf_score - 0.5:+.4f} |")
    for plat, row in per_platform.items():
        if row.get("auc_lr_score") is None:
            continue
        a = row["auc_lr_score"]
        print(f"| Per-platform: {plat} | {row['n']} | "
              f"{a:.4f} | {a - 0.5:+.4f} |")

    print("\nTop-20 single metrics by |AUC-0.5|:")
    print("| metric | applied% | AUC | dev |")
    print("|---|---:|---:|---:|")
    for r in (sm_valid.sort_values("abs_dev_chance", ascending=False)
              .head(20).itertuples(index=False)):
        print(f"| {r.metric} | {r.coverage_applied*100:.1f}% "
              f"| {r.auc:.4f} | {r.auc - 0.5:+.4f} |")


if __name__ == "__main__":
    main()
