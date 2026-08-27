"""AUC v3: bank + a500..a517 + a518..a522. Pooled and per-platform LR.

Compares against v2 baseline (bank + a500..a517). NO COSINE / NO SIM in
features. Outputs JSON + a brief markdown report.
"""
from __future__ import annotations

import json
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
IN_PATH = REPO / "outputs/v2_analysis/comp_unified_claude_bank_scores_v3.parquet"
OUT_JSON = REPO / "outputs/v2_analysis/algorithmic_metrics_auc.json"
OUT_MD = REPO / "outputs/v2_analysis/algorithmic_metrics_report.md"
RNG = 17

V2_IDS = ["a500", "a501", "a502", "a503", "a504",
          "a510", "a511", "a512", "a513", "a514",
          "a515", "a516", "a517"]
NEW_IDS = ["a518", "a519", "a520", "a521", "a522"]


def safe_auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def joint_cv_auc(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RNG)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, C=1.0,
                                      solver="liblinear")),
        ])
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict_proba(X[te])[:, 1]
    return safe_auc(y, oof)


def main():
    df = pd.read_parquet(IN_PATH)
    print(f"shape: {df.shape}")

    # Leakage guard
    bad = [c for c in df.columns
           if any(k in c.lower() for k in ("cos", "sim", "embed", "max_sim"))]
    assert not bad, f"LEAKAGE: {bad}"

    y = df["claude_label"].values.astype(int)

    score_cols_all = sorted(c for c in df.columns if c.endswith("_score"))
    applied_cols_all = sorted(c for c in df.columns if c.endswith("_applied"))

    new_score = [f"{a}_score" for a in NEW_IDS]
    new_appl = [f"{a}_applied" for a in NEW_IDS]
    v2_score = [c for c in score_cols_all if c not in new_score]
    v2_appl = [c for c in applied_cols_all if c not in new_appl]
    print(f"v2 score cols: {len(v2_score)}, new: {len(new_score)}")

    out = {
        "n_total": int(len(df)),
        "pos_rate": float(y.mean()),
        "leakage_guard_passed": True,
        "v2_ids": V2_IDS, "new_ids": NEW_IDS,
    }

    # --- Pooled LR: v2 (bank+a500..a517) vs v3 (v2 + a518..a522) ---
    X_v2 = df[v2_score].values.astype(float)
    X_v3 = df[v2_score + new_score].values.astype(float)
    a_v2 = joint_cv_auc(X_v2, y)
    a_v3 = joint_cv_auc(X_v3, y)
    print(f"\nPOOLED LR")
    print(f"  v2 (score only)        : {a_v2:.4f}")
    print(f"  v3 (v2+new score only) : {a_v3:.4f}  delta={a_v3-a_v2:+.4f}")
    out["pooled"] = {
        "v2_score_only": a_v2,
        "v3_score_only": a_v3,
        "delta": a_v3 - a_v2,
    }

    # Per-platform
    out["per_platform"] = {}
    for plat, sub in df.groupby("source"):
        if len(sub) < 50 or sub["claude_label"].nunique() < 2:
            continue
        ys = sub["claude_label"].values.astype(int)
        X_o = sub[v2_score].values.astype(float)
        X_a = sub[v2_score + new_score].values.astype(float)
        X_n = sub[new_score].values.astype(float)
        a_o = joint_cv_auc(X_o, ys)
        a_a = joint_cv_auc(X_a, ys)
        a_n = joint_cv_auc(X_n, ys)
        out["per_platform"][plat] = {
            "n": int(len(sub)),
            "v2_auc": a_o,
            "v3_auc": a_a,
            "delta": a_a - a_o,
            "new_only_auc": a_n,
        }
        print(f"\n{plat} (n={len(sub)})")
        print(f"  v2          : {a_o:.4f}")
        print(f"  v3 (v2+new) : {a_a:.4f}  delta={a_a-a_o:+.4f}")
        print(f"  new only    : {a_n:.4f}")

    # Per-new-metric single AUC
    print("\nPer-new-metric single AUC")
    out["per_new_metric_single_auc"] = {}
    for aid in NEW_IDS:
        col = f"{aid}_score"
        v = df[col].values.astype(float)
        med = np.nanmedian(v) if np.any(~np.isnan(v)) else 0.0
        v_imp = np.where(np.isnan(v), med, v)
        auc = safe_auc(y, v_imp) if np.unique(v_imp).size >= 2 else float("nan")
        cov_app = float(df[f"{aid}_applied"].mean())
        per_p = {}
        for plat, sub in df.groupby("source"):
            if len(sub) < 50 or sub["claude_label"].nunique() < 2:
                continue
            vp = sub[col].values.astype(float)
            medp = np.nanmedian(vp) if np.any(~np.isnan(vp)) else 0.0
            vp_imp = np.where(np.isnan(vp), medp, vp)
            yp = sub["claude_label"].values.astype(int)
            per_p[plat] = (safe_auc(yp, vp_imp)
                           if np.unique(vp_imp).size >= 2 else float("nan"))
        out["per_new_metric_single_auc"][aid] = {
            "applied_pct": cov_app,
            "auc_pooled": auc,
            "auc_per_platform": per_p,
        }
        per_p_s = ", ".join(f"{k}={v:.3f}" for k, v in per_p.items())
        print(f"  {aid} appl={cov_app:.2f} AUC={auc:.4f}  [{per_p_s}]")

    # Also report v2's baseline pooled/per-platform AUCs for the Δ table
    # (these come from cpp_new_metrics_auc.json prior result)
    try:
        with open(REPO / "outputs/v2_analysis/cpp_new_metrics_auc.json") as f:
            v2_json = json.load(f)
        out["v2_baseline_recorded"] = {
            "pooled_v2": v2_json["pooled"]["bank_plus_new_score_only"],
            "per_platform_v2": {k: v["bank_plus_new_auc"]
                                for k, v in v2_json["per_platform"].items()},
        }
    except Exception as e:
        out["v2_baseline_recorded_error"] = str(e)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}")

    # Build a brief markdown report
    md = []
    md.append("# Algorithmic-fingerprint metrics (a518–a522) AUC report\n")
    md.append(f"- n = {len(df)} pairs, claude_label pos_rate = {y.mean():.3f}")
    md.append("- features: score columns only (NO cosine / sim / embed)\n")
    md.append("## Headline Δ\n")
    md.append("| Cell | Before (v2) | After (v3) | Δ |")
    md.append("|---|---:|---:|---:|")
    md.append(f"| Pooled | {a_v2:.4f} | {a_v3:.4f} | {a_v3-a_v2:+.4f} |")
    for plat, d in out["per_platform"].items():
        md.append(
            f"| {plat} (n={d['n']}) | {d['v2_auc']:.4f} | {d['v3_auc']:.4f} | "
            f"{d['delta']:+.4f} |")
    md.append("\n## Per-new-metric single AUC\n")
    md.append("| Metric | applies% | pooled | cc | luogu |")
    md.append("|---|---:|---:|---:|---:|")
    for aid in NEW_IDS:
        rec = out["per_new_metric_single_auc"][aid]
        cc = rec["auc_per_platform"].get("cc", float("nan"))
        lu = rec["auc_per_platform"].get("luogu", float("nan"))
        md.append(f"| {aid} | {rec['applied_pct']:.2f} | "
                  f"{rec['auc_pooled']:.4f} | {cc:.4f} | {lu:.4f} |")
    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
