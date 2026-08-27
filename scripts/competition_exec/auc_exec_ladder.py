"""AUC ladder for execution-based features on LC-2K (and CC when available).

For each cell with exec coverage:
  exec-only / bank-only / bank+exec, StratifiedGroupKFold(5) by problem.

Writes outputs/v2_analysis/comp_exec_auc.{json,md}.
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
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path("/Users/spangher/Projects/stanford-research/norm-research")
LC_SAMP = REPO / "outputs/v2_analysis/lc_python_2k_sample.parquet"
LC_LAB = REPO / "outputs/v2_analysis/lc_python_2k_claude_labels.parquet"
LC_BANK = REPO / "outputs/v2_analysis/lc_python_metric_scores.parquet"
LC_EXEC = REPO / "outputs/v2_analysis/comp_exec_features_lc.parquet"

OUT_JSON = REPO / "outputs/v2_analysis/comp_exec_auc.json"
OUT_MD = REPO / "outputs/v2_analysis/comp_exec_auc.md"

RNG = 17
N_FOLDS = 5


def safe_auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def _make_lr():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, C=1.0, solver="liblinear",
                                  class_weight="balanced")),
    ])


def _make_rf():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                      n_jobs=-1, class_weight="balanced",
                                      random_state=RNG)),
    ])


def cv_auc(X, y, splits, kind="lr"):
    oof = np.full(len(y), np.nan)
    fold = []
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        pipe = _make_lr() if kind == "lr" else _make_rf()
        pipe.fit(X[tr], y[tr])
        p = pipe.predict_proba(X[te])[:, 1]
        oof[te] = p
        fold.append(safe_auc(y[te], p))
    fold = [a for a in fold if not math.isnan(a)]
    pooled = safe_auc(y[~np.isnan(oof)], oof[~np.isnan(oof)])
    return {
        "auc_oof_pooled": pooled,
        "auc_fold_mean": float(np.mean(fold)) if fold else float("nan"),
        "auc_fold_std": float(np.std(fold)) if fold else float("nan"),
        "n_folds_used": len(fold),
    }


def grouped_split(X, y, groups, n=N_FOLDS):
    try:
        sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=RNG)
        splits = list(sgkf.split(X, y, groups))
        ok = all(
            len(np.unique(y[tr])) == 2 and len(np.unique(y[te])) == 2
            for tr, te in splits
        )
        if ok:
            return splits, "StratifiedGroupKFold"
    except Exception:
        pass
    gkf = GroupKFold(n_splits=n)
    return list(gkf.split(X, y, groups)), "GroupKFold"


def build_exec_features(exec_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = exec_df.copy()
    # Derive flags + numeric features. Handle NaN: candidates without test mapping
    # have NaN exec rows -> downstream imputer fills with medians.
    df["e_compiled"] = df["compiled"].apply(
        lambda v: float(v) if v in (True, False, 0, 1, 0.0, 1.0) else np.nan
    )
    df["e_pass_rate"] = df["pass_rate"].astype(float)
    df["e_n_tests"] = df["n_tests"].astype(float)
    df["e_n_pass"] = df["n_pass"].astype(float)
    df["e_n_fail"] = df["n_fail"].astype(float)
    df["e_n_timeout"] = df["n_timeout"].astype(float)
    df["e_n_runtime_err"] = df["n_runtime_err"].astype(float)
    # Derived ratios
    df["e_fail_rate"] = df["e_n_fail"] / df["e_n_tests"].clip(lower=1)
    df["e_timeout_rate"] = df["e_n_timeout"] / df["e_n_tests"].clip(lower=1)
    df["e_runtime_err_rate"] = df["e_n_runtime_err"] / df["e_n_tests"].clip(lower=1)
    df["e_all_pass"] = (df["e_pass_rate"] >= 1.0).astype(float)
    df["e_any_pass"] = (df["e_pass_rate"] > 0.0).astype(float)
    df["e_wall_ms_median"] = df["wall_ms_median"].astype(float)

    cols = [
        "e_compiled", "e_pass_rate", "e_fail_rate", "e_timeout_rate",
        "e_runtime_err_rate", "e_all_pass", "e_any_pass",
        "e_n_tests", "e_wall_ms_median",
    ]
    return df, cols


def main():
    print("[1] loading LC tables…", flush=True)
    samp = pd.read_parquet(LC_SAMP)
    lab = pd.read_parquet(LC_LAB)
    bank = pd.read_parquet(LC_BANK)
    exec_df = pd.read_parquet(LC_EXEC)
    print(f"  samp={samp.shape} lab={lab.shape} bank={bank.shape} exec={exec_df.shape}")

    # Join: candidate_id is the primary key for bank and exec
    samp["candidate_id"] = samp["candidate_id"].astype(str)
    bank["candidate_id"] = bank["candidate_id"].astype(str)
    exec_df["candidate_id"] = exec_df["candidate_id"].astype(str)

    df = samp.merge(lab, on="pair_id", how="inner")
    df = df.merge(bank, on="candidate_id", how="left")
    # Add exec features (drop redundant question_slug from exec)
    exec_keep = [c for c in exec_df.columns if c not in ("question_slug",)]
    df = df.merge(exec_df[exec_keep], on="candidate_id", how="left",
                  suffixes=("", "_exec"))
    print(f"  joined: {df.shape}")

    df, exec_cols = build_exec_features(df)
    print(f"  exec_cols: {exec_cols}")
    # Bank cols
    score_cols = sorted(c for c in df.columns if c.endswith("_score") and not c.startswith("e_"))
    applied_cols = sorted(c for c in df.columns if c.endswith("_applied") and not c.startswith("e_"))
    bank_cols = score_cols + applied_cols
    print(f"  bank cols: {len(bank_cols)} (score={len(score_cols)}, applied={len(applied_cols)})")

    y = df["label"].astype(int).values
    groups = df["question_slug"].values
    pos_rate = y.mean()
    print(f"  n={len(df)} pos_rate={pos_rate:.3f} "
          f"n_groups={len(np.unique(groups))}")

    # Exec coverage
    exec_cov_n = int(df["e_pass_rate"].notna().sum())
    print(f"  exec coverage: {exec_cov_n}/{len(df)} "
          f"({exec_cov_n/len(df)*100:.1f}%)")

    # Build splits (one set used for all evaluations to make ladders comparable)
    X_dummy = np.zeros((len(df), 1))
    splits, splitter = grouped_split(X_dummy, y, groups)
    print(f"  splitter={splitter}")

    # Feature matrices
    X_exec = df[exec_cols].astype(float).values
    X_bank = df[bank_cols].astype(float).values
    X_both = np.concatenate([X_bank, X_exec], axis=1)

    # Also compute a "labeled-with-exec" subset cell, for cleaner reporting
    has_exec = df["e_pass_rate"].notna().values

    log = {
        "lc_2k": {
            "n": int(len(df)),
            "pos_rate": float(pos_rate),
            "n_groups": int(len(np.unique(groups))),
            "exec_coverage": exec_cov_n / len(df),
            "splitter": splitter,
            "n_bank_features": len(bank_cols),
            "n_exec_features": len(exec_cols),
        }
    }

    print("\n=== LC-2K (all 1995) grouped CV ===")
    log["lc_2k"]["exec_lr"] = cv_auc(X_exec, y, splits, "lr")
    log["lc_2k"]["exec_rf"] = cv_auc(X_exec, y, splits, "rf")
    log["lc_2k"]["bank_lr"] = cv_auc(X_bank, y, splits, "lr")
    log["lc_2k"]["bank_rf"] = cv_auc(X_bank, y, splits, "rf")
    log["lc_2k"]["bank_plus_exec_lr"] = cv_auc(X_both, y, splits, "lr")
    log["lc_2k"]["bank_plus_exec_rf"] = cv_auc(X_both, y, splits, "rf")
    for k in ["exec_lr","exec_rf","bank_lr","bank_rf","bank_plus_exec_lr","bank_plus_exec_rf"]:
        v = log["lc_2k"][k]
        print(f"  {k:25s}: pooled={v['auc_oof_pooled']:.4f}  "
              f"fold={v['auc_fold_mean']:.4f}±{v['auc_fold_std']:.4f}")

    # Subset cell: only candidates with executable tests
    if has_exec.sum() > 200:
        idx = np.where(has_exec)[0]
        Xb_s = X_bank[idx]; Xe_s = X_exec[idx]; Xbe_s = X_both[idx]
        y_s = y[idx]; g_s = groups[idx]
        sub_splits, sub_splitter = grouped_split(Xb_s, y_s, g_s)
        log["lc_with_exec"] = {
            "n": int(len(idx)),
            "pos_rate": float(y_s.mean()),
            "n_groups": int(len(np.unique(g_s))),
            "splitter": sub_splitter,
        }
        log["lc_with_exec"]["exec_lr"] = cv_auc(Xe_s, y_s, sub_splits, "lr")
        log["lc_with_exec"]["exec_rf"] = cv_auc(Xe_s, y_s, sub_splits, "rf")
        log["lc_with_exec"]["bank_lr"] = cv_auc(Xb_s, y_s, sub_splits, "lr")
        log["lc_with_exec"]["bank_rf"] = cv_auc(Xb_s, y_s, sub_splits, "rf")
        log["lc_with_exec"]["bank_plus_exec_lr"] = cv_auc(Xbe_s, y_s, sub_splits, "lr")
        log["lc_with_exec"]["bank_plus_exec_rf"] = cv_auc(Xbe_s, y_s, sub_splits, "rf")
        print(f"\n=== LC-with-exec (n={len(idx)}) grouped CV ===")
        for k in ["exec_lr","exec_rf","bank_lr","bank_rf","bank_plus_exec_lr","bank_plus_exec_rf"]:
            v = log["lc_with_exec"][k]
            print(f"  {k:25s}: pooled={v['auc_oof_pooled']:.4f}")

    # Exec health summary
    e = df[df["e_pass_rate"].notna()]
    log["exec_health"] = {
        "n": int(len(e)),
        "compile_rate": float(e["e_compiled"].mean()) if "e_compiled" in e else None,
        "pass_rate_mean": float(e["e_pass_rate"].mean()),
        "all_pass_rate": float((e["e_pass_rate"] >= 1.0).mean()),
        "any_pass_rate": float((e["e_pass_rate"] > 0.0).mean()),
        "median_n_tests": float(e["e_n_tests"].median()),
    }
    print(f"\nExec health (n={len(e)}):")
    for k, v in log["exec_health"].items():
        print(f"  {k}: {v}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(log, indent=2, default=float))
    print(f"\nWrote {OUT_JSON}")

    # Markdown
    md = []
    md.append("# Execution-based features: AUC ladder\n")
    md.append("Grouped CV (StratifiedGroupKFold k=5) by problem (question_slug for "
              "LC). Static `bank` = 134-metric editorial-similarity bank. `exec` = "
              "9 features derived from actually running the candidate against "
              "real tests (compiled, pass_rate, fail/timeout/runtime_err rates, "
              "all_pass, any_pass, n_tests, wall_ms_median).\n")
    md.append("## LC-2K (n=1995, pos=0.189, problem-grouped)\n")
    md.append("| Model | exec-only | bank-only | bank+exec | Δ(exec) |")
    md.append("|---|---:|---:|---:|---:|")
    for k_pref, label in [("lr", "LogisticReg"), ("rf", "RandomForest")]:
        e = log["lc_2k"][f"exec_{k_pref}"]["auc_oof_pooled"]
        b = log["lc_2k"][f"bank_{k_pref}"]["auc_oof_pooled"]
        be = log["lc_2k"][f"bank_plus_exec_{k_pref}"]["auc_oof_pooled"]
        md.append(f"| {label} | {e:.4f} | {b:.4f} | {be:.4f} | {be-b:+.4f} |")

    if "lc_with_exec" in log:
        md.append(f"\n## LC-with-exec (n={log['lc_with_exec']['n']}, pos={log['lc_with_exec']['pos_rate']:.3f})\n")
        md.append("| Model | exec-only | bank-only | bank+exec | Δ(exec) |")
        md.append("|---|---:|---:|---:|---:|")
        for k_pref, label in [("lr", "LogisticReg"), ("rf", "RandomForest")]:
            e = log["lc_with_exec"][f"exec_{k_pref}"]["auc_oof_pooled"]
            b = log["lc_with_exec"][f"bank_{k_pref}"]["auc_oof_pooled"]
            be = log["lc_with_exec"][f"bank_plus_exec_{k_pref}"]["auc_oof_pooled"]
            md.append(f"| {label} | {e:.4f} | {b:.4f} | {be:.4f} | {be-b:+.4f} |")

    md.append("\n## Exec health\n")
    md.append("| Metric | Value |")
    md.append("|---|---:|")
    for k, v in log["exec_health"].items():
        md.append(f"| {k} | {v} |")

    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
