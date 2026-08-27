"""AUC ladder for execution-based features on CodeChef pools.

Two cells:
  (1) CC-Claude (n~299, claude_label gold) — joins to local v2 bank scores.
  (2) CC-Phase1b (n~14234, qwen_label silver, optionally sample 5K).

For each cell:
  exec-only / bank-only / bank+exec with StratifiedGroupKFold(5) by canonical_pid.
  LR (class_weight=balanced, liblinear) + RF (500 trees).

Writes outputs/v2_analysis/comp_exec_auc_cc.{json,md}.
"""
from __future__ import annotations

import json
import math
import os
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
V2 = REPO / "outputs/v2_analysis"

# Local: Claude v2 bank scores (has claude_label, pair_id, ~134 metrics)
CLAUDE_BANK = V2 / "comp_unified_claude_bank_scores_v2.parquet"
# Local copies of exec features (we'll pull these down from sk3)
CLAUDE_EXEC = V2 / "comp_exec_features_cc_claude.parquet"
PHASE1B_EXEC = V2 / "comp_exec_features_cc_phase1b.parquet"
# Local Phase1b bank scores (we'll pull from sk3)
PHASE1B_BANK = V2 / "comp_qwen_phase1b_bank_scores.parquet"
# Local Phase1b candidates (with canonical_pid groups + qwen_label)
PHASE1B_CANDS = V2 / "cc_phase1b_with_code.parquet"

OUT_JSON = V2 / "comp_exec_auc_cc.json"
OUT_MD = V2 / "comp_exec_auc_cc.md"

RNG = 17
N_FOLDS = 5
SAMPLE_PHASE1B = 5000  # cap if larger


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
    df["e_compiled"] = df["compiled"].apply(
        lambda v: float(v) if v in (True, False, 0, 1, 0.0, 1.0) else np.nan
    )
    df["e_pass_rate"] = df["pass_rate"].astype(float)
    df["e_n_tests"] = df["n_tests"].astype(float)
    df["e_n_pass"] = df["n_pass"].astype(float)
    df["e_n_fail"] = df["n_fail"].astype(float)
    df["e_n_timeout"] = df["n_timeout"].astype(float)
    df["e_n_runtime_err"] = df["n_runtime_err"].astype(float)
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


def run_cell(name: str, df: pd.DataFrame, label_col: str, bank_cols: list[str],
             exec_cols: list[str], group_col: str = "canonical_pid") -> dict:
    log = {}
    y = df[label_col].astype(int).values
    groups = df[group_col].values
    log["n"] = int(len(df))
    log["pos_rate"] = float(y.mean())
    log["n_groups"] = int(pd.Series(groups).nunique())
    log["n_bank_features"] = len(bank_cols)
    log["n_exec_features"] = len(exec_cols)

    has_exec = df["e_pass_rate"].notna().values
    log["exec_coverage"] = float(has_exec.mean())

    X_exec = df[exec_cols].astype(float).values
    X_bank = df[bank_cols].astype(float).values
    X_both = np.concatenate([X_bank, X_exec], axis=1)

    X_dummy = np.zeros((len(df), 1))
    splits, splitter = grouped_split(X_dummy, y, groups)
    log["splitter"] = splitter

    print(f"\n=== {name} (n={len(df)} pos={y.mean():.3f} groups={log['n_groups']}) ===")
    for k_pref in ["lr", "rf"]:
        for tag, X in [("exec", X_exec), ("bank", X_bank), ("bank_plus_exec", X_both)]:
            log[f"{tag}_{k_pref}"] = cv_auc(X, y, splits, k_pref)
            v = log[f"{tag}_{k_pref}"]
            print(f"  {tag}_{k_pref:2s}: pooled={v['auc_oof_pooled']:.4f}  fold={v['auc_fold_mean']:.4f}±{v['auc_fold_std']:.4f}")

    # Subset cell: only with-exec
    if has_exec.sum() > 200:
        idx = np.where(has_exec)[0]
        y_s = y[idx]; g_s = groups[idx]
        sub_splits, sub_splitter = grouped_split(np.zeros((len(idx),1)), y_s, g_s)
        sub = {
            "n": int(len(idx)), "pos_rate": float(y_s.mean()),
            "n_groups": int(pd.Series(g_s).nunique()), "splitter": sub_splitter,
        }
        for k_pref in ["lr", "rf"]:
            for tag, X in [("exec", X_exec[idx]), ("bank", X_bank[idx]),
                           ("bank_plus_exec", X_both[idx])]:
                sub[f"{tag}_{k_pref}"] = cv_auc(X, y_s, sub_splits, k_pref)
        print(f"  --- with-exec subset (n={len(idx)}) ---")
        for k_pref in ["lr","rf"]:
            for tag in ["exec","bank","bank_plus_exec"]:
                v = sub[f"{tag}_{k_pref}"]
                print(f"  {tag}_{k_pref:2s}: pooled={v['auc_oof_pooled']:.4f}")
        log["with_exec_subset"] = sub
    return log


def main():
    log = {}

    # ---------------- Claude-CC ----------------
    print("[1] Claude-CC")
    cb = pd.read_parquet(CLAUDE_BANK)
    cc_b = cb[cb["source"] == "cc"].copy()
    print(f"  Claude bank cc rows: {len(cc_b)}")

    keep_ex = ["pair_id","compiled","n_tests","n_pass","n_fail","n_timeout",
               "n_runtime_err","wall_ms_median","wall_ms_max","pass_rate",
               "first_error","canonical_pid","language"]
    ex_5k = pd.read_parquet(CLAUDE_EXEC)
    ex_5k = ex_5k[[c for c in keep_ex if c in ex_5k.columns]]
    # Split-300 exec features (1620 rows, of which ~150 cc are part of Claude-449)
    split300_exec_fp = V2 / "comp_exec_features_cc_split300.parquet"
    ex_300 = pd.read_parquet(split300_exec_fp)
    ex_300 = ex_300[[c for c in keep_ex if c in ex_300.columns]]
    ex = pd.concat([ex_5k, ex_300], ignore_index=True).drop_duplicates("pair_id")

    df = cc_b.merge(ex, on="pair_id", how="left")
    # Drop the 'language' column conflict if both bank and exec have it
    if "language_x" in df.columns and "language_y" in df.columns:
        df["language"] = df["language_x"].fillna(df["language_y"])
        df = df.drop(columns=["language_x","language_y"])
    # Filter rows with canonical_pid (group key) — non-cc TACO-missing problems can't be grouped
    df = df[df["canonical_pid"].notna()].copy()
    print(f"  after canonical_pid filter: {len(df)} of cc_b={len(cc_b)}")
    df, exec_cols = build_exec_features(df)

    score_cols = sorted(c for c in df.columns if c.endswith("_score")
                        and not c.startswith("e_"))
    applied_cols = sorted(c for c in df.columns if c.endswith("_applied")
                          and not c.startswith("e_"))
    bank_cols = score_cols + applied_cols

    log["cc_claude"] = run_cell("CC-Claude", df, "claude_label", bank_cols, exec_cols)

    # ---------------- CC-Phase1b ----------------
    print("\n[2] CC-Phase1b")
    pb = pd.read_parquet(PHASE1B_BANK)
    pb_cc = pb[pb["platform"] == "cc"].copy()
    print(f"  Phase1b bank cc rows: {len(pb_cc)}")

    p_cands = pd.read_parquet(PHASE1B_CANDS)  # has canonical_pid, qwen_label
    df = pb_cc.merge(p_cands[["pair_id","canonical_pid","language"]],
                     on="pair_id", how="left")

    # Cap to SAMPLE_PHASE1B for tractable RF
    if len(df) > SAMPLE_PHASE1B:
        df = df.sample(n=SAMPLE_PHASE1B, random_state=RNG).reset_index(drop=True)
        print(f"  sampled to {len(df)}")

    pex = pd.read_parquet(PHASE1B_EXEC)
    pex = pex[[c for c in keep_ex if c in pex.columns]]
    df = df.merge(pex, on="pair_id", how="left", suffixes=("","_e"))
    # canonical_pid may now have _e if both sides had it; prefer original
    if "canonical_pid_e" in df.columns:
        df["canonical_pid"] = df["canonical_pid"].fillna(df["canonical_pid_e"])
        df = df.drop(columns=["canonical_pid_e"])
    if "language_e" in df.columns:
        df["language"] = df["language"].fillna(df["language_e"])
        df = df.drop(columns=["language_e"])

    df = df[df["canonical_pid"].notna()].copy()
    print(f"  after canonical_pid filter: {len(df)}")

    df, exec_cols = build_exec_features(df)
    score_cols = sorted(c for c in df.columns if c.endswith("_score")
                        and not c.startswith("e_"))
    applied_cols = sorted(c for c in df.columns if c.endswith("_applied")
                          and not c.startswith("e_"))
    bank_cols = score_cols + applied_cols

    log["cc_phase1b"] = run_cell("CC-Phase1b", df, "qwen_label", bank_cols, exec_cols)

    # ---------------- Exec health ----------------
    health = {}
    for name, fp in [("claude", CLAUDE_EXEC), ("phase1b", PHASE1B_EXEC)]:
        d = pd.read_parquet(fp)
        e = d[d["compiled"].notna()]
        health[name] = {
            "n_attempted": int(len(d)),
            "n_with_tests": int(len(e)),
            "compile_rate": float(e["compiled"].astype(float).mean()) if len(e) else None,
            "all_pass_rate": float((e["pass_rate"] >= 1.0).mean()) if len(e) else None,
            "any_pass_rate": float((e["pass_rate"] > 0.0).mean()) if len(e) else None,
            "mean_pass_rate": float(e["pass_rate"].mean()) if len(e) else None,
            "median_n_tests": float(e["n_tests"].median()) if len(e) else None,
        }
    log["exec_health"] = health

    OUT_JSON.write_text(json.dumps(log, indent=2, default=float))
    print(f"\nWrote {OUT_JSON}")

    # Markdown
    md = []
    md.append("# Execution-based features on CodeChef: AUC ladder\n")
    md.append("Grouped CV (StratifiedGroupKFold k=5) by `canonical_pid`.\n")
    md.append("`bank` = per-cell editorial-similarity rubric metrics (varies by pool); "
              "`exec` = 9 features from running candidates against TACO test cases.\n")
    for cell in ["cc_claude", "cc_phase1b"]:
        c = log[cell]
        md.append(f"\n## {cell} (n={c['n']}, pos={c['pos_rate']:.3f}, groups={c['n_groups']}, exec_cov={c['exec_coverage']:.3f})\n")
        md.append("| Model | exec-only | bank-only | bank+exec | Δ(exec) |")
        md.append("|---|---:|---:|---:|---:|")
        for k_pref, lbl in [("lr", "LogisticReg"), ("rf", "RandomForest")]:
            e = c[f"exec_{k_pref}"]["auc_oof_pooled"]
            b = c[f"bank_{k_pref}"]["auc_oof_pooled"]
            be = c[f"bank_plus_exec_{k_pref}"]["auc_oof_pooled"]
            md.append(f"| {lbl} | {e:.4f} | {b:.4f} | {be:.4f} | {be-b:+.4f} |")
        if "with_exec_subset" in c:
            s = c["with_exec_subset"]
            md.append(f"\n### with-exec subset (n={s['n']}, pos={s['pos_rate']:.3f})\n")
            md.append("| Model | exec-only | bank-only | bank+exec | Δ(exec) |")
            md.append("|---|---:|---:|---:|---:|")
            for k_pref, lbl in [("lr", "LogisticReg"), ("rf", "RandomForest")]:
                e = s[f"exec_{k_pref}"]["auc_oof_pooled"]
                b = s[f"bank_{k_pref}"]["auc_oof_pooled"]
                be = s[f"bank_plus_exec_{k_pref}"]["auc_oof_pooled"]
                md.append(f"| {lbl} | {e:.4f} | {b:.4f} | {be:.4f} | {be-b:+.4f} |")

    md.append("\n## Exec health\n")
    md.append("| Pool | attempted | with_tests | compile_rate | all_pass | any_pass | mean_pass | median_n_tests |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, h in log["exec_health"].items():
        md.append(f"| {name} | {h['n_attempted']} | {h['n_with_tests']} | "
                  f"{h['compile_rate']:.3f} | {h['all_pass_rate']:.3f} | "
                  f"{h['any_pass_rate']:.3f} | {h['mean_pass_rate']:.3f} | "
                  f"{h['median_n_tests']:.1f} |")

    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
