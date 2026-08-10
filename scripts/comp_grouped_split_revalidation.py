"""Problem-grouped re-validation of competition editorial-similarity AUCs.

Goal: existing AUC scripts (comp_unified_claude_bank_auc_v3.py etc.) use
StratifiedKFold with no problem grouping. Multiple candidates of the same
problem can land in train + test, so static bank metrics may fingerprint the
problem rather than the candidate-vs-editorial idea. This script:
  1. Joins pair -> editorial_id -> canonical_pid (group key).
  2. Reproduces random-CV AUCs (must match v3 numbers).
  3. Re-runs the same cells under StratifiedGroupKFold(5).
  4. Adds V0 execution features (verdict one-hot + within-problem percentile
     of runtime/memory if available) and reports the ladder:
        V0-only, bank-only, bank+V0  (all under grouped CV).
  5. Writes JSON + a small markdown table.

Cells: CC-449, Luogu-555, pooled-1004, LC-2K (~1995), phase1b-CC (sk3-only,
run there separately).
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
from sklearn.model_selection import (
    GroupKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path("/Users/spangher/Projects/stanford-research/norm-research")
V3_PATH = REPO / "outputs/v2_analysis/comp_unified_claude_bank_scores_v3.parquet"
EDLAB_PATH = REPO / "outputs/v2_analysis/comp_unified_editorial_labels.parquet"
EDPID_PATH = Path("/tmp/editorial_to_pid.parquet")
CAND_PATH = Path("/tmp/candidates_subset.parquet")
CLAUDE5K_MAP = Path("/tmp/claude5k_mapping.parquet")
CLAUDE3900_MAP = Path("/tmp/claude3900_mapping.parquet")

LC_BANK_PATH = REPO / "outputs/v2_analysis/lc_python_metric_scores.parquet"
LC_SAMP_PATH = REPO / "outputs/v2_analysis/lc_python_2k_sample.parquet"
LC_LAB_PATH = REPO / "outputs/v2_analysis/lc_python_2k_claude_labels.parquet"

OUT_JSON = REPO / "outputs/v2_analysis/comp_grouped_split_revalidation.json"
OUT_MD = REPO / "outputs/v2_analysis/comp_grouped_split_revalidation.md"

RNG = 17
N_FOLDS = 5

# ---------------------- helpers ---------------------- #


def safe_auc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
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
        ("lr", LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")),
    ])


def _make_rf():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                      n_jobs=-1, class_weight="balanced",
                                      random_state=RNG)),
    ])


def cv_auc(X, y, splitter_splits, model_kind="lr"):
    """Generic OOF-pooled AUC across folds; splitter_splits is an iterable
    of (train_idx, test_idx) tuples."""
    oof = np.full(len(y), np.nan)
    fold_aucs = []
    for tr, te in splitter_splits:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        pipe = _make_lr() if model_kind == "lr" else _make_rf()
        pipe.fit(X[tr], y[tr])
        p = pipe.predict_proba(X[te])[:, 1]
        oof[te] = p
        fold_aucs.append(safe_auc(y[te], p))
    fold_aucs = [a for a in fold_aucs if not math.isnan(a)]
    pooled = safe_auc(y[~np.isnan(oof)], oof[~np.isnan(oof)])
    return {
        "auc_oof_pooled": pooled,
        "auc_fold_mean": float(np.mean(fold_aucs)) if fold_aucs else float("nan"),
        "auc_fold_std": float(np.std(fold_aucs)) if fold_aucs else float("nan"),
        "n_folds_used": len(fold_aucs),
    }


def random_split(X, y, n=N_FOLDS):
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=RNG)
    return list(skf.split(X, y))


def grouped_split(X, y, groups, n=N_FOLDS):
    """Try StratifiedGroupKFold, fall back to GroupKFold."""
    try:
        sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=RNG)
        splits = list(sgkf.split(X, y, groups))
        # Validate that no fold has degenerate class
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


# ---------------------- V0 feature builder ---------------------- #

VERDICT_CATS = ["AC", "WA", "TLE", "RE", "CE", "MLE", "OTHER", "unknown"]


def build_v0_features(cand_subset: pd.DataFrame, cand_meta: pd.DataFrame):
    """For each row in cand_subset (must have candidate_id, canonical_pid),
    return a DataFrame with V0 features aligned to its index.

    cand_meta is the full candidates table (per platform), used to compute
    within-problem percentile of runtime_ms / memory_kb.
    """
    # Join verdict
    m = cand_subset.merge(
        cand_meta[["candidate_id", "verdict", "runtime_ms", "memory_kb"]],
        on="candidate_id",
        how="left",
    )

    # Verdict one-hot + known flag
    v = m["verdict"].fillna("unknown")
    for cat in VERDICT_CATS:
        m[f"v0_verdict_{cat}"] = (v == cat).astype(int)
    m["v0_verdict_known"] = (v != "unknown").astype(int)

    # Within-problem percentile (over all candidates with same canonical_pid
    # in cand_meta). For platforms missing runtime/memory entirely we leave NaN.
    # Build problem-level rank tables only over cand_meta rows with non-null
    # values.
    def _within_problem_pct(col):
        # Group by canonical_pid; for each row, percentile = rank / count
        s = cand_meta[["candidate_id", "canonical_pid", col]].dropna(
            subset=["canonical_pid", col]
        )
        s = s.copy()
        s["pct"] = s.groupby("canonical_pid")[col].rank(pct=True)
        out = m.merge(
            s[["candidate_id", "pct"]].rename(columns={"pct": f"v0_pct_{col}"}),
            on="candidate_id",
            how="left",
        )[f"v0_pct_{col}"]
        return out.values

    m["v0_pct_runtime"] = _within_problem_pct("runtime_ms")
    m["v0_pct_memory"] = _within_problem_pct("memory_kb")

    v0_cols = [f"v0_verdict_{c}" for c in VERDICT_CATS] + [
        "v0_verdict_known", "v0_pct_runtime", "v0_pct_memory",
    ]
    return m, v0_cols


# ---------------------- pipelines per cell ---------------------- #


def eval_cell(name, df, score_cols, applied_cols, v0_cols, groups, log):
    """Run the 5 evaluations for a single cell:
      - random-CV bank (LR + RF)
      - grouped-CV bank (LR + RF)
      - grouped-CV V0 only (LR + RF)
      - grouped-CV bank+V0 (LR + RF)
    `score_cols`/`applied_cols` are bank feature columns. The bank feature set
    mirrors v3: score + applied. (v3 paper used score-only; we report both
    flavors of "bank-only", labeled.)
    """
    y = df["claude_label" if "claude_label" in df.columns
           else "label"].astype(int).values
    bank_feats = score_cols + applied_cols
    X_bank = df[bank_feats].astype(float).values
    X_bank_score = df[score_cols].astype(float).values

    out = {
        "n": int(len(df)),
        "pos_rate": float(y.mean()),
        "n_groups": int(pd.unique(groups).size),
        "max_pairs_per_group": int(pd.Series(groups).value_counts().max()),
    }

    # Random CV
    rand_splits = random_split(X_bank, y)
    grp_splits, splitter_name = grouped_split(X_bank, y, groups)
    out["splitter_used"] = splitter_name

    # Bank only (score+applied), random
    out["bank_random_lr"] = cv_auc(X_bank, y, rand_splits, "lr")
    out["bank_random_rf"] = cv_auc(X_bank, y, rand_splits, "rf")
    # Bank only (score only), random — to match v3 baseline directly
    out["bank_score_only_random_lr"] = cv_auc(X_bank_score, y, rand_splits, "lr")

    # Bank only, grouped
    out["bank_grouped_lr"] = cv_auc(X_bank, y, grp_splits, "lr")
    out["bank_grouped_rf"] = cv_auc(X_bank, y, grp_splits, "rf")
    out["bank_score_only_grouped_lr"] = cv_auc(X_bank_score, y, grp_splits, "lr")

    # V0
    if v0_cols:
        X_v0 = df[v0_cols].astype(float).values
        X_bv = np.concatenate([X_bank, X_v0], axis=1)
        out["v0_grouped_lr"] = cv_auc(X_v0, y, grp_splits, "lr")
        out["v0_grouped_rf"] = cv_auc(X_v0, y, grp_splits, "rf")
        out["bank_plus_v0_grouped_lr"] = cv_auc(X_bv, y, grp_splits, "lr")
        out["bank_plus_v0_grouped_rf"] = cv_auc(X_bv, y, grp_splits, "rf")
        # V0 coverage
        out["v0_coverage"] = {
            c: float(df[c].notna().mean()) for c in v0_cols
        }
    log[name] = out
    print(f"\n=== {name} (n={len(df)}, pos_rate={y.mean():.3f}, "
          f"n_groups={out['n_groups']}, max_pairs/group={out['max_pairs_per_group']}, "
          f"splitter={splitter_name}) ===")
    print(f"  bank score-only random-CV LR : "
          f"{out['bank_score_only_random_lr']['auc_oof_pooled']:.4f}  "
          f"fold={out['bank_score_only_random_lr']['auc_fold_mean']:.4f}"
          f"±{out['bank_score_only_random_lr']['auc_fold_std']:.4f}")
    print(f"  bank score-only grouped-CV LR: "
          f"{out['bank_score_only_grouped_lr']['auc_oof_pooled']:.4f}  "
          f"fold={out['bank_score_only_grouped_lr']['auc_fold_mean']:.4f}"
          f"±{out['bank_score_only_grouped_lr']['auc_fold_std']:.4f}")
    print(f"  bank score+applied grouped LR: "
          f"{out['bank_grouped_lr']['auc_oof_pooled']:.4f}")
    if v0_cols:
        print(f"  V0-only grouped LR           : "
              f"{out['v0_grouped_lr']['auc_oof_pooled']:.4f}")
        print(f"  bank+V0 grouped LR           : "
              f"{out['bank_plus_v0_grouped_lr']['auc_oof_pooled']:.4f}")


# ---------------------- driver ---------------------- #


def main():
    log = {}

    # ---- Load CC+Luogu cell (v3 bank scores) ---- #
    print("[1] loading v3 bank scores + editorial labels...", flush=True)
    df = pd.read_parquet(V3_PATH)
    ed_lab = pd.read_parquet(EDLAB_PATH)
    ed_pid = pd.read_parquet(EDPID_PATH)
    cand_meta = pd.read_parquet(CAND_PATH)
    print(f"  v3: {df.shape}, ed_lab: {ed_lab.shape}, ed_pid: {ed_pid.shape}, "
          f"candidates: {cand_meta.shape}")

    # Join pair_id -> candidate_id -> canonical_pid
    # ed_lab has pair_id (cc_/luogu_/usaco_ style), and 5k/3900 mappings cover
    # the clauded5k_* / etc pair_ids.
    map_parts = [ed_lab[["pair_id", "candidate_id", "editorial_id",
                         "platform"]]]
    if CLAUDE5K_MAP.exists():
        map_parts.append(pd.read_parquet(CLAUDE5K_MAP))
    if CLAUDE3900_MAP.exists():
        map_parts.append(pd.read_parquet(CLAUDE3900_MAP)[
            ["pair_id", "candidate_id", "editorial_id", "platform"]])
    pair_map = pd.concat(map_parts, ignore_index=True).drop_duplicates(
        subset=["pair_id"], keep="first")
    print(f"  pair_map combined: {pair_map.shape}")
    df = df.merge(pair_map, on="pair_id", how="left")
    # Get canonical_pid via editorial_id (more direct than via candidate_id,
    # but we'll cross-check by candidate too)
    df = df.merge(
        ed_pid[["editorial_id", "canonical_pid"]],
        on="editorial_id", how="left",
    )
    # Fallback: get canonical_pid via candidate too (must agree)
    cand_pid_map = cand_meta[["candidate_id", "canonical_pid"]].drop_duplicates(
        subset=["candidate_id"]
    ).rename(columns={"canonical_pid": "canonical_pid_from_cand"})
    df = df.merge(cand_pid_map, on="candidate_id", how="left")
    miss_ed = df["canonical_pid"].isna().sum()
    miss_cd = df["canonical_pid_from_cand"].isna().sum()
    print(f"  canonical_pid missing (via editorial): {miss_ed}")
    print(f"  canonical_pid missing (via candidate): {miss_cd}")
    df["canonical_pid"] = df["canonical_pid"].fillna(df["canonical_pid_from_cand"])
    n_missing = df["canonical_pid"].isna().sum()
    print(f"  pairs without canonical_pid: {n_missing}")
    df = df.dropna(subset=["canonical_pid"]).reset_index(drop=True)
    print(f"  retained: {len(df)} rows")

    # Sanity: pairs per problem
    for plat in ["cc", "luogu"]:
        sub = df[df["source"] == plat]
        v = sub.groupby("canonical_pid").size()
        print(f"  {plat}: n={len(sub)}, n_problems={v.size}, "
              f"max_pairs/problem={v.max() if len(v) else 0}, "
              f"mean={v.mean() if len(v) else 0:.2f}")

    # Build V0 features for CC+Luogu
    df, v0_cols = build_v0_features(df, cand_meta)
    print(f"  V0 cols: {len(v0_cols)}")
    # V0 coverage per platform
    for plat in ["cc", "luogu"]:
        sub = df[df["source"] == plat]
        cov = {c: float(sub[c].notna().mean()) for c in
               ["v0_verdict_AC", "v0_verdict_known",
                "v0_pct_runtime", "v0_pct_memory"]}
        print(f"  V0 coverage {plat}: {cov}")

    # ---- Determine bank score/applied cols (v3 style) ---- #
    score_cols = sorted(c for c in df.columns
                        if c.endswith("_score") and not c.startswith("v0"))
    applied_cols = sorted(c for c in df.columns
                          if c.endswith("_applied") and not c.startswith("v0"))
    print(f"  bank score cols: {len(score_cols)}, applied: {len(applied_cols)}")

    # ---- Cell 1: pooled CC+Luogu ---- #
    eval_cell(
        "pooled_cc_luogu_1004",
        df, score_cols, applied_cols, v0_cols,
        df["canonical_pid"].values, log,
    )

    # ---- Cell 2: CC-449 ---- #
    df_cc = df[df["source"] == "cc"].reset_index(drop=True)
    eval_cell(
        "cc_449",
        df_cc, score_cols, applied_cols, v0_cols,
        df_cc["canonical_pid"].values, log,
    )

    # ---- Cell 3: Luogu-555 ---- #
    df_lu = df[df["source"] == "luogu"].reset_index(drop=True)
    eval_cell(
        "luogu_555",
        df_lu, score_cols, applied_cols, v0_cols,
        df_lu["canonical_pid"].values, log,
    )

    # ---- Cell 4: LC-2K ---- #
    print("\n[2] loading LC-2K cell...", flush=True)
    try:
        lc_samp = pd.read_parquet(LC_SAMP_PATH)
        lc_lab = pd.read_parquet(LC_LAB_PATH)
        lc_bank = pd.read_parquet(LC_BANK_PATH)
        # Normalize candidate_id dtype (lc_bank may be int64)
        lc_samp["candidate_id"] = lc_samp["candidate_id"].astype(str)
        lc_bank["candidate_id"] = lc_bank["candidate_id"].astype(str)
        # Merge
        lc = lc_samp.merge(lc_lab, on="pair_id", how="inner")
        lc = lc.merge(lc_bank, on="candidate_id", how="left")
        print(f"  LC after join: {lc.shape}")
        # group key = question_slug (LC's problem id).
        # LC 2K corpus uses int candidate_ids from a separate LC scrape (no
        # verdict/runtime info; these are forum solution snippets, not graded
        # submissions). V0 features cannot be applied here -> v0_cols=[].
        v0_cols_lc = []
        # rename label to claude_label for uniform handling
        lc = lc.rename(columns={"label": "claude_label"})
        # bank score/applied cols
        lc_score = sorted(c for c in lc.columns
                          if c.endswith("_score") and not c.startswith("v0"))
        lc_appl = sorted(c for c in lc.columns
                         if c.endswith("_applied") and not c.startswith("v0"))
        eval_cell(
            "lc_2k",
            lc, lc_score, lc_appl, v0_cols_lc,
            lc["question_slug"].values, log,
        )
    except Exception as e:
        log["lc_2k_error"] = str(e)
        print(f"  LC-2K cell failed: {e}")

    # ---- Write outputs ---- #
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(log, f, indent=2, default=float)
    print(f"\nWrote {OUT_JSON}")

    # Markdown report
    md = []
    md.append("# Problem-grouped re-validation of competition AUCs\n")
    md.append("Re-runs existing CC/Luogu/LC editorial-similarity cells under "
              "StratifiedGroupKFold(5) with canonical_pid as the group key, "
              "then adds V0 execution features (verdict one-hot + within-problem "
              "runtime/memory percentile).\n")
    md.append("## Random vs Grouped CV (bank score-only, LR)\n")
    md.append("| Cell | n | n_problems | max/problem | Random | Grouped | Δ |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for name in ["cc_449", "luogu_555", "pooled_cc_luogu_1004", "lc_2k"]:
        if name not in log:
            continue
        d = log[name]
        r = d["bank_score_only_random_lr"]["auc_oof_pooled"]
        g = d["bank_score_only_grouped_lr"]["auc_oof_pooled"]
        md.append(
            f"| {name} | {d['n']} | {d['n_groups']} | "
            f"{d['max_pairs_per_group']} | {r:.4f} | {g:.4f} | {g-r:+.4f} |"
        )

    md.append("\n## Grouped CV ladder: V0 / bank / bank+V0 (LR)\n")
    md.append("| Cell | V0-only | bank | bank+V0 | Δ(V0) |")
    md.append("|---|---:|---:|---:|---:|")
    for name in ["cc_449", "luogu_555", "pooled_cc_luogu_1004", "lc_2k"]:
        if name not in log or "v0_grouped_lr" not in log[name]:
            continue
        d = log[name]
        v = d["v0_grouped_lr"]["auc_oof_pooled"]
        b = d["bank_grouped_lr"]["auc_oof_pooled"]
        bv = d["bank_plus_v0_grouped_lr"]["auc_oof_pooled"]
        md.append(f"| {name} | {v:.4f} | {b:.4f} | {bv:.4f} | {bv-b:+.4f} |")

    md.append("\n## V0 coverage (fraction non-null per cell)\n")
    md.append("| Cell | verdict_known | pct_runtime | pct_memory |")
    md.append("|---|---:|---:|---:|")
    for name in ["cc_449", "luogu_555", "pooled_cc_luogu_1004", "lc_2k"]:
        if name not in log or "v0_coverage" not in log[name]:
            continue
        c = log[name]["v0_coverage"]
        md.append(f"| {name} | {c.get('v0_verdict_known', 0):.3f} | "
                  f"{c.get('v0_pct_runtime', 0):.3f} | "
                  f"{c.get('v0_pct_memory', 0):.3f} |")

    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
