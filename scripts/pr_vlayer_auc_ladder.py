"""
AUC tier increments for the PR static V-layer on the within-repo-balanced stage0 pool.

Ladders (label = pr_merged):
  A (pooled):   GroupKFold(5) grouped by repo_full, LR(std-scaled) + RF(500)
  B (per-repo): StratifiedKFold(5) within each repo with >=50 balanced PRs

Tiers:
  T1     = CSV metadata (n_files, additions, deletions, total_changes, language one-hot)
  T1+2   = + 44 diff-text features (pr_tier2_features_pool.parquet)
  T1+2+3 = + lint-delta features (pr_tier3_shards/*.jsonl), evaluated on the subset
           with Tier-3 coverage AND, for comparability, T1 / T1+2 re-run on that subset.

Usage: python pr_vlayer_auc_ladder.py [--t3-only]
Writes outputs/v2_analysis/pr_vlayer_auc_ladder.json (+ per-repo parquet).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")
OUT = ROOT / "outputs/v2_analysis"
SHARDS = OUT / "pr_tier3_shards"
SEED = 42


def load():
    pool = pd.read_parquet(OUT / "pr_stage0_pool.parquet")
    t2 = pd.read_parquet(OUT / "pr_tier2_features_pool.parquet")
    df = pool.merge(t2, on=["owner", "repo", "pr_number"], how="left", validate="1:1")
    # tier 3
    rows = []
    for f in sorted(SHARDS.glob("*.jsonl")):
        for line in open(f):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    t3 = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["owner", "repo", "pr_number"])
    if len(t3):
        t3 = t3.drop_duplicates(["owner", "repo", "pr_number"], keep="last")
    df = df.merge(t3, on=["owner", "repo", "pr_number"], how="left", validate="1:1")
    # size-normalized lint deltas (violations introduced per added LOC)
    denom = df.get("t2_loc_added", pd.Series(0, index=df.index)).clip(lower=0) + 30.0
    for c in ["t3_ruff_total_delta", "t3_ruff_S_delta", "t3_ruff_D_delta", "t3_ruff_E_delta",
              "t3_ruff_F_delta", "t3_ruff_N_delta", "t3_ruff_C90_delta", "t3_ruff_B_delta",
              "t3_radon_cc_sum_delta", "t3_eslint_total_delta", "t3_gofmt_bad_files_delta"]:
        if c in df.columns:
            df[c + "_perloc"] = df[c] / denom
    return df


def tier_columns(df):
    t1_num = ["n_files_in_pr", "pr_total_changes", "pr_additions", "pr_deletions",
              "pr_changed_files"]
    lang_dum = pd.get_dummies(df["file_language"], prefix="lang")
    t2_cols = [c for c in df.columns if c.startswith("t2_")]
    t3_cols = [c for c in df.columns
               if c.startswith("t3_") and (c.endswith("_delta") or c.endswith("_head")
                                           or c.endswith("_perloc")
                                           or c in ("t3_n_src_changed", "t3_n_src_skipped"))]
    return t1_num, lang_dum, t2_cols, t3_cols


def xy(df, cols, lang_dum):
    X = pd.concat([df[cols].astype(float), lang_dum.loc[df.index].astype(float)], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df["pr_merged"].astype(int).values
    return X.values, y, list(X.columns)


def pooled_auc(X, y, groups):
    gkf = GroupKFold(n_splits=5)
    res = {}
    for name, mk in [("lr", lambda: LogisticRegression(max_iter=2000, C=1.0)),
                     ("rf", lambda: RandomForestClassifier(
                         n_estimators=500, min_samples_leaf=5, n_jobs=16,
                         random_state=SEED))]:
        oof = np.full(len(y), np.nan)
        for tr, te in gkf.split(X, y, groups):
            if name == "lr":
                sc = StandardScaler().fit(X[tr])
                m = mk().fit(sc.transform(X[tr]), y[tr])
                oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
            else:
                m = mk().fit(X[tr], y[tr])
                oof[te] = m.predict_proba(X[te])[:, 1]
        res[name] = float(roc_auc_score(y, oof))
    return res


def per_repo_auc(df, cols, lang_dum, min_n=50):
    out = []
    for repo, grp in df.groupby("repo_full"):
        if len(grp) < min_n or grp.pr_merged.nunique() < 2:
            continue
        X, y, _ = xy(grp, cols, lang_dum)
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        oof = np.full(len(y), np.nan)
        try:
            for tr, te in skf.split(X, y):
                m = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                           n_jobs=8, random_state=SEED).fit(X[tr], y[tr])
                oof[te] = m.predict_proba(X[te])[:, 1]
            out.append({"repo": repo, "n": len(grp), "auc": float(roc_auc_score(y, oof))})
        except ValueError:
            continue
    return pd.DataFrame(out)


def main():
    df = load()
    t1_num, lang_dum, t2_cols, t3_cols = tier_columns(df)
    groups = df["repo_full"].values
    result = {"n_pool": len(df), "n_repos": int(df.repo_full.nunique())}

    # full-pool ladders: T1, T1+2
    for tier, cols in [("T1", t1_num), ("T1+2", t1_num + t2_cols)]:
        X, y, _ = xy(df, cols, lang_dum)
        result[f"pooled_{tier}"] = pooled_auc(X, y, groups)
        pr = per_repo_auc(df, cols, lang_dum)
        result[f"per_repo_{tier}"] = {
            "n_repos": len(pr), "median": float(pr.auc.median()) if len(pr) else None,
            "p25": float(pr.auc.quantile(.25)) if len(pr) else None,
            "p75": float(pr.auc.quantile(.75)) if len(pr) else None}
        pr.to_parquet(OUT / f"pr_vlayer_perrepo_{tier.replace('+','')}.parquet")
        print(tier, result[f"pooled_{tier}"], result[f"per_repo_{tier}"], flush=True)

    # T3 subset: rows with t3_status ok/no_src_files (delta defined; no_src -> 0)
    if "t3_status" in df.columns:
        sub = df[df.t3_status.isin(["ok", "no_src_files"])].copy()
        # rebalance check: keep repos that still have both classes
        keep = sub.groupby("repo_full").pr_merged.transform(lambda s: s.nunique() == 2)
        sub = sub[keep]
        result["t3_subset_n"] = len(sub)
        result["t3_subset_repos"] = int(sub.repo_full.nunique())
        result["t3_subset_merged_rate"] = float(sub.pr_merged.mean())
        g2 = sub["repo_full"].values
        for tier, cols in [("T1@sub", t1_num), ("T1+2@sub", t1_num + t2_cols),
                           ("T1+2+3@sub", t1_num + t2_cols + t3_cols),
                           ("T3only@sub", t3_cols)]:
            X, y, _ = xy(sub, cols, lang_dum)
            result[f"pooled_{tier}"] = pooled_auc(X, y, g2)
            pr = per_repo_auc(sub, cols, lang_dum)
            result[f"per_repo_{tier}"] = {
                "n_repos": len(pr), "median": float(pr.auc.median()) if len(pr) else None,
                "p25": float(pr.auc.quantile(.25)) if len(pr) else None,
                "p75": float(pr.auc.quantile(.75)) if len(pr) else None}
            pr.to_parquet(OUT / f"pr_vlayer_perrepo_{tier.replace('+','').replace('@','_')}.parquet")
            print(tier, result[f"pooled_{tier}"], result[f"per_repo_{tier}"], flush=True)

        # RF feature importances for the full stack on subset
        X, y, names = xy(sub, t1_num + t2_cols + t3_cols, lang_dum)
        rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, n_jobs=16,
                                    random_state=SEED).fit(X, y)
        imp = sorted(zip(names, rf.feature_importances_), key=lambda t: -t[1])[:25]
        result["top_importances_T123"] = [[n, float(v)] for n, v in imp]

    with open(OUT / "pr_vlayer_auc_ladder.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
