"""
Three-stage corrective re-run of within-repo-balanced GitHub PR experiment.

Inputs (already on sk3):
  outputs/v2_analysis/pr_within_repo_balanced_pool.parquet       (baseline 10,424)
  outputs/v2_analysis/dense_full_diff_single_file.parquet         (140,548 w/ pr_title, scope, dates)
  outputs/v2_analysis/dense_full_diff_bank_scores.parquet         (166 bank features)
  datasets/code-review/code_review_modeling_dataset.csv.gz        (author, author_association)
  datasets/code-review/pr_descriptions.csv.gz                     (pr_body)
  datasets/code-review/pr_merge_status.csv.gz                     (pr_state fallback)

Stages (serial, additive):
  Stage 0: bot-merge label fix + bot-author drop + temporal-gap repo drop, then rebalance
           and re-run Ladder A (GroupKFold 5 by owner_repo) + Ladder B (per-repo 5-fold).
  Stage A: + cheap scope/author/PR-meta controls (additive on Stage-0 pool). Same ladders.
  Stage B: matched pairs (1:1 same-repo on log_loc/n_files/year) on Stage-0 pool.
           Ladder A only on matched corpus (bank + TF-IDF) -> pure code residual.

Outputs:
  outputs/v2_analysis/pr_stage0_pool.parquet
  outputs/v2_analysis/pr_stage0_ladder_a.parquet
  outputs/v2_analysis/pr_stage0_ladder_b.parquet
  outputs/v2_analysis/pr_stageA_features.parquet       (Stage-0 pool augmented with controls)
  outputs/v2_analysis/pr_stageA_ladder_a.parquet
  outputs/v2_analysis/pr_stageA_ladder_b.parquet
  outputs/v2_analysis/pr_stageB_matched.parquet
  outputs/v2_analysis/pr_stageB_ladder_a.parquet
  outputs/v2_analysis/pr_3stage_summary.json
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")
OUT = ROOT / "outputs/v2_analysis"
OUT.mkdir(parents=True, exist_ok=True)

DENSE = OUT / "dense_full_diff_single_file.parquet"
BANK = OUT / "dense_full_diff_bank_scores.parquet"
BASELINE_POOL = OUT / "pr_within_repo_balanced_pool.parquet"

MOD_CSV = ROOT / "datasets/code-review/code_review_modeling_dataset.csv.gz"
DESC_CSV = ROOT / "datasets/code-review/pr_descriptions.csv.gz"
MERGE_CSV = ROOT / "datasets/code-review/pr_merge_status.csv.gz"

RANDOM_STATE = 42
MIN_PER_CLASS_PER_REPO = 10
MIN_BALANCED_PR_PER_REPO_LADDER_B = 50
DIFF_CHAR_CAP = 20_000
TEMPORAL_GAP_MONTHS = 12

T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


# ---------- title leakage strip (carry over from prior run) ----------
LEAK_PATTERNS = [
    r"^\s*\[\s*wip\b[^\]]*\]\s*[:\-]?\s*",
    r"^\s*\[\s*draft\b[^\]]*\]\s*[:\-]?\s*",
    r"^\s*\[\s*dnm\s*\]\s*[:\-]?\s*",
    r"^\s*\[\s*do\s*not\s*merge\s*\]\s*[:\-]?\s*",
    r"^\s*\[\s*do\s*not\s*review\s*\]\s*[:\-]?\s*",
    r"^\s*\[\s*review\s*only\s*\]\s*[:\-]?\s*",
    r"^\s*\[\s*rfc\s*\]\s*[:\-]?\s*",
    r"^\s*\[\s*poc\s*\]\s*[:\-]?\s*",
    r"^\s*wip\s*:\s*",
    r"^\s*draft\s*:\s*",
    r"^\s*dnm\s*:\s*",
    r"^\s*\[?merged by bors\]?\s*[\-:]?\s*",
    r"\bdo\s+not\s+merge\b",
    r"\bdo\s+not\s+review\b",
]
LEAK_COMP = [re.compile(p, re.IGNORECASE) for p in LEAK_PATTERNS]

def strip_title_flags(s):
    if not isinstance(s, str):
        return ""
    out = s
    for _ in range(3):
        prev = out
        for c in LEAK_COMP:
            out = c.sub(" ", out, count=1)
        out = out.strip()
        if out == prev:
            break
    return out


# ---------- Stage-0 patterns ----------
BOT_MERGE_TITLE_PATTERNS = {
    "bors_brackets": re.compile(r"^\s*\[\s*merged\s+by\s+bors\s*\]\s*[\-:]?\s*", re.IGNORECASE),
    "bors_plain": re.compile(r"^\s*merged\s+by\s+bors\b", re.IGNORECASE),
    "bors_bot": re.compile(r"^\s*bors\[bot\]", re.IGNORECASE),
    "auto_merge_brackets": re.compile(r"^\s*\[\s*auto[-\s]?merge\s*\]\s*[\-:]?\s*", re.IGNORECASE),
    "auto_merge_plain": re.compile(r"^\s*auto[-\s]?merge\s*[:\-]\s*", re.IGNORECASE),
    "squashed_and_merged": re.compile(r"^\s*squashed\s+and\s+merged\s+by\b", re.IGNORECASE),
    "mergify_automerge": re.compile(r"^\s*mergify\s+automerge\b", re.IGNORECASE),
    "kx_bot": re.compile(r"^\s*kx-bot\s*:", re.IGNORECASE),
    "automerge_colon": re.compile(r"^\s*automerge\s*:", re.IGNORECASE),
    "merge_pull_request": re.compile(r"^\s*merge\s+pull\s+request\s+#\d+\s+from\b", re.IGNORECASE),
}

KNOWN_BOT_NAMES = [
    "dependabot", "renovate", "github-actions", "bors", "mergify",
    "kx-bot", "step-security-bot",
]
KNOWN_BOT_RE = re.compile(
    r"(?i)^(?:" + "|".join(re.escape(k) for k in KNOWN_BOT_NAMES) + r")(?:\[bot\])?$"
)
BOT_SUFFIX_RE = re.compile(r"\[bot\]$", re.IGNORECASE)


# ---------- AUC helper ----------
def auc_safe(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


# ---------- Ladder runners ----------
def make_tfidf(texts):
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        min_df=10, max_df=0.95,
        max_features=100_000, sublinear_tf=True, lowercase=True,
    )
    return vec, vec.fit_transform(texts)


def lr_clf():
    return LogisticRegression(max_iter=1000, solver="liblinear", C=1.0)


def run_ladder_a(name_prefix, X_dict, y, groups, n_splits=5):
    """X_dict maps cell name -> sparse/dense matrix. All same length, same group."""
    gkf = GroupKFold(n_splits=n_splits)
    folds = list(gkf.split(np.arange(len(y)), y, groups))
    rows = []
    for cell, X in X_dict.items():
        aucs = []
        ts = time.time()
        for k, (tr, va) in enumerate(folds):
            try:
                clf = lr_clf()
                clf.fit(X[tr], y[tr])
                p = clf.predict_proba(X[va])[:, 1]
                aucs.append(auc_safe(y[va], p))
            except Exception as e:
                log(f"  [{name_prefix}/{cell}] fold {k} error: {e}")
                aucs.append(np.nan)
        m = float(np.nanmean(aucs))
        sd = float(np.nanstd(aucs))
        log(f"  [{name_prefix}] {cell}: mean={m:.4f} sd={sd:.4f} ({time.time()-ts:.0f}s)")
        rows.append({"cell": cell, "mean_auc": m, "std_auc": sd,
                     "fold_aucs": [float(x) for x in aucs]})
    return pd.DataFrame(rows)


def run_ladder_b(name_prefix, X, y, groups, min_n=MIN_BALANCED_PR_PER_REPO_LADDER_B):
    """Per-repo 5-fold stratified CV with bank+tfidf x LR. X assumed bank+tfidf."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    df = pd.DataFrame({"repo": groups, "y": y, "i": np.arange(len(y))})
    rows = []
    eligible = []
    for repo, g in df.groupby("repo"):
        if len(g) < min_n:
            continue
        if g["y"].nunique() < 2:
            continue
        eligible.append(repo)
    log(f"  [{name_prefix}] eligible repos (>={min_n} balanced PRs): {len(eligible)}")
    for i, repo in enumerate(eligible):
        idx = df.loc[df["repo"] == repo, "i"].to_numpy()
        Xr = X[idx]
        yr = y[idx]
        aucs = []
        try:
            for tr, va in skf.split(np.arange(len(yr)), yr):
                clf = lr_clf()
                clf.fit(Xr[tr], yr[tr])
                p = clf.predict_proba(Xr[va])[:, 1]
                aucs.append(auc_safe(yr[va], p))
        except Exception as e:
            log(f"  [{name_prefix}/{repo}] error: {e}")
            aucs = [np.nan]
        rows.append({
            "repo": repo, "n_balanced": int(len(idx)),
            "mean_auc": float(np.nanmean(aucs)) if aucs else np.nan,
            "fold_aucs": [float(x) for x in aucs],
        })
        if (i + 1) % 50 == 0:
            log(f"    processed {i+1}/{len(eligible)}")
    return pd.DataFrame(rows)


# ---------- Stage 0 ----------
def stage0_label_fix(df_dense, mod_df):
    """
    Apply three Stage-0 corrections:
      (1) bot-merge title pattern -> flip pr_merged to True
      (2) drop bot-authored PRs
      (3) drop repos with >12mo creation-date gap between class-1 and class-0 medians
    Returns: corrected df + diagnostics dict.
    """
    diag = {}
    # Make sure pr_title is raw string (no strip).
    df = df_dense.copy()
    df["pr_title"] = df["pr_title"].fillna("")
    df["pr_merged"] = df["pr_merged"].astype(bool).astype(int)
    df["repo_full"] = df["owner"] + "/" + df["repo"]

    # (1) bot-merge title patterns
    flip_counts = {}
    initial_rejected = (df["pr_merged"] == 0).sum()
    n_flipped_total = 0
    flip_mask = pd.Series(False, index=df.index)
    for name, pat in BOT_MERGE_TITLE_PATTERNS.items():
        m = df["pr_title"].str.contains(pat, na=False)
        # only count first-match flips (don't double-count if title matches multiple)
        new_m = m & ~flip_mask & (df["pr_merged"] == 0)
        flip_counts[name] = int(new_m.sum())
        flip_mask |= new_m
        n_flipped_total += int(new_m.sum())
    df.loc[flip_mask, "pr_merged"] = 1
    diag["bot_merge_title_flips"] = flip_counts
    diag["bot_merge_total_flipped"] = n_flipped_total
    diag["initial_rejected"] = int(initial_rejected)
    log(f"  Stage 0.1: title-bot flips by pattern: {flip_counts}")
    log(f"  Stage 0.1: {n_flipped_total} PRs flipped from rejected->merged via title regex")

    # (2) drop bot authors
    df = df.merge(mod_df[["owner", "repo", "pr_number", "author", "author_association"]],
                  on=["owner", "repo", "pr_number"], how="left")
    has_author = df["author"].notna()
    is_bot_suffix = df["author"].astype(str).str.contains(BOT_SUFFIX_RE, regex=True, na=False)
    is_known_bot = df["author"].astype(str).str.match(KNOWN_BOT_RE, na=False)
    bot_mask = (is_bot_suffix | is_known_bot) & has_author
    n_bot_drop = int(bot_mask.sum())
    bot_authors = df.loc[bot_mask, "author"].value_counts().head(20).to_dict()
    diag["bot_authors_dropped_count"] = n_bot_drop
    diag["bot_authors_examples"] = bot_authors
    diag["author_coverage_before_drop"] = float(has_author.mean())
    df = df[~bot_mask].copy()
    log(f"  Stage 0.2: dropped {n_bot_drop} bot-authored PRs. Examples: {list(bot_authors.items())[:5]}")

    # (3) temporal-gap repo drop
    df["pr_created_at_dt"] = pd.to_datetime(df["pr_created_at"], errors="coerce", utc=True)
    repo_gaps = {}
    repos_to_drop = []
    for repo, g in df.groupby("repo_full"):
        n1 = (g["pr_merged"] == 1).sum()
        n0 = (g["pr_merged"] == 0).sum()
        if n1 == 0 or n0 == 0:
            continue
        med1 = g.loc[g["pr_merged"] == 1, "pr_created_at_dt"].median()
        med0 = g.loc[g["pr_merged"] == 0, "pr_created_at_dt"].median()
        if pd.isna(med1) or pd.isna(med0):
            continue
        gap_days = abs((med1 - med0).total_seconds() / 86400.0)
        gap_months = gap_days / 30.44
        if gap_months > TEMPORAL_GAP_MONTHS:
            repo_gaps[repo] = round(gap_months, 1)
            repos_to_drop.append(repo)
    diag["temporal_gap_repos_dropped"] = repo_gaps
    diag["temporal_gap_n_repos"] = len(repos_to_drop)
    df_clean = df[~df["repo_full"].isin(repos_to_drop)].copy()
    log(f"  Stage 0.3: dropped {len(repos_to_drop)} repos for temporal gap > {TEMPORAL_GAP_MONTHS}mo")
    if repos_to_drop:
        log(f"    sample: {list(repo_gaps.items())[:10]}")

    return df_clean, diag


def within_repo_balance(df, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    parts = []
    kept = []
    dropped = 0
    for repo, g in df.groupby("repo_full", sort=False):
        n1 = int((g["pr_merged"] == 1).sum())
        n0 = int((g["pr_merged"] == 0).sum())
        if n1 < MIN_PER_CLASS_PER_REPO or n0 < MIN_PER_CLASS_PER_REPO:
            dropped += 1
            continue
        k = min(n1, n0)
        pos = g[g["pr_merged"] == 1]
        neg = g[g["pr_merged"] == 0]
        if len(pos) > k:
            pos = pos.sample(n=k, random_state=seed)
        if len(neg) > k:
            neg = neg.sample(n=k, random_state=seed)
        parts.append(pd.concat([pos, neg]))
        kept.append(repo)
    bal = pd.concat(parts, ignore_index=True).reset_index(drop=True)
    return bal, kept, dropped


# ---------- Stage A: cheap controls ----------
TOP_DIRS = ["src/", "tests/", "test/", "docs/", "vendor/",
            "migrations/", ".github/", "examples/", "lib/", "internal/"]

def stageA_build_features(bal_df, desc_df):
    """Augment bal_df with scope/author/meta features. Returns numpy matrix + feature names."""
    df = bal_df.copy()
    # 1) PR-scope (already present as columns: n_files_in_pr, pr_total_changes, etc.)
    df["log_n_files"] = np.log1p(df["n_files_in_pr"].fillna(0).astype(float))
    df["log_total_changes"] = np.log1p(df["pr_total_changes"].fillna(0).astype(float))
    df["log_additions"] = np.log1p(df["pr_additions"].fillna(0).astype(float))
    df["log_deletions"] = np.log1p(df["pr_deletions"].fillna(0).astype(float))
    df["largest_file_share"] = df["largest_file_share"].fillna(0.0).astype(float)
    df["file_changes_on_largest"] = df["file_changes"].fillna(0).astype(float)

    # 2) Author features
    df["author_association"] = df["author_association"].fillna("MISSING")
    aa_dummies = pd.get_dummies(df["author_association"], prefix="aa")

    # first PR per (repo, author) by creation date
    df["pr_created_at_dt"] = pd.to_datetime(df["pr_created_at"], errors="coerce", utc=True)
    df = df.sort_values(["repo_full", "author", "pr_created_at_dt"])
    df["is_first_pr_by_author_in_repo"] = (
        ~df.duplicated(subset=["repo_full", "author"], keep="first")
    ).astype(int)
    # restore original order via index (don't, we'll keep sorted then merge back)
    df = df.sort_index()

    # 3) PR-meta
    df = df.merge(desc_df, on=["owner", "repo", "pr_number"], how="left")
    df["pr_body"] = df["pr_body"].fillna("")
    df["pr_title_full"] = df["pr_title"].fillna("") + " " + df["pr_body"]
    linked_re = re.compile(r"(?i)(?:#|fix(?:es)?|close[sd]?|resolve[sd]?)\s*#?\d+")
    df["has_linked_issue"] = df["pr_title_full"].str.contains(linked_re, na=False).astype(int)
    df["title_len_chars"] = df["pr_title"].fillna("").str.len()
    df["body_len_chars"] = df["pr_body"].str.len()

    # 4) top-dir
    fp = df["file_path"].fillna("")
    for d in TOP_DIRS:
        key = d.strip("/").replace(".", "_").replace("/", "_") or "root"
        df[f"dir_{key}"] = fp.str.startswith(d).astype(int)

    feat_cols = [
        "log_n_files", "log_total_changes", "log_additions", "log_deletions",
        "largest_file_share", "file_changes_on_largest",
        "is_first_pr_by_author_in_repo",
        "has_linked_issue", "title_len_chars", "body_len_chars",
    ] + [f"dir_{d.strip('/').replace('.', '_').replace('/', '_') or 'root'}" for d in TOP_DIRS] \
      + list(aa_dummies.columns)

    X_df = pd.concat([df[feat_cols[:-len(aa_dummies.columns)]],
                      aa_dummies.reset_index(drop=True)], axis=1)
    # ensure same row order as bal_df keys
    X_df = X_df.reindex(bal_df.index)
    # cast numeric
    X = X_df.to_numpy(dtype=np.float32)
    # scale (StandardScaler -> still numpy)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X = scaler.fit_transform(X).astype(np.float32)
    return X, feat_cols, df  # df has has_linked_issue etc preserved on bal index


# ---------- Stage B: matched pairs ----------
def stageB_matched(bal_df, max_loc_rel=0.2, max_files_abs=2, max_year_abs=1):
    """For each accepted PR, find a same-repo rejected PR matching scope+era.
       Hungarian per repo on L2 distance over (log_loc, n_files, year)."""
    df = bal_df.copy().reset_index(drop=True)
    df["log_loc"] = np.log1p(df["pr_total_changes"].fillna(0).astype(float))
    df["n_files"] = df["n_files_in_pr"].fillna(0).astype(float)
    df["year"] = pd.to_datetime(df["pr_created_at"], errors="coerce", utc=True).dt.year.astype(float)
    df = df.dropna(subset=["year"]).reset_index(drop=True)

    matched_idx = []
    per_repo_stats = []
    for repo, g in df.groupby("repo_full", sort=False):
        pos = g[g["pr_merged"] == 1]
        neg = g[g["pr_merged"] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        p = pos[["log_loc", "n_files", "year"]].to_numpy()
        n = neg[["log_loc", "n_files", "year"]].to_numpy()
        # cost = L2 distance with normalization across the joint repo set
        joint = np.vstack([p, n])
        mu = joint.mean(0)
        sd = joint.std(0) + 1e-6
        pp = (p - mu) / sd
        nn = (n - mu) / sd
        # large infeasibility cost (rectangular Hungarian)
        from scipy.spatial.distance import cdist
        D = cdist(pp, nn, metric="euclidean")
        # mask infeasible matches
        loc_acc = np.exp(p[:, 0:1]) - 1
        loc_rej = (np.exp(n[:, 0:1]) - 1).T
        loc_max = np.maximum(loc_acc, loc_rej)
        loc_max = np.where(loc_max < 1, 1, loc_max)
        loc_ok = (np.abs(loc_acc - loc_rej) <= max_loc_rel * loc_max)
        files_ok = (np.abs(p[:, 1:2] - n[:, 1:2].T) <= max_files_abs)
        year_ok = (np.abs(p[:, 2:3] - n[:, 2:3].T) <= max_year_abs)
        feasible = loc_ok & files_ok & year_ok
        BIG = 1e6
        D2 = np.where(feasible, D, BIG)
        # rectangular Hungarian
        try:
            row_idx, col_idx = linear_sum_assignment(D2)
        except Exception:
            continue
        n_pairs = 0
        for r, c in zip(row_idx, col_idx):
            if D2[r, c] >= BIG:
                continue
            matched_idx.append(pos.index[r])
            matched_idx.append(neg.index[c])
            n_pairs += 1
        per_repo_stats.append({
            "repo": repo, "n_pos": int(len(pos)), "n_neg": int(len(neg)),
            "n_pairs": int(n_pairs),
        })

    matched = df.loc[matched_idx].reset_index(drop=True)
    stats_df = pd.DataFrame(per_repo_stats)
    return matched, stats_df


# ---------- main ----------
def main():
    log("Loading dense parquet ...")
    cols = ["owner", "repo", "pr_number", "paper_id", "pr_title", "file_text",
            "file_path", "file_language", "pr_merged", "pr_created_at",
            "pr_closed_at", "n_files_in_pr", "pr_total_changes", "pr_additions",
            "pr_deletions", "pr_changed_files", "largest_file_share",
            "file_changes", "split"]
    df = pd.read_parquet(DENSE, columns=cols)
    df = df.drop_duplicates(["owner", "repo", "pr_number"]).reset_index(drop=True)
    log(f"  dense unique PRs: {len(df)}")

    log("Loading bank scores ...")
    bank = pd.read_parquet(BANK)
    score_cols = [c for c in bank.columns if c.endswith("_score") and c.startswith("a")]
    applied_cols = [c for c in bank.columns if c.endswith("_applied") and c.startswith("a")]
    bank = bank[["owner", "repo", "pr_number"] + score_cols + applied_cols].drop_duplicates(
        ["owner", "repo", "pr_number"]
    )
    for c in score_cols:
        bank[c] = bank[c].fillna(0.0).astype(np.float32)
    for c in applied_cols:
        bank[c] = bank[c].fillna(0).astype(np.float32)
    log(f"  bank: {bank.shape}, {len(score_cols)} score, {len(applied_cols)} applied")

    log("Loading modeling dataset for author/author_association ...")
    mod = pd.read_csv(MOD_CSV,
                      usecols=["owner", "repo", "pr_number", "author", "author_association"])
    mod = mod.drop_duplicates(["owner", "repo", "pr_number"])
    log(f"  mod unique PRs: {len(mod)}")

    log("Loading PR descriptions ...")
    desc = pd.read_csv(DESC_CSV, usecols=["owner", "repo", "pr_number", "pr_body"])
    desc = desc.drop_duplicates(["owner", "repo", "pr_number"])

    # ============================================================
    # STAGE 0
    # ============================================================
    log("=" * 60)
    log("STAGE 0: bot-merge label fix")
    log("=" * 60)
    df0, diag0 = stage0_label_fix(df, mod)
    log(f"  after Stage 0: {len(df0)} PRs across {df0['repo_full'].nunique()} repos")

    # Now within-repo balance with seed 42
    bal0, kept_repos0, dropped0 = within_repo_balance(df0, seed=RANDOM_STATE)
    log(f"  Stage-0 balanced pool: {len(bal0)} PRs across {len(kept_repos0)} repos "
        f"(dropped {dropped0} repos for <{MIN_PER_CLASS_PER_REPO}/class)")

    # title strip + doc text
    bal0["pr_title_raw"] = bal0["pr_title"]
    bal0["pr_title_stripped"] = bal0["pr_title"].map(strip_title_flags)
    bal0["file_text"] = bal0["file_text"].fillna("").str.slice(0, DIFF_CHAR_CAP)
    bal0["doc_text"] = bal0["pr_title_stripped"] + "\n\n" + bal0["file_text"]

    # merge bank
    bal0 = bal0.merge(bank, on=["owner", "repo", "pr_number"], how="left")
    for c in score_cols:
        bal0[c] = bal0[c].fillna(0.0).astype(np.float32)
    for c in applied_cols:
        bal0[c] = bal0[c].fillna(0).astype(np.float32)

    # save Stage-0 pool meta
    keep_meta = ["owner", "repo", "repo_full", "pr_number", "paper_id",
                 "pr_merged", "pr_title_raw", "pr_title_stripped",
                 "author", "author_association",
                 "file_language", "n_files_in_pr", "pr_total_changes",
                 "pr_additions", "pr_deletions", "pr_changed_files",
                 "pr_created_at", "pr_closed_at", "split"]
    bal0[keep_meta].to_parquet(OUT / "pr_stage0_pool.parquet", index=False)
    log("  wrote pr_stage0_pool.parquet")

    # Build matrices for Stage 0
    log("Stage 0: build TF-IDF...")
    vec0, X_tfidf0 = make_tfidf(bal0["doc_text"].values)
    log(f"  X_tfidf0 {X_tfidf0.shape} nnz={X_tfidf0.nnz}")
    X_bank0_dense = bal0[score_cols + applied_cols].to_numpy(dtype=np.float32)
    X_bank0 = sparse.csr_matrix(X_bank0_dense)
    X_bt0 = sparse.hstack([X_bank0, X_tfidf0], format="csr")
    y0 = bal0["pr_merged"].to_numpy()
    g0 = bal0["repo_full"].to_numpy()

    log("Stage 0: Ladder A ...")
    cells0 = {
        "tfidf_charwb_LR": X_tfidf0,
        "bank_LR": X_bank0,
        "bank_tfidf_LR": X_bt0,
    }
    la0 = run_ladder_a("S0", cells0, y0, g0)
    la0.to_parquet(OUT / "pr_stage0_ladder_a.parquet", index=False)

    log("Stage 0: Ladder B ...")
    lb0 = run_ladder_b("S0", X_bt0, y0, g0)
    lb0.to_parquet(OUT / "pr_stage0_ladder_b.parquet", index=False)

    # ============================================================
    # STAGE A
    # ============================================================
    log("=" * 60)
    log("STAGE A: scope/author/meta controls (additive on Stage-0)")
    log("=" * 60)
    XA_dense, A_feat_names, balA = stageA_build_features(bal0, desc)
    log(f"  Stage-A controls: {XA_dense.shape}, {len(A_feat_names)} features")
    XA = sparse.csr_matrix(XA_dense)

    # Save augmented metadata
    keep_metaA = keep_meta + ["has_linked_issue", "title_len_chars", "body_len_chars",
                              "is_first_pr_by_author_in_repo", "largest_file_share",
                              "file_changes"]
    keep_metaA = [c for c in keep_metaA if c in balA.columns]
    balA[keep_metaA].to_parquet(OUT / "pr_stageA_features.parquet", index=False)

    # cells: Stage-0 cells + controls appended
    X_bank_A = sparse.hstack([X_bank0, XA], format="csr")
    X_tfidf_A = sparse.hstack([X_tfidf0, XA], format="csr")
    X_bt_A = sparse.hstack([X_bt0, XA], format="csr")

    cellsA = {
        "tfidf_charwb_LR_plusA": X_tfidf_A,
        "bank_LR_plusA": X_bank_A,
        "bank_tfidf_LR_plusA": X_bt_A,
        "controlsA_only_LR": XA,
    }
    log("Stage A: Ladder A ...")
    laA = run_ladder_a("SA", cellsA, y0, g0)
    laA.to_parquet(OUT / "pr_stageA_ladder_a.parquet", index=False)

    log("Stage A: Ladder B ...")
    lbA = run_ladder_b("SA", X_bt_A, y0, g0)
    lbA.to_parquet(OUT / "pr_stageA_ladder_b.parquet", index=False)

    # Top coefs on full pool (no CV) for interpretation
    log("Stage A: fit full-pool LR for top coefs ...")
    clf_full = LogisticRegression(max_iter=1000, solver="liblinear", C=1.0)
    clf_full.fit(X_bt_A, y0)
    coefs = clf_full.coef_[0]
    # The Stage-A controls are the LAST len(A_feat_names) features
    A_start = X_bt_A.shape[1] - XA.shape[1]
    A_coefs = coefs[A_start:]
    top_idx = np.argsort(-np.abs(A_coefs))[:5]
    top_A_feats = [{"feature": A_feat_names[i], "coef": float(A_coefs[i])}
                   for i in top_idx]
    log(f"  Top-5 Stage-A features: {top_A_feats}")

    # ============================================================
    # STAGE B
    # ============================================================
    log("=" * 60)
    log("STAGE B: matched pairs (on Stage-0 pool, no controls)")
    log("=" * 60)
    matched, match_stats = stageB_matched(bal0)
    log(f"  matched corpus: {len(matched)} rows ({len(matched)//2} pairs across "
        f"{match_stats['n_pairs'].astype(bool).sum()} repos with >=1 match)")
    matched.to_parquet(OUT / "pr_stageB_matched.parquet", index=False)

    # rebuild matrices on matched (use same TF-IDF vocabulary as Stage 0 for stability,
    # but refit since the corpus is much smaller -> stronger signal locally).
    log("Stage B: build TF-IDF on matched corpus ...")
    vecB, X_tfidfB = make_tfidf(matched["doc_text"].values if "doc_text" in matched.columns
                                else (matched["pr_title_stripped"].fillna("") + "\n\n" + matched["file_text"].fillna("").str.slice(0, DIFF_CHAR_CAP)).values)
    X_bankB_dense = matched[score_cols + applied_cols].to_numpy(dtype=np.float32)
    X_bankB = sparse.csr_matrix(X_bankB_dense)
    X_btB = sparse.hstack([X_bankB, X_tfidfB], format="csr")
    yB = matched["pr_merged"].to_numpy()
    gB = matched["repo_full"].to_numpy()
    log(f"  X_btB: {X_btB.shape}, label rate {yB.mean():.3f}")

    cellsB = {
        "tfidf_charwb_LR_matched": X_tfidfB,
        "bank_LR_matched": X_bankB,
        "bank_tfidf_LR_matched": X_btB,
    }
    log("Stage B: Ladder A on matched corpus ...")
    laB = run_ladder_a("SB", cellsB, yB, gB)
    laB.to_parquet(OUT / "pr_stageB_ladder_a.parquet", index=False)

    # Per-repo Ladder B comparison: get spacemesh + sonarqube for baseline/Stage0/StageA
    prev_top5 = ["spacemeshos/go-spacemesh", "TheAlgorithms/Python",
                 "apache/accumulo", "indico/indico", "apache/airflow"]
    prev_bot5 = ["SonarSource/sonarqube", "SpongePowered/SpongeAPI",
                 "SkriptLang/Skript", "mulesoft/mule", "red-hat-storage/ocs-ci"]
    def lookup(df_lb, repos):
        rows = []
        for r in repos:
            sub = df_lb[df_lb["repo"] == r]
            if len(sub) == 0:
                rows.append({"repo": r, "n_balanced": None, "mean_auc": None})
            else:
                rows.append({"repo": r,
                             "n_balanced": int(sub.iloc[0]["n_balanced"]),
                             "mean_auc": float(sub.iloc[0]["mean_auc"])})
        return rows

    summary = {
        "timing_seconds": time.time() - T0,
        "comparison_table": {
            "baseline": {
                "pool_size": 10424, "ladder_b_repos": 39,
                "ladder_a_best": 0.541, "ladder_b_median": 0.607,
            },
            "stage0": {
                "pool_size": int(len(bal0)),
                "ladder_b_repos": int(((lb0["n_balanced"] >= MIN_BALANCED_PR_PER_REPO_LADDER_B) & lb0["mean_auc"].notna()).sum()),
                "ladder_a_best": float(la0["mean_auc"].max()),
                "ladder_b_median": float(lb0["mean_auc"].median()),
            },
            "stageA": {
                "pool_size": int(len(bal0)),
                "ladder_b_repos": int(((lbA["n_balanced"] >= MIN_BALANCED_PR_PER_REPO_LADDER_B) & lbA["mean_auc"].notna()).sum()),
                "ladder_a_best": float(laA["mean_auc"].max()),
                "ladder_b_median": float(lbA["mean_auc"].median()),
            },
            "stageB": {
                "pool_size": int(len(matched)),
                "ladder_b_repos": None,
                "ladder_a_best": float(laB["mean_auc"].max()),
                "ladder_b_median": None,
            },
        },
        "stage0_diag": diag0,
        "stage0_pool_size": int(len(bal0)),
        "stage0_n_repos_kept": int(len(kept_repos0)),
        "stage0_n_repos_dropped_minclass": int(dropped0),
        "stage0_ladder_a_cells": la0.to_dict(orient="records"),
        "stageA_ladder_a_cells": laA.to_dict(orient="records"),
        "stageA_top5_features": top_A_feats,
        "stageA_n_features": len(A_feat_names),
        "stageA_feature_names": A_feat_names,
        "stageB_ladder_a_cells": laB.to_dict(orient="records"),
        "stageB_match_stats": {
            "total_matched_rows": int(len(matched)),
            "total_pairs": int(len(matched) // 2),
            "n_repos_with_matches": int((match_stats["n_pairs"] > 0).sum()) if len(match_stats) else 0,
            "match_rate_pairs_per_repo_mean": float(match_stats["n_pairs"].mean()) if len(match_stats) else 0,
            "match_rate_pairs_per_repo_median": float(match_stats["n_pairs"].median()) if len(match_stats) else 0,
            "per_repo_stats_top10": match_stats.sort_values("n_pairs", ascending=False).head(10).to_dict(orient="records") if len(match_stats) else [],
        },
        "per_repo_changes": {
            "prev_top5_baseline_aucs": {
                "spacemeshos/go-spacemesh": 0.8052,
                "TheAlgorithms/Python": 0.7602,
                "apache/accumulo": 0.7216,
                "indico/indico": 0.704,
                "apache/airflow": 0.6985,
            },
            "prev_bot5_baseline_aucs": {
                "SonarSource/sonarqube": 0.2938,
                "SpongePowered/SpongeAPI": 0.3511,
                "SkriptLang/Skript": 0.4104,
                "mulesoft/mule": 0.4493,
                "red-hat-storage/ocs-ci": 0.5095,
            },
            "stage0_prev_top5": lookup(lb0, prev_top5),
            "stage0_prev_bot5": lookup(lb0, prev_bot5),
            "stageA_prev_top5": lookup(lbA, prev_top5),
            "stageA_prev_bot5": lookup(lbA, prev_bot5),
        },
        "stage0_ladder_b_top5": lb0.sort_values("mean_auc", ascending=False).head(5).to_dict(orient="records"),
        "stage0_ladder_b_bot5": lb0.sort_values("mean_auc", ascending=True).head(5).to_dict(orient="records"),
        "stageA_ladder_b_top5": lbA.sort_values("mean_auc", ascending=False).head(5).to_dict(orient="records"),
        "stageA_ladder_b_bot5": lbA.sort_values("mean_auc", ascending=True).head(5).to_dict(orient="records"),
    }
    with open(OUT / "pr_3stage_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"Wrote pr_3stage_summary.json")
    log("DONE.")


if __name__ == "__main__":
    main()
