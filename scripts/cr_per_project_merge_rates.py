"""Group code_review PRs by project. Report merge rates, distribution,
inspect a few manual examples per representative project.

Goal: understand whether project-baseline merge-rate differences (not code
quality) dominate the signal — and identify a "median merge rate" subset
where within-project signal is meaningful.
"""
from __future__ import annotations
import re
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
import numpy as np

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
DENSE = REPO / "datasets/code-review/code_review_dense_4096tok.csv.gz"


def main():
    print("Loading dense_4096tok (all cols)...")
    df = pd.read_csv(DENSE, nrows=5)
    print(f"  columns: {list(df.columns)}")
    print()

    # Load full with the columns we need
    cols = ["text", "judgement", "language"]
    extra = [c for c in df.columns if c not in cols
             and any(k in c.lower() for k in ("repo", "id", "url", "owner",
                                              "project", "title"))]
    cols += extra
    print(f"  using cols: {cols}")

    df = pd.read_csv(DENSE, usecols=cols)
    print(f"\nfull rows: {len(df)}, label balance y=1: {df['judgement'].mean():.3f}")

    # Extract repo from any URL-bearing column or from paths inside diff text
    # First: extract `paper_id` or any explicit ID column
    print("\n=== Extracting project ID ===")
    if "paper_id" in df.columns:
        # paper_id format: owner/repo#PR_number → owner/repo
        df["repo"] = df["paper_id"].astype(str).str.split("#").str[0]
    elif "url" in df.columns:
        df["repo"] = df["url"].astype(str).str.extract(
            r"github\.com/([^/]+/[^/]+)", expand=False)
    else:
        # Fall back: extract from first diff --git line a/path
        # We expect leading path components to encode repo for monorepos
        # which doesn't help, so try to extract from PR URL string in title
        def find_repo(text):
            m = re.search(r"github\.com/([^/]+/[^/]+)", text)
            if m:
                return m.group(1)
            return None
        df["repo"] = df["text"].astype(str).str[:1000].map(find_repo)

    n_with_repo = df["repo"].notna().sum()
    print(f"  PRs with repo identified: {n_with_repo}/{len(df)} "
          f"({100*n_with_repo/len(df):.1f}%)")

    # Sample 30 to confirm extraction is sane
    if n_with_repo == 0:
        print("ERROR: could not extract repo from any rows")
        # Print column-by-column sample to debug
        for c in cols:
            print(f"\n  Sample {c}:")
            for v in df[c].astype(str).head(3):
                print(f"    {v[:160]}")
        return

    repo_stats = df.groupby("repo", dropna=True).agg(
        n=("judgement", "size"),
        merge_rate=("judgement", "mean"),
    ).sort_values("n", ascending=False)
    print(f"\nunique repos: {len(repo_stats)}")

    print("\n=== Top 20 repos by PR count ===")
    print(f"  {'repo':<55} {'n':>5} {'merge_rate':>10}")
    for repo, row in repo_stats.head(20).iterrows():
        print(f"  {str(repo):<55} {int(row['n']):>5} {row['merge_rate']:>9.2%}")

    # Distribution of repo merge rates (weighted by n)
    print("\n=== Repo merge-rate distribution (weighted by PR count) ===")
    rates = repo_stats["merge_rate"].values
    ns = repo_stats["n"].values
    # Histogram bins
    bins = [0, 0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 1.001]
    bin_labels = ["<10%", "10-30%", "30-50%", "50-70%", "70-85%",
                  "85-95%", "95-100%"]
    weighted_hist = [0] * (len(bins) - 1)
    repo_hist = [0] * (len(bins) - 1)
    for r, n in zip(rates, ns):
        for i in range(len(bins) - 1):
            if bins[i] <= r < bins[i+1]:
                weighted_hist[i] += n
                repo_hist[i] += 1
                break
    print(f"  {'merge_rate bucket':<14} {'# repos':>8} {'# PRs':>8} "
          f"{'PR pct':>7}")
    total_prs = sum(weighted_hist)
    for lbl, nr, np_ in zip(bin_labels, repo_hist, weighted_hist):
        print(f"  {lbl:<14} {nr:>8} {np_:>8} {100*np_/max(total_prs,1):>6.1f}%")

    # === Subset with "median" merge rate: 30-70% ===
    print("\n=== 'Median' subset: repos with merge rate 30-70% ===")
    median_repos = repo_stats[(repo_stats["merge_rate"] >= 0.30)
                              & (repo_stats["merge_rate"] <= 0.70)
                              & (repo_stats["n"] >= 30)].sort_values("n",
                                                                     ascending=False)
    print(f"  # repos with merge_rate in [30%,70%] and n>=30: {len(median_repos)}")
    print(f"  total PRs in this subset: {median_repos['n'].sum()}")
    print(f"\n  Top 15 such repos:")
    print(f"  {'repo':<55} {'n':>5} {'merge_rate':>10}")
    for repo, row in median_repos.head(15).iterrows():
        print(f"  {str(repo):<55} {int(row['n']):>5} {row['merge_rate']:>9.2%}")

    # === Show a few example titles per representative repo ===
    print("\n=== Sample PR titles per representative 'median' repo ===")
    # Pick 4 mid-merge-rate repos that have ≥40 PRs
    sample_repos = median_repos.head(4).index.tolist()
    for repo in sample_repos:
        sub = df[df["repo"] == repo].sample(
            n=min(8, (df["repo"] == repo).sum()), random_state=42)
        print(f"\n  Repo: {repo} (n={int(repo_stats.loc[repo, 'n'])}, "
              f"merge_rate={repo_stats.loc[repo, 'merge_rate']:.2%})")
        for _, row in sub.iterrows():
            t = str(row["text"])
            # Find title line
            m = re.search(r"## PR Title\s*\n(.+?)(?:\n|$)", t)
            title = m.group(1).strip() if m else "?"
            print(f"    [{row['judgement']}] {title[:100]}")

    # === Compare AUC: pooled vs within-project (subset) ===
    print("\n=== Within-project AUC sanity check ===")
    print("Comparing pooled TF-IDF AUC vs within-project (median repos only)")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    import warnings
    warnings.filterwarnings("ignore")

    SEED = 42
    sub = df[df["repo"].isin(median_repos.index)].copy()
    if len(sub) < 200:
        print("  too few PRs in median subset for AUC comparison")
        return

    print(f"  median-subset n: {len(sub)}, y=1 rate: {sub['judgement'].mean():.3f}")

    # Pooled AUC on the median subset
    y = sub["judgement"].astype(int).values
    Xtr_idx, Xte_idx, ytr, yte = train_test_split(
        sub.index, y, test_size=0.20, stratify=y, random_state=SEED)
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                          min_df=5, max_df=0.95, sublinear_tf=True)
    Xt = vec.fit_transform(sub.loc[Xtr_idx, "text"].astype(str).tolist())
    Xe = vec.transform(sub.loc[Xte_idx, "text"].astype(str).tolist())
    lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                            solver="liblinear")
    lr.fit(Xt, ytr)
    p = lr.predict_proba(Xe)[:, 1]
    pooled_auc = roc_auc_score(yte, p)
    print(f"  Pooled TF-IDF AUC on median subset: {pooled_auc:.3f}")

    # Per-repo AUC then averaged (only repos with both classes in test set)
    sub["fold"] = (sub.index % 5)
    per_repo_aucs = []
    for repo in median_repos.head(20).index:
        rsub = sub[sub["repo"] == repo]
        if len(rsub) < 30 or rsub["judgement"].nunique() < 2:
            continue
        yr = rsub["judgement"].astype(int).values
        # Within-project: leave-one-out-fashion, but simpler: 80/20 split
        try:
            Xtr_i, Xte_i, ytr_r, yte_r = train_test_split(
                rsub.index, yr, test_size=0.20, stratify=yr,
                random_state=SEED)
        except ValueError:
            continue
        vec_r = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                                min_df=2, max_df=0.95, sublinear_tf=True)
        try:
            Xt_r = vec_r.fit_transform(rsub.loc[Xtr_i, "text"].astype(str).tolist())
            Xe_r = vec_r.transform(rsub.loc[Xte_i, "text"].astype(str).tolist())
            lr_r = LogisticRegression(C=1.0, max_iter=2000,
                                      class_weight="balanced",
                                      solver="liblinear")
            lr_r.fit(Xt_r, ytr_r)
            pr = lr_r.predict_proba(Xe_r)[:, 1]
            arc = roc_auc_score(yte_r, pr)
            per_repo_aucs.append((repo, arc, len(rsub), rsub["judgement"].mean()))
        except Exception:
            continue

    print("\n  Within-project AUC per repo (top median-merge-rate repos):")
    print(f"  {'repo':<50} {'n':>5} {'mrate':>7} {'AUC':>6}")
    for repo, arc, n, mr in sorted(per_repo_aucs, key=lambda x: -x[1]):
        print(f"  {str(repo):<50} {n:>5} {mr:>6.2%} {arc:>5.3f}")
    if per_repo_aucs:
        mean_within = np.mean([x[1] for x in per_repo_aucs])
        print(f"\n  Mean within-project AUC: {mean_within:.3f}")
        print(f"  Pooled AUC (with project labels visible): {pooled_auc:.3f}")
        print(f"  Drop when project identity is held out: {pooled_auc - mean_within:.3f}")


if __name__ == "__main__":
    main()
