"""Deeper cleaning audit for code_review_dense_4096tok.

Beyond title markers and project bimodality, check for:
  (1) Bot-merge mislabels: PRs whose title contains 'merged by bors' /
      'dependabot' / 'renovate' / 'github-actions' — likely mis-labeled.
  (2) Per-language merge-rate skew.
  (3) "Workable" subset candidates at different filter cutoffs:
        - merge_rate ∈ [40%, 90%], n≥50  (proposed default)
        - merge_rate ∈ [30%, 95%], n≥30  (more permissive)
        - merge_rate ∈ [50%, 80%], n≥100 (strictest)
  (4) After applying each filter, distribution & label balance.
  (5) Within-filtered-subset bag-of-words AUC vs pooled to confirm
      that meaningful per-PR signal remains.
"""
from __future__ import annotations
import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
DENSE = REPO / "datasets/code-review/code_review_dense_4096tok.csv.gz"
SEED = 42


def main():
    print("Loading dense_4096tok...")
    df = pd.read_csv(DENSE, usecols=["text", "judgement", "language", "paper_id",
                                     "num_files", "num_comments",
                                     "pr_additions", "pr_deletions"])
    df["repo"] = df["paper_id"].astype(str).str.split("#").str[0]
    df["title"] = df["text"].str.extract(r"## PR Title\s*\n(.+?)(?:\n|$)",
                                          expand=False)
    df["title"] = df["title"].fillna("").astype(str)
    print(f"  rows: {len(df)}, y=1: {df['judgement'].mean():.3f}, "
          f"unique repos: {df['repo'].nunique()}")

    # === (1) Bot-merge mislabel scan ===
    print("\n=== Bot-merge mislabel scan ===")
    bot_patterns = {
        "merged by bors": r"\bmerged by bors\b",
        "dependabot bot": r"dependabot\[bot\]|^\s*dependabot:",
        "renovate bot": r"renovate\[bot\]|^\s*renovate:",
        "github-actions bot": r"github-actions\[bot\]",
        "automated PR": r"\bautomated (pr|update|bump)\b",
        "bump deps": r"^bump\b|^chore\(deps?\):",
    }
    title_l = df["title"].str.lower()
    print(f"  {'pattern':<25} {'hits':>5} {'y=1|hit':>9} {'y=1|no':>9} {'delta':>7} {'expected'}")
    for name, pat in bot_patterns.items():
        hit = title_l.str.contains(pat, regex=True, na=False)
        if hit.sum() < 3:
            continue
        ph = df.loc[hit, "judgement"].mean()
        pn = df.loc[~hit, "judgement"].mean()
        expected = "should be ~1 if bot-merged"
        print(f"  {name:<25} {hit.sum():>5} {ph:>8.3f} {pn:>8.3f} "
              f"{ph - pn:>+6.3f}  {expected}")

    # === (2) Per-language ===
    print("\n=== Per-language merge rate ===")
    lang = df.groupby("language", dropna=True).agg(
        n=("judgement", "size"), p1=("judgement", "mean")
    ).sort_values("n", ascending=False)
    print(lang.head(15).to_string())

    # === (3) Filter candidates ===
    repo_stats = df.groupby("repo").agg(
        n=("judgement", "size"), p1=("judgement", "mean")
    )
    filters = {
        "merge_rate [40%,90%], n>=50": (0.40, 0.90, 50),
        "merge_rate [30%,95%], n>=30": (0.30, 0.95, 30),
        "merge_rate [50%,80%], n>=100": (0.50, 0.80, 100),
        "merge_rate [40%,90%], n>=30": (0.40, 0.90, 30),
        "merge_rate [35%,85%], n>=50": (0.35, 0.85, 50),
    }
    print("\n=== Subset candidates ===")
    print(f"  {'filter':<35} {'#repos':>7} {'#PRs':>7} {'y=1':>6}")
    subsets = {}
    for name, (lo, hi, nmin) in filters.items():
        mask = (repo_stats["p1"] >= lo) & (repo_stats["p1"] <= hi) & (repo_stats["n"] >= nmin)
        repos = repo_stats[mask].index
        sub = df[df["repo"].isin(repos)]
        subsets[name] = sub
        print(f"  {name:<35} {len(repos):>7} {len(sub):>7} {sub['judgement'].mean():>5.3f}")

    # === (4) Apply bot-merge filter as well ===
    print("\n=== After also stripping bot/status markers ===")
    strip_pat = re.compile(
        r"\bwip\b|\[wip\]|\bdraft\b|\[draft\]|do[ -]?not[ -]?merge|\[dnm\]|"
        r"\bblocked\b|\[rfc\]|\bmerged by bors\b|dependabot|renovate",
        re.IGNORECASE,
    )
    leak_mask = df["title"].str.contains(strip_pat, regex=True, na=False)
    print(f"  PRs with leak marker in title: {leak_mask.sum()} "
          f"({100*leak_mask.mean():.2f}%)")
    df_clean = df[~leak_mask]
    print(f"  After strip: {len(df_clean)} PRs, y=1: {df_clean['judgement'].mean():.3f}")

    # Combined: median-merge-rate repos AND leak-stripped
    base_filter = subsets["merge_rate [40%,90%], n>=50"]
    n_with_leak = len(base_filter)
    n_no_leak = len(base_filter[~base_filter["title"].str.contains(
        strip_pat, regex=True, na=False)])
    print(f"\n  Default filter + leak-strip:")
    print(f"  PRs: {n_no_leak} ({100*n_no_leak/len(df):.1f}% of original)")

    # === (5) Within-subset signal check ===
    print("\n=== Bag-of-words AUC on best subset ===")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    sub = subsets["merge_rate [40%,90%], n>=50"]
    sub = sub[~sub["title"].str.contains(strip_pat, regex=True, na=False)]
    print(f"  Subset for eval: {len(sub)} PRs, y=1 rate: {sub['judgement'].mean():.3f}")

    y = sub["judgement"].astype(int).values
    if len(y) < 200:
        print("  too small")
        return
    try:
        Xtr, Xte, ytr, yte = train_test_split(
            sub.index, y, test_size=0.20, stratify=y, random_state=SEED)
    except ValueError:
        print("  cannot stratify")
        return
    for section_name, regex in [
        ("title only", r"## PR Title\s*\n(.+?)(?:\n|$)"),
        ("description only", r"## (?:Description|Body)\s*\n(.+?)(?=\n##|\Z)"),
        ("diff only", "DIFF"),  # special
        ("full text", "FULL"),
    ]:
        if regex == "DIFF":
            texts = sub["text"].apply(
                lambda t: t[t.find("diff --git"):] if "diff --git" in t else "")
        elif regex == "FULL":
            texts = sub["text"]
        else:
            texts = sub["text"].str.extract(regex, expand=False,
                                            flags=re.DOTALL).fillna("")
        texts_tr = texts.loc[Xtr].astype(str).tolist()
        texts_te = texts.loc[Xte].astype(str).tolist()
        if max(len(t) for t in texts_tr) < 10:
            continue
        vec = TfidfVectorizer(max_features=15000, ngram_range=(1, 2),
                              min_df=3, max_df=0.95, sublinear_tf=True)
        try:
            Xt = vec.fit_transform(texts_tr)
        except ValueError:
            continue
        Xe = vec.transform(texts_te)
        lr = LogisticRegression(C=1.0, max_iter=2000,
                                class_weight="balanced", solver="liblinear")
        lr.fit(Xt, ytr)
        p = lr.predict_proba(Xe)[:, 1]
        auc = roc_auc_score(yte, p)
        coef = lr.coef_[0]
        names = np.array(vec.get_feature_names_out())
        top_pos = list(names[np.argsort(-coef)[:5]])
        top_neg = list(names[np.argsort(coef)[:5]])
        print(f"  {section_name:<18} AUC={auc:.3f}  "
              f"↑merge:{top_pos}  ↓:{top_neg}")

    print("\nDone.")


if __name__ == "__main__":
    main()
