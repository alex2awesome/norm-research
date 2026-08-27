"""Audit the dense_4096tok dataset for spurious features the dense model
might be exploiting.

Checks:
  1. Label balance (per-language, full set, v2 subset).
  2. Title-marker leakage: are merge/reject labels predictable from title
     keywords like [WIP], [DO NOT MERGE], conventional-commit prefixes,
     bug-tracker IDs?
  3. Description-length leakage: do longer/shorter descriptions correlate
     with label?
  4. Per-section TF-IDF + LR baseline AUC:
        - Title only
        - Description only
        - Diff only
        - Full text
  5. Top tokens by coefficient — what's the bag-of-words actually learning?
  6. Per-author / per-repo signal if we can extract that from text.
"""
from __future__ import annotations
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
DENSE = REPO / "datasets/code-review/code_review_dense_4096tok.csv.gz"
SEED = 42


def extract_sections(text: str):
    """Split dense text into title / description / diff."""
    title_m = re.search(r"## PR Title\s*\n(.*?)(?=\n##|\Z)", text, re.DOTALL)
    desc_m = re.search(r"## PR (?:Description|Body)\s*\n(.*?)(?=\n##|\Z)",
                       text, re.DOTALL)
    diff_idx = text.find("diff --git")
    diff = text[diff_idx:] if diff_idx != -1 else ""
    return ((title_m.group(1).strip() if title_m else ""),
            (desc_m.group(1).strip() if desc_m else ""),
            diff)


def main():
    print("Loading dense_4096tok (full)...")
    df = pd.read_csv(DENSE, usecols=["text", "judgement", "language"])
    print(f"  full rows: {len(df)}, label balance y=1: {df['judgement'].mean():.3f}")
    print(f"  full row count by language (top 10):")
    print(df["language"].value_counts().head(10).to_string())
    print()
    print("=== per-language label rate (top 12 by count) ===")
    lang_stats = df.groupby("language").agg(
        n=("judgement", "size"), p1=("judgement", "mean")
    ).sort_values("n", ascending=False).head(12)
    print(lang_stats.to_string())

    # Slim to a representative working slice
    print("\nSampling 20K rows for the analysis...")
    df = df.sample(n=min(20000, len(df)), random_state=SEED).reset_index(drop=True)
    print(f"  sampled: {len(df)}, label balance y=1: {df['judgement'].mean():.3f}")

    print("\nExtracting title / description / diff sections...")
    sects = df["text"].map(extract_sections)
    df["title"] = sects.map(lambda s: s[0])
    df["desc"] = sects.map(lambda s: s[1])
    df["diff_text"] = sects.map(lambda s: s[2])
    df["title_len"] = df["title"].str.len()
    df["desc_len"] = df["desc"].str.len()
    df["diff_len"] = df["diff_text"].str.len()
    print(f"  title/desc/diff median lens: {df['title_len'].median():.0f}/"
          f"{df['desc_len'].median():.0f}/{df['diff_len'].median():.0f}")

    # === Title-marker leakage ===
    print("\n=== TITLE-MARKER LEAKAGE ===")
    patterns = {
        "[wip]": r"\bwip\b|\[wip\]",
        "[draft]": r"\[draft\]|\bdraft\b",
        "do not merge": r"do[ -]?not[ -]?merge|\bdnm\b",
        "rfc": r"\brfc\b",
        "[wip-do-not-merge]": r"wip.*not.*merge|not.*merge.*wip",
        "fixes #N": r"fix(?:es)?\s+#\d+",
        "issue link": r"https?://.*?/issues?/\d+",
        "JIRA-style": r"\b[A-Z]+-\d{2,}\b",
        "conv-commit (feat:/fix:)": r"^(feat|fix|chore|docs|refactor|test|build|ci|perf|style)(\([^)]+\))?:",
        "[bracket prefix]": r"^\[[^\]]+\]",
    }
    title_lower = df["title"].str.lower()
    desc_lower = df["desc"].str.lower()
    full_lower = df["text"].str.lower()
    print(f"  {'marker':<35} {'hits':>6} {'pct':>5} {'p(y=1|hit)':>12} "
          f"{'p(y=1|no)':>10} {'|delta|':>8}")
    for name, pat in patterns.items():
        hit = title_lower.str.contains(pat, regex=True, na=False)
        if hit.sum() == 0:
            continue
        py_hit = df.loc[hit, "judgement"].mean()
        py_no = df.loc[~hit, "judgement"].mean()
        print(f"  {name:<35} {hit.sum():>6} {hit.mean() * 100:>4.1f}% "
              f"{py_hit:>11.3f} {py_no:>10.3f} {abs(py_hit - py_no):>7.3f}")

    # === TF-IDF baselines by section ===
    print("\n=== TF-IDF + LR baselines (held-out 20%) ===")
    y = df["judgement"].astype(int).values
    Xtr, Xte, ytr, yte = train_test_split(
        df.index, y, test_size=0.20, stratify=y, random_state=SEED)
    for section_name, col in [("title", "title"), ("description", "desc"),
                              ("diff only", "diff_text"),
                              ("full text", "text")]:
        texts_tr = df.loc[Xtr, col].fillna("").astype(str).tolist()
        texts_te = df.loc[Xte, col].fillna("").astype(str).tolist()
        if max(len(t) for t in texts_tr) < 5:
            continue
        # Conservative TF-IDF
        vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                              min_df=5, max_df=0.95, sublinear_tf=True)
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
        # top features for the merged direction
        coef = lr.coef_[0]
        names = np.array(vec.get_feature_names_out())
        top_pos = names[np.argsort(-coef)[:6]]
        top_neg = names[np.argsort(coef)[:6]]
        print(f"  {section_name:<14}  n_train={len(texts_tr):>5}  "
              f"AUC={auc:.3f}")
        print(f"     ↑merged: {list(top_pos)}")
        print(f"     ↓merged: {list(top_neg)}")

    # === Trivial scalar features ===
    print("\n=== Trivial scalar features AUC ===")
    for col in ["title_len", "desc_len", "diff_len"]:
        try:
            auc = roc_auc_score(yte, df.loc[Xte, col].fillna(0).values)
            print(f"  {col:<15} AUC={auc:.3f}  "
                  f"(inverted={1 - auc:.3f})")
        except Exception as e:
            print(f"  {col:<15} failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
