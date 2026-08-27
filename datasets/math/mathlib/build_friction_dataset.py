#!/usr/bin/env python3
"""Build the mathlib review-friction dataset from the GraphQL PR fetch.

Input: pr_reviews_mathlib4.jsonl (37,249 closed PRs, full review detail).
Label (README §3a + label-mechanics findings): among MERGED PRs,
  y = 1  <=>  zero review threads (frictionless merge)
  y = 0  <=>  >=1 review thread (community articulated objections)
CHANGES_REQUESTED is unusable in mathlib (99.4% zero) — threads carry review.

Era-aware merge detection: mergedAt (pre-Bors era) OR '[Merged by Bors]'
title prefix OR ready-to-merge label. The prefix is then STRIPPED from titles
(strip-don't-drop, per github-pr-within-repo-balance-2026-06-05).

Confound handling (measured 2026-06-10 on the first 8K): bin-match classes
within (size-quartile x title-prefix x year x author-association) cells.

Output: friction_dataset.csv.gz (balanced) + friction_full.csv.gz (all merged,
unbalanced, for regression targets) + manifest.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

BORS_PREFIX = re.compile(r"^(\[Merged by Bors\]\s*-\s*)+")  # + : re-landed PRs carry a doubled prefix (18 rows, audit 2026-06-10)
CONV_PREFIX = re.compile(
    r"^(feat|fix|chore|refactor|doc|docs|style|perf|test|ci|golf|move|"
    r"split|delete|deprecate)\b")


def load(path: str) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            labels = {l["name"] for l in d["labels"]["nodes"]}
            merged = (bool(d["mergedAt"])
                      or d["title"].startswith("[Merged by Bors]")
                      or "ready-to-merge" in labels)
            reviews = [r for r in d["reviews"]["nodes"] if r]
            first_commit = (d["commits"]["nodes"] or [{}])[0].get("commit", {})
            title = BORS_PREFIX.sub("", d["title"])
            m = CONV_PREFIX.match(title)
            created = d["createdAt"]
            closed = d["closedAt"]
            days_open = None
            if closed:
                t0 = datetime.fromisoformat(created.rstrip("Z"))
                t1 = datetime.fromisoformat(closed.rstrip("Z"))
                days_open = round((t1 - t0).total_seconds() / 86400, 3)
            rows.append(dict(
                number=d["number"],
                title=title,
                merged=merged,
                state=d["state"],
                created_at=created, closed_at=closed,
                days_open=days_open,
                year=int(created[:4]),
                author=(d["author"] or {}).get("login"),
                author_association=d["authorAssociation"],
                additions=d["additions"] or 0,
                deletions=d["deletions"] or 0,
                changed_files=d["changedFiles"] or 0,
                size=(d["additions"] or 0) + (d["deletions"] or 0),
                conv_prefix=m.group(1) if m else "OTHER",
                labels="|".join(sorted(labels)),
                easy="easy" in labels,
                new_contributor="new-contributor" in labels,
                llm_generated="LLM-generated" in labels,
                topic_labels="|".join(sorted(l for l in labels
                                             if l.startswith("t-"))),
                n_reviews=d["reviews"]["totalCount"],
                n_changes_requested=sum(
                    1 for r in reviews if r["state"] == "CHANGES_REQUESTED"),
                n_review_threads=d["reviewThreads"]["totalCount"],
                n_issue_comments=d["comments"]["totalCount"],
                n_force_pushes=d["timelineItems"]["totalCount"],
                n_commits=d["commits"]["totalCount"],
                first_commit_oid=first_commit.get("oid"),
                head_oid=d["headRefOid"],
                merge_commit_oid=(d["mergeCommit"] or {}).get("oid"),
            ))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="pr_reviews_mathlib4.jsonl")
    ap.add_argument("--out-balanced", default="friction_dataset.csv.gz")
    ap.add_argument("--out-full", default="friction_full.csv.gz")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df = load(args.input)
    manifest = {"input_rows": len(df), "build_date": str(datetime.now())}
    merged = df[df.merged].copy()
    manifest["merged_rows"] = len(merged)
    merged["y"] = (merged.n_review_threads == 0).astype(int)
    manifest["p_frictionless"] = float(merged.y.mean())

    # full (unbalanced) file with continuous targets for regression variants
    merged.to_csv(args.out_full, index=False, compression="gzip")

    # bin-match within (size-quartile x conv_prefix x year x association)
    edges = np.quantile(merged["size"], [0, .25, .5, .75, 1.0])
    edges[-1] += 1
    merged["size_bin"] = np.searchsorted(edges[1:-1], merged["size"])
    # primary math-area label in the key: audit found area imbalance
    # (t-combinatorics P(y=1)=0.36 vs t-category-theory 0.56) leaking area
    # tokens into title models
    merged["primary_topic"] = merged.topic_labels.fillna("").map(
        lambda s: s.split("|")[0] if s else "none")
    merged["cell"] = list(zip(merged.size_bin, merged.conv_prefix,
                              merged.year, merged.author_association,
                              merged.primary_topic))
    kept, stats = [], Counter()
    for cell, g in merged.groupby("cell"):
        pos, neg = g[g.y == 1], g[g.y == 0]
        n = min(len(pos), len(neg))
        if n == 0:
            stats["dropped_cells"] += 1
            stats["dropped_rows"] += len(g)
            continue
        kept += list(rng.choice(pos.index, n, replace=False))
        kept += list(rng.choice(neg.index, n, replace=False))
        stats["kept_cells"] += 1
    bal = merged.loc[kept].sample(frac=1.0, random_state=args.seed)
    # split by PR number hash (stable, no grouping needed — PRs independent)
    import hashlib
    def split_of(n):
        h = int(hashlib.md5(str(n).encode()).hexdigest(), 16) % 100
        return "train" if h < 80 else ("eval" if h < 90 else "test")
    bal["split"] = bal.number.map(split_of)
    bal.drop(columns=["cell"]).to_csv(args.out_balanced, index=False,
                                      compression="gzip")
    manifest["cells"] = dict(stats)
    manifest["balanced_rows"] = len(bal)
    manifest["balanced_pos_rate"] = float(bal.y.mean())
    manifest["year_dist"] = {int(y): int(c) for y, c in
                             bal.year.value_counts().sort_index().items()}
    manifest["splits"] = {s: int(c) for s, c in bal.split.value_counts().items()}
    json.dump(manifest, open(args.out_balanced.replace(".csv.gz",
                                                       ".manifest.json"), "w"),
              indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
