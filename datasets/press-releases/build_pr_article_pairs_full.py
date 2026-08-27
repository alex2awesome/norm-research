#!/usr/bin/env python
"""Build pr_article_pairs_full.jsonl from local cleaned PR dataset.

Steps:
  1. Load positive pairs (judgement=1) from train/eval/test, explode news_article_id lists.
  2. Drop pairs already in pr_article_pairs.jsonl.
  3. Stream news_articles.csv to collect needed article texts.
  4. Apply length / overlap filters.
  5. (Optional) stratified down-sample to ~200K by domain.
  6. Write JSONL with the agreed schema.
"""
from __future__ import annotations
import ast
import csv
import gzip
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research/datasets/press-releases")
MODEL_DIR = ROOT / "press_release_modeling_dataset.csv"  # gz versions are canonical
NEWS_CSV = ROOT / "news_articles.csv"
PREBUILT = ROOT / "pr_article_pairs.jsonl"
OUT_PATH = ROOT / "pr_article_pairs_full.jsonl"
TARGET_MAX = 200_000

PR_MIN, PR_MAX = 200, 12_000
ART_MIN, ART_MAX = 300, 15_000


def load_done_pair_ids() -> set[str]:
    done = set()
    if PREBUILT.exists():
        with open(PREBUILT) as f:
            for line in f:
                done.add(json.loads(line)["pair_id"])
    return done


def load_positive_pairs() -> pd.DataFrame:
    frames = []
    for split in ["train", "eval", "test"]:
        df = pd.read_csv(
            MODEL_DIR / f"{split}.csv.gz",
            usecols=[
                "press_release_id", "text", "press_release_url",
                "press_release_date", "press_release_company",
                "news_article_id", "news_article_domain", "judgement",
            ],
        )
        df = df[df["judgement"] == 1].copy()
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    print(f"[load] positive rows across splits: {len(df):,}", file=sys.stderr)
    # Explode article id list
    df["news_article_id"] = df["news_article_id"].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) and s.startswith("[") else []
    )
    df["news_article_domain"] = df["news_article_domain"].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) and s.startswith("[") else []
    )
    # Pair (article_id, domain) positionally
    rows = []
    for r in df.itertuples(index=False):
        ids = r.news_article_id or []
        doms = r.news_article_domain or []
        for i, aid in enumerate(ids):
            dom = doms[i] if i < len(doms) else (doms[0] if doms else None)
            rows.append({
                "press_release_id": int(r.press_release_id),
                "article_id": int(aid),
                "pr_text": r.text,
                "pr_url": r.press_release_url,
                "pr_date": r.press_release_date,
                "pr_company": r.press_release_company,
                "outlet": dom,
                "split": r.split,
            })
    exploded = pd.DataFrame(rows)
    print(f"[load] exploded pairs: {len(exploded):,}", file=sys.stderr)
    exploded = exploded.drop_duplicates(["press_release_id", "article_id"])
    print(f"[load] after dedup (pr,article): {len(exploded):,}", file=sys.stderr)
    return exploded


def filter_pr_length(df: pd.DataFrame) -> pd.DataFrame:
    L = df["pr_text"].str.len()
    keep = (L >= PR_MIN) & (L <= PR_MAX)
    print(f"[filter] PR length kept {keep.sum():,}/{len(df):,}", file=sys.stderr)
    return df[keep].copy()


def collect_article_texts(needed_ids: set[int]) -> dict[int, dict]:
    """Stream news_articles.csv and pick only rows whose id is needed."""
    print(f"[news] need {len(needed_ids):,} article ids; streaming {NEWS_CSV.name}", file=sys.stderr)
    out: dict[int, dict] = {}
    # Use python's csv module with large field size to handle long article bodies.
    csv.field_size_limit(sys.maxsize)
    with open(NEWS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            if i % 1_000_000 == 0:
                print(f"  scanned {i:,} rows, matched {len(out):,}", file=sys.stderr)
            try:
                aid = int(row["news_article_id"])
            except (TypeError, ValueError):
                continue
            if aid in needed_ids:
                out[aid] = {
                    "text": row["news_article_text"] or "",
                    "url": row["news_article_url"],
                    "date": row["news_article_date"],
                    "source": row["news_article_source"],
                }
                if len(out) == len(needed_ids):
                    print(f"  all matched at row {i:,}", file=sys.stderr)
                    break
    print(f"[news] matched {len(out):,}/{len(needed_ids):,}", file=sys.stderr)
    return out


def overlap_ratio(a: str, b: str) -> float:
    """Cheap token-overlap ratio (article tokens shared with PR / article tokens)."""
    if not a or not b:
        return 0.0
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sb:
        return 0.0
    return len(sa & sb) / len(sb)


def build():
    done = load_done_pair_ids()
    print(f"[done] {len(done):,} pair_ids already extracted", file=sys.stderr)

    pairs = load_positive_pairs()
    pairs["pair_id"] = "pair_" + pairs["press_release_id"].astype(str) + "_" + pairs["article_id"].astype(str)
    pairs = pairs[~pairs["pair_id"].isin(done)].copy()
    print(f"[filter] after dropping done: {len(pairs):,}", file=sys.stderr)

    pairs = filter_pr_length(pairs)

    needed = set(pairs["article_id"].astype(int).tolist())
    art = collect_article_texts(needed)

    # Attach article fields
    pairs["article_text"] = pairs["article_id"].map(lambda i: art.get(i, {}).get("text", ""))
    pairs["article_url"] = pairs["article_id"].map(lambda i: art.get(i, {}).get("url"))
    pairs["article_date"] = pairs["article_id"].map(lambda i: art.get(i, {}).get("date"))
    pairs["article_source"] = pairs["article_id"].map(lambda i: art.get(i, {}).get("source"))

    has_text = pairs["article_text"].str.len() > 0
    print(f"[filter] article text found: {has_text.sum():,}/{len(pairs):,}", file=sys.stderr)
    pairs = pairs[has_text].copy()

    AL = pairs["article_text"].str.len()
    keep = (AL >= ART_MIN) & (AL <= ART_MAX)
    print(f"[filter] article length kept {keep.sum():,}/{len(pairs):,}", file=sys.stderr)
    pairs = pairs[keep].copy()

    # Drop near-duplicate article==PR
    print("[filter] computing PR/article overlap...", file=sys.stderr)
    overlaps = [
        overlap_ratio(pr, art_) for pr, art_ in zip(pairs["pr_text"], pairs["article_text"])
    ]
    pairs["_overlap"] = overlaps
    same = pairs["pr_text"] == pairs["article_text"]
    keep = (~same) & (pairs["_overlap"] < 0.85)
    print(f"[filter] non-duplicate kept {keep.sum():,}/{len(pairs):,}", file=sys.stderr)
    pairs = pairs[keep].drop(columns=["_overlap"]).copy()

    # Stratified downsample by outlet if too many
    if len(pairs) > TARGET_MAX:
        print(f"[sample] downsampling {len(pairs):,} -> ~{TARGET_MAX:,} stratified by outlet", file=sys.stderr)
        counts = pairs["outlet"].fillna("__none__").value_counts()
        frac = TARGET_MAX / len(pairs)
        # Per-outlet quota = max(min(count, ceil(count*frac+1)), ...) ; use groupby sample
        def take(g):
            n = max(1, int(round(len(g) * frac)))
            n = min(n, len(g))
            return g.sample(n=n, random_state=42)
        pairs = pairs.groupby(pairs["outlet"].fillna("__none__"), group_keys=False).apply(take)
        print(f"[sample] after downsample: {len(pairs):,}", file=sys.stderr)

    # Write JSONL
    print(f"[write] {OUT_PATH}", file=sys.stderr)
    domain_counter: Counter[str] = Counter()
    with open(OUT_PATH, "w") as fout:
        for r in pairs.itertuples(index=False):
            pr_text = r.pr_text
            art_text = r.article_text
            text = (
                f"PRESS RELEASE (id={r.press_release_id}):\n{pr_text}\n\n"
                f"---\n\nNEWS ARTICLE COVERING IT (id={r.article_id}):\n{art_text}"
            )
            rec = {
                "pair_id": r.pair_id,
                "press_release_id": int(r.press_release_id),
                "article_id": int(r.article_id),
                "pr_text": pr_text,
                "article_text": art_text,
                "pr_company": r.pr_company,
                "outlet": r.outlet,
                "article_source": r.article_source,
                "article_url": r.article_url,
                "pr_url": r.pr_url,
                "pr_date": r.pr_date,
                "article_date": r.article_date,
                "judgement": 1,
                "text": text,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            domain_counter[str(r.outlet)] += 1

    print(f"\n[done] wrote {sum(domain_counter.values()):,} pairs", file=sys.stderr)
    print("Top 10 outlets:", file=sys.stderr)
    for dom, n in domain_counter.most_common(10):
        print(f"  {dom}: {n:,}", file=sys.stderr)


if __name__ == "__main__":
    build()
