"""
Rebuild the (wiki problem page -> forum link) mapping that build_priority_queue.py
extracted but did not persist.

For every kind=="problem" page in wiki/*.jsonl.gz, emit one row per extracted
forum link:
    contest, year, variant, problem_n, link_kind ("topic"|"post"), link_id

Post-id links are NOT resolved here (no network). They are resolved on sk3
against the crawl shards, where every crawled post carries (post_id, topic_id).

Output: wiki/wiki_forum_links.csv
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
WIKI = REPO / "wiki"

TOPIC_PATS = [
    re.compile(r"/community/[a-z]?\d*h(\d+)", re.I),
    re.compile(r"viewtopic\.php\?[^\s\]\|<>\"'\)}]*?[?&]?t=(\d+)", re.I),
]
POST_PATS = [
    re.compile(r"/community/p(\d+)", re.I),
    re.compile(r"viewtopic\.php\?[^\s\]\|<>\"'\)}]*?[?&]p=(\d+)", re.I),
]


def main() -> int:
    rows = []
    n_pages = n_problem_pages = 0
    for shard in sorted(WIKI.glob("*.jsonl.gz")):
        for line in gzip.open(shard, "rt"):
            rec = json.loads(line)
            n_pages += 1
            if rec.get("kind") != "problem" or rec.get("status") != "ok":
                continue
            n_problem_pages += 1
            wt = rec.get("wikitext") or ""
            base = dict(
                contest=rec["contest"],
                year=int(rec["year"]),
                variant=rec["variant"],
                problem_n=int(rec["problem"]),
            )
            topic_ids, post_ids = set(), set()
            for pat in TOPIC_PATS:
                topic_ids.update(int(m) for m in pat.findall(wt))
            for pat in POST_PATS:
                post_ids.update(int(m) for m in pat.findall(wt))
            for t in sorted(topic_ids):
                rows.append({**base, "link_kind": "topic", "link_id": t})
            for p in sorted(post_ids):
                rows.append({**base, "link_kind": "post", "link_id": p})

    df = pd.DataFrame(rows).drop_duplicates()
    out = WIKI / "wiki_forum_links.csv"
    df.to_csv(out, index=False)

    probs = pd.read_parquet(WIKI / "problems.parquet")
    key = ["contest", "year", "variant", "problem_n"]
    linked = df[key].drop_duplicates()
    merged = probs[key].merge(linked.assign(has_link=1), on=key, how="left")
    print(f"pages scanned: {n_pages}  problem pages: {n_problem_pages}")
    print(f"link rows: {len(df)}  (topic: {(df.link_kind=='topic').sum()}, "
          f"post: {(df.link_kind=='post').sum()})")
    print(f"unique topic ids: {df.loc[df.link_kind=='topic','link_id'].nunique()}")
    print(f"unique post ids:  {df.loc[df.link_kind=='post','link_id'].nunique()}")
    print(f"problems in problems.parquet with >=1 link of any kind: "
          f"{int(merged.has_link.fillna(0).sum())} / {len(probs)}")
    print(merged.assign(has_link=merged.has_link.fillna(0))
          .groupby(probs.contest).has_link.mean().round(3).to_string())
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
