"""Merge per-contest jsonl files into contest_corpus.jsonl.

Applies the shared normalization (normalize.normalize_text) UNIFORMLY to the
title, author, and text fields of every row regardless of source, so that no
source-specific typography survives into the modeling fields. raw_text keeps
the unnormalized extraction.

Run after: scrape_wergle.py, scrape_hull.py, scrape_erma.py
"""
import json
import os
from collections import Counter

from fetch_util import STAGING
from normalize import normalize_text

SOURCES = ["wergle.jsonl", "hull.jsonl", "erma.jsonl"]
OUT = os.path.join(STAGING, "contest_corpus.jsonl")

FIELDS = ["contest", "cycle", "rank_tier", "category", "title", "author",
          "url", "text", "raw_text", "text_available"]


def main():
    rows = []
    for src in SOURCES:
        path = os.path.join(STAGING, src)
        if not os.path.exists(path):
            print(f"!! missing {path} — run its scraper first")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                r["title"] = normalize_text(r.get("title"))
                r["author"] = normalize_text(r.get("author"), strip_wrapper_quotes=False)
                r["text"] = normalize_text(r.get("raw_text"))
                r["text_available"] = bool(r["text"])
                rows.append({k: r.get(k) for k in FIELDS})

    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} rows -> {OUT}")
    print("rows with text:", sum(1 for r in rows if r["text_available"]))
    print("\nper contest x cycle:")
    for (c, y), n in sorted(Counter((r["contest"], r["cycle"]) for r in rows).items()):
        nt = sum(1 for r in rows if r["contest"] == c and r["cycle"] == y and r["text_available"])
        print(f"  {c:18s} {y}  rows={n:3d}  with_text={nt:3d}")
    print("\nper tier:")
    for (c, t), n in sorted(Counter((r["contest"], r["rank_tier"]) for r in rows).items()):
        print(f"  {c:18s} {t:22s} {n}")


if __name__ == "__main__":
    main()
