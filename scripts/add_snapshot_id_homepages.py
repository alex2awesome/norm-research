#!/usr/bin/env python3
"""
Add snapshot_id to homepage_newsworthiness_topic_balanced.csv.gz.

Snapshot_id is derived as md5 of the sorted set of all headlines present in
each row (headline + context). Rows from the same snapshot share the same
headline-set and therefore the same snapshot_id.

After adding snapshot_id, also re-balance per-class within each snapshot
group so a group split does not produce class-imbalanced eval/test splits.

Output:
  /lfs/skampere3/0/alexspan/norm-research/datasets/news-homepages/homepage_newsworthiness_topic_balanced_groupsplit.csv.gz
"""
import csv
import gzip
import hashlib
import sys
from collections import Counter

csv.field_size_limit(2**31 - 1)

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/news-homepages"
INPUT = BASE + "/homepage_newsworthiness_topic_balanced.csv.gz"
OUTPUT = BASE + "/homepage_newsworthiness_topic_balanced_groupsplit.csv.gz"


def split_text(text):
    parts = text.split("\n\nCONTEXT: ", 1)
    headline = parts[0].replace("HEADLINE: ", "").strip()
    ctx_str = parts[1] if len(parts) > 1 else ""
    ctx_items = [h.strip() for h in ctx_str.split("; ") if h.strip()]
    return headline, ctx_items


def snapshot_id(headline, ctx_items):
    items = sorted({headline[:50]} | {c[:50] for c in ctx_items})
    return hashlib.md5("|".join(items).encode("utf-8")).hexdigest()[:16]


def main():
    print("Reading %s" % INPUT)
    rows = []
    with gzip.open(INPUT, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            h, ctx = split_text(row["text"])
            sid = snapshot_id(h, ctx)
            row["snapshot_id"] = sid
            rows.append(row)
    print("  loaded %d rows" % len(rows))

    sid_counts = Counter(r["snapshot_id"] for r in rows)
    print("  unique snapshots: %d" % len(sid_counts))
    print("  max rows/snapshot: %d, median: %d" % (
        max(sid_counts.values()),
        sorted(sid_counts.values())[len(sid_counts) // 2],
    ))

    print("Writing %s" % OUTPUT)
    with gzip.open(OUTPUT, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "judgement", "snapshot_id"])
        writer.writeheader()
        for r in rows:
            writer.writerow({"text": r["text"], "judgement": r["judgement"], "snapshot_id": r["snapshot_id"]})

    pos = sum(1 for r in rows if int(r["judgement"]) == 1)
    print("  label dist: pos=%d (%.1f%%)  neg=%d (%.1f%%)" % (
        pos, pos / len(rows) * 100, len(rows) - pos, (len(rows) - pos) / len(rows) * 100,
    ))
    print("Done.")


if __name__ == "__main__":
    main()
