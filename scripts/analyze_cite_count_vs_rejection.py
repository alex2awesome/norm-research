#!/usr/bin/env python3
"""Are apps with fewer examiner-cited refs less likely to be rejected?

For each app in our balanced patents file, count examiner citations from OARD.
Bucket by count. For each bucket, report:
  - n apps
  - rejected_102 rate
  - rejected_103 rate
  - any-prior-art-rejection rate (102 OR 103)
  - first_draft_approved rate
"""
import csv
import gzip
import os
import re
import sys
from collections import Counter, defaultdict

csv.field_size_limit(2**31 - 1)

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
BALANCED = f"{BASE}/patents_first_draft_cpc_balanced_with_rejections.csv.gz"
OARD_CITES = f"{BASE}/raw/oard/oard_citations.csv"


def bucket(n):
    if n == 0: return "0"
    if n <= 2: return "1-2"
    if n <= 5: return "3-5"
    if n <= 10: return "6-10"
    if n <= 20: return "11-20"
    if n <= 50: return "21-50"
    return "50+"


def main():
    # 1. load app_id set and per-app labels from balanced file
    print("Loading balanced patents...", file=sys.stderr)
    app_to_labels = {}
    with gzip.open(BALANCED, "rt") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            app = r["app_id"].strip()
            if not app: continue
            app_to_labels[app] = {
                "first_draft_approved": int(r["judgement"]),
                "rejected_101": int(r["rejected_101"]),
                "rejected_102": int(r["rejected_102"]),
                "rejected_103": int(r["rejected_103"]),
                "rejected_112b": int(r["rejected_112b"]),
            }
    print(f"  {len(app_to_labels):,} apps loaded", file=sys.stderr)

    # 2. count examiner cites per app
    print("Counting examiner cites per app...", file=sys.stderr)
    cite_count = Counter()
    with open(OARD_CITES) as f:
        rdr = csv.DictReader(f)
        for n, r in enumerate(rdr, 1):
            app = r.get("app_id", "").strip()
            if app in app_to_labels:
                cite_count[app] += 1
            if n % 10_000_000 == 0:
                print(f"  scanned {n:,}", file=sys.stderr)
    print(f"  apps with at least 1 cite: {len(cite_count):,}", file=sys.stderr)

    # 3. bucket + aggregate
    bucket_stats = defaultdict(lambda: {
        "n": 0,
        "first_draft_approved": 0,
        "rejected_101": 0,
        "rejected_102": 0,
        "rejected_103": 0,
        "rejected_112b": 0,
    })
    for app, labels in app_to_labels.items():
        cnt = cite_count.get(app, 0)
        b = bucket(cnt)
        bucket_stats[b]["n"] += 1
        for k in ("first_draft_approved", "rejected_101", "rejected_102", "rejected_103", "rejected_112b"):
            bucket_stats[b][k] += labels[k]

    order = ["0", "1-2", "3-5", "6-10", "11-20", "21-50", "50+"]
    print()
    print(f"{'cites':>7s} | {'n':>9s} | {'1st_draft_ok':>12s} | {'§102':>6s} | {'§103':>6s} | {'§101':>6s} | {'§112b':>6s} | {'any_PA':>7s}")
    for b in order:
        d = bucket_stats[b]
        if d["n"] == 0:
            continue
        n = d["n"]
        approve_rate = d["first_draft_approved"] / n
        r102 = d["rejected_102"] / n
        r103 = d["rejected_103"] / n
        r101 = d["rejected_101"] / n
        r112b = d["rejected_112b"] / n
        any_pa = (d["rejected_102"] + d["rejected_103"]) / n  # overestimate; can both
        print(f"{b:>7s} | {n:>9,d} | {approve_rate*100:>11.1f}% | {r102*100:>5.1f}% | {r103*100:>5.1f}% | "
              f"{r101*100:>5.1f}% | {r112b*100:>5.1f}% | {any_pa*100:>6.1f}%")


if __name__ == "__main__":
    main()
