#!/usr/bin/env python3
"""Cite-count distribution within our CURRENT balanced patents file
(patents_first_draft_cpc_balanced.csv.gz, 547K rows).

The balanced file doesn't carry pgpub_id, so we join to patents_dataset.jsonl.gz
by first ~200 chars of abstract (same trick used in analyze_patents_year_drift.py).

Reports cumulative count + pos rate at each threshold.
"""
import csv
import gzip
import json
from collections import defaultdict

csv.field_size_limit(2**31 - 1)

BALANCED = ("/lfs/skampere3/0/alexspan/norm-research/datasets/patents/"
            "patents_first_draft_cpc_balanced.csv.gz")
JSONL = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents/patents_dataset.jsonl.gz"


def abstract_key(text):
    i = text.find("ABSTRACT:\n")
    if i < 0:
        return text[:200]
    return text[i + len("ABSTRACT:\n"):i + len("ABSTRACT:\n") + 200]


def main():
    print("Loading balanced file → abstract-prefix → label map ...")
    by_key = {}
    with gzip.open(BALANCED, "rt") as f:
        for r in csv.DictReader(f):
            k = abstract_key(r["text"])
            by_key[k] = int(r["judgement"])
    print(f"  loaded {len(by_key):,} balanced rows")

    print("\nStreaming JSONL to recover cite counts ...")
    rows = []  # (cite_count, label)
    n_matched = 0
    with gzip.open(JSONL, "rt") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            ab = d.get("pg_abstract") or d.get("g_abstract") or ""
            k = ab[:200]
            if k not in by_key:
                continue
            cites = d.get("applicant_citations") or []
            cnt = len(cites) if isinstance(cites, list) else 0
            rows.append((cnt, by_key[k]))
            n_matched += 1
            if n_matched % 100_000 == 0:
                print(f"  matched {n_matched:,}")
    print(f"  matched {n_matched:,} / {len(by_key):,} balanced rows")
    print()

    # Sort by cite count and compute cumulative
    rows.sort(key=lambda x: x[0])
    n_total = len(rows)
    pos_total = sum(l for _, l in rows)
    print(f"Total balanced rows recovered: {n_total:,}  (pos: {pos_total:,}, "
          f"overall pos rate: {pos_total/n_total*100:.1f}%)")
    print()

    # Thresholds for cumulative buckets
    thresholds = [0, 2, 5, 10, 20, 50, 100, 9999]
    print(f"  {'cite ≤':>8s} | {'pos_rate':>9s} | {'# patents':>11s} | {'% of total':>10s}")
    for t in thresholds:
        sub = [(c, l) for c, l in rows if c <= t]
        n_sub = len(sub)
        if n_sub == 0:
            continue
        pos_sub = sum(l for _, l in sub)
        pos_rate = pos_sub / n_sub
        pct = n_sub / n_total * 100
        label = "all" if t == 9999 else str(t)
        print(f"  {label:>8s} | {pos_rate*100:>8.1f}% | {n_sub:>11,d} | {pct:>9.1f}%")


if __name__ == "__main__":
    main()
