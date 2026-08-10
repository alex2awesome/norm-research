#!/usr/bin/env python3
"""Audit year drift in patents_first_draft_with_applicant_cites_balanced.csv.gz.

The balanced CSV doesn't carry pgpub_id, so we join back to patents_dataset.jsonl.gz
by matching the first 200 chars of the abstract (unique enough as a quasi-key).

Checks:
 1. Filing year distribution by label.
 2. MI(year, label) — is year alone predictive?
 3. Length × year × label cross-tab — is the length skew time-driven?
 4. Per-year positive rate variance.
"""
import csv
import gzip
import json
import math
import statistics
from collections import Counter, defaultdict

csv.field_size_limit(2**31 - 1)

BALANCED = ("/lfs/skampere3/0/alexspan/norm-research/datasets/patents/"
            "patents_first_draft_with_applicant_cites_balanced.csv.gz")
JSONL = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents/patents_dataset.jsonl.gz"


def abstract_key(text):
    """Pull the first ~200 chars after ABSTRACT: as the quasi-key."""
    i = text.find("ABSTRACT:\n")
    if i < 0:
        return text[:200]
    return text[i + len("ABSTRACT:\n"):i + len("ABSTRACT:\n") + 200]


def main():
    # 1. Load balanced rows, build map from abstract-prefix → label.
    print("Loading balanced CSV...")
    by_key = {}
    label_counts = Counter()
    with gzip.open(BALANCED, "rt") as f:
        for r in csv.DictReader(f):
            k = abstract_key(r["text"])
            by_key[k] = int(r["judgement"])
            label_counts[int(r["judgement"])] += 1
    print(f"  Loaded {len(by_key):,} balanced rows (label dist: {dict(label_counts)})")

    # 2. Stream JSONL, match by abstract prefix, capture date_published.
    print("Streaming JSONL to recover years (matching by abstract prefix)...")
    by_year = defaultdict(lambda: {0: 0, 1: 0})
    by_year_len = defaultdict(lambda: {0: [], 1: []})
    n_matched = 0
    n_skipped = 0
    with gzip.open(JSONL, "rt") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            ab = d.get("pg_abstract") or d.get("g_abstract") or ""
            k = ab[:200]
            if k not in by_key:
                n_skipped += 1
                continue
            label = by_key[k]
            year = (d.get("date_published") or "")[:4]
            if not year.isdigit():
                continue
            year = int(year)
            by_year[year][label] += 1
            # Use claim/abstract length as proxy for text length
            claim_len = len(d.get("pg_claims") or "")
            by_year_len[year][label].append(claim_len)
            n_matched += 1
            if n_matched % 50000 == 0:
                print(f"  matched {n_matched:,} so far ...")
    print(f"  matched {n_matched:,} / {len(by_key):,} balanced rows")
    print(f"  unmatched JSONL rows skipped: {n_skipped:,}")
    print()

    # 3. Per-year pos rate
    print("--- Per-year label distribution ---")
    print(f"  {'year':6s} | {'pos':>8s} | {'neg':>8s} | {'total':>8s} | {'pos_rate':>9s}")
    rows = []
    for y in sorted(by_year):
        d = by_year[y]
        tot = d[0] + d[1]
        if tot < 100:
            continue
        rate = d[1] / tot
        print(f"  {y:6d} | {d[1]:>8d} | {d[0]:>8d} | {tot:>8d} | {rate:>8.1%}")
        rows.append((y, tot, rate))
    print()

    # 4. MI(year, label)
    if rows:
        tot_all = sum(t for _, t, _ in rows)
        mean = sum(t * p for _, t, p in rows) / tot_all
        h_l = (-mean * math.log(mean) - (1 - mean) * math.log(1 - mean)) if 0 < mean < 1 else 0
        h_l_given_y = 0
        for _, t, p in rows:
            w = t / tot_all
            if 0 < p < 1:
                h_l_given_y += w * (-p * math.log(p) - (1 - p) * math.log(1 - p))
        mi = h_l - h_l_given_y
        print(f"H(L) = {h_l:.4f} nats")
        print(f"H(L|year) = {h_l_given_y:.4f} nats")
        print(f"MI(label, year) = {mi:.4f} ({mi/h_l*100:.1f}% of H(L))")
        # Variance of pos_rate weighted by year volume
        var = sum(t * (p - mean) ** 2 for _, t, p in rows) / tot_all
        print(f"Weighted pos-rate mean: {mean:.3f}, std across years: {math.sqrt(var):.3f}")
        print()

    # 5. Length × year × label
    print("--- Mean claim length by year × label (for length-confound check) ---")
    print(f"  {'year':6s} | {'len_pos':>9s} | {'len_neg':>9s} | {'len_ratio':>9s}")
    for y in sorted(by_year_len):
        d = by_year_len[y]
        if len(d[0]) < 50 or len(d[1]) < 50:
            continue
        lp = statistics.mean(d[1])
        ln = statistics.mean(d[0])
        ratio = lp / ln if ln else 0
        print(f"  {y:6d} | {lp:>9.0f} | {ln:>9.0f} | {ratio:>9.3f}")


if __name__ == "__main__":
    main()
