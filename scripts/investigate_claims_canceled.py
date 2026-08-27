#!/usr/bin/env python3
"""Check whether 'claims canceled' in pg_claims is a real leak.

For each application in patents_dataset.jsonl.gz:
  1. Count occurrences of 'claims canceled' / 'canceled' / '(canceled)' patterns.
  2. Compare pos rate within rows that have these markers vs without.
  3. Look at pg_claims vs g_claims — if pg has cancellation markers, are they
     in the originally-filed version or in a re-publication of amendments?
  4. Sample 5 rows with the marker, show context around each match.

This tells us:
  - If pos rate within "has marker" rows is ~50%, no leak (just a stylistic
    artifact present at filing).
  - If pos rate is >>50% (e.g., 80%), it IS a leak — the marker is post-
    prosecution content sneaking into the pre-grant text.
"""
import gzip
import json
import re
import sys
from collections import Counter

JSONL = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents/patents_dataset.jsonl.gz"

# Patterns to check
PATTERNS = {
    "claims_canceled":      re.compile(r"claims?\s+canceled", re.IGNORECASE),
    "canceled":             re.compile(r"\bcanceled\b", re.IGNORECASE),
    "cancelled":            re.compile(r"\bcancelled\b", re.IGNORECASE),
    "paren_canceled":       re.compile(r"\(\s*canceled\s*\)", re.IGNORECASE),
    "previously_presented": re.compile(r"previously\s+presented", re.IGNORECASE),
    "currently_amended":    re.compile(r"currently\s+amended", re.IGNORECASE),
    "new_added":            re.compile(r"\bnew\s*\)", re.IGNORECASE),
}


def main():
    print(f"Scanning {JSONL} ...")
    n_total = 0
    n_with_label = 0
    counts = {p: {0: 0, 1: 0} for p in PATTERNS}  # n_rows where pattern hit
    pos_total = 0
    neg_total = 0

    samples = []
    sample_target = 5
    seen_with_marker_g_only = 0
    seen_with_marker_pg = 0

    with gzip.open(JSONL, "rt") as f:
        for line in f:
            n_total += 1
            try:
                d = json.loads(line)
            except Exception:
                continue
            label = d.get("first_draft_approved")
            if label is None:
                continue
            n_with_label += 1
            label = int(bool(label))
            if label == 1: pos_total += 1
            else:          neg_total += 1
            pg = d.get("pg_claims") or ""
            g  = d.get("g_claims") or ""

            for name, pat in PATTERNS.items():
                if pat.search(pg):
                    counts[name][label] += 1

            # Track whether marker is in pg only vs g only
            has_pg = PATTERNS["claims_canceled"].search(pg) or PATTERNS["paren_canceled"].search(pg)
            has_g  = PATTERNS["claims_canceled"].search(g)  or PATTERNS["paren_canceled"].search(g)
            if has_pg: seen_with_marker_pg += 1
            if has_g and not has_pg: seen_with_marker_g_only += 1

            # Collect samples
            if len(samples) < sample_target and has_pg:
                # find first match position
                m = (PATTERNS["claims_canceled"].search(pg) or
                     PATTERNS["paren_canceled"].search(pg))
                if m:
                    s, e = m.start(), m.end()
                    samples.append({
                        "pgpub_id": d.get("pgpub_id"),
                        "label": label,
                        "first_draft_approved": d.get("first_draft_approved"),
                        "n_office_actions": d.get("n_office_actions"),
                        "first_oa_date": d.get("first_oa_date"),
                        "first_allow_date": d.get("first_allow_date"),
                        "context": pg[max(0, s - 200):e + 200],
                    })

            if n_total % 500_000 == 0:
                print(f"  scanned {n_total:,}", file=sys.stderr)

    print(f"\nScanned {n_total:,} JSONL rows ({n_with_label:,} with label).")
    print(f"Overall pos rate: {pos_total / (pos_total + neg_total) * 100:.1f}% "
          f"(pos={pos_total:,}, neg={neg_total:,})")
    base_pos_rate = pos_total / (pos_total + neg_total)
    print()
    print("--- Pattern occurrence rates by label ---")
    print(f"  {'pattern':22s} | {'pos_hits':>10s} {'neg_hits':>10s} {'p(label|has)':>14s} {'lift_vs_base':>14s}")
    for name in PATTERNS:
        p_h = counts[name][1]
        n_h = counts[name][0]
        tot = p_h + n_h
        if tot == 0: continue
        rate_given = p_h / tot
        lift = rate_given - base_pos_rate
        print(f"  {name:22s} | {p_h:>10d} {n_h:>10d} {rate_given:>13.1%} {lift:>+13.3f}")
    print()
    print(f"--- pg vs g_claims marker presence ---")
    print(f"  rows where marker is in pg_claims: {seen_with_marker_pg:,}")
    print(f"  rows where marker is in g_claims ONLY (not pg): {seen_with_marker_g_only:,}")
    print()
    print("--- Sample rows with 'claims canceled' in pg_claims ---")
    for i, s in enumerate(samples):
        print(f"\n[Sample {i + 1}] pgpub_id={s['pgpub_id']}  "
              f"first_draft_approved={s['first_draft_approved']}  "
              f"n_office_actions={s['n_office_actions']}  "
              f"first_oa_date={s['first_oa_date']}  "
              f"first_allow_date={s['first_allow_date']}")
        print(f"  context: ...{s['context']}...")


if __name__ == "__main__":
    main()
