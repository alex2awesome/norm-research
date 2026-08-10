#!/usr/bin/env python3
"""Examine distribution of len(applicant_citations) per patent application.

Why this matters:
  - applicant_citations comes from g_us_patent_citation.tsv (PatentsView
    GRANTED citations table). If this is *only* populated for granted
    patents, it's a label leak: any non-zero count → label=1 trivially.
  - Even if populated for both, count is likely a proxy for applicant
    sophistication (more cites = experienced IDS practice).

Computes:
  1. Distribution of citation counts in the full JSONL.
  2. Pos rate by citation-count bucket.
  3. Cross-tab: cite count × first_draft_approved.
  4. Zero-citation rate by label.
"""
import gzip
import json
import statistics
import sys
from collections import Counter, defaultdict

JSONL = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents/patents_dataset.jsonl.gz"


def bucket(n):
    if n == 0: return "0"
    if n <= 2: return "1-2"
    if n <= 5: return "3-5"
    if n <= 10: return "6-10"
    if n <= 20: return "11-20"
    if n <= 50: return "21-50"
    if n <= 100: return "51-100"
    return "100+"


def main():
    print(f"Scanning {JSONL} ...")
    counts_by_label = defaultdict(list)
    bucket_label = defaultdict(lambda: {0: 0, 1: 0})
    n_total = 0
    is_granted_zero_cites = {True: 0, False: 0}
    is_granted_nonzero = {True: 0, False: 0}
    first_draft_zero_cites = {0: 0, 1: 0}
    first_draft_nonzero = {0: 0, 1: 0}

    with gzip.open(JSONL, "rt") as f:
        for line in f:
            n_total += 1
            try:
                d = json.loads(line)
            except Exception:
                continue
            cites = d.get("applicant_citations") or []
            cnt = len(cites) if isinstance(cites, list) else 0
            label = d.get("first_draft_approved")
            is_gr = d.get("is_granted")

            if label is not None:
                label = int(bool(label))
                counts_by_label[label].append(cnt)
                bucket_label[bucket(cnt)][label] += 1
                if cnt == 0:
                    first_draft_zero_cites[label] += 1
                else:
                    first_draft_nonzero[label] += 1

            if is_gr is not None:
                is_gr = bool(is_gr)
                if cnt == 0:
                    is_granted_zero_cites[is_gr] += 1
                else:
                    is_granted_nonzero[is_gr] += 1

            if n_total % 500_000 == 0:
                print(f"  scanned {n_total:,}", file=sys.stderr)

    print(f"\nScanned {n_total:,} rows.")
    print()

    print("--- 1. Citation count summary by first_draft_approved label ---")
    for label in (0, 1):
        cs = counts_by_label[label]
        if cs:
            mean = statistics.mean(cs)
            med = statistics.median(cs)
            p90 = sorted(cs)[int(0.9 * len(cs))]
            p99 = sorted(cs)[int(0.99 * len(cs))]
            n_zero = sum(1 for x in cs if x == 0)
            print(f"  label={label} (first_draft_approved={'True' if label else 'False'}): "
                  f"n={len(cs):,}  mean={mean:.1f} med={med} p90={p90} p99={p99}  "
                  f"zero_rate={n_zero/len(cs)*100:.1f}%")
    print()

    print("--- 2. Distribution by bucket ---")
    print(f"  {'bucket':>8s} | {'pos':>10s} {'neg':>10s} {'total':>10s} {'pos_rate':>9s}")
    base = sum(counts_by_label[1]) and len(counts_by_label[1]) / (len(counts_by_label[0]) + len(counts_by_label[1])) or 0
    for b in ["0", "1-2", "3-5", "6-10", "11-20", "21-50", "51-100", "100+"]:
        d = bucket_label[b]
        tot = d[0] + d[1]
        if tot == 0: continue
        rate = d[1] / tot
        print(f"  {b:>8s} | {d[1]:>10d} {d[0]:>10d} {tot:>10d} {rate:>8.1%}")
    print()

    print("--- 3. Zero-cite vs nonzero-cite breakdown ---")
    print(f"  By first_draft_approved label:")
    print(f"    cite count = 0:    pos={first_draft_zero_cites[1]:,}  neg={first_draft_zero_cites[0]:,}")
    print(f"    cite count > 0:    pos={first_draft_nonzero[1]:,}  neg={first_draft_nonzero[0]:,}")
    print(f"  By is_granted flag (broader pool):")
    print(f"    cite count = 0:    granted={is_granted_zero_cites[True]:,}  not_granted={is_granted_zero_cites[False]:,}")
    print(f"    cite count > 0:    granted={is_granted_nonzero[True]:,}  not_granted={is_granted_nonzero[False]:,}")
    print()

    # Key diagnostic: is cite count zero for abandoned patents?
    zero_g = is_granted_zero_cites[True]
    zero_ng = is_granted_zero_cites[False]
    nz_g = is_granted_nonzero[True]
    nz_ng = is_granted_nonzero[False]
    if zero_g + nz_g > 0 and zero_ng + nz_ng > 0:
        zero_rate_granted = zero_g / (zero_g + nz_g)
        zero_rate_not = zero_ng / (zero_ng + nz_ng)
        print("--- 4. Diagnostic: is applicant_citations populated for non-granted patents? ---")
        print(f"  zero-cite rate among GRANTED: {zero_rate_granted*100:.1f}%")
        print(f"  zero-cite rate among NOT GRANTED: {zero_rate_not*100:.1f}%")
        if zero_rate_not > 0.95:
            print("  → WARNING: non-granted patents almost always have 0 cites.")
            print("    This suggests applicant_citations is granted-only → label leak.")
        elif abs(zero_rate_not - zero_rate_granted) > 0.2:
            print("  → Big gap in zero-cite rate by grant status. Cite-presence is a partial label proxy.")
        else:
            print("  → Zero-cite rates similar → no leak from population coverage.")


if __name__ == "__main__":
    main()
