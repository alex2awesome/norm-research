#!/usr/bin/env python3
"""
Rebuild notice_and_comment dataset with per-agency rebalancing to remove
the agency-identity confound surfaced by analyze_nc_leakage.py (per-agency
pos-rate variance from 49% → 89%, MI(agency, label) = 5.1% of H(L)).

Pipeline:
  1. Load all 18 agencies' per-agency labeled rows from agencies/<ag>/{train,eval,test}.csv.gz.
  2. Tag each row with agency.
  3. Per-agency balance: 2 * min(n_pos, n_neg) per agency.
  4. Cap per-agency contribution at PER_AGENCY_CAP rows so CMS/EPA can't dominate.
  5. Output text + judgement + agency columns.

The `agency` column is carried so a future GroupShuffleSplit on agency
(or on rule_id if we re-derive it) can prevent agency identity from
straddling train/eval/test.
"""
import csv
import gzip
import os
import random
from collections import defaultdict

csv.field_size_limit(2**31 - 1)

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/notice-and-comment"
AGENCIES_DIR = BASE + "/agencies"
OUTPUT = BASE + "/notice_and_comment_agency_balanced.csv.gz"

PER_AGENCY_CAP = 100_000  # per-agency cap on BALANCED rows (i.e. <= 50K each class)
SEED = 42

random.seed(SEED)


def load_agency(agency):
    """Concat train+eval+test for one agency."""
    rows = []
    for split in ("train", "eval", "test"):
        path = f"{AGENCIES_DIR}/{agency}/{split}.csv.gz"
        if not os.path.exists(path):
            continue
        with gzip.open(path, "rt") as f:
            for r in csv.DictReader(f):
                rows.append({
                    "text": r["text"],
                    "judgement": int(r["judgement"]),
                    "agency": agency,
                })
    return rows


def main():
    agencies = sorted(
        d for d in os.listdir(AGENCIES_DIR)
        if os.path.isdir(os.path.join(AGENCIES_DIR, d))
    )
    print(f"Found {len(agencies)} agency dirs")

    out = []
    per_agency_stats = []
    for ag in agencies:
        rows = load_agency(ag)
        if not rows:
            continue
        pos = [r for r in rows if r["judgement"] == 1]
        neg = [r for r in rows if r["judgement"] == 0]
        n_balanced = min(len(pos), len(neg))
        if n_balanced == 0:
            print(f"  [skip] {ag}: pos={len(pos)} neg={len(neg)}")
            continue

        # Cap per-class to PER_AGENCY_CAP // 2
        n_take = min(n_balanced, PER_AGENCY_CAP // 2)
        random.shuffle(pos)
        random.shuffle(neg)
        chosen_pos = pos[:n_take]
        chosen_neg = neg[:n_take]
        out.extend(chosen_pos)
        out.extend(chosen_neg)
        per_agency_stats.append((ag, len(rows), len(pos), len(neg), n_take * 2))
        print(f"  {ag}: total={len(rows):,} pos={len(pos):,} neg={len(neg):,} "
              f"→ balanced_taken={n_take * 2:,}")

    random.shuffle(out)
    print(f"\nTotal output rows: {len(out):,} (50/50 across {len(per_agency_stats)} agencies)")

    print(f"\nWriting {OUTPUT}")
    with gzip.open(OUTPUT, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "judgement", "agency"])
        w.writeheader()
        for r in out:
            w.writerow(r)

    # Verify
    pos_total = sum(1 for r in out if r["judgement"] == 1)
    print(f"Verify: pos={pos_total} ({pos_total/len(out)*100:.1f}%)")
    agency_dist = defaultdict(int)
    for r in out:
        agency_dist[r["agency"]] += 1
    print("\nFinal per-agency rows in output:")
    for ag in sorted(agency_dist, key=lambda x: -agency_dist[x]):
        print(f"  {ag}: {agency_dist[ag]:,}")


if __name__ == "__main__":
    main()
