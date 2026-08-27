#!/usr/bin/env python3
"""Enumerate the cited prior-art union and check on-disk coverage.

Sources:
  - JSONL applicant_citations (each app's IDS cite list, granted-patent IDs)
  - OARD oard_citations.csv (examiner cite records, pre-grant pub IDs)

For each cited ID, classify:
  - Format: US pre-grant pub (11 digits, looks like 20190123456) vs
            US granted (7-8 digits, looks like 10123456) vs
            foreign (EP, WO, JP, CN, ...)
  - On-disk coverage: is this pgpub_id present in our patents_dataset.jsonl.gz?

Output: stats per source + per format + coverage gap.
"""
import csv
import gzip
import json
import re
import sys
from collections import Counter, defaultdict

csv.field_size_limit(2**31 - 1)

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
JSONL = f"{BASE}/patents_dataset.jsonl.gz"
OARD_CITES = f"{BASE}/raw/oard/oard_citations.csv"


def classify(cite_id: str) -> str:
    """Classify a citation ID by surface format."""
    s = cite_id.strip().upper()
    if not s:
        return "empty"
    # Strip common prefix
    m = re.match(r"^(US|EP|WO|JP|CN|KR|DE|FR|GB|CA)([0-9].*)?$", s)
    if m:
        prefix = m.group(1)
        rest = m.group(2) or ""
        if prefix == "US":
            digits = re.sub(r"[^0-9]", "", rest)
            if len(digits) == 11 and digits[:4] >= "1980":
                return "us_pgpub"  # like 20190123456
            elif 6 <= len(digits) <= 9:
                return "us_granted"
            else:
                return "us_unknown_format"
        else:
            return f"foreign_{prefix.lower()}"
    # Plain numeric — assume US
    digits = re.sub(r"[^0-9]", "", s)
    if len(digits) == 11:
        return "us_pgpub"
    elif 6 <= len(digits) <= 9:
        return "us_granted"
    return "other"


def normalize(cite_id: str) -> str:
    """Normalize cite id to bare digit string for comparison with JSONL pgpub_id."""
    s = cite_id.strip().upper()
    s = re.sub(r"^US", "", s)
    s = re.sub(r"[A-Z]\d?$", "", s)
    s = re.sub(r"[^0-9]", "", s)
    return s


def main():
    # Pass 1: enumerate applicant cites from JSONL.
    print("=== Pass 1: applicant_citations from JSONL ===")
    applicant_cites = Counter()
    applicant_total = 0
    n_apps = 0
    with gzip.open(JSONL, "rt") as f:
        for line in f:
            n_apps += 1
            try:
                d = json.loads(line)
            except Exception:
                continue
            ac = d.get("applicant_citations") or []
            applicant_total += len(ac)
            for c in ac:
                applicant_cites[str(c).strip()] += 1
            if n_apps % 500_000 == 0:
                print(f"  scanned {n_apps:,} apps, distinct cites so far: {len(applicant_cites):,}", file=sys.stderr)
    print(f"  apps scanned: {n_apps:,}")
    print(f"  total applicant cite records: {applicant_total:,}")
    print(f"  distinct applicant cites: {len(applicant_cites):,}")
    print()

    # Pass 2: enumerate examiner cites from OARD csv.
    print("=== Pass 2: examiner cites from OARD ===")
    examiner_cites = Counter()
    examiner_total = 0
    with open(OARD_CITES) as f:
        rdr = csv.DictReader(f)
        for n, r in enumerate(rdr, 1):
            cid = r.get("citation_pat_pgpub_id") or ""
            cid = cid.strip()
            if cid:
                examiner_cites[cid] += 1
                examiner_total += 1
            if n % 5_000_000 == 0:
                print(f"  scanned {n:,} OARD rows, distinct examiner cites: {len(examiner_cites):,}", file=sys.stderr)
    print(f"  total examiner cite records: {examiner_total:,}")
    print(f"  distinct examiner cites: {len(examiner_cites):,}")
    print()

    # Union
    union = set(applicant_cites) | set(examiner_cites)
    inter = set(applicant_cites) & set(examiner_cites)
    print(f"=== Union ===")
    print(f"  applicant only:           {len(applicant_cites) - len(inter):,}")
    print(f"  examiner only:            {len(examiner_cites) - len(inter):,}")
    print(f"  both:                     {len(inter):,}")
    print(f"  total distinct (union):   {len(union):,}")
    print()

    # Classify by format
    fmt = Counter(classify(c) for c in union)
    print("=== Format breakdown of union ===")
    for k, v in sorted(fmt.items(), key=lambda x: -x[1]):
        print(f"  {k:25s} {v:>12,d}  ({v/len(union)*100:.1f}%)")
    print()

    # Coverage: which cited pgpub_ids are in our JSONL?
    print("=== Coverage check: pass 3 over JSONL to enumerate pgpub_ids ===")
    on_disk_pgpubs = set()
    with gzip.open(JSONL, "rt") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            pid = str(d.get("pgpub_id", "")).strip()
            if pid:
                on_disk_pgpubs.add(pid)
            # Also index by patent_id (granted)
            ptid = str(d.get("patent_id", "")).strip()
            if ptid:
                on_disk_pgpubs.add(ptid)
    print(f"  on-disk pgpub+patent IDs: {len(on_disk_pgpubs):,}")
    print()

    # Compute coverage
    union_normalized = {normalize(c) for c in union if c}
    covered = union_normalized & on_disk_pgpubs
    missing = union_normalized - on_disk_pgpubs
    print(f"=== Coverage of union ===")
    print(f"  cited refs in our JSONL:  {len(covered):,}  ({len(covered)/len(union_normalized)*100:.1f}%)")
    print(f"  cited refs NOT in JSONL:  {len(missing):,}  ({len(missing)/len(union_normalized)*100:.1f}%)")
    print()

    # Breakdown of MISSING by format
    raw_missing_by_norm = defaultdict(list)
    for c in union:
        if c and normalize(c) in missing:
            raw_missing_by_norm[classify(c)].append(c)
    print("=== Missing by format ===")
    for k, lst in sorted(raw_missing_by_norm.items(), key=lambda x: -len(x[1])):
        examples = lst[:3]
        print(f"  {k:25s} {len(lst):>12,d}  examples: {examples}")


if __name__ == "__main__":
    main()
