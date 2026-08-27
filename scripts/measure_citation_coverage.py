#!/usr/bin/env python3
"""Measure citation coverage after all available text sources are loaded.

Sources, in priority order:
  1. patents_dataset.jsonl.gz (pgpub_id + patent_id keys)
  2. granted_patents_claim1.parquet (patent_id keys)
  3. pgpub_claims1.parquet (pgpub_id keys)
  4. bigquery_supplement.parquet (publication_number keys, from BigQuery fetch)

For each cited reference, mark covered=True if its normalized ID appears in
any source's key set. Report total coverage % and missing-by-format breakdown.

Also writes the still-missing IDs to a CSV for the next fetch iteration:
  /processed/missing_after_local_sources.csv
"""
import csv
import gzip
import json
import os
import re
import sys
from collections import Counter

import pyarrow.parquet as pq

csv.field_size_limit(2**31 - 1)

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
JSONL = f"{BASE}/patents_dataset.jsonl.gz"
OARD_CITES = f"{BASE}/raw/oard/oard_citations.csv"
GRANTED_PARQUET = f"{BASE}/processed/granted_patents_claim1.parquet"
PGPUB_PARQUET = f"{BASE}/processed/pgpub_claims1.parquet"
LEGACY_CLAIM1_LOOKUP = f"{BASE}/processed/claim1_lookup.parquet"
BQ_SUPPL = f"{BASE}/processed/bigquery_supplement.parquet"
MISSING_OUT = f"{BASE}/processed/missing_after_local_sources.csv"


def normalize(s):
    if not s: return ""
    s = s.strip().upper()
    # Keep design prefix
    is_design = s.startswith("D") and not s.startswith("DE")
    s = re.sub(r"^US", "", s)
    if not is_design:
        s = re.sub(r"[A-Z]\d?$", "", s)
    return re.sub(r"[^0-9D]", "", s) if is_design else re.sub(r"[^0-9]", "", s)


def classify(cite_id):
    s = cite_id.strip().upper()
    if not s: return "empty"
    m = re.match(r"^(US|EP|WO|JP|CN|KR|DE|FR|GB|CA)([0-9].*)?$", s)
    if m:
        prefix = m.group(1)
        rest = m.group(2) or ""
        if prefix == "US":
            digits = re.sub(r"[^0-9]", "", rest)
            if len(digits) == 11 and digits[:4] >= "1980":
                return "us_pgpub"
            elif 6 <= len(digits) <= 9:
                return "us_granted"
            return "us_unknown"
        return f"foreign_{prefix.lower()}"
    digits = re.sub(r"[^0-9]", "", s)
    if len(digits) == 11: return "us_pgpub"
    if 6 <= len(digits) <= 9: return "us_granted"
    return "other"


def main():
    # Collect available IDs from each source.
    available = set()

    print("Loading IDs from JSONL ...", file=sys.stderr)
    with gzip.open(JSONL, "rt") as f:
        for n, line in enumerate(f, 1):
            try: d = json.loads(line)
            except Exception: continue
            for k in ("pgpub_id", "patent_id"):
                v = str(d.get(k) or "").strip()
                if v: available.add(v)
            if n % 500_000 == 0:
                print(f"  scanned {n:,}, available {len(available):,}", file=sys.stderr)
    print(f"  +JSONL: {len(available):,}", file=sys.stderr)

    if os.path.exists(GRANTED_PARQUET):
        t = pq.read_table(GRANTED_PARQUET, columns=["patent_id"])
        for pid in t.column("patent_id").to_pylist():
            if pid: available.add(str(pid).strip())
        print(f"  +granted_parquet: {len(available):,}", file=sys.stderr)

    if os.path.exists(LEGACY_CLAIM1_LOOKUP):
        t = pq.read_table(LEGACY_CLAIM1_LOOKUP, columns=["patent_id"])
        for pid in t.column("patent_id").to_pylist():
            if pid: available.add(str(pid).strip())
        print(f"  +legacy_claim1_lookup: {len(available):,}", file=sys.stderr)

    if os.path.exists(PGPUB_PARQUET):
        t = pq.read_table(PGPUB_PARQUET, columns=["pgpub_id"])
        for pid in t.column("pgpub_id").to_pylist():
            if pid: available.add(str(pid).strip())
        print(f"  +pgpub_parquet: {len(available):,}", file=sys.stderr)

    if os.path.exists(BQ_SUPPL):
        t = pq.read_table(BQ_SUPPL, columns=["raw_id"])
        for pid in t.column("raw_id").to_pylist():
            if pid: available.add(str(pid).strip())
        print(f"  +bq_supplement: {len(available):,}", file=sys.stderr)

    # Now enumerate the union of cited IDs (from applicant_citations + OARD).
    print("\nCollecting cited-reference union ...", file=sys.stderr)
    cited_raw = set()
    with gzip.open(JSONL, "rt") as f:
        for n, line in enumerate(f, 1):
            try: d = json.loads(line)
            except Exception: continue
            for c in (d.get("applicant_citations") or []):
                cited_raw.add(str(c).strip())
            if n % 500_000 == 0:
                print(f"  applicant cites: {len(cited_raw):,}", file=sys.stderr)
    print(f"  applicant union: {len(cited_raw):,}", file=sys.stderr)

    with open(OARD_CITES) as f:
        rdr = csv.DictReader(f)
        for n, r in enumerate(rdr, 1):
            cid = (r.get("citation_pat_pgpub_id") or "").strip()
            if cid: cited_raw.add(cid)
            if n % 5_000_000 == 0:
                print(f"  OARD scanned {n:,}, total cited: {len(cited_raw):,}", file=sys.stderr)
    print(f"  full cited union: {len(cited_raw):,}", file=sys.stderr)

    # Cover check.
    covered_raw = set()
    missing_raw = set()
    for c in cited_raw:
        n = normalize(c)
        if n in available:
            covered_raw.add(c)
        else:
            missing_raw.add(c)

    total = len(cited_raw)
    pct = len(covered_raw) / total * 100 if total else 0
    print()
    print(f"=== Coverage ===")
    print(f"  total cited refs:   {total:,}")
    print(f"  covered:            {len(covered_raw):,}  ({pct:.1f}%)")
    print(f"  missing:            {len(missing_raw):,}  ({(100-pct):.1f}%)")
    print()

    fmt_missing = Counter(classify(c) for c in missing_raw)
    print("Missing breakdown by format:")
    for k, v in sorted(fmt_missing.items(), key=lambda x: -x[1]):
        print(f"  {k:25s} {v:>12,d}")

    # Write missing IDs to CSV for next iteration
    print(f"\nWriting missing IDs to {MISSING_OUT} ...")
    os.makedirs(os.path.dirname(MISSING_OUT), exist_ok=True)
    with open(MISSING_OUT, "w") as f:
        w = csv.writer(f)
        w.writerow(["raw_id", "normalized_id", "format"])
        for c in missing_raw:
            w.writerow([c, normalize(c), classify(c)])
    print(f"  wrote {len(missing_raw):,} missing IDs")

    # Exit code so the loop runner can detect coverage threshold
    if pct >= 90:
        print(f"\n[STOP] coverage {pct:.1f}% >= 90% — threshold reached.")
        sys.exit(0)
    else:
        print(f"\n[CONTINUE] coverage {pct:.1f}% < 90% — more fetch needed.")
        sys.exit(2)


if __name__ == "__main__":
    main()
