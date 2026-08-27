#!/usr/bin/env python3
"""Extract claim-1 text for ALL granted patents from PatentsView g_claims_*.tsv.zip.

The g_claims_YYYY.tsv.zip files have one row per claim, with columns:
  patent_id, claim_sequence, claim_text, dependent, ind_flg

INDEXING BUG FIXED 2026-07-08: PatentsView's granted-claims table is 0-INDEXED
(claim_sequence "0" = claim 1; verified against g_claims_2015 raw — seq-0 rows are
"I claim the ornamental design..." single-claim design patents, and seq-1 rows begin "2 .").
The pre-grant table (pg_claims_*) is 1-indexed, so extract_pgpub_corpus.py was correct.
The original run of this script filtered seq=="1" and therefore extracted CLAIM 2 of every
granted patent into granted_patents_claim1.parquet (kept on disk, do not reuse for claim-1
semantics). This fixed version filters seq=="0" and writes a NEW file:

Output: granted_patents_claim1_v2.parquet
  Columns: patent_id (str), claim_text (str), claim_year (int)
"""
import csv
import glob
import io
import os
import re
import sys
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq

csv.field_size_limit(2**31 - 1)

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
G_DIR = f"{BASE}/raw/patentsview_grant"
OUT = f"{BASE}/processed/granted_patents_claim1_v2.parquet"

os.makedirs(f"{BASE}/processed", exist_ok=True)


def yield_rows():
    """Stream claim-1 from each year file."""
    files = sorted(glob.glob(f"{G_DIR}/g_claims_*.tsv.zip"))
    print(f"Found {len(files)} year files", file=sys.stderr)
    for fp in files:
        year_match = re.search(r"g_claims_(\d{4})\.tsv\.zip", fp)
        year = int(year_match.group(1)) if year_match else 0
        print(f"  Processing {fp} (year {year})...", file=sys.stderr)
        n = 0; kept = 0
        with zipfile.ZipFile(fp) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as fh:
                tr = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
                rdr = csv.DictReader(tr, delimiter="\t")
                seen_patents = set()
                for r in rdr:
                    n += 1
                    pid = r.get("patent_id", "").strip()
                    seq = r.get("claim_sequence", "").strip()
                    if not pid:
                        continue
                    if pid in seen_patents:
                        continue
                    # Keep first claim — g_claims is 0-INDEXED (seq "0" = claim 1)
                    if seq and seq != "0":
                        continue
                    text = (r.get("claim_text") or "").strip()
                    if not text:
                        continue
                    seen_patents.add(pid)
                    kept += 1
                    yield {
                        "patent_id": pid,
                        "claim_text": text[:4000],  # cap claim length
                        "claim_year": year,
                    }
        print(f"    {fp}: {n:,} rows, kept {kept:,} claim-1 records", file=sys.stderr)


def main():
    rows = list(yield_rows())
    print(f"\nTotal claim-1 records: {len(rows):,}")
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, OUT, compression="zstd")
    sz = os.path.getsize(OUT) / 1e9
    print(f"Wrote {OUT} ({sz:.2f} GB)")


if __name__ == "__main__":
    main()
