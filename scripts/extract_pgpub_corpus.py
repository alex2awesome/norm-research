#!/usr/bin/env python3
"""Extract claim-1 text for ALL pre-grant pubs from PatentsView pg_claims_*.tsv.zip.

Columns: pgpub_id, claim_sequence, claim_text, dependent, claim_number
We keep only claim_sequence=1 (or first claim per pgpub).

Output: pgpub_claims1.parquet
  Columns: pgpub_id (str), claim_text (str), claim_year (int)
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
PG_DIR = f"{BASE}/raw/patentsview_pg"
OUT = f"{BASE}/processed/pgpub_claims1.parquet"

os.makedirs(f"{BASE}/processed", exist_ok=True)


def yield_rows():
    files = sorted(glob.glob(f"{PG_DIR}/pg_claims_*.tsv.zip"))
    print(f"Found {len(files)} year files", file=sys.stderr)
    for fp in files:
        year_match = re.search(r"pg_claims_(\d{4})\.tsv\.zip", fp)
        year = int(year_match.group(1)) if year_match else 0
        print(f"  Processing {fp} (year {year})...", file=sys.stderr)
        n = 0; kept = 0
        with zipfile.ZipFile(fp) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as fh:
                tr = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
                rdr = csv.DictReader(tr, delimiter="\t")
                seen = set()
                for r in rdr:
                    n += 1
                    pid = r.get("pgpub_id", "").strip()
                    seq = r.get("claim_sequence", "").strip()
                    if not pid:
                        continue
                    if pid in seen:
                        continue
                    if seq and seq != "1":
                        continue
                    text = (r.get("claim_text") or "").strip()
                    if not text:
                        continue
                    seen.add(pid)
                    kept += 1
                    yield {
                        "pgpub_id": pid,
                        "claim_text": text[:4000],
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
