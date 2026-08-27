#!/usr/bin/env python3
"""Batched BQ fetch — avoids staging table (free-storage quota exhausted).

For each missing ID, generate 1-2 candidate publication_number strings,
batch into groups of 1000, run WHERE IN UNNEST(@ids) queries, stream results.

Reduces candidates per ID from 10 to 2 (most common kind codes only).
"""
import csv
import os
import re
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import bigquery

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
MISSING_CSV = f"{BASE}/processed/missing_after_local_sources.csv"
OUT_PARQUET = f"{BASE}/processed/bigquery_supplement.parquet"
PROJECT = "usc-research"
BATCH = 1000  # ids per BQ query

# Reduced kind codes per format type — most common only.
KIND_VARIANTS = {
    "us_granted":    ["A", "B2"],
    "us_pgpub":      ["A1"],
    "us_unknown":    ["A", "B2"],
    "foreign_jp":    ["A"],
    "foreign_wo":    ["A1"],
    "foreign_ep":    ["A1", "B1"],
    "foreign_de":    ["A1"],
    "foreign_cn":    ["A"],
    "foreign_kr":    ["A"],
    "foreign_gb":    ["A"],
    "foreign_fr":    ["A1"],
    "foreign_ca":    ["A1"],
    "other":         ["A1"],
}


def variants_for(raw_id, fmt):
    raw = raw_id.strip().upper()
    m = re.match(r"^(US|EP|WO|JP|CN|KR|DE|FR|GB|CA)?[\s-]*([0-9]+)", raw)
    if not m: return []
    country = m.group(1) or "US"
    digits = m.group(2)
    if not digits: return []
    kinds = KIND_VARIANTS.get(fmt, ["A", "A1"])
    return [f"{country}-{digits}-{k}" for k in kinds]


def main():
    client = bigquery.Client(project=PROJECT)

    print(f"Reading missing IDs from {MISSING_CSV} ...")
    raw_to_candidates = {}
    candidates_to_raw = {}
    with open(MISSING_CSV) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            raw = r["raw_id"]
            fmt = r["format"]
            v = variants_for(raw, fmt)
            if v:
                raw_to_candidates[raw] = v
                for c in v:
                    candidates_to_raw[c] = raw
    all_candidates = list(candidates_to_raw)
    print(f"  {len(raw_to_candidates):,} missing IDs -> {len(all_candidates):,} candidates")

    print(f"Running batched WHERE-IN queries (batch={BATCH:,}) ...")
    n_batches = (len(all_candidates) + BATCH - 1) // BATCH
    results = {}
    t0 = time.time()
    for i in range(n_batches):
        chunk = all_candidates[i * BATCH:(i + 1) * BATCH]
        q = """
        SELECT
          publication_number,
          (SELECT cl.text FROM UNNEST(claims_localized) AS cl
           WHERE cl.language = 'en' LIMIT 1) AS claims_text
        FROM `patents-public-data.patents.publications`
        WHERE publication_number IN UNNEST(@pubs)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("pubs", "STRING", chunk)
            ]
        )
        try:
            for row in client.query(q, job_config=job_config).result():
                raw = candidates_to_raw.get(row.publication_number)
                if raw and raw not in results and row.claims_text:
                    txt = re.sub(r"<[^>]+>", " ", row.claims_text)
                    txt = re.sub(r"\s+", " ", txt).strip()
                    # Take first claim only
                    spl = re.split(r"\s+2\s*[.,]\s+", txt, maxsplit=1)
                    results[raw] = spl[0][:4000]
        except Exception as e:
            print(f"  batch {i+1}/{n_batches} failed: {e}", file=sys.stderr)
            continue
        if (i + 1) % 10 == 0 or i == n_batches - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 60
            eta_min = (n_batches - i - 1) / rate
            print(f"  batch {i+1:,}/{n_batches:,} | {len(results):,} resolved | "
                  f"{rate:.0f} batch/min | ETA {eta_min:.0f} min",
                  file=sys.stderr, flush=True)

    print(f"\nResolved {len(results):,} unique cited refs from BigQuery.")
    rows = [{"raw_id": k, "claim_text": v} for k, v in results.items()]
    if rows:
        os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, OUT_PARQUET, compression="zstd")
        sz = os.path.getsize(OUT_PARQUET) / 1e6
        print(f"Wrote {OUT_PARQUET} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
