#!/usr/bin/env python3
"""Fetch missing patent claim-1 from BigQuery public patents dataset.

Source: patents-public-data.patents.publications
This has US + foreign patents going back decades, including pre-2001 US
patents that PatentsView's bulk files don't carry.

Approach:
  1. Read missing_after_local_sources.csv → list of missing IDs + formats.
  2. Build candidate publication_number variants per missing ID:
       - US granted '5003263' -> 'US-5003263-A', 'US-5003263-B1', 'US-5003263-B2'
       - US pgpub '20100158389' -> 'US-20100158389-A1', 'US-20100158389-A2'
       - Foreign 'EP0897788' -> 'EP-0897788-A1', 'EP-0897788-B1'
  3. Upload candidate list to a BigQuery TEMP table.
  4. JOIN against publications: SELECT publication_number, claims_localized_html
     where publication_number IN (candidates).
  5. Pull English claims text out of claims_localized_html.
  6. Write to bigquery_supplement.parquet with columns (raw_id, claim_text).
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
DATASET = "patents_research"
TMP_TABLE = "missing_patent_ids"


KIND_VARIANTS = {
    "us_granted":    ["A", "A1", "A2", "B1", "B2", "P", "P1", "P2", "S", "S1"],
    "us_pgpub":      ["A1", "A2", "A9", "A11"],
    "us_unknown":    ["A", "A1", "B1", "B2"],
    "foreign_jp":    ["A", "A1", "B2"],
    "foreign_wo":    ["A1", "A2", "A3"],
    "foreign_ep":    ["A1", "A2", "B1", "B2"],
    "foreign_de":    ["A1", "B4", "C2", "T2"],
    "foreign_cn":    ["A", "B", "U"],
    "foreign_kr":    ["B1", "A"],
    "foreign_gb":    ["A", "B"],
    "foreign_fr":    ["A1", "B1"],
    "foreign_ca":    ["A1", "C"],
    "other":         ["A", "A1", "B1", "B2"],
}


def variants_for(raw_id, fmt):
    """Return list of candidate publication_number strings (BQ format US-XXX-A1)."""
    raw = raw_id.strip().upper()
    # Extract country + digits
    m = re.match(r"^(US|EP|WO|JP|CN|KR|DE|FR|GB|CA)?[\s-]*([0-9]+)", raw)
    if not m:
        return []
    country = m.group(1) or "US"
    digits = m.group(2)
    if not digits:
        return []
    kinds = KIND_VARIANTS.get(fmt, ["A", "A1", "B1", "B2"])
    return [f"{country}-{digits}-{k}" for k in kinds]


def main():
    client = bigquery.Client(project=PROJECT)

    # Make sure dataset exists in our project for the staging table.
    ds_ref = bigquery.Dataset(f"{PROJECT}.{DATASET}")
    ds_ref.location = "US"
    try:
        client.create_dataset(ds_ref, exists_ok=True)
    except Exception as e:
        print(f"WARN: dataset create: {e}", file=sys.stderr)

    print(f"Reading missing IDs from {MISSING_CSV} ...")
    candidates = []
    raw_id_to_candidates = {}
    with open(MISSING_CSV) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            raw = r["raw_id"]
            fmt = r["format"]
            v = variants_for(raw, fmt)
            if v:
                raw_id_to_candidates[raw] = v
                for c in v:
                    candidates.append((raw, c))
    print(f"  {len(raw_id_to_candidates):,} missing IDs -> {len(candidates):,} candidate publication_numbers")

    # Upload candidates to BQ staging table.
    print("Uploading candidate list to BigQuery staging ...")
    tbl_ref = f"{PROJECT}.{DATASET}.{TMP_TABLE}"
    schema = [
        bigquery.SchemaField("raw_id", "STRING"),
        bigquery.SchemaField("publication_number", "STRING"),
    ]
    job_config = bigquery.LoadJobConfig(
        schema=schema, write_disposition="WRITE_TRUNCATE"
    )
    job = client.load_table_from_json(
        [{"raw_id": r, "publication_number": c} for r, c in candidates],
        tbl_ref,
        job_config=job_config,
    )
    job.result()
    print(f"  uploaded {job.output_rows:,} rows to {tbl_ref}")

    # Run the join query.
    print("\nRunning BigQuery JOIN to fetch claim text ...")
    q = f"""
    WITH cand AS (
      SELECT raw_id, publication_number FROM `{tbl_ref}`
    )
    SELECT
      cand.raw_id,
      p.publication_number,
      (
        SELECT cl.text
        FROM UNNEST(p.claims_localized) AS cl
        WHERE cl.language = 'en'
        LIMIT 1
      ) AS claims_text
    FROM `patents-public-data.patents.publications` AS p
    JOIN cand USING (publication_number)
    WHERE p.country_code = SUBSTR(publication_number, 1, 2)
    """
    df = client.query(q).to_dataframe()
    print(f"  matched {len(df):,} publication_numbers")

    # Deduplicate to first claim per raw_id, strip HTML, keep first claim.
    print("Processing claims (first claim, strip HTML) ...")
    cleaned = {}
    for _, r in df.iterrows():
        raw = r["raw_id"]
        txt = r["claims_text"] or ""
        if not txt or raw in cleaned:
            continue
        # Strip HTML tags
        txt_plain = re.sub(r"<[^>]+>", " ", txt)
        txt_plain = re.sub(r"\s+", " ", txt_plain).strip()
        # Take first claim only (USPTO claims are usually numbered "1." through "N.")
        first_split = re.split(r"\s+2\s*[.,]\s+", txt_plain, maxsplit=1)
        first_claim = first_split[0]
        cleaned[raw] = first_claim[:4000]
    print(f"  cleaned {len(cleaned):,} unique claim-1 records")

    rows = [{"raw_id": k, "claim_text": v} for k, v in cleaned.items()]
    if rows:
        os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, OUT_PARQUET, compression="zstd")
        sz = os.path.getsize(OUT_PARQUET) / 1e6
        print(f"\nWrote {OUT_PARQUET} ({sz:.1f} MB, {len(rows):,} records)")
    else:
        print("\nNo records to write.")


if __name__ == "__main__":
    main()
