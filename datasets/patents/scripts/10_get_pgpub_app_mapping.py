#!/usr/bin/env python3
"""Get pgpub_id -> application_number mapping from BigQuery."""
from google.cloud import bigquery
import pandas as pd
from pathlib import Path

OUT = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents/processed/pgpub_to_appnum.parquet"

client = bigquery.Client(project="usc-research")

q = """
SELECT
  publication_number,
  application_number
FROM `patents-public-data.patents.publications`
WHERE country_code = 'US'
  AND kind_code = 'A1'
  AND application_number != ''
"""

print("Querying BigQuery for US pre-grant pubs...")
df = client.query(q).to_dataframe()
print(f"Downloaded {len(df):,} rows")

# publication_number looks like 'US-20100158439-A1', extract the numeric part
df["pgpub_id"] = df["publication_number"].str.replace(r"^US-|-(A1|A2|A9)$", "", regex=True)
# application_number looks like 'US-12345678-A', extract numeric part
df["app_num"] = df["application_number"].str.replace(r"^US-|-[A-Z]\d*$", "", regex=True)

print(df[["pgpub_id", "app_num"]].head(10))
print(f"Unique pgpub_ids: {df.pgpub_id.nunique():,}")

Path(OUT).parent.mkdir(parents=True, exist_ok=True)
df[["pgpub_id", "app_num"]].to_parquet(OUT, index=False)
print(f"Saved to {OUT}")
