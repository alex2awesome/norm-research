#!/usr/bin/env python3
"""Query OARD rejections table, aggregate to per-application rejection flags.

Output: /lfs/.../patents/raw/oard/oard_rejections_by_app.csv
Columns: app_id, rejected_101, rejected_102, rejected_103, rejected_112a,
         rejected_112b, rejected_112d, rejected_112f, rejected_dp
         (each 0/1 — true if the app ever received that rejection ground in
          any OA)
"""
from pathlib import Path
from google.cloud import bigquery

OUT = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents/raw/oard/oard_rejections_by_app.csv"

QUERY = """
SELECT
  app_id,
  CAST(MAX(CASE WHEN action_type = '101' THEN 1 ELSE 0 END) AS INT64) AS rejected_101,
  CAST(MAX(CASE WHEN action_type = '102' THEN 1 ELSE 0 END) AS INT64) AS rejected_102,
  CAST(MAX(CASE WHEN action_type = '103' THEN 1 ELSE 0 END) AS INT64) AS rejected_103,
  CAST(MAX(CASE WHEN action_type = '112' AND action_subtype = 'a' THEN 1 ELSE 0 END) AS INT64) AS rejected_112a,
  CAST(MAX(CASE WHEN action_type = '112' AND action_subtype = 'b' THEN 1 ELSE 0 END) AS INT64) AS rejected_112b,
  CAST(MAX(CASE WHEN action_type = '112' AND action_subtype = 'd' THEN 1 ELSE 0 END) AS INT64) AS rejected_112d,
  CAST(MAX(CASE WHEN action_type = '112' AND action_subtype = 'f' THEN 1 ELSE 0 END) AS INT64) AS rejected_112f,
  CAST(MAX(CASE WHEN action_type = 'nonstatutory double patenting' THEN 1 ELSE 0 END) AS INT64) AS rejected_dp
FROM `patents-public-data.uspto_oce_office_actions.rejections`
GROUP BY app_id
"""


def main():
    client = bigquery.Client(project="usc-research")
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    print(f"Querying OARD per-app rejection flags ...")
    df = client.query(QUERY).to_dataframe()
    print(f"  rows: {len(df):,}")
    print(f"  per-flag positives:")
    for c in df.columns:
        if c == "app_id": continue
        print(f"    {c}: {df[c].sum():,}  ({df[c].mean()*100:.1f}% of apps)")
    df.to_csv(OUT, index=False)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
