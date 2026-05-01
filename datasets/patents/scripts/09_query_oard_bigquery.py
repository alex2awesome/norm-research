#!/usr/bin/env python3
"""
Query USPTO OARD citations from BigQuery public dataset.
Extracts per-application citation records covering both granted and abandoned apps.
"""
import argparse
import sys
from pathlib import Path

from google.cloud import bigquery


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="usc-research")
    p.add_argument("--output", default="/lfs/skampere3/0/alexspan/norm-research/datasets/patents/raw/oard/oard_citations.csv")
    p.add_argument("--dry-run", action="store_true", help="Just count rows, don't download")
    args = p.parse_args()

    client = bigquery.Client(project=args.project)

    if args.dry_run:
        q = "SELECT COUNT(*) as n FROM `patents-public-data.uspto_oce_office_actions.citations`"
        result = list(client.query(q).result())
        print(f"Total rows: {result[0].n:,}")
        q2 = """
        SELECT column_name, data_type
        FROM `patents-public-data.uspto_oce_office_actions.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = 'citations'
        """
        cols = list(client.query(q2).result())
        print("Columns:")
        for c in cols:
            print(f"  {c.column_name}: {c.data_type}")
        return

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("Querying OARD citations from BigQuery...")
    q = "SELECT * FROM `patents-public-data.uspto_oce_office_actions.citations`"
    df = client.query(q).to_dataframe()
    print(f"Downloaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print(df.head())

    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
