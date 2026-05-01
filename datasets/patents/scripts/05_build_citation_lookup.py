#!/usr/bin/env python3
"""
Build a patent_id -> claim_1_text lookup table.

This is used to enrich draft patent text with cited prior-art claim text
(both applicant-disclosed and examiner-cited).

We need claim 1 only (the broadest, most informative claim) to keep context
budget reasonable: each cited patent contributes ~500-2000 chars.
"""
import argparse
import zipfile
from pathlib import Path

import pandas as pd

DEFAULT_GRANT_DIR = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents/raw/patentsview_grant"
DEFAULT_OUTPUT = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents/processed/claim1_lookup.parquet"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--grant-dir", default=DEFAULT_GRANT_DIR)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--year-min", type=int, default=2001)
    p.add_argument("--year-max", type=int, default=2025)
    return p.parse_args()


def read_tsv_zip(zip_path, **kwargs):
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".tsv")]
        with zf.open(names[0]) as fh:
            return pd.read_csv(fh, sep="\t", on_bad_lines="skip", low_memory=False, **kwargs)


def main():
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    grant_dir = Path(args.grant_dir)

    chunks = []
    for year in range(args.year_min, args.year_max + 1):
        path = grant_dir / f"g_claims_{year}.tsv.zip"
        if not path.exists():
            continue
        print(f"Reading {path.name}...")
        df = read_tsv_zip(
            path,
            usecols=["patent_id", "claim_sequence", "claim_text"],
            dtype={"patent_id": "str"},
        )
        # claim_sequence == 0 is the first claim in PatentsView (zero-indexed)
        df = df[df["claim_sequence"] == 0]
        df = df[["patent_id", "claim_text"]].rename(columns={"claim_text": "claim_1"})
        df = df.dropna(subset=["claim_1"])
        chunks.append(df)
        print(f"  -> {len(df):,} patents with claim 1 from {year}")

    combined = pd.concat(chunks, ignore_index=True)
    # Deduplicate by patent_id (keep first occurrence — patents granted in only one year)
    combined = combined.drop_duplicates(subset=["patent_id"], keep="first")
    print(f"\nTotal unique patents with claim 1: {len(combined):,}")

    # Truncate very long claim 1 to keep context budget reasonable
    combined["claim_1"] = combined["claim_1"].astype(str).str[:1500]

    combined.to_parquet(args.output, index=False)
    print(f"Wrote lookup to {args.output}")


if __name__ == "__main__":
    main()
