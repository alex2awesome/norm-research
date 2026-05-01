#!/usr/bin/env python3
"""
Join PatEx labels with PatentsView text to build the final patent dataset.

Pipeline:
1. Load labels (application_number -> first_draft_approved, final_outcome, ...)
2. Load PG metadata (application_number -> pgpub_id, filing_date)
3. Load PG abstracts and PG claims (year-split) -> aggregate per pgpub
4. Load granted patent text (claims + abstract) for granted apps
5. Load IDS / applicant citations (from g_us_patent_citation, filtered to
   "cited by applicant" rows — these are the non-leaky features)
6. Output:
   - patents_dataset.jsonl.gz (rich, all fields)
   - patents_first_draft.csv.gz (text/judgement for Task A)
   - patents_final_outcome.csv.gz (text/judgement for Task B)
"""
import argparse
import csv
import gzip
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

PATENTS_DIR = Path("/lfs/skampere3/0/alexspan/norm-research/datasets/patents")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default=str(PATENTS_DIR / "processed/labels.parquet"))
    p.add_argument("--pg-dir", default=str(PATENTS_DIR / "raw/patentsview_pg"))
    p.add_argument("--g-dir", default=str(PATENTS_DIR / "raw/patentsview_grant"))
    p.add_argument("--output-jsonl", default=str(PATENTS_DIR / "patents_dataset.jsonl.gz"))
    p.add_argument(
        "--output-task-a", default=str(PATENTS_DIR / "patents_first_draft.csv.gz")
    )
    p.add_argument(
        "--output-task-b", default=str(PATENTS_DIR / "patents_final_outcome.csv.gz")
    )
    p.add_argument(
        "--year-min",
        type=int,
        default=2010,
        help="Earliest filing year to include (default 2010 to keep dataset modern)",
    )
    p.add_argument(
        "--year-max",
        type=int,
        default=2024,
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="If set, cap total output rows for testing",
    )
    return p.parse_args()


def read_tsv_zip(zip_path, **kwargs):
    """Read a single .tsv inside a .tsv.zip."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".tsv")]
        if not names:
            raise FileNotFoundError(f"No .tsv inside {zip_path}")
        with zf.open(names[0]) as fh:
            return pd.read_csv(fh, sep="\t", on_bad_lines="skip", low_memory=False, **kwargs)


def normalize_app_number(s):
    """Normalize application_number to a consistent string format.

    PatEx uses '12345678' style; PatentsView sometimes uses '12/345,678'.
    Strip non-digits and leading zeros for matching.
    """
    if pd.isna(s):
        return None
    s = str(s)
    digits = "".join(c for c in s if c.isdigit())
    return digits.lstrip("0") or "0"


def load_labels(path, year_min, year_max):
    print(f"Loading labels from {path}...")
    labels = pd.read_parquet(path)
    print(f"  Loaded {len(labels):,} labels")
    # Normalize app numbers
    labels["app_norm"] = labels["application_number"].map(normalize_app_number)
    # Filter to year range based on filing date if available, else by first_allow_date
    # We'll do year filtering after joining with pg_published_application
    return labels


def load_pg_metadata(pg_dir):
    """Load pre-grant publication metadata: app number, pgpub id, dates.

    Columns in pg_published_application: pgpub_id, application_id, filing_date,
    patent_type, filing_type, published_date, wipo_kind, series_code,
    application_title, rule_47_flag, filename.
    """
    path = Path(pg_dir) / "pg_published_application.tsv.zip"
    if not path.exists():
        raise FileNotFoundError(path)
    print(f"Loading PG metadata from {path}...")
    df = read_tsv_zip(
        path,
        usecols=lambda c: c
        in (
            "pgpub_id",
            "application_id",
            "filing_date",
            "published_date",
            "application_title",
            "patent_type",
        ),
    )
    print(f"  Loaded {len(df):,} PG records")
    # PatentsView uses application_id; rename for consistency
    df = df.rename(columns={"application_id": "application_number", "application_title": "title", "published_date": "date_published"})
    df["app_norm"] = df["application_number"].map(normalize_app_number)
    df["filing_year"] = pd.to_datetime(df["filing_date"], errors="coerce").dt.year
    return df


def load_pg_abstracts(pg_dir):
    """Load PG abstracts."""
    path = Path(pg_dir) / "pg_published_application_abstract.tsv.zip"
    if not path.exists():
        print(f"  WARN: no abstract file at {path}")
        return None
    print(f"Loading PG abstracts from {path}...")
    df = read_tsv_zip(path)
    abstract_col = next((c for c in df.columns if "abstract" in c.lower()), None)
    if abstract_col is None:
        print(f"  WARN: no abstract column found in {list(df.columns)}")
        return None
    print(f"  Loaded {len(df):,} abstracts (text col: {abstract_col}, cols: {list(df.columns)})")
    return df[["pgpub_id", abstract_col]].rename(columns={abstract_col: "pg_abstract"})


def load_pg_claims_aggregated(pg_dir, years):
    """Load pg_claims for given years and aggregate per pgpub_id into one text blob."""
    print(f"Loading PG claims for years {years[0]}-{years[-1]}...")
    aggregates = []
    for year in years:
        path = Path(pg_dir) / f"pg_claims_{year}.tsv.zip"
        if not path.exists():
            continue
        print(f"  {path.name}...")
        df = read_tsv_zip(path)
        # Find columns: pgpub_id and claim_text
        pgpub_col = next((c for c in df.columns if "pgpub" in c.lower()), None)
        text_col = next(
            (c for c in df.columns if "text" in c.lower() or c.lower() == "claim_text"),
            None,
        )
        seq_col = next(
            (c for c in df.columns if "sequence" in c.lower() or "num" in c.lower()),
            None,
        )
        if not pgpub_col or not text_col:
            print(f"    skipping — missing cols (have: {list(df.columns)})")
            continue
        # Sort by claim sequence so concatenation is in order
        if seq_col:
            df = df.sort_values([pgpub_col, seq_col])
        # Aggregate
        df[text_col] = df[text_col].fillna("").astype(str)
        agg = df.groupby(pgpub_col)[text_col].apply(lambda s: "\n\n".join(x for x in s if x)).reset_index()
        agg.columns = ["pgpub_id", "pg_claims"]
        aggregates.append(agg)
        print(f"    aggregated to {len(agg):,} pgpub_ids")
    if not aggregates:
        return None
    combined = pd.concat(aggregates, ignore_index=True)
    print(f"  Combined: {len(combined):,} pgpub_ids with claims")
    return combined


def load_g_application(g_dir):
    """Map patent_id <-> application_number for granted patents.

    Columns: application_id, patent_id, patent_application_type, filing_date,
    series_code, rule_47_flag.
    """
    path = Path(g_dir) / "g_application.tsv.zip"
    if not path.exists():
        return None
    print(f"Loading granted application map from {path}...")
    df = read_tsv_zip(path, usecols=["application_id", "patent_id", "filing_date"])
    df = df.rename(columns={"application_id": "application_number"})
    print(f"  Loaded {len(df):,} granted application records")
    df["app_norm"] = df["application_number"].map(normalize_app_number)
    return df


def load_g_claims_aggregated(g_dir, years):
    """Load granted claims and aggregate per patent_id."""
    print(f"Loading granted claims for years {years[0]}-{years[-1]}...")
    aggregates = []
    for year in years:
        path = Path(g_dir) / f"g_claims_{year}.tsv.zip"
        if not path.exists():
            continue
        print(f"  {path.name}...")
        df = read_tsv_zip(path)
        pat_col = "patent_id" if "patent_id" in df.columns else next(
            (c for c in df.columns if "patent" in c.lower()), None
        )
        text_col = next(
            (c for c in df.columns if "text" in c.lower() or c.lower() == "claim_text"),
            None,
        )
        seq_col = next(
            (c for c in df.columns if "sequence" in c.lower() or "num" in c.lower()),
            None,
        )
        if not pat_col or not text_col:
            print(f"    skipping — missing cols (have: {list(df.columns)})")
            continue
        if seq_col:
            df = df.sort_values([pat_col, seq_col])
        df[text_col] = df[text_col].fillna("").astype(str)
        agg = df.groupby(pat_col)[text_col].apply(lambda s: "\n\n".join(x for x in s if x)).reset_index()
        agg.columns = ["patent_id", "g_claims"]
        aggregates.append(agg)
        print(f"    aggregated to {len(agg):,} patent_ids")
    if not aggregates:
        return None
    return pd.concat(aggregates, ignore_index=True)


def load_g_abstracts(g_dir):
    path = Path(g_dir) / "g_patent_abstract.tsv.zip"
    if not path.exists():
        return None
    print(f"Loading granted abstracts from {path}...")
    df = read_tsv_zip(path)
    abstract_col = next((c for c in df.columns if "abstract" in c.lower()), None)
    return df[["patent_id", abstract_col]].rename(columns={abstract_col: "g_abstract"})


def load_pg_to_patent_crosswalk(pg_dir):
    path = Path(pg_dir) / "pg_granted_pgpubs_crosswalk.tsv.zip"
    if not path.exists():
        return None
    print(f"Loading PG <-> patent crosswalk from {path}...")
    df = read_tsv_zip(path)
    print(f"  Loaded {len(df):,} crosswalk records (cols: {list(df.columns)})")
    return df


def load_applicant_citations(g_dir):
    """
    Load g_us_patent_citation, filter to applicant-disclosed (IDS) citations,
    and aggregate per patent_id.

    Columns: patent_id, citation_sequence, citation_patent_id, citation_date,
    record_name, wipo_kind, citation_category.

    Applicant-disclosed citations (category contains 'applicant') are non-leaky
    for tasks A/B; examiner-cited citations are leaky (imply rejection happened).
    """
    path = Path(g_dir) / "g_us_patent_citation.tsv.zip"
    if not path.exists():
        return None
    print(f"Loading IDS citations from {path}...")
    df = read_tsv_zip(
        path,
        usecols=["patent_id", "citation_patent_id", "citation_category"],
    )
    print(f"  Loaded {len(df):,} total citation records")
    print(f"  Category distribution: {df['citation_category'].value_counts().head(10).to_dict()}")
    applicant = df[df["citation_category"].astype(str).str.contains("applicant", case=False, na=False)]
    print(f"  Applicant-disclosed: {len(applicant):,}")
    agg = (
        applicant.groupby("patent_id")["citation_patent_id"]
        .apply(list)
        .reset_index()
    )
    agg.columns = ["patent_id", "applicant_citations"]
    print(f"  Aggregated {len(agg):,} patents with applicant citations")
    return agg


def main():
    args = parse_args()
    years = list(range(args.year_min, args.year_max + 1))

    # ---- 1) Labels
    labels = load_labels(args.labels, args.year_min, args.year_max)

    # ---- 2) PG metadata (app -> pgpub_id, filing_year)
    pg_meta = load_pg_metadata(args.pg_dir)

    # Year filter: keep only apps filed within target window
    pg_meta = pg_meta[
        (pg_meta["filing_year"] >= args.year_min)
        & (pg_meta["filing_year"] <= args.year_max)
    ]
    print(f"After year filter [{args.year_min}-{args.year_max}]: {len(pg_meta):,} PG records")

    # Join labels with PG metadata
    df = pg_meta.merge(labels, on="app_norm", how="inner", suffixes=("_pg", "_lbl"))
    print(f"After joining labels: {len(df):,} apps")

    # ---- 3) PG abstracts and claims
    pg_abs = load_pg_abstracts(args.pg_dir)
    if pg_abs is not None:
        df = df.merge(pg_abs, on="pgpub_id", how="left")

    pg_claims = load_pg_claims_aggregated(args.pg_dir, years)
    if pg_claims is not None:
        df = df.merge(pg_claims, on="pgpub_id", how="left")

    # ---- 4) Granted text (for the granted subset)
    g_app = load_g_application(args.g_dir)  # app_number -> patent_id
    g_claims = load_g_claims_aggregated(args.g_dir, years)
    g_abs = load_g_abstracts(args.g_dir)

    if g_app is not None and "patent_id" in g_app.columns:
        # Left join: app_norm -> patent_id
        df = df.merge(g_app[["app_norm", "patent_id"]], on="app_norm", how="left")
        if g_claims is not None:
            df = df.merge(g_claims, on="patent_id", how="left")
        if g_abs is not None:
            df = df.merge(g_abs, on="patent_id", how="left")

    # ---- 5) IDS citations
    ids = load_applicant_citations(args.g_dir)
    if ids is not None and "patent_id" in df.columns:
        df = df.merge(ids, on="patent_id", how="left")

    # ---- 6) Filter to apps that have draft text
    print(f"Before draft-text filter: {len(df):,} rows")
    df = df[df["pg_claims"].fillna("").str.len() > 50]
    print(f"After requiring pg_claims: {len(df):,} rows")

    if args.max_rows:
        df = df.head(args.max_rows)
        print(f"Capped at --max-rows={args.max_rows}: {len(df):,}")

    print("Final label distribution:")
    print(df["first_draft_approved"].value_counts())
    print(df["final_outcome"].value_counts())

    # ---- 7) Write outputs
    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing rich JSONL to {out_jsonl}...")

    keep_cols = [
        c
        for c in df.columns
        if c
        in (
            "application_number",
            "pgpub_id",
            "patent_id",
            "filing_date",
            "filing_year",
            "date_published",
            "title",
            "pg_abstract",
            "pg_claims",
            "g_abstract",
            "g_claims",
            "applicant_citations",
            "first_draft_approved",
            "final_outcome",
            "n_office_actions",
            "first_oa_date",
            "first_allow_date",
            "is_granted",
            "is_abandoned",
        )
    ]

    # Write CSVs FIRST so training data is saved even if JSONL fails later
    print(f"Writing Task A CSV to {args.output_task_a}...")
    task_a = pd.DataFrame(
        {
            "text": "ABSTRACT:\n"
            + df["pg_abstract"].fillna("").astype(str)
            + "\n\nCLAIMS:\n"
            + df["pg_claims"].fillna("").astype(str),
            "judgement": df["first_draft_approved"].astype(int),
        }
    )
    task_a = task_a[task_a["text"].str.len() > 100]
    task_a.to_csv(args.output_task_a, index=False, compression="gzip")
    print(f"  {len(task_a):,} rows; label balance: {task_a['judgement'].value_counts().to_dict()}")

    print(f"Writing Task B CSV to {args.output_task_b}...")
    df_b = df[df["final_outcome"].isin(["granted", "abandoned"])]
    task_b = pd.DataFrame(
        {
            "text": "ABSTRACT:\n"
            + df_b["pg_abstract"].fillna("").astype(str)
            + "\n\nCLAIMS:\n"
            + df_b["pg_claims"].fillna("").astype(str),
            "judgement": (df_b["final_outcome"] == "granted").astype(int),
        }
    )
    task_b = task_b[task_b["text"].str.len() > 100]
    task_b.to_csv(args.output_task_b, index=False, compression="gzip")
    print(f"  {len(task_b):,} rows; label balance: {task_b['judgement'].value_counts().to_dict()}")

    # Now write rich JSONL (may be slow for 4M rows)
    def _clean_value(v):
        if isinstance(v, (list, tuple)):
            return [str(x) if x is not None else None for x in v]
        if v is None:
            return None
        # scalar null check
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(v, (bool, int, float, str)):
            return v
        return str(v)

    with gzip.open(out_jsonl, "wt") as f:
        for rec in df[keep_cols].to_dict(orient="records"):
            clean = {k: _clean_value(v) for k, v in rec.items()}
            f.write(json.dumps(clean) + "\n")

    print("Done!")


if __name__ == "__main__":
    main()
