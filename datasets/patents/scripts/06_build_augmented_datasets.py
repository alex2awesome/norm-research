#!/usr/bin/env python3
"""
Build citation-augmented training CSVs for patent prediction tasks.

Loads:
- The rich JSONL with per-app metadata (output of 04_build_dataset.py)
- claim1_lookup.parquet (output of 05_build_citation_lookup.py)
- Citation sources:
  - g_us_patent_citation.tsv.zip (PatentsView, granted patents only)
  - OARD oard_citations.csv (BigQuery, covers granted AND abandoned apps)

Outputs two non-leaky task variants:

1. Rough-draft prediction + applicant (author) cites
   - Uses g_us_patent_citation (keyed by patent_id, granted only — acceptable)
   - File: patents_first_draft_with_applicant_cites.csv.gz

2. Final-draft prediction + examiner cites
   - Uses OARD (keyed by app_id, covers both granted AND abandoned)
   - CRITICAL: g_us_patent_citation only has granted patents, so abandoned apps
     get zero examiner cites => catastrophic leakage. OARD fixes this.
   - File: patents_final_outcome_with_examiner_cites.csv.gz
"""
import argparse
import csv
import gzip
import json
import zipfile
from pathlib import Path

import pandas as pd

PATENTS_DIR = Path("/lfs/skampere3/0/alexspan/norm-research/datasets/patents")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rich-jsonl", default=str(PATENTS_DIR / "patents_dataset.jsonl.gz"))
    p.add_argument("--claim1-lookup", default=str(PATENTS_DIR / "processed/claim1_lookup.parquet"))
    p.add_argument("--g-citation",
                   default=str(PATENTS_DIR / "raw/patentsview_grant/g_us_patent_citation.tsv.zip"))
    p.add_argument("--oard-citations",
                   default=str(PATENTS_DIR / "raw/oard/oard_citations.csv"),
                   help="OARD citations CSV from BigQuery (covers granted+abandoned)")
    p.add_argument("--output-dir", default=str(PATENTS_DIR))
    p.add_argument("--top-k-cites", type=int, default=5,
                   help="How many cited patents to include per app (max claim 1 budget)")
    p.add_argument("--max-text-chars", type=int, default=20000,
                   help="Cap total text length per row")
    p.add_argument("--require-oa", action="store_true",
                   help="For Task B + examiner cites, only include apps that "
                        "had at least 1 office action (removes leakage from "
                        "'no examiner cites => first-draft approved => granted').")
    p.add_argument("--task-b-only", action="store_true",
                   help="Only rebuild Task B + examiner cites (skip Task A)")
    return p.parse_args()


def read_tsv_zip(zip_path, **kwargs):
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".tsv")]
        with zf.open(names[0]) as fh:
            return pd.read_csv(fh, sep="\t", on_bad_lines="skip", low_memory=False, **kwargs)


def load_citations_with_categories(g_citation_path, top_k):
    """
    Load per-patent citation lists separated by category (applicant vs examiner).
    Returns DataFrame: patent_id -> {applicant_cites: [...], examiner_cites: [...]}.
    """
    print(f"Loading citations from {g_citation_path}...")
    df = read_tsv_zip(
        g_citation_path,
        usecols=["patent_id", "citation_sequence", "citation_patent_id", "citation_category"],
        dtype={"patent_id": "str", "citation_patent_id": "str"},
    )
    print(f"  Loaded {len(df):,} citation records")

    # Sort by sequence so .head(top_k) takes the first K cited patents
    df = df.sort_values(["patent_id", "citation_sequence"])

    cat_lower = df["citation_category"].astype(str).str.lower()
    applicant_mask = cat_lower.str.contains("applicant", na=False)
    examiner_mask = cat_lower.str.contains("examiner", na=False)

    print("Aggregating applicant citations (top-k per patent)...")
    applicant_cites = (
        df[applicant_mask]
        .groupby("patent_id")["citation_patent_id"]
        .apply(lambda s: list(s.head(top_k)))
        .rename("applicant_cites")
    )
    print(f"  {len(applicant_cites):,} patents with applicant cites")

    print("Aggregating examiner citations (top-k per patent)...")
    examiner_cites = (
        df[examiner_mask]
        .groupby("patent_id")["citation_patent_id"]
        .apply(lambda s: list(s.head(top_k)))
        .rename("examiner_cites")
    )
    print(f"  {len(examiner_cites):,} patents with examiner cites")

    return applicant_cites, examiner_cites


def load_oard_examiner_cites(oard_path, top_k):
    """Load per-application examiner cites from OARD (covers granted AND abandoned)."""
    print(f"Loading OARD citations from {oard_path}...")
    df = pd.read_csv(
        oard_path,
        usecols=["app_id", "citation_pat_pgpub_id", "form892"],
        dtype={"app_id": "str", "citation_pat_pgpub_id": "str"},
    )
    print(f"  Loaded {len(df):,} citation records")
    examiner = df[df["form892"] == 1].copy()
    print(f"  Examiner cites (form892=1): {len(examiner):,}")
    del df

    examiner_cites = (
        examiner
        .groupby("app_id")["citation_pat_pgpub_id"]
        .apply(lambda s: list(s.head(top_k)))
        .rename("oard_examiner_cites")
    )
    print(f"  {len(examiner_cites):,} apps with examiner cites")
    return examiner_cites


def load_rich_jsonl(path):
    """Load the rich JSONL into a DataFrame."""
    print(f"Loading {path}...")
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) % 500000 == 0:
                print(f"  read {len(rows):,} rows")
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df):,} apps")
    return df


def lookup_cite_texts(cite_ids, claim1_dict):
    """Resolve a list of citation IDs to claim 1 texts. Returns list of (id, text) tuples."""
    if not isinstance(cite_ids, list):
        return []
    out = []
    for cid in cite_ids:
        text = claim1_dict.get(str(cid))
        if text:
            out.append((cid, text))
    return out


def format_with_cites(abstract, claims, cite_texts, max_chars):
    """Build full text with cited prior-art context. Truncates to fit budget."""
    base = f"ABSTRACT:\n{abstract or ''}\n\nCLAIMS:\n{claims or ''}"
    if not cite_texts:
        return base[:max_chars]

    # Allocate ~half of remaining budget to citations
    cite_budget = max(2000, (max_chars - len(base)) // 1)
    cite_chunks = []
    used = 0
    for i, (cid, text) in enumerate(cite_texts):
        chunk = f"\n[Cited Patent {cid}, claim 1]:\n{text}"
        if used + len(chunk) > cite_budget:
            break
        cite_chunks.append(chunk)
        used += len(chunk)

    if cite_chunks:
        return (base + "\n\nCITED PRIOR ART:" + "".join(cite_chunks))[:max_chars]
    return base[:max_chars]


def write_csv(rows, out_path):
    print(f"Writing {len(rows):,} rows to {out_path}...")
    with gzip.open(out_path, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "judgement"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)

    # Load claim 1 lookup as dict for fast access
    print(f"Loading claim 1 lookup from {args.claim1_lookup}...")
    lookup_df = pd.read_parquet(args.claim1_lookup)
    print(f"  Loaded {len(lookup_df):,} patents")
    claim1_dict = dict(zip(lookup_df["patent_id"].astype(str), lookup_df["claim_1"]))
    del lookup_df

    # Load applicant cites from PatentsView (keyed by patent_id, granted only — OK for Task A)
    applicant_cites, _ = load_citations_with_categories(
        args.g_citation, args.top_k_cites
    )

    # Load examiner cites from OARD (keyed by app_id, covers granted AND abandoned)
    oard_examiner_cites = load_oard_examiner_cites(
        args.oard_citations, args.top_k_cites
    )

    # Build pgpub_id -> application_number mapping from pg_published_application
    # so we can join OARD cites (keyed by app_id) to JSONL rows (keyed by pgpub_id)
    pg_app_path = Path(args.claim1_lookup).parent.parent / "raw/patentsview_pg/pg_published_application.tsv.zip"
    print(f"Loading pgpub -> application_number mapping from {pg_app_path}...")
    pg_app = read_tsv_zip(str(pg_app_path), usecols=["pgpub_id", "application_id"],
                          dtype={"pgpub_id": "str", "application_id": "str"})
    pg_app = pg_app.rename(columns={"application_id": "application_number"})
    pgpub_to_appnum = dict(zip(pg_app["pgpub_id"].astype(str), pg_app["application_number"].astype(str)))
    print(f"  {len(pgpub_to_appnum):,} pgpub -> app_number mappings")
    del pg_app

    # Re-key OARD examiner cites from app_id -> pgpub_id
    # (JSONL rows are keyed by pgpub_id, OARD by application_number)
    appnum_to_pgpub = {v: k for k, v in pgpub_to_appnum.items()}
    print("Re-keying OARD examiner cites from app_id to pgpub_id...")
    oard_by_pgpub = {}
    n_mapped = 0
    for app_id, cite_list in oard_examiner_cites.items():
        pgpub = appnum_to_pgpub.get(str(app_id))
        if pgpub:
            oard_by_pgpub[pgpub] = cite_list
            n_mapped += 1
    oard_examiner_by_pgpub = pd.Series(oard_by_pgpub, name="oard_examiner_cites")
    oard_examiner_by_pgpub.index.name = "pgpub_id"
    print(f"  Mapped {n_mapped:,} / {len(oard_examiner_cites):,} to pgpub_id")
    del pgpub_to_appnum, appnum_to_pgpub, oard_by_pgpub

    # Load main dataset
    df = load_rich_jsonl(args.rich_jsonl)

    # Make sure join keys are strings
    if "patent_id" in df.columns:
        df["patent_id"] = df["patent_id"].astype(str)

    # Join applicant cites on patent_id (PatentsView)
    df = df.join(applicant_cites, on="patent_id")
    # Join examiner cites on pgpub_id (OARD, re-keyed via pg_published_application)
    if "pgpub_id" in df.columns:
        df["pgpub_id"] = df["pgpub_id"].astype(str)
    df = df.join(oard_examiner_by_pgpub, on="pgpub_id")
    print(f"After join: {df.shape}")
    print(f"  with applicant_cites:      {df['applicant_cites'].apply(lambda x: isinstance(x, list)).sum():,}")
    print(f"  with oard_examiner_cites:  {df['oard_examiner_cites'].apply(lambda x: isinstance(x, list)).sum():,}")

    # ---- Build the two non-leaky augmented variants ----
    print("\nBuilding augmented texts...")
    rows_rough_author, rows_final_examiner = [], []

    n_skipped_no_oa = 0

    for i, row in enumerate(df.itertuples(index=False)):
        if (i + 1) % 250000 == 0:
            print(f"  processed {i+1:,} rows")

        abstract = getattr(row, "pg_abstract", "") or ""
        claims = getattr(row, "pg_claims", "") or ""
        if len(claims) < 50:
            continue

        first_draft_approved = bool(getattr(row, "first_draft_approved", False))
        final_outcome = getattr(row, "final_outcome", "")
        n_oa = int(getattr(row, "n_office_actions", 0) or 0)

        applicant_list = getattr(row, "applicant_cites", None)
        examiner_list = getattr(row, "oard_examiner_cites", None)

        applicant_texts = lookup_cite_texts(applicant_list, claim1_dict)
        examiner_texts = lookup_cite_texts(examiner_list, claim1_dict)

        # ---- Variant 1: rough-draft prediction + applicant cites ----
        if not args.task_b_only:
            text_rough = format_with_cites(abstract, claims, applicant_texts, args.max_text_chars)
            rows_rough_author.append(
                {"text": text_rough, "judgement": int(first_draft_approved)}
            )

        # ---- Variant 2: final-draft prediction + examiner cites ----
        # Label: granted vs abandoned (only meaningful when outcome is decided)
        # Examiner cites happen during prosecution, BEFORE final outcome is decided
        if final_outcome in ("granted", "abandoned"):
            # Optional: skip apps with no OA (residual leakage signal)
            if args.require_oa and n_oa == 0:
                n_skipped_no_oa += 1
                continue
            text_final = format_with_cites(abstract, claims, examiner_texts, args.max_text_chars)
            rows_final_examiner.append(
                {"text": text_final, "judgement": int(final_outcome == "granted")}
            )

    if args.require_oa:
        print(f"  Skipped {n_skipped_no_oa:,} Task B rows with no office action (--require-oa)")

    if not args.task_b_only:
        write_csv(rows_rough_author, out_dir / "patents_first_draft_with_applicant_cites.csv.gz")
    write_csv(rows_final_examiner, out_dir / "patents_final_outcome_with_examiner_cites.csv.gz")

    print("\nDone! Output files:")
    files = []
    if not args.task_b_only:
        files.append("patents_first_draft_with_applicant_cites.csv.gz")
    files.append("patents_final_outcome_with_examiner_cites.csv.gz")
    for f in files:
        path = out_dir / f
        if path.exists():
            print(f"  {f}: {path.stat().st_size / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
