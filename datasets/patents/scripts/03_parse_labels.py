#!/usr/bin/env python3
"""
Parse PatEx to derive per-application labels.

Sources:
- transactions.csv: event log (13GB). Use to detect office actions / allowance.
- application_data.csv: per-app metadata (3.3GB). Use to get grant status (patent_number).

Output: parquet file with per-application labels.

Event code logic (verified against PatEx event_codes.csv):
- Office actions:   CTNF, CTFR, CTAV, CTEQ
- Allowance:        MN/=., N/=., N/=A, N/=F
- Abandonment:      any code starting with 'ABN' or 'MABN'

Label: first_draft_approved = True iff a Notice of Allowance occurred AND no
office action event occurred before it.

Grant status comes from application_data.patent_number (not transactions).

LABEL-LOGIC CAVEATS (dual audit Codex+Fable 2026-07-13 — outputs frozen, documented here):
- CTAV (advisory action: exists only if the applicant files after final = applicant-strategy
  event) and CTEQ (Quayle: substance already allowable) are counted as office actions. This
  distorts first_draft_approved and inflates n_office_actions; events are also counted as raw
  rows with no (code, date) dedup. Corrected, dated, risk-set-conditioned outcomes live in
  scripts/patents_event_panel.py (n_oa_rounds counts CTNF/CTFR only).
- Abandonment = ANY historical ABN*/MABN* event; status_pending is computed but ignored in
  final_outcome — a revived, currently-pending app can be labeled abandoned. ~3.3% of
  abandoned apps in the outcome cohort had a Notice of Allowance BEFORE abandoning.
- No examiner/art-unit fields are read here (or anywhere downstream of this file before
  2026-07-13) — examiner-leniency variance is uncontrolled in every consumer. The event
  panel joins examiner_full_name/examiner_art_unit and computes LOO leniency rates.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


OFFICE_ACTION_CODES = {
    "CTNF",   # Non-Final Rejection
    "CTFR",   # Final Rejection
    "CTAV",   # Advisory Action
    "CTEQ",   # Quayle action (formality/minor rejection)
}

ALLOWANCE_CODES = {
    "MN/=.",  # Mail Notice of Allowance
    "N/=.",   # Notice of Allowance
    "N/=A",   # Notice of Allowance with Examiner's Amendment
    "N/=F",   # Notice of Allowance with Issue Fee Charged
}

# Abandonment codes use prefix matching: anything starting with ABN or MABN
ABANDON_PREFIXES = ("ABN", "MABN")

ALL_EXACT_CODES = OFFICE_ACTION_CODES | ALLOWANCE_CODES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--transactions",
        default="/lfs/skampere3/0/alexspan/norm-research/datasets/patents/raw/patex/transactions.csv",
    )
    p.add_argument(
        "--application-data",
        default="/lfs/skampere3/0/alexspan/norm-research/datasets/patents/raw/patex/application_data.csv",
    )
    p.add_argument(
        "--output",
        default="/lfs/skampere3/0/alexspan/norm-research/datasets/patents/processed/labels.parquet",
    )
    p.add_argument(
        "--summary-output",
        default="/lfs/skampere3/0/alexspan/norm-research/datasets/patents/processed/labels_summary.json",
    )
    p.add_argument("--chunksize", type=int, default=2_000_000)
    return p.parse_args()


def load_transactions_filtered(path, chunksize):
    """Stream transactions.csv, keep only office action and allowance events."""
    print(f"Streaming transactions from {path} (chunksize={chunksize:,})...")
    chunks = []
    total_read = 0
    for i, chunk in enumerate(
        pd.read_csv(
            path,
            chunksize=chunksize,
            usecols=["application_number", "event_code", "recorded_date"],
            dtype={"application_number": "str", "event_code": "str"},
            on_bad_lines="skip",
            low_memory=False,
        )
    ):
        total_read += len(chunk)
        # Keep exact-match OA + allowance, plus any code starting with ABN/MABN
        mask_exact = chunk["event_code"].isin(ALL_EXACT_CODES)
        mask_abn = chunk["event_code"].str.startswith(("ABN", "MABN"), na=False)
        chunk = chunk[mask_exact | mask_abn]
        chunks.append(chunk)
        if (i + 1) % 5 == 0:
            kept = sum(len(c) for c in chunks)
            print(f"  Read {total_read:,} rows, kept {kept:,} relevant ({kept/total_read*100:.2f}%)")

    df = pd.concat(chunks, ignore_index=True)
    print(f"Total events read: {total_read:,}; kept {len(df):,} relevant ({len(df)/total_read*100:.2f}%)")
    return df


def derive_event_labels(transactions):
    """Compute first-draft-approved and related flags from event log."""
    print("Parsing dates...")
    transactions["recorded_date"] = pd.to_datetime(
        transactions["recorded_date"], errors="coerce"
    )
    transactions = transactions.dropna(subset=["recorded_date"])

    print("Computing first office action dates...")
    oa_mask = transactions["event_code"].isin(OFFICE_ACTION_CODES)
    first_oa = (
        transactions[oa_mask]
        .groupby("application_number")["recorded_date"]
        .min()
        .rename("first_oa_date")
    )

    print("Computing first allowance dates...")
    allow_mask = transactions["event_code"].isin(ALLOWANCE_CODES)
    first_allow = (
        transactions[allow_mask]
        .groupby("application_number")["recorded_date"]
        .min()
        .rename("first_allow_date")
    )

    print("Counting office actions per application...")
    oa_counts = (
        transactions[oa_mask]
        .groupby("application_number")
        .size()
        .rename("n_office_actions")
    )

    print("Flagging abandonment events...")
    abn_mask = transactions["event_code"].str.startswith(("ABN", "MABN"), na=False)
    has_abandon = set(transactions[abn_mask]["application_number"].unique())

    # Build per-application event summary
    print("Building per-application event table...")
    apps = pd.Series(transactions["application_number"].unique(), name="application_number")
    ev = pd.DataFrame({"application_number": apps})
    ev = ev.merge(first_oa, on="application_number", how="left")
    ev = ev.merge(first_allow, on="application_number", how="left")
    ev = ev.merge(oa_counts, on="application_number", how="left")
    ev["n_office_actions"] = ev["n_office_actions"].fillna(0).astype(int)
    ev["had_abandonment_event"] = ev["application_number"].isin(has_abandon)

    # First-draft approved: allowance occurred AND (no OA OR allowance before OA)
    has_allow = ev["first_allow_date"].notna()
    no_oa_before_allow = ev["first_oa_date"].isna() | (
        ev["first_allow_date"] < ev["first_oa_date"]
    )
    ev["first_draft_approved"] = has_allow & no_oa_before_allow

    return ev


def load_application_grant_status(path):
    """Use application_data.patent_number to determine granted vs not."""
    print(f"Loading application_data from {path}...")
    df = pd.read_csv(
        path,
        usecols=[
            "application_number",
            "patent_number",
            "appl_status_desc",
            "filing_date",
            "application_invention_type",
        ],
        dtype={"application_number": "str", "patent_number": "str"},
        on_bad_lines="skip",
        low_memory=False,
    )
    print(f"  Loaded {len(df):,} applications")

    df["is_granted"] = df["patent_number"].notna() & (df["patent_number"].str.strip() != "")
    df["filing_year"] = pd.to_datetime(df["filing_date"], errors="coerce").dt.year

    # Identify abandoned from status text
    status_lower = df["appl_status_desc"].fillna("").str.lower()
    df["status_abandoned"] = status_lower.str.contains("abandon", na=False)
    df["status_pending"] = status_lower.str.contains("pending|preexam|dispatched|docketed", na=False)

    # Keep only utility applications; drop provisional / design / plant if possible
    if "application_invention_type" in df.columns:
        before = len(df)
        df = df[df["application_invention_type"].fillna("").str.upper().isin(["UTILITY", "REISSUE", ""])]
        print(f"  Filtered to utility/reissue: {len(df):,} (was {before:,})")

    return df


def summarize(labels):
    total = len(labels)
    n_first_draft = int(labels["first_draft_approved"].fillna(False).sum())
    n_granted = int(labels["is_granted"].sum())
    n_abandoned = int(labels["status_abandoned"].sum())
    pending = total - n_granted - n_abandoned

    summary = {
        "total_applications": total,
        "granted": n_granted,
        "abandoned": n_abandoned,
        "other_or_pending": pending,
        "first_draft_approved_any": n_first_draft,
        "first_draft_approved_and_granted": int(
            (labels["first_draft_approved"].fillna(False) & labels["is_granted"]).sum()
        ),
        "pct_first_draft_approved_of_total": round(n_first_draft / total * 100, 2),
        "pct_first_draft_approved_of_granted": round(
            int((labels["first_draft_approved"].fillna(False) & labels["is_granted"]).sum())
            / max(n_granted, 1)
            * 100,
            2,
        ),
        "n_office_actions_distribution": (
            labels["n_office_actions"].fillna(0).astype(int).value_counts().sort_index().head(10).to_dict()
        ),
        "filing_year_distribution": (
            labels["filing_year"].dropna().astype(int).value_counts().sort_index().tail(20).to_dict()
        ),
    }
    return summary


def main():
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Event-based labels
    tx = load_transactions_filtered(args.transactions, args.chunksize)
    ev = derive_event_labels(tx)

    # Grant status from application_data
    app = load_application_grant_status(args.application_data)

    # Merge
    print("Merging event labels + app metadata...")
    labels = app.merge(ev, on="application_number", how="left")
    labels["n_office_actions"] = labels["n_office_actions"].fillna(0).astype(int)
    labels["first_draft_approved"] = labels["first_draft_approved"].fillna(False)
    labels["had_abandonment_event"] = labels["had_abandonment_event"].fillna(False)

    # Final outcome
    def outcome(row):
        if row["is_granted"]:
            return "granted"
        if row["status_abandoned"] or row["had_abandonment_event"]:
            return "abandoned"
        return "pending"

    labels["final_outcome"] = labels.apply(outcome, axis=1)

    summary = summarize(labels)
    print("Summary:")
    print(json.dumps(summary, indent=2, default=str))

    print(f"Writing labels to {args.output}...")
    labels.to_parquet(args.output, index=False)

    with open(args.summary_output, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote summary to {args.summary_output}")


if __name__ == "__main__":
    main()
