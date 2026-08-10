#!/usr/bin/env python3
"""Query USPTO OARD rejections table to count applications per rejection ground.

The rejections table has one row per (office_action, rejection_type) pair.
Each row indicates whether that OA included a rejection on a given § ground.

We want: count of DISTINCT applications that received at least one rejection
on each ground.

In our first-draft-approval task:
  - label=1 (first_draft_approved=True): zero office actions → 0 rejections on every ground.
  - label=0: received ≥1 OA → 1+ rejection grounds across the OAs.
So the per-§ count below is the count of label=0 applications by ground.
"""
from google.cloud import bigquery

client = bigquery.Client(project="usc-research")

# Schema discovery first.
q = """
SELECT column_name, data_type
FROM `patents-public-data.uspto_oce_office_actions.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'rejections'
ORDER BY ordinal_position
"""
print("=== schema ===")
cols = list(client.query(q).result())
for c in cols:
    print(f"  {c.column_name}  ({c.data_type})")

# Count distinct apps overall and per-rejection.
# OARD's rejections table column names should be like rejection_101, rejection_102,
# alice (for §101), rejection_112_para_a, etc. Use schema output to confirm.
q2 = "SELECT COUNT(*) n, COUNT(DISTINCT app_id) n_apps FROM `patents-public-data.uspto_oce_office_actions.rejections`"
res = list(client.query(q2).result())[0]
print(f"\n=== overall ===")
print(f"  Total rejection rows: {res.n:,}")
print(f"  Distinct applications: {res.n_apps:,}  (= label=0 pool in OARD coverage)")

# Per-ground counts (uses BOOLEAN columns the schema reveals).
# Try the standard rejection-ground column names.
# Auto-detect rejection-ground columns from schema (boolean columns starting with 'rejection_' or 'alice')
existing_cols = [c for c in cols if c.data_type in ("BOOL", "BOOLEAN", "INT64") and
                 (c.column_name.startswith("rejection_") or c.column_name in ("alice",))]
print("\n=== detected rejection-ground columns ===")
for c in existing_cols:
    print(f"  {c.column_name}  ({c.data_type})")

# Per-ground distinct app counts (single query for efficiency)
flag_exprs = []
for c in existing_cols:
    flag_exprs.append(
        f"COUNT(DISTINCT IF({c.column_name} IS TRUE OR {c.column_name} = 1, app_id, NULL)) AS {c.column_name}"
    )
if flag_exprs:
    q3 = (
        "SELECT COUNT(DISTINCT app_id) AS total_apps,\n  " +
        ",\n  ".join(flag_exprs) +
        "\nFROM `patents-public-data.uspto_oce_office_actions.rejections`"
    )
    print("\n=== running aggregate query ===")
    r = list(client.query(q3).result())[0]
    print(f"\n  Total distinct apps in rejections table: {r.total_apps:,}")
    print(f"\n  {'column':30s} | {'distinct_apps':>14s} | {'% of label=0':>14s}")
    for c in existing_cols:
        n = getattr(r, c.column_name)
        pct = n / r.total_apps * 100 if r.total_apps else 0
        print(f"  {c.column_name:30s} | {n:>14,d} | {pct:>13.1f}%")
