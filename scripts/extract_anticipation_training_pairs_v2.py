#!/usr/bin/env python3
"""v2 of extract_anticipation_training_pairs.py.

Same logic but uses BOTH text sources:
  - patents_dataset.jsonl.gz (4.7M pre-grant pubs)
  - granted_patents_claim1.parquet (12M granted patents' claim-1)

This should close most of the 78% cited-ref gap that v1 had.

Mapping logic:
  - The "anchor" (rejected app) is identified by app_id → pgpub_id (via PatEx)
    → text from JSONL (pg_claims).
  - The "positive" (cited reference) is identified by a raw cite string from
    OARD. Normalized cite id is tried first against JSONL pgpub_id, then
    against granted patent_id.
"""
import csv
import gzip
import json
import os
import re
import sys
from collections import defaultdict

import pyarrow.parquet as pq

csv.field_size_limit(2**31 - 1)

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
JSONL = f"{BASE}/patents_dataset.jsonl.gz"
PATEX = f"{BASE}/raw/patex/application_data.csv"
OARD_REJ = f"{BASE}/raw/oard/oard_rejections_by_app.csv"
OARD_CITES = f"{BASE}/raw/oard/oard_citations.csv"
GRANTED_PARQUET = f"{BASE}/processed/granted_patents_claim1.parquet"
PGPUB_PARQUET = f"{BASE}/processed/pgpub_claims1.parquet"
LEGACY_CLAIM1_LOOKUP = f"{BASE}/processed/claim1_lookup.parquet"
OUTPUT = f"{BASE}/processed/anticipation_training_pairs_v2.jsonl.gz"


def normalize_pgpub(s):
    if not s: return ""
    s = s.strip().upper()
    s = re.sub(r"^US", "", s)
    s = re.sub(r"[A-Z]\d$", "", s)
    return re.sub(r"[^0-9]", "", s)


def main():
    print("Loading PatEx pgpub→app mapping...", file=sys.stderr)
    pgpub_to_app = {}
    app_to_pgpub = {}
    with open(PATEX) as f:
        rdr = csv.DictReader(f)
        for n, r in enumerate(rdr, 1):
            pgpub = normalize_pgpub(r.get("earliest_pgpub_number", ""))
            app = r.get("application_number", "").strip()
            if pgpub and app:
                pgpub_to_app[pgpub] = app
                app_to_pgpub[app] = pgpub
    print(f"  {len(pgpub_to_app):,} pgpub→app mappings", file=sys.stderr)

    print("Loading apps with rejected_102=True...", file=sys.stderr)
    apps_with_102 = set()
    with open(OARD_REJ) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            if int(r.get("rejected_102", 0) or 0) == 1:
                apps_with_102.add(r["app_id"].strip())
    print(f"  {len(apps_with_102):,} apps with §102 rejection", file=sys.stderr)

    print("Enumerating examiner cites for §102 apps...", file=sys.stderr)
    app_to_cites = defaultdict(set)
    with open(OARD_CITES) as f:
        rdr = csv.DictReader(f)
        for n, r in enumerate(rdr, 1):
            app = r.get("app_id", "").strip()
            if app not in apps_with_102:
                continue
            cited = (r.get("citation_pat_pgpub_id") or "").strip()
            if cited:
                app_to_cites[app].add(normalize_pgpub(cited))
            if n % 10_000_000 == 0:
                print(f"  scanned {n:,} OARD rows", file=sys.stderr)
    print(f"  {len(app_to_cites):,} apps have cited refs", file=sys.stderr)

    # Build needed-pgpub set (anchors + positives)
    needed_ids = set()
    for app in app_to_cites:
        pid = app_to_pgpub.get(app)
        if pid: needed_ids.add(pid)
    for cites in app_to_cites.values():
        needed_ids.update(cites)
    print(f"  {len(needed_ids):,} distinct IDs needed for text", file=sys.stderr)

    # Source 1: JSONL pre-grant pubs (anchor = app's pg_claims)
    print("Streaming JSONL for pgpub text...", file=sys.stderr)
    id_to_text = {}
    with gzip.open(JSONL, "rt") as f:
        for n, line in enumerate(f, 1):
            try:
                d = json.loads(line)
            except Exception:
                continue
            pid = str(d.get("pgpub_id", "")).strip()
            if pid in needed_ids and pid not in id_to_text:
                t = (d.get("pg_claims") or "").strip()
                if t:
                    id_to_text[pid] = t[:4000]
            # Also use patent_id (granted) if cited
            ptid = str(d.get("patent_id", "")).strip()
            if ptid in needed_ids and ptid not in id_to_text:
                t = (d.get("g_claims") or d.get("pg_claims") or "").strip()
                if t:
                    id_to_text[ptid] = t[:4000]
            if n % 500_000 == 0:
                print(f"  JSONL scanned {n:,}, resolved {len(id_to_text):,}", file=sys.stderr)
    print(f"  resolved via JSONL: {len(id_to_text):,}", file=sys.stderr)

    # Source 2: granted patent parquet (claim-1 by patent_id)
    if os.path.exists(GRANTED_PARQUET):
        print(f"Loading granted parquet {GRANTED_PARQUET} ...", file=sys.stderr)
        table = pq.read_table(GRANTED_PARQUET)
        added = 0
        for row in table.to_pylist():
            pid = str(row["patent_id"]).strip()
            if pid in needed_ids and pid not in id_to_text:
                t = row["claim_text"]
                if t:
                    id_to_text[pid] = t[:4000]
                    added += 1
        print(f"  resolved via granted parquet: +{added:,} (total: {len(id_to_text):,})", file=sys.stderr)
    else:
        print(f"  NOTE: {GRANTED_PARQUET} not found; skipping granted source", file=sys.stderr)

    # Source 3: pgpub parquet (claim-1 by pgpub_id)
    if os.path.exists(PGPUB_PARQUET):
        print(f"Loading pgpub parquet {PGPUB_PARQUET} ...", file=sys.stderr)
        table = pq.read_table(PGPUB_PARQUET)
        added = 0
        for row in table.to_pylist():
            pid = str(row["pgpub_id"]).strip()
            if pid in needed_ids and pid not in id_to_text:
                t = row["claim_text"]
                if t:
                    id_to_text[pid] = t[:4000]
                    added += 1
        print(f"  resolved via pgpub parquet: +{added:,} (total: {len(id_to_text):,})", file=sys.stderr)
    else:
        print(f"  NOTE: {PGPUB_PARQUET} not found; skipping pgpub source", file=sys.stderr)

    # Source 4: legacy claim1 lookup (older + design + plant + reissue patents)
    if os.path.exists(LEGACY_CLAIM1_LOOKUP):
        print(f"Loading legacy claim1 lookup {LEGACY_CLAIM1_LOOKUP} ...", file=sys.stderr)
        table = pq.read_table(LEGACY_CLAIM1_LOOKUP)
        added = 0
        for row in table.to_pylist():
            pid = str(row["patent_id"]).strip()
            if pid in needed_ids and pid not in id_to_text:
                t = row["claim_1"]
                if t:
                    id_to_text[pid] = t[:4000]
                    added += 1
        print(f"  resolved via legacy lookup: +{added:,} (total: {len(id_to_text):,})", file=sys.stderr)

    # Emit pairs
    print(f"\nWriting pairs to {OUTPUT}...", file=sys.stderr)
    n_pairs = 0
    n_apps_skip = 0
    n_cites_skip = 0
    with gzip.open(OUTPUT, "wt") as fout:
        for app, cites in app_to_cites.items():
            anchor_pid = app_to_pgpub.get(app)
            if not anchor_pid or anchor_pid not in id_to_text:
                n_apps_skip += 1
                continue
            anchor_text = id_to_text[anchor_pid]
            for cited_pid in cites:
                if cited_pid not in id_to_text:
                    n_cites_skip += 1
                    continue
                pos_text = id_to_text[cited_pid]
                fout.write(json.dumps({
                    "anchor_text": anchor_text,
                    "positive_text": pos_text,
                    "anchor_pgpub_id": anchor_pid,
                    "positive_pgpub_id": cited_pid,
                    "rejected_app_id": app,
                }) + "\n")
                n_pairs += 1
                if n_pairs % 200_000 == 0:
                    print(f"  written {n_pairs:,}", file=sys.stderr)

    print(f"\nDone.")
    print(f"  pairs written:                       {n_pairs:,}")
    print(f"  apps with no anchor text:            {n_apps_skip:,}")
    print(f"  cites with no positive text:         {n_cites_skip:,}")


if __name__ == "__main__":
    main()
