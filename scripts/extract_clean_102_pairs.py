#!/usr/bin/env python3
"""Extract CLEAN §102 training pairs using action_type='102' filter.

For each row in oard_citations.csv with action_type='102':
  - rejected app's pgpub_id (via PatEx app_id → pgpub mapping)
  - cited reference's pgpub_id (the citation_pat_pgpub_id field)
  - look up both texts (from JSONL + parquets)
  - emit pair

Output: processed/clean_102_pairs.jsonl.gz
  Each row: {anchor_pgpub_id, anchor_text, positive_pgpub_id, positive_text,
             rejected_app_id, ifw_number}
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
OARD_CITES = f"{BASE}/raw/oard/oard_citations.csv"
GRANTED_PARQUET = f"{BASE}/processed/granted_patents_claim1.parquet"
PGPUB_PARQUET = f"{BASE}/processed/pgpub_claims1.parquet"
LEGACY_PARQUET = f"{BASE}/processed/claim1_lookup.parquet"
OUTPUT = f"{BASE}/processed/clean_102_pairs.jsonl.gz"


def normalize_pgpub(s):
    if not s: return ""
    s = s.strip().upper()
    return re.sub(r"^US|[A-Z]\d$", "", s).lstrip("0")


def main():
    # 1. Filter OARD citations to action_type='102'
    print("Filtering OARD citations to action_type='102' ...", file=sys.stderr)
    pairs_raw = []
    with open(OARD_CITES) as f:
        rdr = csv.DictReader(f)
        for n, r in enumerate(rdr, 1):
            if r.get("action_type") != "102":
                continue
            app = r.get("app_id", "").strip()
            cited = (r.get("citation_pat_pgpub_id") or "").strip()
            ifw = r.get("ifw_number", "").strip()
            if app and cited:
                pairs_raw.append((app, cited, ifw))
            if n % 10_000_000 == 0:
                print(f"  scanned {n:,}  102-pairs so far: {len(pairs_raw):,}", file=sys.stderr)
    print(f"  total §102 pairs (app, cited, ifw): {len(pairs_raw):,}")

    # 2. PatEx mapping app_id → pgpub_id
    print("Loading PatEx app→pgpub ...", file=sys.stderr)
    app_to_pgpub = {}
    with open(PATEX) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            pgpub = normalize_pgpub(r.get("earliest_pgpub_number", ""))
            app = r.get("application_number", "").strip()
            if pgpub and app:
                app_to_pgpub[app] = pgpub
    print(f"  {len(app_to_pgpub):,} mappings")

    # 3. Resolve anchor pgpubs + cited pgpubs
    needed = set()
    valid = []
    for app, cited, ifw in pairs_raw:
        anchor_pid = app_to_pgpub.get(app)
        cited_norm = normalize_pgpub(cited)
        if not anchor_pid or not cited_norm:
            continue
        needed.add(anchor_pid)
        needed.add(cited_norm)
        valid.append((app, anchor_pid, cited_norm, ifw))
    print(f"  pairs with both anchor + cited normalized: {len(valid):,}")

    # 4. Text lookup from multiple sources, one pass each
    texts = {}
    print("Streaming JSONL ...", file=sys.stderr)
    with gzip.open(JSONL, "rt") as f:
        for line in f:
            try: d = json.loads(line)
            except Exception: continue
            pid = str(d.get("pgpub_id", "")).strip().lstrip("0")
            if pid in needed and pid not in texts:
                claims = (d.get("pg_claims") or "").strip()
                if claims:
                    texts[pid] = claims[:8000]
            ptid = str(d.get("patent_id", "")).strip().lstrip("0")
            if ptid in needed and ptid not in texts:
                claims = (d.get("g_claims") or d.get("pg_claims") or "").strip()
                if claims:
                    texts[ptid] = claims[:8000]
            if len(texts) >= len(needed):
                break

    for parquet_path, id_col, text_col in [
        (GRANTED_PARQUET, "patent_id", "claim_text"),
        (PGPUB_PARQUET, "pgpub_id", "claim_text"),
        (LEGACY_PARQUET, "patent_id", "claim_1"),
    ]:
        if not os.path.exists(parquet_path): continue
        print(f"Loading {parquet_path} ...", file=sys.stderr)
        tbl = pq.read_table(parquet_path, columns=[id_col, text_col])
        for row in tbl.to_pylist():
            pid = str(row[id_col]).strip().lstrip("0")
            if pid in needed and pid not in texts:
                t = row[text_col]
                if t:
                    texts[pid] = t[:8000]
        print(f"  resolved so far: {len(texts):,}/{len(needed):,}")

    # 5. Write pairs
    print(f"Writing pairs to {OUTPUT} ...", file=sys.stderr)
    n_written = 0
    n_drop = 0
    with gzip.open(OUTPUT, "wt") as fout:
        for app, anchor_pid, cited_norm, ifw in valid:
            if anchor_pid in texts and cited_norm in texts:
                fout.write(json.dumps({
                    "anchor_text": texts[anchor_pid],
                    "positive_text": texts[cited_norm],
                    "anchor_pgpub_id": anchor_pid,
                    "positive_pgpub_id": cited_norm,
                    "rejected_app_id": app,
                    "ifw_number": ifw,
                }) + "\n")
                n_written += 1
            else:
                n_drop += 1

    print(f"Done. {n_written:,} pairs written. {n_drop:,} dropped (no text)")


if __name__ == "__main__":
    main()
