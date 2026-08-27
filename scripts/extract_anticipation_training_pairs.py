#!/usr/bin/env python3
"""Extract (rejected_claim, anticipating_cite) pairs from OARD + JSONL.

Strategy: for each application with rejected_102=True, pair its first claim
text with each of its examiner-cited references that we also have text for.

Noisy but high-recall: some cites listed in OARD for an app where §102
appeared may have actually triggered §103 instead. The (app_id, ifw_number)
join would give cleaner pairs but at the cost of recall; we go noisy here
to maximize training-pair count per user's "as many as possible" request.

Inputs:
  - patents_dataset.jsonl.gz (4.7M apps with pg_abstract + pg_claims)
  - PatEx application_data.csv (pgpub_id → app_id mapping)
  - oard_rejections_by_app.csv (per-app rejection flags)
  - oard_citations.csv (per-OA citations with citation_pat_pgpub_id)

Output: training_pairs.jsonl with {anchor_text, positive_text, anchor_pgpub_id, positive_pgpub_id}
"""
import csv
import gzip
import json
import re
import sys
from collections import defaultdict

csv.field_size_limit(2**31 - 1)

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
JSONL = f"{BASE}/patents_dataset.jsonl.gz"
PATEX = f"{BASE}/raw/patex/application_data.csv"
OARD_REJ = f"{BASE}/raw/oard/oard_rejections_by_app.csv"
OARD_CITES = f"{BASE}/raw/oard/oard_citations.csv"
OUTPUT = f"{BASE}/processed/anticipation_training_pairs.jsonl.gz"

import os
os.makedirs(f"{BASE}/processed", exist_ok=True)


def normalize_pgpub(s):
    if not s: return ""
    s = s.strip().upper()
    s = re.sub(r"^US", "", s)
    s = re.sub(r"[A-Z]\d$", "", s)
    return re.sub(r"[^0-9]", "", s)


def main():
    # Step 1: PatEx pgpub → app_id mapping (we built this before, redo here)
    print("Loading PatEx pgpub_id → app_id mapping...", file=sys.stderr)
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
            if n % 2_000_000 == 0:
                print(f"  PatEx scanned {n:,}, mapped {len(pgpub_to_app):,}", file=sys.stderr)
    print(f"  {len(pgpub_to_app):,} pgpub→app mappings", file=sys.stderr)

    # Step 2: load apps with rejected_102=True
    print("Loading apps with rejected_102=True...", file=sys.stderr)
    apps_with_102 = set()
    with open(OARD_REJ) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            if int(r.get("rejected_102", 0) or 0) == 1:
                apps_with_102.add(r["app_id"].strip())
    print(f"  {len(apps_with_102):,} apps with §102 rejection", file=sys.stderr)

    # Step 3: enumerate examiner-cited references for those apps
    print("Enumerating examiner cites for §102 apps...", file=sys.stderr)
    app_to_cites = defaultdict(set)
    n_lines = 0
    with open(OARD_CITES) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            n_lines += 1
            app = r.get("app_id", "").strip()
            if app not in apps_with_102:
                continue
            cited = (r.get("citation_pat_pgpub_id") or "").strip()
            if cited:
                app_to_cites[app].add(normalize_pgpub(cited))
            if n_lines % 5_000_000 == 0:
                print(f"  scanned {n_lines:,} OARD rows, {len(app_to_cites):,} apps with cites", file=sys.stderr)
    print(f"  {len(app_to_cites):,} apps have cited refs", file=sys.stderr)
    avg_cites = sum(len(c) for c in app_to_cites.values()) / max(1, len(app_to_cites))
    print(f"  avg cites per app: {avg_cites:.1f}", file=sys.stderr)

    # Step 4: stream JSONL once, build pgpub → text map for ALL pgpubs we might need
    needed_pgpubs = set()
    # Source side (the rejected app)
    for app in app_to_cites:
        pid = app_to_pgpub.get(app)
        if pid: needed_pgpubs.add(pid)
    # Target side (cited refs)
    for cites in app_to_cites.values():
        needed_pgpubs.update(cites)
    print(f"  {len(needed_pgpubs):,} distinct pgpub_ids needed for text lookup", file=sys.stderr)

    print("Streaming JSONL for text...", file=sys.stderr)
    pgpub_to_text = {}
    with gzip.open(JSONL, "rt") as f:
        for n_apps_seen, line in enumerate(f, 1):
            try:
                d = json.loads(line)
            except Exception:
                continue
            pid = str(d.get("pgpub_id", "")).strip()
            if pid in needed_pgpubs:
                claims = (d.get("pg_claims") or "").strip()
                abstract = (d.get("pg_abstract") or "").strip()
                if claims or abstract:
                    pgpub_to_text[pid] = {
                        "claims": claims[:4000],   # truncate to keep memory reasonable
                        "abstract": abstract[:1000],
                    }
            if n_apps_seen % 500_000 == 0:
                print(f"  scanned {n_apps_seen:,} JSONL rows, indexed {len(pgpub_to_text):,}", file=sys.stderr)
    print(f"  {len(pgpub_to_text):,} pgpub_ids resolved to text", file=sys.stderr)

    # Step 5: emit pairs
    print(f"Writing pairs to {OUTPUT}...", file=sys.stderr)
    n_pairs = 0
    n_apps_skipped_no_source = 0
    n_skipped_no_text = 0
    with gzip.open(OUTPUT, "wt") as fout:
        for app, cites in app_to_cites.items():
            anchor_pid = app_to_pgpub.get(app)
            if not anchor_pid or anchor_pid not in pgpub_to_text:
                n_apps_skipped_no_source += 1
                continue
            anchor_text = pgpub_to_text[anchor_pid]["claims"]
            if not anchor_text:
                continue
            for cited_pid in cites:
                if cited_pid not in pgpub_to_text:
                    n_skipped_no_text += 1
                    continue
                pos_text = pgpub_to_text[cited_pid]["claims"] or pgpub_to_text[cited_pid]["abstract"]
                if not pos_text:
                    continue
                fout.write(json.dumps({
                    "anchor_text": anchor_text,
                    "positive_text": pos_text,
                    "anchor_pgpub_id": anchor_pid,
                    "positive_pgpub_id": cited_pid,
                    "rejected_app_id": app,
                }) + "\n")
                n_pairs += 1
                if n_pairs % 100_000 == 0:
                    print(f"  written {n_pairs:,}", file=sys.stderr)

    print(f"\nDone.")
    print(f"  pairs written: {n_pairs:,}")
    print(f"  apps skipped (no source text):     {n_apps_skipped_no_source:,}")
    print(f"  cites skipped (no cited text):     {n_skipped_no_text:,}")


if __name__ == "__main__":
    main()
