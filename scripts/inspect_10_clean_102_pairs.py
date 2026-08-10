#!/usr/bin/env python3
"""Pull 10 CLEAN §102 pairs (action_type='102' from OARD) and print:
  - Anchor's claims (rejected app)
  - Cited ref's FULL claims block, broken into individual claims
  - Cited ref's abstract

For manual inspection: where in the cited document does the overlap with
the anchor's claim live?  Claim-1?  A later claim?  The abstract?
Or not visibly in the cited text at all (→ spec-buried, needs LLM)?
"""
import csv
import gzip
import json
import os
import random
import re
import sys
from collections import defaultdict

import pyarrow.parquet as pq

csv.field_size_limit(2**31 - 1)

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
JSONL = f"{BASE}/patents_dataset.jsonl.gz"
PATEX = f"{BASE}/raw/patex/application_data.csv"
OARD_CITES = f"{BASE}/raw/oard/oard_citations.csv"


def normalize_pgpub(s):
    if not s: return ""
    s = s.strip().upper()
    return re.sub(r"^US|[A-Z]\d$", "", s).lstrip("0")


def split_claims(claims_block: str) -> list:
    """Split full claims block into individual claims by leading number+dot."""
    if not claims_block:
        return []
    parts = re.split(r"\n\s*(\d+)\s*\.\s+", "\n" + claims_block)
    # parts looks like ['', 'N1', 'text1', 'N2', 'text2', ...]
    out = []
    for i in range(1, len(parts) - 1, 2):
        n, txt = parts[i], parts[i + 1].strip()
        if txt:
            out.append((int(n), txt))
    return out


def main():
    # 1. Get clean §102 pairs
    print("Streaming OARD citations to find §102 pairs ...", file=sys.stderr)
    pairs = []
    rng = random.Random(42)
    with open(OARD_CITES) as f:
        rdr = csv.DictReader(f)
        for n, r in enumerate(rdr, 1):
            if r.get("action_type") != "102":
                continue
            app = r.get("app_id", "").strip()
            cited = (r.get("citation_pat_pgpub_id") or "").strip()
            if not app or not cited:
                continue
            # Reservoir sample
            if len(pairs) < 1000:
                pairs.append((app, cited))
            else:
                k = rng.randrange(len(pairs))
                if k < 1000:
                    pairs[k] = (app, cited)
            if n % 5_000_000 == 0:
                print(f"  scanned {n:,}, sampled pool {len(pairs):,}", file=sys.stderr)
    print(f"  done. Sampled pool: {len(pairs):,}", file=sys.stderr)

    # 2. Build PatEx app_id → pgpub_id mapping (so we can find anchor's text)
    print("Loading PatEx app→pgpub mapping ...", file=sys.stderr)
    app_to_pgpub = {}
    with open(PATEX) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            pgpub = normalize_pgpub(r.get("earliest_pgpub_number", ""))
            app = r.get("application_number", "").strip()
            if pgpub and app:
                app_to_pgpub[app] = pgpub
    print(f"  {len(app_to_pgpub):,} mappings", file=sys.stderr)

    # 3. For each pair, get anchor's pgpub_id; collect needed pgpub_ids
    needed = set()
    valid_pairs = []
    for app, cited in pairs:
        anchor_pid = app_to_pgpub.get(app)
        cited_norm = normalize_pgpub(cited)
        if not anchor_pid or not cited_norm:
            continue
        needed.add(anchor_pid)
        needed.add(cited_norm)
        valid_pairs.append((app, anchor_pid, cited_norm, cited))
    print(f"  valid pairs (anchor in PatEx): {len(valid_pairs):,}", file=sys.stderr)

    # 4. One-pass JSONL to pull texts
    print("Streaming JSONL for text lookup ...", file=sys.stderr)
    texts = {}
    with gzip.open(JSONL, "rt") as f:
        for line in f:
            try: d = json.loads(line)
            except Exception: continue
            pid = str(d.get("pgpub_id", "")).strip().lstrip("0")
            if pid in needed and pid not in texts:
                texts[pid] = {
                    "claims": (d.get("pg_claims") or "").strip(),
                    "abstract": (d.get("pg_abstract") or "").strip(),
                }
            if len(texts) >= len(needed):
                break
    print(f"  resolved {len(texts):,} / {len(needed):,} pgpub_ids in JSONL", file=sys.stderr)

    # 5. Find 10 pairs where BOTH anchor and cited resolve
    final = []
    for app, anchor_pid, cited_norm, cited_raw in valid_pairs:
        if anchor_pid in texts and cited_norm in texts:
            if texts[anchor_pid]["claims"] and texts[cited_norm]["claims"]:
                final.append((app, anchor_pid, cited_norm, cited_raw))
        if len(final) >= 10:
            break
    print(f"\nFound {len(final)} fully-resolvable §102 pairs", file=sys.stderr)
    print()

    # 6. Print each pair for manual inspection
    for i, (app, anchor_pid, cited_norm, cited_raw) in enumerate(final, 1):
        anchor_claims = texts[anchor_pid]["claims"]
        cited_claims = texts[cited_norm]["claims"]
        cited_abstract = texts[cited_norm]["abstract"]

        print("=" * 100)
        print(f"PAIR #{i}  app={app}  anchor_pgpub={anchor_pid}  cited_pgpub={cited_raw}")
        print("=" * 100)
        print(f"\n--- ANCHOR (rejected app) claim 1 (first 1200 chars) ---")
        anchor_first = split_claims(anchor_claims)
        if anchor_first:
            print(f"  {anchor_first[0][1][:1200]}")
        else:
            print(f"  (couldn't parse) {anchor_claims[:1200]}")

        print(f"\n--- CITED REF abstract ---")
        print(f"  {cited_abstract[:600]}")

        cited_claims_list = split_claims(cited_claims)
        print(f"\n--- CITED REF has {len(cited_claims_list)} claims; showing first 5 ---")
        for n, txt in cited_claims_list[:5]:
            print(f"\n  [claim {n}]")
            print(f"    {txt[:800]}")
        print()


if __name__ == "__main__":
    main()
