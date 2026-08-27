#!/usr/bin/env python3
"""Fetch missing patent claim-1 from Google Patents (HTML scraping).

Target: pre-2001 US granted patents (IDs 4000000-6999999), plus design
(D-prefix). These are the patents PatentsView's bulk files don't carry.

Strategy:
  - Multi-threaded HTTP requests (8 concurrent), polite delay between batches
  - Stream results to parquet, checkpoint every 1000 patents
  - Resume support: skip patents already in output parquet
"""
import csv
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from bs4 import BeautifulSoup

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
MISSING_CSV = f"{BASE}/processed/missing_after_local_sources.csv"
OUT_PARQUET = f"{BASE}/processed/google_patents_supplement.parquet"

CONCURRENCY = 8
RATE_LIMIT_SEC = 0.15  # ~6 req/sec across threads = manageable

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
HEADERS = {"User-Agent": UA, "Accept": "text/html,application/xhtml+xml"}


def load_target_ids():
    """Filter to pre-2001 US granted (7-digit, 4000000-6999999) + design patents (D...)."""
    targets = []
    with open(MISSING_CSV) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            raw = r["raw_id"].strip().upper()
            fmt = r["format"]
            if fmt == "us_granted":
                digits = re.sub(r"[^0-9]", "", r["normalized_id"])
                if len(digits) == 7 and digits[0] in "456":
                    targets.append(("US" + digits, raw))
                elif raw.startswith("D"):  # design
                    targets.append((raw.replace(" ", ""), raw))
            elif raw.startswith("D"):
                targets.append((raw.replace(" ", ""), raw))
    return targets


def fetch_one(pid_tuple):
    pid_clean, raw_id = pid_tuple
    url = f"https://patents.google.com/patent/{pid_clean}/en"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            return raw_id, None
        soup = BeautifulSoup(r.content, "html.parser")
        sec = soup.find("section", itemprop="claims")
        if not sec:
            return raw_id, None
        text = sec.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()
        # Strip leading "Claims (NN)" header
        text = re.sub(r"^Claims\s*\(\d+\)\s*", "", text)
        # First claim only — split on " 2 . " or similar
        spl = re.split(r"\s+2\s*[.,]\s+", text, maxsplit=1)
        first = spl[0]
        return raw_id, first[:4000] if first else None
    except Exception:
        return raw_id, None


def main():
    print("Loading targets...")
    targets = load_target_ids()
    print(f"  {len(targets):,} target patents (pre-2001 US granted + design)")

    # Resume
    seen = set()
    if os.path.exists(OUT_PARQUET):
        t = pq.read_table(OUT_PARQUET, columns=["raw_id"])
        seen = set(t.column("raw_id").to_pylist())
        print(f"  resuming with {len(seen):,} already fetched")
    todo = [p for p in targets if p[1] not in seen]
    random.shuffle(todo)
    print(f"  {len(todo):,} to fetch")

    results = {}
    fetched_since_save = 0
    t0 = time.time()
    n_done = 0
    n_ok = 0

    def flush(force=False):
        nonlocal fetched_since_save
        if not results:
            return
        if not force and fetched_since_save < 1000:
            return
        # Merge with existing
        if os.path.exists(OUT_PARQUET):
            prev = pq.read_table(OUT_PARQUET).to_pylist()
            existing = {r["raw_id"]: r["claim_text"] for r in prev}
        else:
            existing = {}
        existing.update(results)
        tbl = pa.Table.from_pylist(
            [{"raw_id": k, "claim_text": v} for k, v in existing.items() if v]
        )
        pq.write_table(tbl, OUT_PARQUET, compression="zstd")
        results.clear()
        fetched_since_save = 0

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = {ex.submit(fetch_one, p): p for p in todo}
        for fut in as_completed(futures):
            n_done += 1
            raw, txt = fut.result()
            if txt:
                results[raw] = txt
                n_ok += 1
                fetched_since_save += 1
            if n_done % 100 == 0:
                rate = n_done / max(1, time.time() - t0)
                eta_h = (len(todo) - n_done) / rate / 3600
                print(f"  {n_done:,}/{len(todo):,}  ok={n_ok:,}  "
                      f"{rate:.1f} req/s  ETA {eta_h:.1f}h",
                      file=sys.stderr, flush=True)
            flush()
            # Politeness
            time.sleep(RATE_LIMIT_SEC / CONCURRENCY)

    flush(force=True)
    print(f"\nDONE. {n_ok:,} new claim texts saved.")


if __name__ == "__main__":
    main()
