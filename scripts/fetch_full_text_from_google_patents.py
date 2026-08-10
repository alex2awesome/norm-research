#!/usr/bin/env python3
"""Re-scrape Google Patents for FULL TEXT (all claims + description/spec).

The earlier scraper grabbed only claim 1. This one grabs:
  - abstract
  - all claims block
  - full description (spec)

Same target list: pre-2001 US granted + design patents (~1.7M).
Independent output file so it doesn't overwrite the existing claim-1 scrape.
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
OUT_PARQUET = f"{BASE}/processed/google_patents_full_text.parquet"

CONCURRENCY = 8
RATE_LIMIT = 0.15

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"
HEADERS = {"User-Agent": UA, "Accept": "text/html,application/xhtml+xml"}


def load_target_ids():
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
                elif raw.startswith("D"):
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
            return raw_id, None, None, None
        soup = BeautifulSoup(r.content, "html.parser")

        abstract = ""
        abs_sec = soup.find("section", itemprop="abstract")
        if abs_sec:
            abstract = re.sub(r"\s+", " ", abs_sec.get_text(separator=" ")).strip()[:3000]

        claims = ""
        cl_sec = soup.find("section", itemprop="claims")
        if cl_sec:
            claims = re.sub(r"\s+", " ", cl_sec.get_text(separator=" ")).strip()[:30000]

        spec = ""
        desc_sec = soup.find("section", itemprop="description")
        if desc_sec:
            spec = re.sub(r"\s+", " ", desc_sec.get_text(separator=" ")).strip()[:150000]

        if not claims and not spec:
            return raw_id, None, None, None
        return raw_id, abstract, claims, spec
    except Exception:
        return raw_id, None, None, None


def main():
    print("Loading targets...")
    targets = load_target_ids()
    print(f"  {len(targets):,} target patents")

    # Resume support
    seen = set()
    if os.path.exists(OUT_PARQUET):
        t = pq.read_table(OUT_PARQUET, columns=["raw_id"])
        seen = set(t.column("raw_id").to_pylist())
        print(f"  resuming with {len(seen):,} fetched")
    todo = [p for p in targets if p[1] not in seen]
    random.shuffle(todo)
    print(f"  {len(todo):,} to fetch")

    results = {}  # raw_id → (abstract, claims, spec)
    flushed_since = 0
    t0 = time.time()
    n_ok = 0; n_done = 0

    def flush(force=False):
        nonlocal flushed_since
        if not results: return
        if not force and flushed_since < 1000: return
        if os.path.exists(OUT_PARQUET):
            prev = pq.read_table(OUT_PARQUET).to_pylist()
            existing = {r["raw_id"]: (r.get("abstract", ""), r["claims"], r["spec"]) for r in prev}
        else:
            existing = {}
        for k, v in results.items():
            existing[k] = v
        rows = [{"raw_id": k, "abstract": v[0], "claims": v[1], "spec": v[2]}
                for k, v in existing.items() if v[1] or v[2]]
        tbl = pa.Table.from_pylist(rows)
        pq.write_table(tbl, OUT_PARQUET, compression="zstd")
        results.clear()
        flushed_since = 0

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = {ex.submit(fetch_one, p): p for p in todo}
        for fut in as_completed(futures):
            n_done += 1
            raw, abstract, claims, spec = fut.result()
            if claims or spec:
                results[raw] = (abstract, claims, spec)
                n_ok += 1
                flushed_since += 1
            if n_done % 100 == 0:
                rate = n_done / max(1, time.time() - t0)
                eta_h = (len(todo) - n_done) / rate / 3600
                print(f"  {n_done:,}/{len(todo):,}  ok={n_ok:,}  "
                      f"{rate:.1f}/s  ETA {eta_h:.1f}h",
                      file=sys.stderr, flush=True)
            flush()
            time.sleep(RATE_LIMIT / CONCURRENCY)

    flush(force=True)
    print(f"\nDone. {n_ok:,} full-text records saved.")


if __name__ == "__main__":
    main()
