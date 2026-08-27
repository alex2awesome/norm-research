#!/usr/bin/env python3
"""Second pass: retry Top50 rows marked 'dead' (many were false-dead due to a
transient archive.org connection-refused window during the first parallel run).

Purges the cached failed Wayback responses for those URLs, then re-fetches with
the improved engine (gentler archive spacing, more retries). Updates
top50_texts.jsonl in place (replacing the dead rows that now succeed).
"""
import glob
import hashlib
import json
import os
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_top50_par import fetch_one, CACHE

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/creative-writing/wigleaf"
WORKERS = 6


def main():
    out_path = os.path.join(BASE, "top50_texts.jsonl")
    rows = [json.loads(l) for l in open(out_path)]
    dead = [r for r in rows if r.get("fetch_status") == "dead"]
    ok = [r for r in rows if r.get("fetch_status") != "dead"]
    print(f"top50 rows: {len(rows)} | dead to retry: {len(dead)} | already ok: {len(ok)}", flush=True)

    # purge cached failed wayback-availability/cdx/wayback fetches for these URLs
    # (any cached entry whose status != 200 or html is null), so they re-fetch
    purged = 0
    for f in glob.glob(os.path.join(CACHE, "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            os.remove(f); purged += 1; continue
        if isinstance(d, dict) and (d.get("status") != 200 or not d.get("html")):
            os.remove(f); purged += 1
    print(f"purged {purged} cached non-200/empty fetch entries", flush=True)

    lock = threading.Lock()
    counts = Counter()
    recovered = {}
    n = [0]

    def work(r):
        st, raw, clean = fetch_one(r["story_url"], int(r["year"]))
        with lock:
            counts[st] += 1
            n[0] += 1
            if st != "dead":
                rr = dict(r); rr.update(fetch_status=st, fetch_source=st,
                                        raw_text=raw, text=clean)
                recovered[r["story_url"]] = rr
            if n[0] % 25 == 0:
                print(f"[{n[0]}/{len(dead)}] {dict(counts)}", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(as_completed([ex.submit(work, r) for r in dead]))

    # rewrite: keep ok rows, replace recovered dead rows, keep still-dead as dead
    new_rows = list(ok)
    for r in dead:
        new_rows.append(recovered.get(r["story_url"], r))
    with open(out_path, "w") as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("retry done:", dict(counts), flush=True)
    final = Counter(r["fetch_status"] for r in new_rows)
    print("final top50 status:", dict(final), flush=True)


if __name__ == "__main__":
    main()
