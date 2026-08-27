#!/usr/bin/env python3
"""RoyalRoad deep-metrics pass: fetch full fiction pages (5-dim scores,
favorites, average views, ratings count, first reviews) for priority subsets:

  1. all STUB fictions (commercial-pickup verdict side)
  2. top-N by followers (dense metric strata)
  3. stable-hash random sample of the remaining population

Raw HTML saved to fiction_pages_raw/<id>.html.gz; parse offline.
"""
import gzip
import hashlib
import json
import os
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 norm-research-academic "
      "(contact: alex2awesome@gmail.com)")
RATE = 1.05
TOP_N = 5000
RANDOM_N = 5000


def targets():
    stubs, rows = [], []
    for line in gzip.open(os.path.join(HERE, "listings_stub.jsonl.gz")):
        stubs.append(json.loads(line)["fiction_id"])
    seen = set()
    for line in gzip.open(os.path.join(HERE, "listings_all.jsonl.gz")):
        d = json.loads(line)
        if d["fiction_id"] in seen:
            continue
        seen.add(d["fiction_id"])
        rows.append((d["fiction_id"], d.get("followers") or 0))
    top = [fid for fid, _ in sorted(rows, key=lambda r: -r[1])[:TOP_N]]
    rest = [fid for fid, _ in rows if fid not in set(top)]
    rnd = sorted(rest, key=lambda f: hashlib.md5(f"deep::{f}".encode()).hexdigest())[:RANDOM_N]
    ordered = list(dict.fromkeys(stubs + top + rnd))
    return ordered


def main():
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA})
    raw_dir = os.path.join(HERE, "fiction_pages_raw")
    os.makedirs(raw_dir, exist_ok=True)
    ids = targets()
    print(f"{len(ids)} target fictions", flush=True)
    for i, fid in enumerate(ids):
        f = os.path.join(raw_dir, f"{fid}.html.gz")
        if os.path.exists(f):
            continue
        for a in range(6):
            try:
                r = sess.get(f"https://www.royalroad.com/fiction/{fid}", timeout=60, allow_redirects=True)
                if r.status_code == 404:
                    with gzip.open(f, "wt") as fh:
                        fh.write("<!-- 404 -->")
                    break
                if r.status_code in (429, 500, 502, 503):
                    raise RuntimeError(f"HTTP {r.status_code}")
                r.raise_for_status()
                with gzip.open(f, "wt") as fh:
                    fh.write(r.text)
                break
            except Exception as e:
                w = min(300, 2 ** a * 5)
                print(f"fid {fid} attempt {a}: {e}; sleep {w}", flush=True)
                time.sleep(w)
        if i % 200 == 0:
            print(f"{i}/{len(ids)}", flush=True)
        time.sleep(RATE)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
