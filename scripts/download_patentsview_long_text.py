#!/usr/bin/env python3
"""Download PatentsView Long Text bulk files from USPTO Open Data Portal.

Products:
  PVGPATTXT — Granted Patent Long Text (1976-2025, ~109 GB)
  PVPGPUBTXT — Pre-Grant Pub Long Text (2001-2025, ~119 GB)

Each product contains year-split files for brief_summary, claims,
detail_description, drawing_description.

We download:
  - g_detail_desc_text_*.tsv.zip  (the spec — what we want most)
  - pg_detail_desc_text_*.tsv.zip
  - (optionally g_brf_sum_text_* and pg_brf_sum_text_* — smaller, useful)

Skips already-downloaded files. Streams to disk. Adds ~228 GB total.
"""
import json
import os
import sys
import time

import requests

KEY = open("/lfs/skampere3/0/alexspan/.uspto-open-data-api-key.txt").read().strip()
HEADERS = {"X-API-KEY": KEY}

G_DIR = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents/raw/patentsview_grant"
PG_DIR = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents/raw/patentsview_pg"

PRODUCTS = [
    ("PVGPATTXT", G_DIR),
    ("PVPGPUBTXT", PG_DIR),
]

# Which fields to download — leave detail_desc + brf_sum, skip drawing (low value)
WANT_KEYWORDS = ["detail_desc_text", "brf_sum_text"]


def list_files(product_id):
    r = requests.get(
        f"https://api.uspto.gov/api/v1/datasets/products/{product_id}",
        headers=HEADERS, timeout=30,
    )
    r.raise_for_status()
    d = r.json()
    return d["bulkDataProductBag"][0]["productFileBag"]["fileDataBag"]


def download_file(file_meta, out_dir):
    fname = file_meta["fileName"]
    if not any(k in fname for k in WANT_KEYWORDS):
        return "skip-keyword", 0
    url = file_meta["fileDownloadURI"]
    expected_size = file_meta["fileSize"]
    out_path = os.path.join(out_dir, fname)
    if os.path.exists(out_path):
        actual = os.path.getsize(out_path)
        if actual == expected_size:
            return "have", actual
        else:
            print(f"  size mismatch {fname}: have {actual:,} want {expected_size:,}, redownloading",
                  file=sys.stderr)
    # Stream download with retry on broken-pipe / chunked errors
    print(f"  fetching {fname} ({expected_size / 1e9:.2f} GB)", file=sys.stderr, flush=True)
    for attempt in range(5):
        try:
            t0 = time.time()
            with requests.get(url, headers=HEADERS, stream=True, timeout=600) as r:
                if r.status_code != 200:
                    return f"http-{r.status_code}", 0
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=4 << 20):  # 4 MB
                        f.write(chunk)
            actual = os.path.getsize(out_path)
            if actual != expected_size:
                print(f"    size mismatch {fname}: got {actual:,} want {expected_size:,}, retrying",
                      file=sys.stderr)
                if os.path.exists(out_path):
                    os.remove(out_path)
                time.sleep(2 ** attempt * 10)
                continue
            rate = actual / max(1, time.time() - t0) / 1e6
            print(f"    done {fname} ({actual / 1e9:.2f} GB, {rate:.1f} MB/s)",
                  file=sys.stderr, flush=True)
            return "ok", actual
        except Exception as e:
            print(f"    attempt {attempt+1}/5 failed for {fname}: {e}", file=sys.stderr)
            if os.path.exists(out_path):
                os.remove(out_path)
            time.sleep(2 ** attempt * 10)  # 10, 20, 40, 80, 160 sec
    return "fail-after-retries", 0


def main():
    total_downloaded = 0
    for product_id, out_dir in PRODUCTS:
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n=== {product_id} → {out_dir} ===")
        try:
            files = list_files(product_id)
        except Exception as e:
            print(f"  ERROR listing {product_id}: {e}")
            continue
        print(f"  {len(files)} files in product")
        n_ok = 0; n_have = 0; n_skip = 0; n_err = 0
        for fm in files:
            status, sz = download_file(fm, out_dir)
            if status == "ok":
                n_ok += 1; total_downloaded += sz
            elif status == "have":
                n_have += 1
            elif status == "skip-keyword":
                n_skip += 1
            else:
                n_err += 1; print(f"    fail {fm['fileName']}: {status}", file=sys.stderr)
        print(f"  {product_id} summary: {n_ok} new, {n_have} already had, "
              f"{n_skip} skipped (not wanted), {n_err} errors")
    print(f"\nTotal new download: {total_downloaded / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
