#!/usr/bin/env python3
"""Fetch office action documents from USPTO ODP for apps with §102/§103 rejections.

For each app_id:
  1. GET /api/v1/patent/applications/{NUM}/documents → document list
  2. Filter to CTNF (Non-Final Rejection) + CTFR (Final Rejection)
  3. Prefer DOCX over PDF for cleaner text extraction
  4. Download + extract text
  5. Save: {app_id, ifw_number, document_code, document_date, text}

Resume support: skip app_ids already in output JSONL.

Output: /lfs/.../patents/processed/office_actions.jsonl.gz
"""
import argparse
import csv
import gzip
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

csv.field_size_limit(2**31 - 1)

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
INPUT_FILE = f"{BASE}/patents_final_outcome_cpc_balanced_with_rejections.csv.gz"
KEY_FILE = "/lfs/skampere3/0/alexspan/.uspto-open-data-api-key.txt"
OUT_JSONL = f"{BASE}/processed/office_actions.jsonl.gz"

WANT_CODES = {"CTNF", "CTFR"}  # non-final + final rejection
CONCURRENCY = 4  # polite: ~4 req/sec
REQUEST_DELAY = 0.0  # in seconds, per-request before next


def get_key():
    return open(KEY_FILE).read().strip()


def list_documents(session, app_num, key):
    url = f"https://api.uspto.gov/api/v1/patent/applications/{app_num}/documents"
    try:
        r = session.get(url, headers={"X-API-KEY": key}, timeout=30)
        if r.status_code != 200:
            return None
        return r.json().get("documentBag", [])
    except Exception:
        return None


def download_and_extract(session, doc, key):
    """Try DOCX first, then PDF; return extracted text or None."""
    opts = {o["mimeTypeIdentifier"]: o for o in doc.get("downloadOptionBag", [])}
    if "MS_WORD" in opts:
        url = opts["MS_WORD"]["downloadUrl"]
        kind = "docx"
    elif "PDF" in opts:
        url = opts["PDF"]["downloadUrl"]
        kind = "pdf"
    else:
        return None
    try:
        r = session.get(url, headers={"X-API-KEY": key}, timeout=60)
        if r.status_code != 200:
            return None
        if kind == "docx":
            from docx import Document
            doc_obj = Document(io.BytesIO(r.content))
            return "\n".join(p.text for p in doc_obj.paragraphs)
        else:  # pdf
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(r.content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return None


def process_app(app_num, session, key):
    """Returns list of OA records for the app, or empty list."""
    docs = list_documents(session, app_num, key)
    if not docs:
        return []
    out = []
    for d in docs:
        if d.get("documentCode") not in WANT_CODES:
            continue
        text = download_and_extract(session, d, key)
        if not text or len(text) < 200:
            continue
        out.append({
            "app_id": app_num,
            "ifw_number": d.get("documentIdentifier"),
            "document_code": d.get("documentCode"),
            "document_date": (d.get("officialDate") or "")[:10],
            "page_count": d.get("downloadOptionBag", [{}])[0].get("pageTotalQuantity"),
            "text": text[:50000],
        })
    return out


def load_target_apps(limit=None):
    """Apps where rejected_102=1 OR rejected_103=1."""
    apps = []
    with gzip.open(INPUT_FILE, "rt") as f:
        for r in csv.DictReader(f):
            if int(r.get("rejected_102", 0)) or int(r.get("rejected_103", 0)):
                apps.append(r["app_id"].strip())
            if limit and len(apps) >= limit:
                break
    return apps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    key = get_key()
    print(f"Key length: {len(key)}", file=sys.stderr)

    print("Loading target apps...")
    targets = load_target_apps(limit=args.limit)
    print(f"  {len(targets):,} apps with §102 or §103 rejection")

    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    done = set()
    if os.path.exists(OUT_JSONL):
        with gzip.open(OUT_JSONL, "rt") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["app_id"])
                except Exception:
                    pass
        print(f"  resuming with {len(done):,} apps already done")
    todo = [a for a in targets if a not in done]
    print(f"  {len(todo):,} to fetch")

    n_ok = 0
    n_records = 0
    t0 = time.time()
    session = requests.Session()
    with gzip.open(OUT_JSONL, "at") as fout, \
         ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = {ex.submit(process_app, a, session, key): a for a in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            recs = fut.result()
            for r in recs:
                fout.write(json.dumps(r) + "\n")
                n_records += 1
            if recs:
                n_ok += 1
            if i % 50 == 0:
                fout.flush()
                rate = i / max(1, time.time() - t0)
                eta_h = (len(todo) - i) / rate / 3600
                print(f"  {i:,}/{len(todo):,}  apps_ok={n_ok:,}  "
                      f"oa_recs={n_records:,}  {rate:.1f} app/s  ETA {eta_h:.1f}h",
                      file=sys.stderr, flush=True)

    print(f"\nDone. {n_ok:,} apps, {n_records:,} OA records.")


if __name__ == "__main__":
    main()
