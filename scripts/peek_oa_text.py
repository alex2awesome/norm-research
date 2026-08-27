#!/usr/bin/env python3
"""Quick peek at OA text content to see how examiners cite refs."""
import io
import re
import sys

import requests
from pypdf import PdfReader

KEY = open("/lfs/skampere3/0/alexspan/.uspto-open-data-api-key.txt").read().strip()
HEADERS = {"X-API-KEY": KEY}

APP = "12618065"
CITED_RAW = "20060242362"

# List docs
url = f"https://api.uspto.gov/api/v1/patent/applications/{APP}/documents"
r = requests.get(url, headers=HEADERS, timeout=30).json()
docs = [d for d in r["documentBag"] if d.get("documentCode") == "CTNF"]

for d in docs:
    opts = {o["mimeTypeIdentifier"]: o for o in d.get("downloadOptionBag", [])}
    pdf_url = opts["PDF"]["downloadUrl"]
    print(f"Downloading {pdf_url} ...")
    r = requests.get(pdf_url, headers=HEADERS, timeout=120)
    print(f"  status: {r.status_code}, content-type: {r.headers.get('content-type')}, size: {len(r.content)}")
    reader = PdfReader(io.BytesIO(r.content))
    text = "\n".join(p.extract_text() or "" for p in reader.pages)
    print(f"  pages: {len(reader.pages)}, text len: {len(text)}")
    print()
    print("=== First 2000 chars ===")
    print(text[:2000])
    print()
    # Look for patterns that might cite our patent
    print(f"=== Hits for '{CITED_RAW}' (exact substring) ===")
    for m in re.finditer(re.escape(CITED_RAW), text):
        s, e = max(0, m.start() - 100), min(len(text), m.end() + 100)
        print(f"  ...{text[s:e]}...")
    print()
    print("=== Hits for '2006/0242362' (slash format) ===")
    for m in re.finditer(r"2006/0242362", text):
        s, e = max(0, m.start() - 100), min(len(text), m.end() + 100)
        print(f"  ...{text[s:e]}...")
    print()
    print("=== Hits for any 'US 200x/0xxxx' pattern ===")
    for m in re.finditer(r"US\s+\d{4}/[\d,]+", text):
        print(f"  {m.group(0)}")
    print()
    print("=== Hits for any 'US X,XXX,XXX' granted pattern ===")
    for m in re.finditer(r"US\s+[\d,]{7,15}\s*[A-Z]?\d?", text):
        print(f"  {m.group(0)}")
    break  # just first OA
