#!/usr/bin/env python3
"""Fetch USPTO Patent Grant Full Text Data for 1976-2001 (pre-PatentsView era).

PatentsView's g_claims_*.tsv.zip starts at 2001. For citations referencing
older granted patents, we need to pull from USPTO Bulk Data:
  https://bulkdata.uspto.gov/data/patent/grant/redbook/fulltext/{year}/

The site publishes one ZIP per week (~150-300 MB each). Each ZIP contains
an XML file (post-2002) or SGML (1976-2001).

We download year-by-year, extract claim 1 for any patent matching our
missing-IDs list, write to uspto_supplement.parquet, then delete the
downloaded files to keep disk usage bounded.

Output:  uspto_supplement.parquet  with columns (patent_id, claim_text, year)
"""
import csv
import os
import re
import shutil
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
import pyarrow as pa
import pyarrow.parquet as pq
from bs4 import BeautifulSoup

csv.field_size_limit(2**31 - 1)

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
MISSING_CSV = f"{BASE}/processed/missing_after_local_sources.csv"
OUT_PARQUET = f"{BASE}/processed/uspto_supplement.parquet"
WORK = f"{BASE}/raw/uspto_bulk_work"
USPTO_BASE = "https://bulkdata.uspto.gov/data/patent/grant/redbook/fulltext"

os.makedirs(WORK, exist_ok=True)
os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)


def load_missing_us_granted_ids() -> set:
    """Read missing IDs CSV; keep only US granted format with 6-8 digit IDs."""
    out = set()
    with open(MISSING_CSV) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            if r["format"] != "us_granted":
                continue
            digits = re.sub(r"[^0-9]", "", r["normalized_id"])
            if 6 <= len(digits) <= 8:
                out.add(digits.lstrip("0"))
    return out


def list_year_files(year: int) -> list:
    """Scrape directory listing for the year."""
    url = f"{USPTO_BASE}/{year}/"
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except Exception as e:
        print(f"  failed to list {year}: {e}", file=sys.stderr)
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    files = []
    for a in soup.find_all("a"):
        href = a.get("href", "")
        if href.endswith(".zip"):
            files.append(urljoin(url, href))
    return files


def extract_patent_id_and_claim1(xml_text: str) -> list:
    """Parse one weekly XML file. Returns list of {patent_id, claim_text}.

    Modern (2002+) XML: <us-patent-grant>...<doc-number>NNNNNNNN</doc-number>...
                          <claims>...<claim id="claim-1"><claim-text>...
    Legacy (1976-2001) SGML: <PATDOC>...<B110> with patent number,
                              claims under <SDOCL>.
    We use regex rather than full XML parse for robustness/speed.
    """
    out = []
    # Modern: split on patent boundaries
    for m in re.finditer(r"<us-patent-grant\b.*?</us-patent-grant>", xml_text, re.DOTALL):
        doc = m.group(0)
        pn = re.search(r"<doc-number>(\d+)</doc-number>", doc)
        if not pn: continue
        pid = pn.group(1).lstrip("0")
        cl = re.search(r"<claim[^>]*\bnum[^>]*>([\s\S]*?)</claim>", doc)
        if not cl:
            cl = re.search(r"<claim[^>]*>([\s\S]*?)</claim>", doc)
        if not cl: continue
        txt = re.sub(r"<[^>]+>", " ", cl.group(1))
        txt = re.sub(r"\s+", " ", txt).strip()
        if txt:
            out.append({"patent_id": pid, "claim_text": txt[:4000]})
    # Legacy SGML: <PATDOC> ... <B110>...
    for m in re.finditer(r"<PATDOC\b[\s\S]*?</PATDOC>", xml_text):
        doc = m.group(0)
        pn = re.search(r"<B110>\s*<DNUM>\s*<PDAT>(\d+)</PDAT>", doc)
        if not pn:
            pn = re.search(r"<B110>[\s\S]*?<DNUM>[\s\S]*?<PDAT>(\d+)</PDAT>", doc)
        if not pn: continue
        pid = pn.group(1).lstrip("0")
        # First claim block — try SDOCL > CL > CLM > PARA
        cl = re.search(r"<SDOCL>[\s\S]*?<CL\b[\s\S]*?</CL>", doc)
        if cl:
            txt = re.sub(r"<[^>]+>", " ", cl.group(0))
            txt = re.sub(r"\s+", " ", txt).strip()
            if txt:
                out.append({"patent_id": pid, "claim_text": txt[:4000]})
    return out


def process_year(year: int, missing_ids: set, results: dict):
    print(f"\n=== Year {year} ===", file=sys.stderr)
    files = list_year_files(year)
    if not files:
        print(f"  no files for {year}", file=sys.stderr)
        return
    print(f"  {len(files)} weekly files", file=sys.stderr)
    for i, url in enumerate(files):
        fname = os.path.basename(url)
        local_zip = os.path.join(WORK, fname)
        try:
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            with open(local_zip, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        except Exception as e:
            print(f"  [{i+1}/{len(files)}] download failed: {fname} ({e})", file=sys.stderr)
            continue
        # Open + parse
        try:
            import zipfile
            with zipfile.ZipFile(local_zip) as zf:
                for inner in zf.namelist():
                    if not (inner.endswith(".xml") or inner.endswith(".sgm") or inner.endswith(".sgml")):
                        continue
                    with zf.open(inner) as fh:
                        text = fh.read().decode("utf-8", errors="replace")
                    found = extract_patent_id_and_claim1(text)
                    added = 0
                    for rec in found:
                        if rec["patent_id"] in missing_ids and rec["patent_id"] not in results:
                            results[rec["patent_id"]] = rec["claim_text"]
                            added += 1
                    if added:
                        print(f"  [{i+1}/{len(files)}] {fname}: +{added} (total: {len(results):,})",
                              file=sys.stderr)
        except Exception as e:
            print(f"  [{i+1}/{len(files)}] parse error: {e}", file=sys.stderr)
        finally:
            os.remove(local_zip)


def main():
    print("Loading missing US granted IDs ...")
    missing = load_missing_us_granted_ids()
    # Strip leading zeros for comparison
    print(f"  {len(missing):,} missing US granted IDs")

    # Load existing results if any
    results = {}
    if os.path.exists(OUT_PARQUET):
        t = pq.read_table(OUT_PARQUET)
        for row in t.to_pylist():
            results[row["patent_id"]] = row["claim_text"]
        print(f"  resumed with {len(results):,} existing results")

    # Process 1976-2001 (pre-PatentsView coverage)
    for year in range(1976, 2002):
        process_year(year, missing, results)
        # Periodic flush
        if results:
            tbl = pa.Table.from_pylist(
                [{"patent_id": k, "claim_text": v, "year": 0} for k, v in results.items()]
            )
            pq.write_table(tbl, OUT_PARQUET, compression="zstd")
        print(f"  end of {year}: {len(results):,} total resolved", file=sys.stderr)

    print(f"\nDONE. {len(results):,} cited refs resolved from USPTO bulk.")


if __name__ == "__main__":
    main()
