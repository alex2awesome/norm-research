#!/usr/bin/env python3
"""Deep-dive on the 10 §102 pairs from the manual check.

For each pair:
  1. Pull the office action text (CTNF/CTFR) from USPTO ODP via API
  2. Find and extract the §102 rejection section discussing the cited ref
  3. Look up the cited reference's FULL claims + abstract (not just claim-1)
  4. Print everything side-by-side for inspection
"""
import gzip
import io
import json
import os
import re
import sys
import time

import requests

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/patents"
JSONL = f"{BASE}/patents_dataset.jsonl.gz"
KEY_FILE = "/lfs/skampere3/0/alexspan/.uspto-open-data-api-key.txt"
KEY = open(KEY_FILE).read().strip()
HEADERS = {"X-API-KEY": KEY}

# The 10 pairs from the manual check
PAIRS = [
    ("12618065", "20060242362", 0.713),
    ("12528547", "20040176588", 0.842),
    ("13924639", "20090128527", 0.738),
    ("13171474", "6807434",     0.548),
    ("13511975", "20060096425", 0.642),
    ("14489336", "20090072025", 0.636),
    ("13642243", "20100324395", 0.660),
    ("13181461", "20100076786", 0.769),
    ("13809154", "20100021421", 0.444),
    ("14334866", "20060091878", 0.553),
]


def list_documents(app):
    url = f"https://api.uspto.gov/api/v1/patent/applications/{app}/documents"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            return None
        return r.json().get("documentBag", [])
    except Exception as e:
        return None


def download_doc(doc):
    """Prefer MS_WORD (real text, requires following redirect). Fall back to PDF
    only if no MS_WORD (PDFs from USPTO are typically scanned images with no
    extractable text). Returns (text, error_msg).
    """
    opts = {o["mimeTypeIdentifier"]: o for o in doc.get("downloadOptionBag", [])}

    if "MS_WORD" in opts:
        url = opts["MS_WORD"]["downloadUrl"]
        # Retry up to 3 times for 429 rate-limit
        for attempt in range(3):
            try:
                r = requests.get(url, headers=HEADERS, timeout=120)
            except Exception as e:
                return None, f"network err (msw): {e}"
            if r.status_code == 200:
                break
            if r.status_code == 429:
                time.sleep(2 ** attempt * 5)  # 5, 10, 20 sec
                continue
            return None, f"HTTP {r.status_code} (msw): {r.text[:200]}"
        else:
            return None, f"HTTP 429 after retries"

        # The response might be:
        #   (a) the actual .docm bytes (starts with PK = ZIP magic)
        #   (b) a text message with a redirect URL (older API behavior)
        content = r.content
        if content[:2] == b"PK":
            # Direct .docm download — use mammoth (more permissive than python-docx)
            try:
                import mammoth
                result = mammoth.extract_raw_text(io.BytesIO(content))
                text = result.value
                if len(text) < 200:
                    return None, f"mammoth text short: {len(text)} chars"
                return text, None
            except Exception as e:
                return None, f"parse err (mammoth): {e}"
        # Maybe a redirect message
        m = re.search(r"https://data-documents\.uspto\.gov/[^\s\"']+", r.text)
        if m:
            redirect_url = m.group(0).rstrip(".,;) ")
            try:
                r2 = requests.get(redirect_url, timeout=120)
            except Exception as e:
                return None, f"network err (redirect): {e}"
            if r2.status_code != 200:
                return None, f"HTTP {r2.status_code} (redirect)"
            try:
                from docx import Document
                d = Document(io.BytesIO(r2.content))
                text = "\n".join(p.text for p in d.paragraphs)
                if len(text) < 200:
                    return None, f"docx text short after redirect: {len(text)} chars"
                return text, None
            except Exception as e:
                return None, f"parse err (docx redirect): {e}"
        return None, f"msw response not PK and no redirect URL: {r.text[:200]}"

    if "PDF" in opts:
        url = opts["PDF"]["downloadUrl"]
        try:
            r = requests.get(url, headers=HEADERS, timeout=120)
        except Exception as e:
            return None, f"network err (pdf): {e}"
        if r.status_code != 200:
            return None, f"HTTP {r.status_code} (pdf)"
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(r.content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            if len(text) < 200:
                return None, f"pdf text suspiciously short ({len(text)} chars) — likely scanned image"
            return text, None
        except Exception as e:
            return None, f"parse err (pdf): {e}"

    return None, "no MS_WORD or PDF download option"


def extract_rejection_section(oa_text, cited_id):
    """Find paragraph(s) discussing the cited ref under §102 or §103."""
    chunks = []
    paras = re.split(r"\n\s*\n", oa_text)
    cited_short = re.sub(r"^US", "", cited_id).lstrip("0")

    # Build a list of possible ID forms the OA might use
    cited_forms = {cited_id, cited_short, cited_id.lstrip("0"), cited_short.lstrip("0")}
    cited_forms = {f for f in cited_forms if f}

    def has_cite(p):
        return any(f in p for f in cited_forms)

    # Tier 1: §102 paragraph mentioning the cite
    for p in paras:
        if "102" in p and has_cite(p):
            chunks.append(("§102", p.strip()))
    # Tier 2: §103 paragraph mentioning the cite
    for p in paras:
        if "103" in p and has_cite(p) and not any(c[1] == p.strip() for c in chunks):
            chunks.append(("§103", p.strip()))
    # Tier 3: any paragraph mentioning the cite with > 200 chars
    if not chunks:
        for p in paras:
            if has_cite(p) and len(p) > 200:
                chunks.append(("any", p.strip()))
    return chunks  # list of (label, text)


def load_cited_full_docs(cited_ids):
    """Single JSONL pass to find ALL cited refs at once.

    JSONL stores pgpub_id and patent_id as INTEGERS. Coerce both sides.
    """
    needed_str = {c.strip() for c in cited_ids}
    # Also try numeric variants for robust matching
    needed_int = set()
    for c in cited_ids:
        try:
            needed_int.add(int(re.sub(r"[^0-9]", "", c)))
        except Exception:
            pass
    needed = needed_str | {str(n) for n in needed_int}
    found = {}
    print(f"  one-shot JSONL pass for {len(cited_ids)} cited refs ...", file=sys.stderr, flush=True)
    with gzip.open(JSONL, "rt") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            for k in ("pgpub_id", "patent_id"):
                v = d.get(k)
                if v is None:
                    continue
                vs = str(v).strip()
                if vs in needed:
                    # Map back to original cited_id format
                    for cid in cited_ids:
                        cid_n = re.sub(r"[^0-9]", "", cid).lstrip("0")
                        vs_n = vs.lstrip("0")
                        if vs_n == cid_n and cid not in found:
                            found[cid] = {
                                "title": d.get("title", ""),
                                "abstract": (d.get("pg_abstract") or d.get("g_abstract") or "")[:2000],
                                "claims": (d.get("pg_claims") or d.get("g_claims") or "")[:8000],
                            }
            if len(found) >= len(cited_ids):
                break
    print(f"  resolved {len(found)}/{len(cited_ids)} cited refs from JSONL", file=sys.stderr, flush=True)
    return found


def main():
    # One-shot JSONL lookup for ALL cited refs (single pass over 6 GB JSONL)
    cited_docs = load_cited_full_docs([c for _, c, _ in PAIRS])

    for i, (app, cited, sim) in enumerate(PAIRS, 1):
        print("=" * 90)
        print(f"PAIR #{i}  similarity={sim:.3f}  app={app} → cited={cited}")
        print("=" * 90)

        # Office actions for this app
        print(f"\n--- Fetching office action list for app {app} ---")
        docs = list_documents(app)
        if docs is None:
            print(f"  ERROR: could not list documents (404/403/timeout)")
            continue

        oa_docs = [d for d in docs if d.get("documentCode") in ("CTNF", "CTFR")]
        print(f"  Found {len(oa_docs)} office action doc(s)")

        # Download each OA, extract the §102 section discussing this cited ref
        oa_102_found = False
        for d in oa_docs:
            date = d.get("officialDate", "")[:10]
            code = d.get("documentCode")
            print(f"\n  Downloading OA: code={code}, date={date}, "
                  f"pages={d.get('downloadOptionBag', [{}])[0].get('pageTotalQuantity')}")
            oa_text, err = download_doc(d)
            if not oa_text:
                print(f"    download failed: {err}")
                continue
            chunks = extract_rejection_section(oa_text, cited)
            if chunks:
                oa_102_found = True
                print(f"\n  ### Rejection paragraph(s) in OA ({code}, {date}) mentioning cite {cited}:")
                for label, txt in chunks[:3]:
                    print(f"\n  [{label}]")
                    print(textwrap_indent(txt[:2500], "    "))
                break  # one OA with this cite is enough
            else:
                print(f"  (no paragraph mentions {cited} in this OA)")

        if not oa_102_found:
            print(f"\n  → NO REJECTION PARAGRAPH FOUND MENTIONING {cited} IN ANY OA")

        # Cited ref's full doc (from pre-loaded dict)
        print(f"\n--- Cited ref {cited} full text ---")
        d = cited_docs.get(cited)
        if d is None:
            print(f"  Not found in JSONL (probably a pre-2002 grant or non-US)")
        else:
            print(f"  Title:    {d['title'][:200]}")
            print(f"  Abstract: {d['abstract'][:600]}")
            print(f"  Claims (first 2000 chars):")
            print(textwrap_indent(d["claims"][:2000], "    "))
        print()
        time.sleep(0.5)  # polite


def textwrap_indent(text, prefix):
    return "\n".join(prefix + line for line in text.splitlines())


if __name__ == "__main__":
    main()
