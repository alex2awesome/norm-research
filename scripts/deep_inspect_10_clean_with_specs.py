#!/usr/bin/env python3
"""Real deep inspection of 10 clean §102 pairs.

For each pair:
  1. Show anchor's claim 1
  2. Show cited ref's ALL claims (not just first 5)
  3. Show cited ref's abstract
  4. Fetch cited ref's FULL SPEC from Google Patents (description section)
  5. Extract distinctive multi-word phrases from anchor's claim 1
  6. Search the cited ref's spec for those phrases, show top hits with context

Cited refs from the prior `inspect_10_clean_102_pairs.py` run:
"""
import json
import re
import sys
import time

import requests
from bs4 import BeautifulSoup

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

# The 10 pairs from inspect_10_clean_102_pairs.py
PAIRS = [
    # (app_id, anchor_pgpub, cited_pgpub)
    ("14325752", "20150016067", "20120243195"),
    ("13118659", "20120050635", "20110007236"),
    ("14420559", "20150221910", "20100273034"),
    ("13348519", "20120176646", "20100245904"),
    ("14261649", "20150143378", "20140149719"),
    ("13436759", "20120301141", "20120106966"),
    ("14278037", "20140376642", "20130202051"),
    ("14560512", "20150223839", "20120059213"),
    ("14097370", "20140267152", "20100253651"),
    ("13937960", "20150018623", "20120296171"),
]


def fetch_google_patents(pgpub_id):
    """Fetch the cited ref's Google Patents page and pull description + claims + abstract."""
    url = f"https://patents.google.com/patent/US{pgpub_id}/en"
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    if r.status_code != 200:
        return None
    soup = BeautifulSoup(r.content, "html.parser")
    out = {"abstract": "", "claims": "", "description": ""}

    # Abstract
    abs_sec = soup.find("section", itemprop="abstract")
    if abs_sec:
        out["abstract"] = re.sub(r"\s+", " ", abs_sec.get_text(separator=" ")).strip()

    # Claims
    cl_sec = soup.find("section", itemprop="claims")
    if cl_sec:
        out["claims"] = re.sub(r"\s+", " ", cl_sec.get_text(separator=" ")).strip()

    # Description — Google Patents puts it in <section itemprop="description">
    desc_sec = soup.find("section", itemprop="description")
    if desc_sec:
        out["description"] = re.sub(r"\s+", " ", desc_sec.get_text(separator=" ")).strip()
    return out


def extract_key_phrases(text, min_words=3, max_words=6, top_n=20):
    """Extract distinctive multi-word phrases (potential limitations) from text.
    Drops phrases that are mostly stop-words.
    """
    STOP = {"a", "the", "of", "in", "and", "or", "to", "by", "for", "is", "are",
            "with", "on", "an", "as", "at", "be", "comprising", "wherein",
            "which", "that", "from", "between", "having", "one", "such",
            "include", "includes", "including", "thereof", "thereto",
            "according", "claim", "said", "first", "second", "third"}
    # Tokenize, simple
    tokens = re.findall(r"[a-z][a-z\-]+", text.lower())
    # Get n-grams
    counter = {}
    for n in range(min_words, max_words + 1):
        for i in range(len(tokens) - n + 1):
            ngram = tokens[i:i + n]
            # Skip if first or last is stopword
            if ngram[0] in STOP or ngram[-1] in STOP:
                continue
            # Skip if all middle words are stopwords too
            if all(w in STOP for w in ngram[1:-1]):
                continue
            phrase = " ".join(ngram)
            counter[phrase] = counter.get(phrase, 0) + 1
    # Sort by length-weighted score
    scored = [(p, c, c * len(p.split())) for p, c in counter.items()]
    scored.sort(key=lambda x: -x[2])
    # Dedupe overlapping shorter ones
    seen = []
    for p, c, _ in scored:
        if not any(p in s or s in p for s, _, _ in seen):
            seen.append((p, c, _))
        if len(seen) >= top_n:
            break
    return [(p, c) for p, c, _ in seen]


def search_phrase(text, phrase, max_hits=3, ctx=120):
    """Find phrase in text, return list of (position, context_string)."""
    out = []
    for m in re.finditer(re.escape(phrase), text, re.IGNORECASE):
        s, e = max(0, m.start() - ctx), min(len(text), m.end() + ctx)
        out.append((m.start(), text[s:e]))
        if len(out) >= max_hits:
            break
    return out


def split_claims(claims_text):
    parts = re.split(r"\b(\d+)\s*\.\s+", claims_text)
    out = []
    for i in range(1, len(parts) - 1, 2):
        try:
            n = int(parts[i])
            txt = parts[i + 1].strip()
            if txt and n <= 100:
                out.append((n, txt))
        except ValueError:
            pass
    return out


def main():
    # We need to load anchor's claim 1 from JSONL (we don't have it cached)
    import gzip
    print("Loading anchor claims from JSONL ...", file=sys.stderr)
    needed_anchors = {a for _, a, _ in PAIRS}
    anchor_claims = {}
    with gzip.open("/lfs/skampere3/0/alexspan/norm-research/datasets/patents/patents_dataset.jsonl.gz", "rt") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            pid = str(d.get("pgpub_id", "")).strip()
            if pid in needed_anchors and pid not in anchor_claims:
                anchor_claims[pid] = (d.get("pg_claims") or "").strip()
            if len(anchor_claims) >= len(needed_anchors):
                break
    print(f"  loaded {len(anchor_claims)}/{len(needed_anchors)} anchor claims", file=sys.stderr)

    for i, (app, anchor_pid, cited_pid) in enumerate(PAIRS, 1):
        print(f"\n{'=' * 100}")
        print(f"PAIR #{i}  app={app}  anchor_pgpub={anchor_pid}  cited_pgpub={cited_pid}")
        print(f"{'=' * 100}")

        # Anchor claim 1
        ac = split_claims(anchor_claims.get(anchor_pid, ""))
        if not ac:
            print("  (couldn't parse anchor claims)")
            continue
        anchor_claim_1 = ac[0][1]
        print(f"\n--- ANCHOR claim 1 ---")
        print(f"  {anchor_claim_1[:1500]}")

        # Fetch cited doc from Google Patents
        print(f"\n--- Fetching cited US{cited_pid} from Google Patents ...")
        doc = fetch_google_patents(cited_pid)
        time.sleep(0.5)
        if doc is None:
            print(f"  fetch failed")
            continue

        print(f"\n--- CITED abstract ---")
        print(f"  {doc['abstract'][:800]}")

        cited_claims = split_claims(doc["claims"])
        print(f"\n--- CITED has {len(cited_claims)} claims (showing all) ---")
        for n, txt in cited_claims:
            print(f"\n  [claim {n}]")
            print(f"    {txt[:600]}{'...' if len(txt) > 600 else ''}")

        # Extract key phrases from anchor's claim 1
        phrases = extract_key_phrases(anchor_claim_1, top_n=15)
        print(f"\n--- Distinctive phrases from anchor claim 1 ---")
        for p, c in phrases:
            print(f"  '{p}' (anchor count={c})")

        # Search each phrase in cited spec
        print(f"\n--- Searching cited's full spec (length {len(doc['description']):,} chars) ---")
        for p, anchor_c in phrases:
            hits = search_phrase(doc["description"], p)
            if hits:
                print(f"\n  ★ '{p}' HITS IN SPEC ({len(hits)}):")
                for pos, ctx in hits:
                    print(f"      [pos {pos}]: ...{ctx}...")
            else:
                # Also search in claims (in case it's there)
                hits_c = search_phrase(doc["claims"], p, max_hits=1)
                if hits_c:
                    print(f"  ◇ '{p}' (not in spec but in cited claims)")
                else:
                    print(f"  · '{p}' (no hit anywhere in cited doc)")


if __name__ == "__main__":
    main()
