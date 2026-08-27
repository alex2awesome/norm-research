#!/usr/bin/env python3
"""Dump full text for each of the 10 §102 pairs to individual files.
   Each file: anchor's claim 1 + cited's abstract + cited's all claims + cited's spec.
"""
import gzip
import json
import os
import re
import sys
import time

import requests
from bs4 import BeautifulSoup

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"
OUT_DIR = "/lfs/skampere3/0/alexspan/norm-research/logs/pair_dumps"
os.makedirs(OUT_DIR, exist_ok=True)

PAIRS = [
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
    url = f"https://patents.google.com/patent/US{pgpub_id}/en"
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    if r.status_code != 200:
        return None
    soup = BeautifulSoup(r.content, "html.parser")
    out = {"abstract": "", "claims": "", "description": ""}
    abs_sec = soup.find("section", itemprop="abstract")
    if abs_sec: out["abstract"] = re.sub(r"\s+", " ", abs_sec.get_text(separator=" ")).strip()
    cl_sec = soup.find("section", itemprop="claims")
    if cl_sec: out["claims"] = re.sub(r"\s+", " ", cl_sec.get_text(separator=" ")).strip()
    desc_sec = soup.find("section", itemprop="description")
    if desc_sec: out["description"] = re.sub(r"\s+", " ", desc_sec.get_text(separator=" ")).strip()
    return out


def split_claims(claims_text):
    parts = re.split(r"\b(\d+)\s*\.\s+", claims_text)
    out = []
    for i in range(1, len(parts) - 1, 2):
        try:
            n = int(parts[i]); txt = parts[i + 1].strip()
            if txt and n <= 200: out.append((n, txt))
        except ValueError: pass
    return out


def main():
    print("Loading anchor claims from JSONL ...", file=sys.stderr)
    needed = {a for _, a, _ in PAIRS}
    anchor = {}
    with gzip.open("/lfs/skampere3/0/alexspan/norm-research/datasets/patents/patents_dataset.jsonl.gz", "rt") as f:
        for line in f:
            try: d = json.loads(line)
            except Exception: continue
            pid = str(d.get("pgpub_id", "")).strip()
            if pid in needed and pid not in anchor:
                anchor[pid] = (d.get("pg_claims") or "").strip()
            if len(anchor) >= len(needed): break

    for i, (app, anchor_pid, cited_pid) in enumerate(PAIRS, 1):
        path = f"{OUT_DIR}/pair_{i:02d}.txt"
        print(f"Writing pair {i} → {path}", file=sys.stderr)
        ac = split_claims(anchor.get(anchor_pid, ""))
        doc = fetch_google_patents(cited_pid)
        time.sleep(0.7)
        with open(path, "w") as f:
            f.write(f"PAIR #{i}  app={app}  anchor_pgpub={anchor_pid}  cited_pgpub={cited_pid}\n")
            f.write("=" * 100 + "\n\n")
            f.write("ANCHOR CLAIM 1:\n")
            f.write("-" * 50 + "\n")
            f.write((ac[0][1] if ac else anchor.get(anchor_pid, "[parse fail]")) + "\n\n")
            if doc is None:
                f.write("[Google Patents fetch failed]\n")
                continue
            f.write(f"CITED ABSTRACT (length {len(doc['abstract']):,}):\n")
            f.write("-" * 50 + "\n")
            f.write(doc["abstract"] + "\n\n")
            f.write(f"CITED CLAIMS (length {len(doc['claims']):,}):\n")
            f.write("-" * 50 + "\n")
            for n, txt in split_claims(doc["claims"]):
                f.write(f"\n[claim {n}]\n{txt}\n")
            f.write(f"\n\nCITED SPEC (length {len(doc['description']):,}):\n")
            f.write("-" * 50 + "\n")
            f.write(doc["description"][:50000])
            f.write("\n")
        print(f"  ok ({os.path.getsize(path):,} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
