#!/usr/bin/env python3
"""Parse ICLR/NeurIPS/ICML membership from the DBLP XML dump.

Fallback for fetch_dblp_lists.py when the DBLP search API throttles too
hard (it rate-limits per IP within minutes). The dump (dblp.xml.gz,
~0.8 GB) is a static download and not throttled.

Membership = <inproceedings> whose <crossref> points to the main-track
proceedings volume of the venue-year. Workshop volumes have distinct
proceedings keys (e.g. conf/iclr/2017w) and are excluded. Run with
--list-crossrefs first to see all volume keys + counts, then the script's
MAIN_TRACK regexes pick main volumes; validate counts against official
accepted numbers.

dblp.xml uses ~100s of named entities defined in dblp.dtd; we prefill the
expat parser's entity table from html.entities.entitydefs.

Output: dblp_papers.csv (same schema as fetch_dblp_lists.py).
"""
import argparse
import csv
import gzip
import html.entities
import re
import sys
import xml.etree.ElementTree as ET

MAIN_TRACK = {
    "ICLR": re.compile(r"^conf/iclr/(20\d\d)$"),
    # NeurIPS: conf/nips/2013 ... conf/nips/2023 (datasets&benchmarks etc.
    # live under suffixed keys)
    "NeurIPS": re.compile(r"^conf/nips/(20\d\d)$"),
    "ICML": re.compile(r"^conf/icml/(20\d\d)$"),
}
YEARS = set(range(2013, 2024))


def records(path):
    parser = ET.XMLParser()
    parser.entity.update(html.entities.entitydefs)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        context = ET.iterparse(f, events=("end",), parser=parser)
        for _, elem in context:
            if elem.tag == "inproceedings":
                yield elem
                elem.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="path to dblp.xml or dblp.xml.gz")
    ap.add_argument("--out", default="dblp_papers.csv")
    ap.add_argument("--list-crossrefs", action="store_true")
    args = ap.parse_args()

    crossref_counts = {}
    rows = []
    n = 0
    for elem in records(args.dump):
        n += 1
        if n % 1000000 == 0:
            print(f"  scanned {n/1e6:.0f}M inproceedings...", file=sys.stderr)
        cr = elem.findtext("crossref") or ""
        if not (cr.startswith("conf/iclr/") or cr.startswith("conf/nips/")
                or cr.startswith("conf/icml/")):
            continue
        crossref_counts[cr] = crossref_counts.get(cr, 0) + 1
        for venue, rx in MAIN_TRACK.items():
            m = rx.match(cr)
            if not m or int(m.group(1)) not in YEARS:
                continue
            title = "".join((elem.find("title")).itertext()).strip()
            rows.append({
                "venue": venue,
                "year": int(m.group(1)),
                "title": title.rstrip("."),
                "dblp_key": elem.get("key"),
                "dblp_type": "Conference and Workshop Papers",
                "dblp_venue": venue,
            })

    if args.list_crossrefs:
        for cr in sorted(crossref_counts):
            print(cr, crossref_counts[cr])
        return

    with open(args.out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=["venue", "year", "title",
                                           "dblp_key", "dblp_type",
                                           "dblp_venue"])
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    per = Counter((r["venue"], r["year"]) for r in rows)
    for k in sorted(per):
        print(k, per[k])
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
