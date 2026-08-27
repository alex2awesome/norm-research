#!/usr/bin/env python3
"""Recover recent CS-flagship venue-years that moved to issue-numbered journals.

PLDI 2023+  -> PACMPL  vol=year-2016, number="PLDI"   (pacmpl mixes POPL/OOPSLA/ICFP/PLDI)
FSE  2024+  -> PACMSE  vol=year-2023, number="FSE"    (pacmse vol2 also has ISSTA)

The plain dump parser keyed on <crossref>/<journal-prefix> alone over-collected
PACMPL (all 4 conferences) so PLDI/FSE recent years were dropped. Here we use the
<number> field to isolate the right conference, then append to dblp_pool.csv.
"""
import csv
import gzip
import html.entities
import xml.etree.ElementTree as ET

import pandas as pd

DUMP = "dblp.xml.gz"
POOL = "dblp_pool.csv"

# (journal_prefix, number_value) -> (venue, vol->year)
ISSUE_MAP = {
    ("pacmpl", "PLDI"): ("PLDI", lambda v: v + 2016),   # vol7=2023
    ("pacmse", "FSE"):  ("FSE",  lambda v: v + 2023),   # vol1=2024
}


def records(path):
    parser = ET.XMLParser()
    parser.entity.update(html.entities.entitydefs)
    with gzip.open(path, "rb") as f:
        ctx = ET.iterparse(f, events=("end",), parser=parser)
        for _, e in ctx:
            if e.tag == "article":
                yield e
                e.clear()


def main():
    want = set()
    for r in pd.read_csv("/tmp/award_venue_years.csv").itertuples():
        want.add((r.venue, int(r.year)))

    existing_keys = set()
    pool = pd.read_csv(POOL)
    existing_keys = set(pool.dblp_key.tolist())
    existing_vy = set(map(tuple, pool[["venue", "year"]].drop_duplicates().values.tolist()))

    new_rows = []
    for e in records(DUMP):
        k = e.get("key") or ""
        if not k.startswith("journals/"):
            continue
        pref = k.split("/")[1] if "/" in k else ""
        num = e.findtext("number")
        key2 = (pref, num)
        if key2 not in ISSUE_MAP:
            continue
        venue, v2y = ISSUE_MAP[key2]
        volt = e.findtext("volume")
        if not volt or not volt.isdigit():
            continue
        year = v2y(int(volt))
        if (venue, year) not in want:
            continue
        if k in existing_keys:
            continue
        te = e.find("title")
        if te is None:
            continue
        title = "".join(te.itertext()).strip().rstrip(".")
        if not title:
            continue
        existing_keys.add(k)
        new_rows.append({"venue": venue, "year": year, "title": title,
                         "dblp_key": k, "dblp_type": "Journal Articles",
                         "dblp_venue": venue})

    print(f"recovered {len(new_rows)} journal papers")
    from collections import Counter
    per = Counter((r["venue"], r["year"]) for r in new_rows)
    for k in sorted(per):
        print("  ", k, per[k])

    if new_rows:
        with open(POOL, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["venue", "year", "title",
                                              "dblp_key", "dblp_type", "dblp_venue"])
            w.writerows(new_rows)
        print(f"appended to {POOL}")


if __name__ == "__main__":
    main()
