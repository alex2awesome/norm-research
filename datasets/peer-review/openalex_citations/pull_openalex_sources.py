#!/usr/bin/env python3
"""Cursor-paginate all OpenAlex works for ICLR/NeurIPS/ICML sources, 2013-2023.

Source ids (resolved 2026-06 via api.openalex.org/sources?search=...):
    NeurIPS  S4306420609  "Neural Information Processing Systems"   (conference)
    ICLR     S4306419637  "International Conference on Learning
                           Representations"                          (conference)
    ICML     S4306419644  "International Conference on Machine
                           Learning"                                 (conference)

KNOWN LIMITATION (documented per-year in the run log): these are MAG-era
sources. Coverage is partial 2013-2021 (e.g. NeurIPS 2019: 761 works vs 1,428
accepted) and collapses to ~zero from 2022 on, because these venues don't
issue DOIs. PMLR / "Advances in NeurIPS" sources hold <10 works each. The
title-level join against DBLP membership (join_and_build.py) fills the gap.

Filter: locations.source.id (catches works whose venue record is not the
primary location), publication_year 2013-2023, type:article.

Output: openalex_source_works.parquet
"""
import json
import time

import pandas as pd
import requests

MAILTO = "alex2awesome@gmail.com"
API = "https://api.openalex.org/works"
SOURCES = {
    "NeurIPS": "S4306420609",
    "ICLR": "S4306419637",
    "ICML": "S4306419644",
}
SELECT = ("id,doi,title,publication_year,type,cited_by_count,"
          "abstract_inverted_index,primary_location")


def decode_abstract(inv):
    if not inv:
        return ""
    pos = [(i, w) for w, idxs in inv.items() for i in idxs]
    pos.sort()
    return " ".join(w for _, w in pos)


def main():
    session = requests.Session()
    session.headers["User-Agent"] = f"norm-research (mailto:{MAILTO})"
    rows = []
    for venue, sid in SOURCES.items():
        cursor = "*"
        n = 0
        while cursor:
            params = {
                "filter": (f"locations.source.id:{sid},"
                           f"publication_year:2013-2023,type:article"),
                "select": SELECT,
                "per-page": "200",
                "cursor": cursor,
                "mailto": MAILTO,
            }
            for attempt in range(6):
                try:
                    r = session.get(API, params=params, timeout=120)
                    if r.status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(2 ** attempt)
            j = r.json()
            for w in j["results"]:
                src = (w.get("primary_location") or {}).get("source") or {}
                rows.append({
                    "venue": venue,
                    "openalex_id": w["id"],
                    "doi": w.get("doi"),
                    "title": w.get("title"),
                    "publication_year": w.get("publication_year"),
                    "type": w.get("type"),
                    "cited_by_count": w.get("cited_by_count"),
                    "abstract": decode_abstract(w.get("abstract_inverted_index")),
                    "primary_source": src.get("display_name"),
                })
            n += len(j["results"])
            cursor = j["meta"].get("next_cursor")
            if not j["results"]:
                break
            time.sleep(0.15)
        print(f"{venue}: {n} works", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet("openalex_source_works.parquet", index=False)
    print(df.groupby(["venue", "publication_year"]).size().to_string())
    print(f"\nwrote {len(df)} rows to openalex_source_works.parquet")


if __name__ == "__main__":
    main()
