#!/usr/bin/env python3
"""Fetch authoritative per-venue-year paper lists for ICLR/NeurIPS/ICML from DBLP.

Why DBLP: OpenAlex's conference sources for these venues are MAG-era relics --
coverage is partial before 2022 and essentially zero after (these venues don't
issue DOIs, so nothing flows in via Crossref). DBLP defines venue x year
membership; OpenAlex later supplies citations + abstracts.

Why toc: queries (not venue:X:): venue field search matches at the term
level, so "venue:ICLR:" also returns ICLR workshop-track papers in years
where dblp labels them venue "ICLR" (2017: 311 hits vs 198 accepted). The
per-year table-of-contents files are exact: toc:db/conf/iclr/iclr2017.bht:
returns 199 = 198 papers + 1 front-matter Editorship (dropped by the type
filter). Verified against official accepted counts: ICLR 2018 = 337, NIPS
2013 = 361, ICML 2017 = 435. NeurIPS TOCs are nips{y}.bht through 2019 and
neurips{y}.bht after; we try both. Workshop / Datasets & Benchmarks /
companion volumes have their own TOC files and are therefore excluded.

Output: dblp_papers.csv -- venue, year, title, dblp_key, dblp_type, dblp_venue
"""
import csv
import time

import requests

API = "https://dblp.org/search/publ/api"
UA = "norm-research (mailto:alex2awesome@gmail.com)"
YEARS = range(2013, 2024)
TOC_CANDIDATES = {
    "ICLR": ["db/conf/iclr/iclr{y}.bht"],
    "NeurIPS": ["db/conf/nips/neurips{y}.bht", "db/conf/nips/nips{y}.bht"],
    "ICML": ["db/conf/icml/icml{y}.bht"],
}
H = 1000  # max page size


def fetch_page(session, q, f):
    for attempt in range(10):
        status = "conn-error"
        try:
            r = session.get(API,
                            params={"q": q, "format": "json", "h": H, "f": f},
                            timeout=60)
            status = r.status_code
            if r.status_code == 200:
                return r.json()["result"]
        except (requests.RequestException, ValueError, KeyError):
            pass
        # DBLP rate-limits hard (429s and dropped connections); back off
        wait = min(180, 10 * 2 ** attempt)
        print(f"    status {status}, backing off {wait}s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"DBLP gave up on {q!r} f={f}")


FIELDS = ["venue", "year", "title", "dblp_key", "dblp_type", "dblp_venue"]


def main():
    session = requests.Session()
    session.headers["User-Agent"] = UA
    # incremental checkpointing: append per venue-year, skip completed ones
    import os
    done_vy, seen_keys = set(), set()
    if os.path.exists("dblp_papers.partial.csv"):
        with open("dblp_papers.partial.csv") as f:
            for row in csv.DictReader(f):
                done_vy.add((row["venue"], int(row["year"])))
                seen_keys.add(row["dblp_key"])
    part_f = open("dblp_papers.partial.csv", "a", newline="")
    part_w = csv.DictWriter(part_f, fieldnames=FIELDS)
    if not done_vy:
        part_w.writeheader()
    rows = []
    for venue, tocs in TOC_CANDIDATES.items():
        for year in YEARS:
            if (venue, year) in done_vy:
                print(f"{venue} {year}: already done, skipping", flush=True)
                continue
            vy_rows = []
            total, got, kept = 0, 0, 0
            for toc in tocs:
                q = f"toc:{toc.format(y=year)}:"
                f = 0
                while True:
                    res = fetch_page(session, q, f)
                    hits = res["hits"]
                    total = int(hits["@total"])
                    if total == 0:
                        break
                    page = hits.get("hit", [])
                    for h in page:
                        info = h["info"]
                        key = info.get("key")
                        if key in seen_keys:
                            continue
                        if info.get("type") != "Conference and Workshop Papers":
                            continue  # drops front-matter Editorship entries
                        seen_keys.add(key)
                        v = info.get("venue")
                        if isinstance(v, list):
                            v = v[0] if v else ""
                        title = info.get("title") or ""
                        vy_rows.append({
                            "venue": venue,
                            "year": year,
                            "title": title.rstrip("."),
                            "dblp_key": key,
                            "dblp_type": info.get("type"),
                            "dblp_venue": v,
                        })
                        kept += 1
                    got += len(page)
                    f += len(page)  # DBLP may serve fewer than h per page
                    time.sleep(3)
                    if got >= total or not page:
                        break
                if total > 0:
                    break  # found the right TOC name; skip alternates
            for row in vy_rows:
                part_w.writerow(row)
            part_f.flush()
            rows.extend(vy_rows)
            print(f"{venue} {year}: toc_total={total} kept={kept}", flush=True)
    part_f.close()

    # final output = the (complete) partial file, deduped
    with open("dblp_papers.partial.csv") as f:
        all_rows = list(csv.DictReader(f))
    with open("dblp_papers.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {len(all_rows)} rows to dblp_papers.csv")


if __name__ == "__main__":
    main()
