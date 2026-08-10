#!/usr/bin/env python3
"""Enumerate DBLP venue-year proceedings membership for the best_papers (acad-B2) venues.

Reuses the TOC-query approach from openalex_citations/fetch_dblp_lists.py but
extends it to all 32 award venues, including the CS-flagship venues that
OpenAlex's venue pools cannot resolve (PLDI/SOSP/STOC/CHI/CVPR/ISCA/SIGCOMM/
SIGGRAPH/...). DBLP defines venue x year membership cleanly via the per-year
table-of-contents (.bht) files.

Journal-published proceedings need year->volume maps:
  VLDB        -> PVLDB        vol = year - 2007  (2008=pvldb1 ... 2024=pvldb17)
  SIGGRAPH    -> ACM TOG      vol = year - 1981  (technical papers track)
  SIGMETRICS  -> POMACS       vol = year - 2016  (2017=pomacs1; pre-2017 conf TOC)
  SIGMOD 23+  -> PACMMOD      vol = year - 2022  (2023=pacmmod1; pre-2023 conf TOC)

Multi-volume conferences (ACL) iterate -1, -2, -3 suffixes.

We try every candidate TOC for a (venue, year); accept the first that returns
>0 hits (conf TOCs) OR union all matching volumes (ACL multi-volume).
Only type == "Conference and Workshop Papers" (or Journal Articles for the
journal-published proceedings) are kept; front-matter Editorship dropped.

Output: dblp_pool.csv  -- venue, year, title, dblp_key, dblp_type, dblp_venue
Resumable via dblp_pool.partial.csv (per venue-year checkpoint).
"""
import csv
import os
import sys
import time

import requests
import pandas as pd

API = "https://dblp.org/search/publ/api"
UA = "norm-research best_papers acad-B2 (mailto:alex2awesome@gmail.com)"
H = 1000


def toc_candidates(venue, year):
    """Return ordered list of (toc_path, accept_journal_articles) candidates."""
    y = year
    c = []
    if venue == "VLDB":
        vol = y - 2007
        if vol >= 1:
            c.append((f"db/journals/pvldb/pvldb{vol}.bht", True))
        # very early VLDB had conference proceedings
        c.append((f"db/conf/vldb/vldb{y}.bht", False))
    elif venue == "SIGGRAPH":
        vol = y - 1981
        c.append((f"db/journals/tog/tog{vol}.bht", True))
    elif venue == "SIGMETRICS":
        if y >= 2017:
            vol = y - 2016
            c.append((f"db/journals/pomacs/pomacs{vol}.bht", True))
        c.append((f"db/conf/sigmetrics/sigmetrics{y}.bht", False))
    elif venue == "SIGMOD":
        if y >= 2023:
            vol = y - 2022
            c.append((f"db/journals/pacmmod/pacmmod{vol}.bht", True))
        c.append((f"db/conf/sigmod/sigmod{y}.bht", False))
    elif venue == "ACL":
        # multi-volume: acl{y}-1, -2, ... ; also legacy plain acl{y}
        for suf in ["-1", "-2", "-3", "-4", "-5", ""]:
            c.append((f"db/conf/acl/acl{y}{suf}.bht", False))
    elif venue == "FSE":
        # FSE lives under sigsoft; older years sometimes fse{y}
        c.append((f"db/conf/sigsoft/fse{y}.bht", False))
        c.append((f"db/conf/fse/fse{y}.bht", False))
    elif venue == "S&P":
        c.append((f"db/conf/sp/sp{y}.bht", False))
        c.append((f"db/conf/sp/sp{y}-1.bht", False))
    elif venue == "OSDI":
        c.append((f"db/conf/osdi/osdi{y}.bht", False))
    elif venue == "NeurIPS":
        c.append((f"db/conf/nips/neurips{y}.bht", False))
        c.append((f"db/conf/nips/nips{y}.bht", False))
    else:
        # generic: db/conf/{key}/{key}{year}.bht  (key = lowercased venue)
        key = venue.lower()
        c.append((f"db/conf/{key}/{key}{y}.bht", False))
        # some venues split into -1/-2 in recent years (CVPR, ICCV, etc.)
        c.append((f"db/conf/{key}/{key}{y}-1.bht", False))
    return c


def fetch_page(session, q, f):
    for attempt in range(10):
        status = "conn-error"
        try:
            r = session.get(API, params={"q": q, "format": "json", "h": H, "f": f},
                            timeout=60)
            status = r.status_code
            if r.status_code == 200:
                return r.json()["result"]
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                wait = float(ra) if ra and ra.replace(".", "").isdigit() else min(180, 10 * 2 ** attempt)
                print(f"    429, backing off {wait}s", flush=True)
                time.sleep(wait)
                continue
        except (requests.RequestException, ValueError, KeyError):
            pass
        wait = min(180, 10 * 2 ** attempt)
        print(f"    status {status}, backing off {wait}s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"DBLP gave up on {q!r} f={f}")


def fetch_toc(session, toc, accept_journal):
    """Return list of (title, key, type, venue) for one TOC, or [] if empty."""
    q = f"toc:{toc}:"
    out = []
    f = 0
    total = 0
    while True:
        res = fetch_page(session, q, f)
        hits = res["hits"]
        total = int(hits["@total"])
        if total == 0:
            return [], 0
        page = hits.get("hit", [])
        for h in page:
            info = h["info"]
            typ = info.get("type")
            ok = typ == "Conference and Workshop Papers" or (
                accept_journal and typ == "Journal Articles")
            if not ok:
                continue
            v = info.get("venue")
            if isinstance(v, list):
                v = v[0] if v else ""
            title = (info.get("title") or "").rstrip(".")
            out.append((title, info.get("key"), typ, v))
        f += len(page)
        time.sleep(3)
        if f >= total or not page:
            break
    return out, total


FIELDS = ["venue", "year", "title", "dblp_key", "dblp_type", "dblp_venue"]


def main():
    vy = pd.read_csv("/tmp/award_venue_years.csv")
    # sort so the previously-broken CS flagships go first (high value)
    PRIORITY = ["PLDI", "SOSP", "STOC", "FOCS", "CHI", "CVPR", "ISCA",
                "SIGCOMM", "SIGGRAPH", "UIST", "PODS", "ICCV", "S&P",
                "INFOCOM", "MOBICOM", "SIGMETRICS", "ACL", "FSE", "WWW",
                "CIKM", "SIGIR", "KDD", "SODA", "ICSE", "SIGMOD", "VLDB",
                "NSDI", "OSDI", "AAAI", "IJCAI", "ICML", "NeurIPS"]
    vy["prio"] = vy["venue"].apply(lambda v: PRIORITY.index(v) if v in PRIORITY else 99)
    vy = vy.sort_values(["prio", "venue", "year"]).reset_index(drop=True)

    done_vy, seen_keys = set(), set()
    if os.path.exists("dblp_pool.partial.csv"):
        with open("dblp_pool.partial.csv") as f:
            for row in csv.DictReader(f):
                done_vy.add((row["venue"], int(row["year"])))
                seen_keys.add(row["dblp_key"])
    # we also record empty venue-years so we don't retry them
    tried_empty = set()
    if os.path.exists("dblp_pool.empty.csv"):
        with open("dblp_pool.empty.csv") as f:
            for row in csv.DictReader(f):
                tried_empty.add((row["venue"], int(row["year"])))

    part_f = open("dblp_pool.partial.csv", "a", newline="")
    part_w = csv.DictWriter(part_f, fieldnames=FIELDS)
    if not done_vy:
        part_w.writeheader()
    empty_f = open("dblp_pool.empty.csv", "a", newline="")
    empty_w = csv.DictWriter(empty_f, fieldnames=["venue", "year", "tried"])
    if not tried_empty:
        empty_w.writeheader()

    session = requests.Session()
    session.headers["User-Agent"] = UA

    n_total = len(vy)
    n_enum = 0
    for i, r in enumerate(vy.itertuples()):
        venue, year = r.venue, int(r.year)
        if (venue, year) in done_vy or (venue, year) in tried_empty:
            continue
        cands = toc_candidates(venue, year)
        rows = []
        used = []
        multivol = venue == "ACL"
        for toc, acc_j in cands:
            recs, total = fetch_toc(session, toc, acc_j)
            if total > 0:
                used.append(f"{toc}({total})")
                for (title, key, typ, dv) in recs:
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    rows.append({"venue": venue, "year": year, "title": title,
                                 "dblp_key": key, "dblp_type": typ,
                                 "dblp_venue": dv})
                if not multivol:
                    break  # found the right single TOC
        if rows:
            for row in rows:
                part_w.writerow(row)
            part_f.flush()
            n_enum += 1
            print(f"[{i+1}/{n_total}] {venue} {year}: kept={len(rows)} via {used}", flush=True)
        else:
            empty_w.writerow({"venue": venue, "year": year,
                              "tried": "|".join(t for t, _ in cands)})
            empty_f.flush()
            print(f"[{i+1}/{n_total}] {venue} {year}: EMPTY (tried {len(cands)} tocs)", flush=True)

    part_f.close()
    empty_f.close()

    # consolidate
    with open("dblp_pool.partial.csv") as f:
        all_rows = list(csv.DictReader(f))
    with open("dblp_pool.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {len(all_rows)} rows to dblp_pool.csv; enumerated {n_enum} new venue-years this run")


if __name__ == "__main__":
    main()
