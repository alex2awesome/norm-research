#!/usr/bin/env python3
"""Parse best_papers (acad-B2) venue-year membership from the DBLP XML dump.

The DBLP search API throttles per IP within minutes (the API fetcher stalled
on conn-errors / 429 backoff). The static dump (dblp.xml.gz, ~1GB) is not
throttled and contains all proceedings membership.

Membership model:
  * Conference proceedings: <inproceedings> whose <crossref> points to the
    main-track proceedings volume of the venue-year, e.g. conf/pldi/2019,
    conf/sosp/2019, conf/chi/2019. Workshop / companion / datasets&benchmarks
    volumes have suffixed crossref keys (conf/iclr/2017w, conf/nips/2021dbt,
    conf/sigmod/2019c, ...) and are excluded by an anchored main-track regex.
  * Journal-published proceedings: <article> in the right journal volume:
        VLDB        -> PVLDB     journals/pvldb/...{vol}     vol=year-2007
        SIGGRAPH    -> ACM TOG   journals/tog/...{vol}       vol=year-1981
        SIGMETRICS  -> POMACS    journals/pomacs/...{vol}    vol=year-2016 (>=2017)
        SIGMOD 23+  -> PACMMOD   journals/pacmmod/...{vol}   vol=year-2022 (>=2023)
    For these we map the article's journal+volume back to (venue, year).

We restrict to the exact (venue, year) pairs that have a best-paper award
(award_venue_years.csv) so the pool is the same universe as the labels.

Output: dblp_pool.csv  -- venue, year, title, dblp_key, dblp_type, dblp_venue
Also prints per-(venue,year) kept counts and total venue-years enumerated.
"""
import argparse
import csv
import gzip
import html.entities
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import pandas as pd

# venue -> compiled crossref regex with a capturing group for the year.
# anchored ^...$ so suffixed workshop/companion volumes are excluded.
CONF = {
    "PLDI":     r"^conf/pldi/(\d{4})$",
    "SOSP":     r"^conf/sosp/(\d{4})$",
    "STOC":     r"^conf/stoc/(\d{4})$",
    "FOCS":     r"^conf/focs/(\d{4})$",
    "CHI":      r"^conf/chi/(\d{4})$",
    "CVPR":     r"^conf/cvpr/(\d{4})$",
    "ISCA":     r"^conf/isca/(\d{4})$",
    "SIGCOMM":  r"^conf/sigcomm/(\d{4})$",
    "UIST":     r"^conf/uist/(\d{4})$",
    "PODS":     r"^conf/pods/(\d{4})$",
    "ICCV":     r"^conf/iccv/(\d{4})$",
    "S&P":      r"^conf/sp/(\d{4})$",
    "INFOCOM":  r"^conf/infocom/(\d{4})$",
    "MOBICOM":  r"^conf/mobicom/(\d{4})$",
    "ACL":      r"^conf/acl/(\d{4})(?:-\d+)?$",   # multi-volume -1/-2/...
    "FSE":      r"^conf/(?:sigsoft|fse)/(?:fse)?(\d{4})$",
    "WWW":      r"^conf/www/(\d{4})$",
    "CIKM":     r"^conf/cikm/(\d{4})$",
    "SIGIR":    r"^conf/sigir/(\d{4})$",
    "KDD":      r"^conf/kdd/(\d{4})$",
    "SODA":     r"^conf/soda/(\d{4})$",
    "ICSE":     r"^conf/icse/(\d{4})$",
    "NSDI":     r"^conf/nsdi/(\d{4})$",
    "OSDI":     r"^conf/osdi/(\d{4})$",
    "AAAI":     r"^conf/aaai/(\d{4})$",
    "IJCAI":    r"^conf/ijcai/(\d{4})$",
    "ICML":     r"^conf/icml/(\d{4})$",
    "NeurIPS":  r"^conf/nips/(\d{4})$",
    "ICLR":     r"^conf/iclr/(\d{4})$",
    # SIGMOD pre-2023 conference; 2023+ in PACMMOD (article path below)
    "SIGMOD":   r"^conf/sigmod/(\d{4})$",
    # SIGMETRICS pre-2017 conference; 2017+ in POMACS (article path below)
    "SIGMETRICS": r"^conf/sigmetrics/(\d{4})$",
    # VLDB had old conference volumes pre-2008
    "VLDB":     r"^conf/vldb/(\d{4})$",
}
CONF_RX = {v: re.compile(p) for v, p in CONF.items()}

# journal article membership: (venue, journal_prefix_regex, vol->year fn)
# article key looks like journals/pvldb/SomeName17 ; we read the <journal> +
# <volume> fields instead of the key to be robust.
JOURNAL = {
    "VLDB":       ("pvldb",   lambda vol: vol + 2007, lambda y: y >= 2008),
    "SIGGRAPH":   ("tog",     lambda vol: vol + 1981, lambda y: True),
    "SIGMETRICS": ("pomacs",  lambda vol: vol + 2016, lambda y: y >= 2017),
    "SIGMOD":     ("pacmmod", lambda vol: vol + 2022, lambda y: y >= 2023),
}
# reverse: journal_prefix -> (venue, vol->year fn)
JPREFIX = {pref: (v, f, g) for v, (pref, f, g) in JOURNAL.items()}


def records(path):
    parser = ET.XMLParser()
    parser.entity.update(html.entities.entitydefs)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        context = ET.iterparse(f, events=("end",), parser=parser)
        for _, elem in context:
            if elem.tag in ("inproceedings", "article"):
                yield elem
                elem.clear()


def journal_prefix_of_key(key):
    # key like journals/pvldb/Foo17 -> pvldb
    m = re.match(r"^journals/([a-z0-9]+)/", key or "")
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--vy", default="/tmp/award_venue_years.csv")
    ap.add_argument("--out", default="dblp_pool.csv")
    args = ap.parse_args()

    want = set()
    vydf = pd.read_csv(args.vy)
    for r in vydf.itertuples():
        want.add((r.venue, int(r.year)))
    want_venues = set(v for v, _ in want)
    want_years = set(y for _, y in want)
    print(f"target: {len(want)} venue-years across {len(want_venues)} venues", file=sys.stderr)

    rows = []
    seen_keys = set()
    n = 0
    for elem in records(args.dump):
        n += 1
        if n % 2000000 == 0:
            print(f"  scanned {n/1e6:.0f}M records, kept {len(rows)}...", file=sys.stderr)

        if elem.tag == "inproceedings":
            cr = elem.findtext("crossref") or ""
            if not cr:
                continue
            matched = None
            for venue, rx in CONF_RX.items():
                if venue not in want_venues:
                    continue
                m = rx.match(cr)
                if m:
                    year = int(m.group(1))
                    if (venue, year) in want:
                        matched = (venue, year)
                    break
            if not matched:
                continue
            venue, year = matched
            key = elem.get("key")
            if key in seen_keys:
                continue
            te = elem.find("title")
            if te is None:
                continue
            title = "".join(te.itertext()).strip().rstrip(".")
            if not title:
                continue
            seen_keys.add(key)
            rows.append({"venue": venue, "year": year, "title": title,
                         "dblp_key": key, "dblp_type": "Conference and Workshop Papers",
                         "dblp_venue": venue})

        else:  # article -- journal-published proceedings
            key = elem.get("key") or ""
            pref = journal_prefix_of_key(key)
            if pref not in JPREFIX:
                continue
            venue, vol2year, yrok = JPREFIX[pref]
            if venue not in want_venues:
                continue
            volt = elem.findtext("volume")
            if not volt or not volt.isdigit():
                continue
            year = vol2year(int(volt))
            if not yrok(year) or (venue, year) not in want:
                continue
            # tog (SIGGRAPH) carries many non-SIGGRAPH papers; keep only
            # technical-papers issues? DBLP doesn't separate cleanly, so we
            # keep all TOG vol papers of that year as the candidate pool.
            if key in seen_keys:
                continue
            te = elem.find("title")
            if te is None:
                continue
            title = "".join(te.itertext()).strip().rstrip(".")
            if not title:
                continue
            seen_keys.add(key)
            rows.append({"venue": venue, "year": year, "title": title,
                         "dblp_key": key, "dblp_type": "Journal Articles",
                         "dblp_venue": venue})

    print(f"scanned {n} records total", file=sys.stderr)
    with open(args.out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=["venue", "year", "title", "dblp_key",
                                           "dblp_type", "dblp_venue"])
        w.writeheader()
        w.writerows(rows)

    per = Counter((r["venue"], r["year"]) for r in rows)
    pv = Counter(r["venue"] for r in rows)
    enum_vy = len(per)
    print(f"\nenumerated {enum_vy}/{len(want)} target venue-years; {len(rows)} papers", file=sys.stderr)
    print("=== per venue (venue-years enumerated / target, papers) ===", file=sys.stderr)
    for v in sorted(want_venues):
        tvy = sum(1 for (vv, _) in want if vv == v)
        evy = sum(1 for (vv, _) in per if vv == v)
        print(f"  {v:12s} {evy:2d}/{tvy:2d} vy   {pv.get(v,0):6d} papers", file=sys.stderr)
    # which target venue-years are missing
    missing = sorted(want - set(per.keys()))
    print(f"\nmissing venue-years ({len(missing)}):", file=sys.stderr)
    print("  " + ", ".join(f"{v}{y}" for v, y in missing[:80]), file=sys.stderr)


if __name__ == "__main__":
    main()
