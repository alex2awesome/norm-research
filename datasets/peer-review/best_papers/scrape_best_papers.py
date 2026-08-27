#!/usr/bin/env python3
"""Scrape Jeff Huang's Best Paper Awards in Computer Science page.

https://jeffhuang.com/best_paper_awards/

Page structure (as of 2026-06): one <table> per year (1996-2025). Within a
table, each venue starts with a <th class="category-name" rowspan=N> cell
containing the venue acronym (in an <a>) and field name (in a <span>);
the N rows it spans are the awarded papers (title cell + authors cell).

Output: best_papers_awards.csv with columns
    venue, field, year, title, authors_raw
"""
import argparse
import csv
import sys

import requests
from bs4 import BeautifulSoup

URL = "https://jeffhuang.com/best_paper_awards/"
UA = "norm-research scraper (mailto:alex2awesome@gmail.com)"


def parse(html: str):
    soup = BeautifulSoup(html, "html.parser")
    records = []
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        # first row is the year header (single th)
        year_text = rows[0].get_text(strip=True)
        try:
            year = int(year_text)
        except ValueError:
            print(f"WARNING: table with non-year header {year_text!r}, skipping",
                  file=sys.stderr)
            continue
        cur_venue, cur_field = None, None
        for r in rows[1:]:
            cells = r.find_all(["td", "th"])
            if not cells:
                continue
            if "category-name" in (cells[0].get("class") or []):
                vc = cells[0]
                a = vc.find("a")
                span = vc.find("span")
                cur_venue = (a.get_text(strip=True) if a
                             else vc.get_text(" ", strip=True))
                cur_field = span.get_text(" ", strip=True) if span else ""
                cells = cells[1:]
            if len(cells) < 1:
                continue
            title = cells[0].get_text(" ", strip=True)
            authors = cells[1].get_text(" ", strip=True) if len(cells) > 1 else ""
            if not title:
                continue
            records.append({
                "venue": cur_venue,
                "field": cur_field,
                "year": year,
                "title": title,
                "authors_raw": authors,
            })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", help="path to pre-downloaded HTML (skips fetch)")
    ap.add_argument("--out", default="best_papers_awards.csv")
    args = ap.parse_args()

    if args.html:
        html = open(args.html).read()
    else:
        resp = requests.get(URL, headers={"User-Agent": UA}, timeout=60)
        resp.raise_for_status()
        html = resp.text

    records = parse(html)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["venue", "field", "year", "title",
                                          "authors_raw"])
        w.writeheader()
        w.writerows(records)

    years = sorted({r["year"] for r in records})
    venues = sorted({r["venue"] for r in records})
    print(f"wrote {len(records)} awards to {args.out}")
    print(f"years: {years[0]}-{years[-1]} ({len(years)})")
    print(f"venues ({len(venues)}): {', '.join(venues)}")


if __name__ == "__main__":
    main()
