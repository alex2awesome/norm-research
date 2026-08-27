"""Scrape Erma Bombeck Writing Competition winners.

NOTE: humorwriters.org (the old Erma Bombeck Writers' Workshop domain named
in some references) has LAPSED and is now casino spam — do not use it. The
competition is run by the Washington-Centerville Public Library (Centerville,
OH) in partnership with the University of Dayton workshop:

  index : https://www.wclibrary.info/erma/winners/
  detail: https://www.wclibrary.info/erma/winners/detail/?id={ID}&year={YYYY}&winner={W|H|F}

Index page structure (a single table):
  td.erma_header    -> 'Winning Entries - 2026'
  td.erma_subheader -> 'Humor Category - Local Area' / 'Humor Category -
                       Global' / 'Human Interest Category - ...'
  td.erma_essay     -> 'First Place: <a>Author</a> - City, ST, Country'
                       (also 'Honorable Mention:' / 'Finalist:')

The live index covers the biennial cycles 2018-2026 only, BUT the detail
pages for older cycles are still served by the live site — only the index
stopped listing them. We therefore also parse a 2019 Wayback snapshot of the
index (same table format, lists 2010-2016) and fetch those essays from the
live site. Cycles before 2010 used per-essay .asp pages with a different
structure and are NOT collected (known gap).

Detail page: h2.heading2-sub = '2026 First Place - Humor - Local',
td.erma_header = '"Title" - Written By: Author - City, ST, Country',
td.erma_essay = full essay, td#erma_bio = author bio (excluded).

Output: erma.jsonl in the staging dir.
"""
import json
import os
import re
import sys
import urllib.parse

from bs4 import BeautifulSoup

from fetch_util import STAGING, fetch
from normalize import normalize_text

BASE = "https://www.wclibrary.info"
INDEX = f"{BASE}/erma/winners/"
# 2019 Wayback capture of the same index page; lists cycles 2010-2016 that
# the live index has since dropped. Detail links are re-pointed at the live
# site, which still serves them.
WAYBACK_INDEX = "https://web.archive.org/web/20191112204602/https://wclibrary.info/erma/winners/"
OUT_PATH = os.path.join(STAGING, "erma.jsonl")

TIER_MAP = {
    "first place": "first_place",
    "honorable mention": "honorable_mention",
    "finalist": "finalist",
}


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def parse_index(index_url: str = INDEX, link_base: str = INDEX) -> list[dict]:
    html = fetch(index_url)
    if html is None:
        raise RuntimeError(f"could not fetch Erma winners index {index_url}")
    soup = BeautifulSoup(html, "lxml")
    rows = []
    year = None
    category = None
    for td in soup.find_all("td", class_=re.compile(r"erma_(header|subheader|essay)")):
        cls = td.get("class", [""])[0]
        text = _clean(td.get_text(" ", strip=True))
        if cls == "erma_header":
            m = re.search(r"Winning Entries\s*-\s*(\d{4})", text)
            year = m.group(1) if m else None
        elif cls == "erma_subheader":
            category = text.replace(" Category", "")  # 'Humor - Local Area'
        elif cls == "erma_essay":
            m = re.match(r"^([A-Za-z ]+):", text)
            if not m or year is None:
                continue
            tier = TIER_MAP.get(m.group(1).strip().lower())
            a = td.find("a", href=True)
            if tier is None or a is None:
                continue
            author = _clean(a.get_text(" ", strip=True))
            href = a["href"].strip()
            # strip any Wayback prefix so detail links hit the live site
            href = re.sub(r"^(?:https?://web\.archive\.org)?/web/\d+(?:[a-z]{2}_)?/", "", href)
            url = urllib.parse.urljoin(link_base, href)
            rows.append({
                "contest": "erma_bombeck",
                "cycle": year,
                "rank_tier": tier,
                "category": category,
                "author": author,
                "url": url,
            })
    return rows


def parse_detail(url: str) -> tuple[str | None, str | None]:
    """Returns (title, raw_essay_text)."""
    html = fetch(url)
    if html is None:
        return None, None
    soup = BeautifulSoup(html, "lxml")
    title = None
    hdr = soup.find("td", class_="erma_header")
    if hdr is not None:
        htext = _clean(hdr.get_text(" ", strip=True))
        m = re.match(r'^[“"](?P<t>.+?)[”"]\s*-\s*Written By', htext)
        if m:
            title = m.group("t").strip()
        else:
            m = re.match(r"^(?P<t>.+?)\s*-\s*Written By", htext)
            title = m.group("t").strip() if m else None

    essay_td = soup.find("td", class_="erma_essay")
    if essay_td is None:
        return title, None
    for br in essay_td.find_all("br"):
        br.insert_before("\n")
        br.unwrap()
    paras = []
    for p in essay_td.find_all("p"):
        t = p.get_text().strip("\n")
        if t.strip():
            paras.append(t)
    if not paras:  # essay may be bare text inside the td
        t = essay_td.get_text()
        if t.strip():
            paras = [t.strip("\n")]
    return title, ("\n\n".join(paras) if paras else None)


def main():
    rows = parse_index()
    live_years = {r["cycle"] for r in rows}
    try:
        wb_rows = [r for r in parse_index(WAYBACK_INDEX, INDEX) if r["cycle"] not in live_years]
        print(f"[erma] wayback snapshot added cycles: {sorted({r['cycle'] for r in wb_rows})}")
        rows.extend(wb_rows)
    except RuntimeError as e:
        print(f"[erma] WARNING: wayback index unavailable, older cycles skipped ({e})")
    from collections import Counter
    print(f"[erma] index rows: {len(rows)} by cycle {dict(Counter(r['cycle'] for r in rows))}")
    for i, row in enumerate(rows):
        title, raw = parse_detail(row["url"])
        row["title"] = title
        row["raw_text"] = raw
        row["text"] = normalize_text(raw)
        row["text_available"] = bool(raw)
        if (i + 1) % 25 == 0:
            print(f"[erma] fetched {i + 1}/{len(rows)} detail pages")
    # uniform field order
    out_rows = [{
        "contest": r["contest"], "cycle": r["cycle"], "rank_tier": r["rank_tier"],
        "category": r["category"], "title": r["title"], "author": r["author"],
        "url": r["url"], "text": r["text"], "raw_text": r["raw_text"],
        "text_available": r["text_available"],
    } for r in rows]
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_text = sum(1 for r in out_rows if r["text_available"])
    print(f"[erma] wrote {len(out_rows)} rows ({n_text} with text) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
