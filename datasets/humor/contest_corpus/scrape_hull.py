"""Scrape To Hull And Back humorous short story competition results
(christopherfielden.com).

Results pages (note: NOT '.../to-hull-and-back-short-story-competition/'):
  https://www.christopherfielden.com/to-hull-and-back-humorous-short-story-competition/results-{YYYY}/
for cycles 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2023, 2025 (annual to
2019, then biennial), discovered from the competition landing page.

Page structure (WordPress + ez-toc, consistent across all years):
  h2 'Winners'           -> h3 tier headings (1st/2nd/3rd Prize, Highly
                            Commended) followed by <p>'Title, by Author'</p>
  h2 'The Shortlist'     -> <p><a>Author</a> - Title</p>  (en dash)
  h2 'The Longlist'      -> same (2016+; disjoint from shortlist)
  h2 'Special Mentions'  -> same (most years)
Any other h2 (videos, bios, judges, notes) ends collection.

Winners appear both under 'Winners' and in 'The Shortlist'; we keep only the
highest tier per (author, title).

Full story text is NOT freely posted on the site (stories are published in
the paid To Hull And Back anthologies), so text_available=False for all rows.

Output: hull.jsonl in the staging dir.
"""
import json
import os
import re
import sys

from bs4 import BeautifulSoup

from fetch_util import STAGING, fetch
from normalize import normalize_text

BASE = "https://www.christopherfielden.com"
LANDING = f"{BASE}/to-hull-and-back-humorous-short-story-competition/"
OUT_PATH = os.path.join(STAGING, "hull.jsonl")

SECTION_MAP = {
    "winners": "winners",
    "the shortlist": "shortlist",
    "the longlist": "longlist",
    "special mentions": "special_mention",
}
TIER_MAP = {
    "1st prize": "first_prize",
    "2nd prize": "second_prize",
    "3rd prize": "third_prize",
    "highly commended": "highly_commended",
}
TIER_ORDER = [
    "first_prize", "second_prize", "third_prize", "highly_commended",
    "shortlist", "longlist", "special_mention",
]


def discover_cycles() -> list[tuple[str, str]]:
    html = fetch(LANDING)
    if html is None:
        raise RuntimeError("could not fetch Hull landing page")
    soup = BeautifulSoup(html, "lxml")
    cycles = []
    seen = set()
    for a in soup.find_all("a", href=True):
        m = re.search(r"/to-hull-and-back-humorous-short-story-competition/results-(\d{4})/?$", a["href"])
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            cycles.append((m.group(1), f"{BASE}/to-hull-and-back-humorous-short-story-competition/results-{m.group(1)}/"))
    return sorted(cycles)


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def parse_results_page(year: str, url: str) -> list[dict]:
    html = fetch(url)
    if html is None:
        print(f"  !! results page failed: {url}", file=sys.stderr)
        return []
    soup = BeautifulSoup(html, "lxml")
    # drop the ez-toc table of contents so its links don't interfere
    for toc in soup.select("#ez-toc-container, .ez-toc-list, nav"):
        toc.decompose()
    body = soup.body

    rows = []
    section = None   # winners | shortlist | longlist | special_mention | None
    tier = None      # within winners
    for el in body.find_all(["h2", "h3", "p"]):
        text = _clean(el.get_text(" ", strip=True))
        if el.name == "h2":
            section = SECTION_MAP.get(text.lower())
            tier = None
            continue
        if el.name == "h3":
            if section == "winners":
                tier = TIER_MAP.get(text.lower())
            else:
                tier = None
            continue
        if section is None or not text:
            continue
        if section == "winners":
            if tier is None:
                continue
            # 'Title, by <a>Author</a>' (2015+) or 'Title by <a>Author</a>'
            # (2014); anchor parse on the author bio link to avoid mis-
            # splitting titles that themselves contain ' by '
            a = el.find("a", href=re.compile(r"^#"))
            if a is None:
                continue
            author = _clean(a.get_text(" ", strip=True))
            before = "".join(
                prev.get_text() if hasattr(prev, "get_text") else str(prev)
                for prev in a.previous_siblings
            )
            m = re.match(r"^(?P<title>.+?),?\s+by\s*$", _clean(before) + " ")
            if not m or not (2 <= len(author) <= 60):
                continue
            rows.append((tier, author, _clean(m.group("title"))))
        else:
            # '<a>Author</a> - Title' with en/em dash; require a bio anchor link
            a = el.find("a", href=re.compile(r"#"))
            if a is None:
                continue
            m = re.match(r"^(?P<author>.+?)\s*[–—-]\s+(?P<title>.+)$", text)
            if not m:
                continue
            author = _clean(m.group("author"))
            if len(author) > 60 or len(author) < 2:
                continue
            rows.append((section, author, _clean(m.group("title"))))

    # dedupe: keep highest tier per (author, title)
    best: dict[tuple[str, str], str] = {}
    for tier, author, title in rows:
        key = (author.lower(), title.lower())
        if key not in best or TIER_ORDER.index(tier) < TIER_ORDER.index(best[key]):
            best[key] = tier
    out = []
    emitted = set()
    for tier, author, title in rows:
        key = (author.lower(), title.lower())
        if key in emitted or best[key] != tier:
            continue
        emitted.add(key)
        out.append({
            "contest": "to_hull_and_back",
            "cycle": year,
            "rank_tier": tier,
            "category": None,
            "title": normalize_text(title),
            "author": normalize_text(author),
            "url": url,
            "text": None,
            "raw_text": None,
            "text_available": False,
        })
    return out


def main():
    cycles = discover_cycles()
    print(f"[hull] {len(cycles)} cycles: {[c[0] for c in cycles]}")
    all_rows = []
    for year, url in cycles:
        rows = parse_results_page(year, url)
        from collections import Counter
        print(f"[hull] {year}: {len(rows)} rows {dict(Counter(r['rank_tier'] for r in rows))}")
        all_rows.extend(rows)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[hull] wrote {len(all_rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
