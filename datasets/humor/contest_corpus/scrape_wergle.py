"""Scrape Wergle Flomp Humor Poetry Contest winners (winningwriters.com).

Cycle archive pages:
  https://winningwriters.com/our-contests/contest-archives/wergle-flomp-humor-poetry-contest-{YYYY}
discovered from https://winningwriters.com/our-contests/contest-archives

Each cycle page lists tiers as <h4> headings ("First Prize $2,000",
"Honorable Mention $100", "Finalists") followed by either a <p> (inside a
.listing-item div, top prizes) or <ul><li> rows of the form
"Author, <a href=/past-winning-entries/...>Title</a>".

Each entry has its own page at /past-winning-entries/<slug> containing the
full poem text (paragraphs of <br>-separated lines after the div.meta block).

Output: one JSON line per winning entry -> wergle.jsonl in the staging dir.
"""
import json
import os
import re
import sys

from bs4 import BeautifulSoup

from fetch_util import STAGING, fetch
from normalize import normalize_text

BASE = "https://winningwriters.com"
ARCHIVE_INDEX = f"{BASE}/our-contests/contest-archives"
OUT_PATH = os.path.join(STAGING, "wergle.jsonl")

TIER_MAP = [
    (re.compile(r"^grand prize", re.I), "grand_prize"),
    (re.compile(r"^first prize", re.I), "first_prize"),
    (re.compile(r"^second prize", re.I), "second_prize"),
    (re.compile(r"^third prize", re.I), "third_prize"),
    (re.compile(r"^fourth prize", re.I), "fourth_prize"),
    (re.compile(r"^fifth prize", re.I), "fifth_prize"),
    (re.compile(r"^honou?rable mention", re.I), "honorable_mention"),
    (re.compile(r"^finalists?", re.I), "finalist"),
    (re.compile(r"^most highly commended", re.I), "most_highly_commended"),
    (re.compile(r"^highly commended", re.I), "highly_commended"),
    (re.compile(r"^commended", re.I), "commended"),
    (re.compile(r"^runner", re.I), "runner_up"),
]
STOP_HEADINGS = re.compile(r"contest judge|about the judge|what's new", re.I)

# Paragraphs on entry pages that are editorial notes, not part of the poem.
NOTE_PATTERNS = re.compile(
    r"^(illustration by|photo by|cartoon by|artwork by|art by|please enjoy|enjoy (this|the) (video|reading|interview)"
    r"|watch (a|the|ms|mr)|listen to|read (an|the) interview|see more of|this poem (first appeared|was|is reprinted)"
    r"|first published in|originally published|reprinted (from|by|with)|published in\b|courtesy of"
    r"|view the assignment|click here)",
    re.I,
)


def discover_cycles() -> list[tuple[str, str]]:
    html = fetch(ARCHIVE_INDEX)
    if html is None:
        raise RuntimeError("could not fetch contest archive index")
    soup = BeautifulSoup(html, "lxml")
    cycles = []
    seen = set()
    for a in soup.find_all("a", href=True):
        m = re.search(r"/contest-archives/(wergle-flomp-humor-poetry-contest-(\d{4}))/?$", a["href"])
        if m and m.group(2) not in seen:
            seen.add(m.group(2))
            cycles.append((m.group(2), f"{BASE}/our-contests/contest-archives/{m.group(1)}"))
    return sorted(cycles)


def classify_tier(h4_text: str) -> str | None:
    for rx, tier in TIER_MAP:
        if rx.search(h4_text.strip()):
            return tier
    return None


def parse_entry_line(el) -> tuple[str, str, str] | None:
    """Parse a <p>/<li> of the form 'Author, <a>Title</a>'. Returns
    (author, title, url) or None if the element doesn't match."""
    a = el.find("a", href=re.compile(r"/past-winning-entries/"))
    if a is None:
        return None
    # the link must be the LAST content in the element (otherwise it's a
    # judge-commentary paragraph that merely mentions the poem)
    after = ""
    for nxt in a.next_siblings:
        after += nxt.get_text() if hasattr(nxt, "get_text") else str(nxt)
    # tolerate a trailing media-type tag (e.g. ', Poetry' in the 2005 cycle)
    after = re.sub(r"^[\s,;]*(poetry|prose|audio|video|mp3)\b", "", after.strip(" \n\t.”\"'"), flags=re.I)
    if after.strip(" \n\t.,;”\"'"):
        return None
    # text before the link must be a short author name ending with a comma
    before = ""
    for prev in a.previous_siblings:
        before = (prev.get_text() if hasattr(prev, "get_text") else str(prev)) + before
    before = before.strip()
    if not before.endswith(",") or len(before) > 80:
        return None
    author = before.rstrip(",").strip()
    title = a.get_text(" ", strip=True)
    url = a["href"]
    if url.startswith("/"):
        url = BASE + url
    return author, title, url


def parse_cycle_page(year: str, url: str) -> list[dict]:
    html = fetch(url)
    if html is None:
        print(f"  !! cycle page failed: {url}", file=sys.stderr)
        return []
    soup = BeautifulSoup(html, "lxml")
    main = soup.find(id="content")
    for aside in main.find_all("aside"):
        aside.decompose()

    rows = []
    tier = None
    for el in main.find_all(["h4", "p", "li"]):
        if el.name == "h4":
            t = el.get_text(" ", strip=True)
            if STOP_HEADINGS.search(t):
                break
            new_tier = classify_tier(t)
            if new_tier:
                tier = new_tier
            continue
        if tier is None:
            continue
        parsed = parse_entry_line(el)
        if parsed is None:
            continue
        author, title, entry_url = parsed
        rows.append({
            "contest": "wergle_flomp",
            "cycle": year,
            "rank_tier": tier,
            "category": None,
            "title": title,
            "author": author,
            "url": entry_url,
        })
    return rows


# Site tombstones served where the poem used to be — NOT poem text.
TOMBSTONE_PATTERNS = re.compile(
    r"(we regret (the text of this poem|this content) is no longer available"
    r"|poem not displayed\??\s*download the pdf)",
    re.I,
)


def extract_pdf_poem(entry_url: str, html: str) -> str | None:
    """Some entries (e.g. 2018 'The Swipe Sonnets') embed the poem as a PDF
    with a 'Poem not displayed? Download the PDF' fallback link. Fetch the
    PDF (cached, same politeness) and extract its text."""
    m = re.search(r'href="(https?://winningwriters\.com/graphics/[^"]+\.pdf)"', html, re.I)
    if not m:
        return None
    from fetch_util import fetch_binary
    blob = fetch_binary(m.group(1))
    if blob is None:
        return None
    try:
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(blob))
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
        return text or None
    except Exception as e:
        print(f"  [pdf] extraction failed for {entry_url}: {e}")
        return None


def extract_poem(entry_url: str) -> str | None:
    html = fetch(entry_url)
    if html is None:
        return None
    soup = BeautifulSoup(html, "lxml")
    main = soup.find(id="content")
    if main is None:
        return None
    for aside in main.find_all("aside"):
        aside.decompose()
    meta = main.find("div", class_="meta")

    paras = []
    started = meta is None  # if no meta block, take everything after h1
    for el in main.children:
        if getattr(el, "name", None) is None:
            continue
        if not started:
            if el is meta:
                started = True
            continue
        if el.name in ("iframe", "script", "style", "form", "hr"):
            continue
        if el.name == "div":
            continue  # photo wrappers, ads
        if el.name in ("p", "blockquote", "h2", "h3", "ul", "ol", "pre"):
            if el.find("img") is not None:
                continue
            for br in el.find_all("br"):
                # insert_before + unwrap (NOT replace_with) so that content
                # mis-nested inside unclosed <br> tags survives
                br.insert_before("\n")
                br.unwrap()
            if el.name in ("ul", "ol"):
                text = "\n".join(li.get_text() for li in el.find_all("li"))
            else:
                text = el.get_text()
            text = text.strip("\n")
            if not text.strip():
                continue
            if NOTE_PATTERNS.search(text.strip()):
                continue
            paras.append(text)
    if not paras:
        return None
    text = "\n\n".join(paras)
    # Tombstone page ("We regret ... no longer available", "Poem not
    # displayed? Download the PDF"): try the PDF fallback, else unavailable.
    if TOMBSTONE_PATTERNS.search(text) and len(text) < 300:
        return extract_pdf_poem(entry_url, html)
    return text


def main(limit_cycles: list[str] | None = None, fetch_texts: bool = True):
    cycles = discover_cycles()
    if limit_cycles:
        cycles = [c for c in cycles if c[0] in limit_cycles]
    print(f"[wergle] {len(cycles)} cycles: {[c[0] for c in cycles]}")

    all_rows = []
    for year, url in cycles:
        rows = parse_cycle_page(year, url)
        print(f"[wergle] {year}: {len(rows)} entries")
        all_rows.extend(rows)

    for i, row in enumerate(all_rows):
        raw = extract_poem(row["url"]) if fetch_texts else None
        # title-echo pages (body is just the title repeated, poem missing,
        # e.g. 2011 'How to Write a Poem for a Journal') -> no usable text
        if raw is not None and raw.strip().lower() == (row["title"] or "").strip().lower():
            raw = None
        row["raw_text"] = raw
        row["text"] = normalize_text(raw)
        row["text_available"] = bool(raw)
        if fetch_texts and (i + 1) % 50 == 0:
            print(f"[wergle] fetched {i + 1}/{len(all_rows)} entry pages")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_text = sum(1 for r in all_rows if r["text_available"])
    print(f"[wergle] wrote {len(all_rows)} rows ({n_text} with text) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
