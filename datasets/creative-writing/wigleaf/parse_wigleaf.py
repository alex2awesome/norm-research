#!/usr/bin/env python3
"""Parse cached Wigleaf Top 50 + longlist HTML into wigleaf_labels.csv.

Top 50 page format (all years 2008-2025, paginated x3):
    AUTHOR&nbsp;NAME,&nbsp;&nbsp; "<a href=URL>Title</a>"  ...  from <span italic>Magazine</span>
Longlist format (single page per year; "...and the rest of the Top 200"):
    [N.] Lastname, Firstname, "Title," Magazine[,] (date-ish)     -- no story links
    2008-2010 wrap the magazine in <small>CAPS</small>; some years number entries.

Usage: parse_wigleaf.py [year ...]   (default: all years in manifest)
"""
import csv
import html as htmllib
import json
import os
import re
import sys
import unicodedata

STAGING = "/Users/spangher/Projects/stanford-research/norm-research/.staging/norm-datasets/creative-writing/wigleaf"
CACHE_DIR = os.path.join(STAGING, "html_cache")
BASE = "https://wigleaf.com/"

CURLY = {"‘": "'", "’": "'", "“": '"', "”": '"',
         "′": "'", "«": '"', "»": '"'}


def norm(s):
    """NFKC-normalize, straighten curly quotes, collapse whitespace."""
    s = unicodedata.normalize("NFKC", s)
    for k, v in CURLY.items():
        s = s.replace(k, v)
    return re.sub(r"\s+", " ", s).strip()


def strip_tags(s):
    return re.sub(r"<[^>]+>", " ", s)


ANCHOR_RE = re.compile(r'<a\s+href="(https?://[^"]+)"[^>]*>(.*?)</a>', re.S | re.I)
MAG_RE = re.compile(r'from\s*(?:&nbsp;|\s)*<(span|i)[^>]*>(.*?)</\1>', re.S | re.I)
AUTHOR_RE = re.compile(r'>([^<>]+?),(?:&nbsp;|\s)*"?\s*$', re.S)


def parse_top50_page(path, year):
    raw = open(path, encoding="utf-8", errors="replace").read()
    raw = re.sub(r"<script.*?</script>", " ", raw, flags=re.S | re.I)
    rows, problems = [], []
    for m in ANCHOR_RE.finditer(raw):
        url, title_html = m.group(1), m.group(2)
        if re.search(r"statcounter|^https?://(www\.)?wigleaf\.com/?$", url, re.I):
            continue
        title = norm(htmllib.unescape(strip_tags(title_html)))
        # author: text run immediately before the anchor, ending ', "'
        before = raw[max(0, m.start() - 300):m.start()]
        am = AUTHOR_RE.search(before)
        author = norm(htmllib.unescape(am.group(1))) if am else ""
        # magazine: 'from <span italic>Mag</span>' within the next 500 chars,
        # but not past the next story anchor
        after = raw[m.end():m.end() + 500]
        nxt = ANCHOR_RE.search(after)
        if nxt:
            after = after[:nxt.start()]
        gm = MAG_RE.search(after)
        magazine = norm(htmllib.unescape(strip_tags(gm.group(2)))) if gm else ""
        if not author or not magazine:
            problems.append((year, os.path.basename(path), title, "missing author/magazine"))
        rows.append({"year": year, "tier": "top50", "title": title, "author": author,
                     "magazine": magazine, "story_url": url,
                     "list_page_url": BASE + os.path.basename(path)})
    return rows, problems


LL_HEAD_RE = re.compile(r'^(?P<author>[^"]+?),?\s*"(?P<title>[^"]+?)[,.]?\s*"\s*(?P<rest>.*)$')
LL_PAREN_DATE_RE = re.compile(r'^(?P<mag>.*?)\s*[,.]?\s*\((?P<date>[^()]+)\)\s*\*?\s*\w{0,3}\s*\.?$')

_MONTH = r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
_SEASON = r'(?:spring|summer|fall|autumn|winter)'
DATEY_RE = re.compile(
    r'^(?:\d{4}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4}'
    rf'|{_MONTH}(?:/{_MONTH})?\.?(?:\s+\d{{1,2}})?(?:\s*,?\s*\d{{4}})?'
    rf'|{_SEASON}(?:\s*[/-]\s*{_SEASON})?\s*,?\s*\d{{0,4}}'
    rf'|{_SEASON}\s*[\w /]*?\d{{4}}(?:\s*/\s*\d{{2,4}})?'
    r'|(?:vol(?:ume)?|issue|no|number)\.?\s*\S{1,8}|\d{1,2}\s+\d{4}|undated)\s*\*?$',
    re.I)

FOOTER_PAT = re.compile(r'w i g|statcounter|\[home\]|^PAGE\b|top ?50|^[A-Z](/[A-Z])*$|'
                        r'^\d{2}-\d{2}-\d{2}$|wigleaf|line-height|^Others from', re.I)


def invert_name(author):
    """'Last, First[, rest]' -> 'First Last[, rest]'."""
    parts = [p.strip() for p in author.split(",")]
    if len(parts) >= 2 and parts[0] and parts[1]:
        out = f"{parts[1]} {parts[0]}"
        if len(parts) > 2:
            out += ", " + ", ".join(parts[2:])
        return out
    return author


def extract_magazine(rest):
    """rest = everything after the closing title quote. Returns (magazine, note)."""
    rest = rest.strip().lstrip(",").strip()
    rest = re.sub(r"https?://\S+\s*$", "", rest).strip()  # stray trailing URL (2014 Gutting)
    # drop a leading parenthetical title-annotation like '(from Winesburg Appendix),'
    rest = re.sub(r"^\([^()]*\),\s*", "", rest)
    if rest.count("(") == rest.count(")") + 1:  # missing close paren typo
        rest += ")"
    m = LL_PAREN_DATE_RE.match(rest)
    if m and m.group("mag").strip():
        return m.group("mag").strip().rstrip(",."), ""
    # no parenthesized date (2018-style: 'Magazine , January 25, 2017'):
    # pop date-ish comma-chunks off the end
    parts = [p.strip() for p in rest.split(",") if p.strip()]
    popped = False
    while len(parts) > 1 and DATEY_RE.match(parts[-1]):
        parts.pop()
        popped = True
    mag = ", ".join(parts).strip().rstrip(",.")
    return mag, ("" if popped or m else "no-date-found")


# Source typos where the title has NO quotation marks at all; comma ambiguity
# makes these unparseable generically. Verified by hand against the cached HTML.
MANUAL_FIXES = {
    "McKimm, Dafydd, Peach Child": ("McKimm, Dafydd", "Peach Child, Woman, Stone", "Flash Fiction Online"),
    "McLeod, Lindz, Upper Bout": ("McLeod, Lindz", "Upper Bout", "Flash Fiction Online"),
    "Scheina, Carol, The Hundred Hidden Kisses": ("Scheina, Carol", "The Hundred Hidden Kisses", "Flash Fiction Online"),
    "Trahan, Leslie Walker, Heirloom": ("Trahan, Leslie Walker", "Heirloom", "Cease, Cows"),
}


def parse_longlist_page(path, year):
    raw = open(path, encoding="utf-8", errors="replace").read()
    raw = re.sub(r"<(script|style).*?</\1>", " ", raw, flags=re.S | re.I)
    # collapse source newlines FIRST so a single mid-entry <br> can't form a
    # fake blank line; then <p>/<br><br> become entry separators
    raw = raw.replace("\r", " ").replace("\n", " ")
    raw = re.sub(r"</?p[^>]*>", "\n\n", raw, flags=re.I)
    raw = re.sub(r"<br[^>]*>", "\n", raw, flags=re.I)
    text = htmllib.unescape(strip_tags(raw))
    for k, v in CURLY.items():
        text = text.replace(k, v)
    chunks = [re.sub(r"\s+", " ", c).strip() for c in re.split(r"\n\s*\n+", text)]
    # adaptive: some years (2018) separate entries with a SINGLE <br>; if the
    # blank-line split yields almost no entry-like chunks, split on single lines
    if sum(1 for c in chunks if c.count('"') >= 2) < 25:
        chunks = [re.sub(r"\s+", " ", c).strip() for c in text.split("\n")]
    rows, problems = [], []
    for c in chunks:
        c = c.lstrip("-– ").strip()
        fix = next((v for k, v in MANUAL_FIXES.items() if c.startswith(k)), None)
        if fix:
            rows.append({"year": year, "tier": "longlist", "title": fix[1],
                         "author": invert_name(fix[0]), "magazine": fix[2],
                         "story_url": "",
                         "list_page_url": BASE + os.path.basename(path)})
            continue
        if not c or len(c) < 15 or '"' not in c or c.startswith("*"):
            if c and not FOOTER_PAT.search(c) and len(c) > 3:
                problems.append((year, os.path.basename(path), c[:80], "skipped chunk"))
            continue
        c0 = re.sub(r"^\d+\.?\s*", "", c)
        note = ""
        m = LL_HEAD_RE.match(c0)
        if m:
            author = norm(m.group("author"))
            title = norm(m.group("title")).rstrip(",.;: ")
            magazine, note = extract_magazine(m.group("rest"))
        elif c0.count('"') == 1:
            # malformed: a missing opening OR closing quote
            pre, post = c0.split('"', 1)
            post = re.sub(r"\([^()]*\)?\.?\s*$", "", post).strip()  # drop (date)
            post = post.rstrip(",. ").strip()
            if "," not in post.rstrip(",. "):
                # missing OPENING quote: 'Last, First, Title," Magazine (date)'
                bits = [b.strip() for b in pre.split(",") if b.strip()]
                author = ", ".join(bits[:2])
                title = norm(", ".join(bits[2:])).strip("',. ")
                magazine = norm(post).strip("',. ")
            else:
                # missing CLOSING quote: 'Last, First, "Title, Mag, (date)'
                author = norm(pre).rstrip(",")
                bits = [b for b in post.rsplit(",", 1) if b.strip()]
                title = norm(bits[0]).strip("',. ")
                magazine = norm(bits[1]).strip("',. ") if len(bits) > 1 else ""
            note = "fallback parse"
        else:
            problems.append((year, os.path.basename(path), c[:80], "unparsed"))
            continue
        if note:
            problems.append((year, os.path.basename(path), c[:80], note))
        magazine = re.sub(r"\(\s+", "(", norm(magazine))
        rows.append({"year": year, "tier": "longlist", "title": title,
                     "author": invert_name(author),
                     "magazine": magazine.title() if magazine.isupper() else magazine,
                     "story_url": "",
                     "list_page_url": BASE + os.path.basename(path)})
    return rows, problems


def main():
    with open(os.path.join(STAGING, "manifest.json")) as f:
        manifest = json.load(f)
    years = sys.argv[1:] or sorted(manifest)
    all_rows, all_problems = [], []
    for year in years:
        info = manifest[str(year)]
        n50 = nll = 0
        for page in info.get("top50_pages", []):
            rows, probs = parse_top50_page(os.path.join(CACHE_DIR, page), int(year))
            all_rows += rows
            all_problems += probs
            n50 += len(rows)
        for page in info.get("longlist_pages", []):
            rows, probs = parse_longlist_page(os.path.join(CACHE_DIR, page), int(year))
            all_rows += rows
            all_problems += probs
            nll += len(rows)
        flags = []
        if n50 != 50:
            flags.append(f"top50 != 50 ({n50})")
        if not info.get("longlist_pages"):
            flags.append("NO LONGLIST PAGE")
        print(f"{year}: top50={n50} longlist={nll} {' '.join(flags)}")

    out = os.path.join(STAGING, "wigleaf_labels.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["year", "tier", "title", "author",
                                          "magazine", "story_url", "list_page_url"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n{len(all_rows)} rows -> {out}")
    if all_problems:
        plog = os.path.join(STAGING, "parse_problems.log")
        with open(plog, "w") as f:
            for p in all_problems:
                f.write("\t".join(str(x) for x in p) + "\n")
        print(f"{len(all_problems)} problems -> {plog}")


if __name__ == "__main__":
    main()
