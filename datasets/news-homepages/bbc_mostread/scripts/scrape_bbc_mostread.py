#!/usr/bin/env python3
"""
Scrape BBC "Most Read" lists from the Wayback Machine (news-C cell: lay crowd,
revealed attention).

Two harvest channels, both server-rendered:

  CHANNEL A  the dedicated Most-Read page  /news/popular/read  (and .fragment)
             markup: ul.most-popular-page__list > li.most-popular-page-list-item
                     > a.most-popular-page-list-item__link  (rank span inside)
             era ~2014-2017.  Gives clean ranked top-10.

  CHANNEL B  the homepage Most-Read module on bbc.com/news & bbc.co.uk/news
             - Morph era (~2017-2023): ol.nw-c-most-read__items > li
                 > a.nw-c-most-read__link  (+ span.nw-c-most-read__rank)
             - React era (~2024-2025): section[data-analytics_group_name="Most read"]
                 with data-testid="cambridge-card" links; rank = DOM order.

For each parsed capture we ALSO harvest the same-page non-most-read article links
(homepage other-headlines for channel B; for channel A the dedicated page only
contains the list, so its controls come from the matched homepage capture nearest
in time on the same day -- handled in the builder, not here).  Here we just emit,
per capture, the most-read list AND every other /news article link + anchor text.

Politeness: <=1 req/sec, retry/backoff on 429/5xx, real UA w/ contact email.
Resumable: appends JSONL; a checkpoint set of done (timestamp,original) keys is
rebuilt from the existing JSONL on startup so re-runs skip finished captures.

Usage:
  python scrape_bbc_mostread.py --out raw/captures.jsonl \
      [--max-captures N] [--from 20140101 --to 20251231] [--collapse timestamp:8]
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.parse
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36 "
      "(research crawler; contact alex2awesome@gmail.com)")
S = requests.Session(); S.headers.update({"User-Agent": UA})
MIN_INTERVAL = 1.1
_last = [0.0]


def throttle():
    dt = time.time() - _last[0]
    if dt < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - dt)
    _last[0] = time.time()


def get(url, max_retries=6, timeout=60):
    for a in range(max_retries):
        throttle()
        try:
            r = S.get(url, timeout=timeout)
        except requests.RequestException as e:
            wait = min(60, 2 ** a)
            print(f"  [retry {a}] {type(e).__name__}; sleep {wait}", file=sys.stderr)
            time.sleep(wait); continue
        if r.status_code in (429, 500, 502, 503, 504):
            wait = min(90, 2 ** a * 3)
            print(f"  [retry {a}] HTTP {r.status_code}; sleep {wait}", file=sys.stderr)
            time.sleep(wait); continue
        return r
    return None


def cdx(url, frm, to, collapse, limit=None):
    base = "http://web.archive.org/cdx/search/cdx"
    p = {"url": url, "output": "json", "from": frm, "to": to,
         "collapse": collapse,
         "fl": "timestamp,original,statuscode,digest,mimetype"}
    if limit:
        p["limit"] = str(limit)
    r = get(base + "?" + urllib.parse.urlencode(p))
    if not r or r.status_code != 200:
        print(f"CDX fail {url}: {r.status_code if r else 'none'}", file=sys.stderr)
        return []
    try:
        d = r.json()
    except Exception:
        return []
    if not d:
        return []
    return [dict(zip(d[0], row)) for row in d[1:]]


def fetch_capture(ts, orig):
    r = get(f"http://web.archive.org/web/{ts}id_/{orig}")
    return r.text if r else None


def _txt(node):
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()


# ---------------------------------------------------------------------------
# PARSERS  -> return list of {rank, headline, href}
# ---------------------------------------------------------------------------
def parse_popular_page(soup):
    """Channel A: dedicated /news/popular/read page (and .fragment)."""
    out = []
    lst = soup.select_one("ul.most-popular-page__list")
    if lst is None:
        # fragment may put items in #comp-most-popular-page
        cont = soup.select_one("#comp-most-popular-page")
        if cont:
            lst = cont.find("ul")
    if lst is None:
        return out
    for li in lst.find_all("li", recursive=False) or lst.find_all("li"):
        a = li.find("a")
        if not a:
            continue
        rank_span = li.select_one(".most-popular-page-list-item__rank")
        rank = None
        if rank_span:
            m = re.search(r"\d+", rank_span.get_text())
            rank = int(m.group()) if m else None
        # headline text = link text minus the rank number
        full = _txt(a)
        if rank_span:
            rt = rank_span.get_text(strip=True)
            full = full[len(rt):].strip() if full.startswith(rt) else full
        full = re.sub(r"^\d{1,2}\s+", "", full)
        href = a.get("href", "")
        if full and len(full) > 6:
            out.append({"rank": rank if rank else len(out) + 1,
                        "headline": full, "href": href})
    return out


def parse_morph_most_read(soup):
    """Channel B Morph era: ol.nw-c-most-read__items."""
    out = []
    ol = soup.select_one("ol.nw-c-most-read__items")
    if ol is None:
        # sometimes the items live directly under .nw-c-most-read
        cont = soup.select_one(".nw-c-most-read")
        if cont:
            ol = cont.find("ol")
    if ol is None:
        return out
    for li in ol.find_all("li"):
        a = li.find("a", class_=re.compile("nw-c-most-read__link")) or li.find("a")
        if not a:
            continue
        rank_span = li.select_one(".nw-c-most-read__rank")
        rank = None
        if rank_span:
            m = re.search(r"\d+", rank_span.get_text())
            rank = int(m.group()) if m else None
        # link text often = "<rank> Headline" or has a hidden rank span; strip it
        head = a.find(class_=re.compile("nw-c-most-read__title")) or a
        full = _txt(head)
        if rank_span:
            rt = rank_span.get_text(strip=True)
            full = full[len(rt):].strip() if full.startswith(rt) else full
        full = re.sub(r"^\d{1,2}\s+", "", full)
        href = a.get("href", "")
        if full and len(full) > 6:
            out.append({"rank": rank if rank else len(out) + 1,
                        "headline": full, "href": href})
    return out


def _react_card_headline(card):
    """Extract the clean headline + href + rank from a React 'cambridge-card'.
    Uses data-testid='card-headline' (an <h2>, headline ONLY) and 'card-order'
    (the rank span); avoids the whole-card text which carries timestamp/section
    metadata suffixes (a class-leak risk)."""
    head_el = card.find(attrs={"data-testid": "card-headline"})
    if head_el is None:
        return None
    headline = _txt(head_el)
    a = card.find("a", attrs={"data-testid": "internal-link"}) or card.find("a")
    href = a.get("href", "") if a else ""
    rank = None
    ro = card.find(attrs={"data-testid": "card-order"})
    if ro:
        m = re.search(r"\d+", ro.get_text())
        rank = int(m.group()) if m else None
    # Strip a leading rank number ONLY if it equals this card's rank -- a blind
    # ^\d+ strip would eat legitimate leading numbers like "50 years ago: ...".
    if rank is not None:
        headline = re.sub(rf"^0*{rank}[.\s]+", "", headline).strip()
    if headline and len(headline) > 6:
        return {"rank": rank, "headline": headline, "href": href}
    return None


def parse_react_most_read(soup):
    """Channel B React era (2024-2025): section data-analytics_group_name='Most read'."""
    out = []
    sec = soup.find("section", attrs={"data-analytics_group_name": "Most read"})
    if sec is None:
        sec = soup.find("section", attrs={"data-analytics_group_name": re.compile(
            r"most read", re.I)})
    if sec is None:
        return out
    cards = sec.find_all(attrs={"data-testid": "cambridge-card"})
    seen = set()
    for c in cards:
        rec = _react_card_headline(c)
        if rec is None or rec["href"] in seen:
            continue
        seen.add(rec["href"])
        if rec["rank"] is None:
            rec["rank"] = len(out) + 1
        out.append(rec)
    return out


def parse_most_read(html):
    """Try all parsers; return (channel, items)."""
    soup = BeautifulSoup(html, "html.parser")
    for name, fn in (("popular_page", parse_popular_page),
                     ("morph", parse_morph_most_read),
                     ("react", parse_react_most_read)):
        try:
            items = fn(soup)
        except Exception as e:
            print(f"  parser {name} err: {e}", file=sys.stderr)
            items = []
        if len(items) >= 3:
            return name, items, soup
    return None, [], soup


# ---------------------------------------------------------------------------
# control headlines from the same homepage capture (channel B only)
# ---------------------------------------------------------------------------
ARTICLE_HREF_RE = re.compile(r"/news/(?:[a-z-]+-)?\d{6,}|/news/articles/[a-z0-9]+", re.I)


def _norm_href(href):
    href = re.sub(r"^https?://web\.archive\.org/web/\d+(?:id_)?/", "", href)
    href = re.sub(r"^https?://(www\.)?bbc\.(co\.uk|com)", "", href)
    return href


def harvest_other_headlines(soup, most_read_hrefs):
    """Every other article link on the homepage with a CLEAN headline.

    React era (any cambridge-card present): harvest ONLY from
    data-testid='card-headline' (<h2>, headline only). Generic anchor text in the
    React homepage bundles headline + dek + 'N hrs ago · Section' metadata, which
    would leak the class vs the clean most-read text -- so we suppress the generic
    fallback whenever the page is React-era.
    Morph/older eras (no cambridge-card): use anchor text (clean there).
    """
    out = {}
    mr = set(_norm_href(h).split("?")[0].rstrip("/") for h in most_read_hrefs)
    react_cards = soup.find_all(attrs={"data-testid": "cambridge-card"})
    is_react = len(react_cards) > 0

    for c in react_cards:
        rec = _react_card_headline(c)
        if not rec:
            continue
        href = _norm_href(rec["href"])
        key = href.split("?")[0].rstrip("/")
        if not ARTICLE_HREF_RE.search(href) or key in mr:
            continue
        txt = rec["headline"]
        if len(txt) < 12 or len(txt.split()) < 3:
            continue
        if key not in out or len(txt) > len(out[key]):
            out[key] = txt

    if not is_react:
        # Morph and older eras: anchor text is the clean headline.
        for a in soup.find_all("a", href=True):
            href = _norm_href(a["href"])
            key = href.split("?")[0].rstrip("/")
            if not ARTICLE_HREF_RE.search(href) or key in mr or key in out:
                continue
            txt = _txt(a)
            if len(txt) < 12 or len(txt.split()) < 3:
                continue
            if txt.lower() in ("read more", "more", "full article", "watch", "listen"):
                continue
            out[key] = txt
    return [{"href": k, "headline": v} for k, v in out.items()]


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done.add((d["timestamp"], d["original"]))
                except Exception:
                    continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--from", dest="frm", default="20140101")
    ap.add_argument("--to", default="20251231")
    ap.add_argument("--collapse", default="timestamp:8")  # daily
    ap.add_argument("--max-captures", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = load_done(args.out)
    print(f"resume: {len(done)} captures already in {args.out}", file=sys.stderr)

    # Build the capture worklist.
    # Channel A: dedicated most-read page (clean), all domain forms.
    # Channel B: homepage (other-headline controls live here).
    url_forms = [
        ("A", "bbc.com/news/popular/read"),
        ("A", "www.bbc.com/news/popular/read"),
        ("A", "bbc.co.uk/news/popular/read"),
        ("A", "www.bbc.co.uk/news/popular/read"),
        ("A", "bbc.co.uk/news/popular/read.fragment"),
        ("B", "bbc.com/news"),
        ("B", "www.bbc.com/news"),
        ("B", "bbc.co.uk/news"),
        ("B", "www.bbc.co.uk/news"),
    ]
    worklist = []
    seen_keys = set()
    for channel, form in url_forms:
        rows = cdx(form, args.frm, args.to, args.collapse)
        rows = [r for r in rows
                if r.get("statuscode") == "200" and "html" in r.get("mimetype", "")]
        print(f"CDX {form}: {len(rows)} captures (200/html)", file=sys.stderr)
        for r in rows:
            k = (r["timestamp"], r["original"])
            if k in seen_keys:
                continue
            seen_keys.add(k)
            r["_channel"] = channel
            worklist.append(r)
    worklist.sort(key=lambda r: r["timestamp"])
    if args.max_captures:
        worklist = worklist[:args.max_captures]
    print(f"worklist: {len(worklist)} captures", file=sys.stderr)

    n_ok = n_empty = n_fail = 0
    with open(args.out, "a") as fout:
        for i, r in enumerate(worklist):
            ts, orig, channel = r["timestamp"], r["original"], r["_channel"]
            if (ts, orig) in done:
                continue
            html = fetch_capture(ts, orig)
            if not html:
                n_fail += 1
                continue
            parser, items, soup = parse_most_read(html)
            rec = {"timestamp": ts, "original": orig, "channel": channel,
                   "parser": parser, "n_mostread": len(items),
                   "most_read": items}
            # Always harvest other-headlines on channel-B homepage captures, even
            # when the most-read module did not parse (2014-2017 era = AJAX-only
            # most-read but the homepage still lists many other articles we use as
            # controls for that day's Channel-A most-read list).
            if channel == "B":
                mr_hrefs = [it["href"] for it in items]
                rec["others"] = harvest_other_headlines(soup, mr_hrefs)
            else:
                rec["others"] = []
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if items:
                n_ok += 1
            else:
                n_empty += 1
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(worklist)}] ok={n_ok} empty={n_empty} "
                      f"fail={n_fail} last={ts} parser={parser} n={len(items)}",
                      file=sys.stderr)
    print(f"DONE: ok={n_ok} empty={n_empty} fail={n_fail}", file=sys.stderr)


if __name__ == "__main__":
    main()
