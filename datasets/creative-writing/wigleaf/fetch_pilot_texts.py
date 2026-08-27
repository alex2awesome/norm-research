#!/usr/bin/env python3
"""Pilot story-text retrieval for the Wigleaf Top 50 corpus.

Samples ~60 stories stratified across years (top50 tier only: longlist entries
are NOT hyperlinked on wigleaf.com, so there is no story URL to fetch), fetches
each story page (live, falling back to the Wayback Machine), extracts the main
story text with a largest-text-block heuristic, and writes
wigleaf_pilot_texts.jsonl.

Politeness: 2s/domain rate limit, academic UA, all fetches cached on disk.
"""
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
import unicodedata
from urllib.parse import urlparse, quote

import requests
from bs4 import BeautifulSoup

STAGING = "/Users/spangher/Projects/stanford-research/norm-research/.staging/norm-datasets/creative-writing/wigleaf"
STORY_CACHE = os.path.join(STAGING, "html_cache", "stories")
UA = "norm-research-academic-crawler/0.1 (alex2awesome@gmail.com; Stanford research)"
SLEEP_S = 2.0
N_PER_YEAR = 3
N_TOTAL = 60
SEED = 7

PARKED_TITLE_PAT = re.compile(
    r"domain (?:is )?for sale|buy this domain|parked free|"
    r"this domain may be for sale|domain expired|seized|account suspended", re.I)
PARKED_DOMAINS = {"hugedomains.com", "sedo.com", "sedoparking.com", "dan.com",
                  "afternic.com", "godaddy.com", "namecheap.com", "porkbun.com"}


def is_parked(html_str, final_url):
    dom = urlparse(final_url).netloc.lower().removeprefix("www.")
    if dom in PARKED_DOMAINS:
        return True
    m = re.search(r"<title[^>]*>(.*?)</title>", html_str[:20000], re.S | re.I)
    return bool(m and PARKED_TITLE_PAT.search(m.group(1)))

_last_fetch = {}


def polite_get(session, url, timeout=30):
    domain = urlparse(url).netloc
    el = time.monotonic() - _last_fetch.get(domain, 0)
    if el < SLEEP_S:
        time.sleep(SLEEP_S - el)
    try:
        r = session.get(url, headers={"User-Agent": UA}, timeout=timeout,
                        allow_redirects=True)
        _last_fetch[domain] = time.monotonic()
        return r, None
    except requests.RequestException as e:
        _last_fetch[domain] = time.monotonic()
        return None, f"{type(e).__name__}"


def cached_fetch(session, url, tag):
    """Returns dict(status, final_url, html, error). Caches 200 bodies."""
    key = hashlib.sha1((tag + url).encode()).hexdigest()[:16]
    cpath = os.path.join(STORY_CACHE, key + ".json")
    if os.path.exists(cpath):
        with open(cpath) as f:
            return json.load(f)
    r, err = polite_get(session, url)
    out = {"url": url}
    if r is None:
        out.update(status=None, final_url=None, html=None, error=err)
    else:
        enc = r.apparent_encoding if r.apparent_encoding else "utf-8"
        try:
            r.encoding = enc
        except Exception:
            pass
        out.update(status=r.status_code, final_url=r.url,
                   html=r.text if r.status_code == 200 else None, error=None)
    with open(cpath, "w") as f:
        json.dump(out, f)
    return out


def looks_like_homepage_redirect(orig_url, final_url):
    """Redirect to site root (or a different site's root) = dead link in disguise."""
    op, fp = urlparse(orig_url), urlparse(final_url)
    if len(op.path.strip("/")) > 1 and len(fp.path.strip("/")) <= 1 and not fp.query:
        return True
    return False


def extract_main_text(html_str):
    """Largest-text-block heuristic (trafilatura not installed)."""
    soup = BeautifulSoup(html_str, "html.parser")
    for sel in ["script", "style", "nav", "header", "footer", "form", "aside",
                "noscript", "iframe", "svg", "button", "figure"]:
        for t in soup.find_all(sel):
            t.decompose()
    # wayback toolbar
    for tid in ["wm-ipp-base", "wm-ipp", "donato"]:
        t = soup.find(id=tid)
        if t:
            t.decompose()
    total_len = len(soup.get_text(" ", strip=True)) or 1
    for t in soup.find_all(attrs={"class": re.compile(
            r"comment|sidebar|share|related|menu|nav-|footer|header|widget|breadcrumb", re.I)}):
        # never decompose structural containers: themes put e.g. "no-sidebar" /
        # "cb-sidebar-none" on <body> or main wrappers, which nuked whole pages
        if t.name in ("body", "html", "main", "article") or t.parent is None:
            continue
        if len(t.get_text(" ", strip=True)) > 0.6 * total_len:
            continue
        t.decompose()
    candidates = soup.find_all(["article", "main", "div", "td", "section", "body"])
    best, best_len = None, 0
    for c in candidates:
        paras = c.find_all(["p", "br"])
        txt = c.get_text(" ", strip=True)
        # prefer paragraph-rich blocks; penalize link-farms
        link_txt = sum(len(a.get_text()) for a in c.find_all("a"))
        score = len(txt) - 2 * link_txt + 50 * min(len(paras), 30)
        if score > best_len:
            best, best_len = c, score
    if best is None:
        return ""
    parts = []
    for block in best.find_all(["p", "h1", "h2", "blockquote", "li"]) or [best]:
        t = block.get_text(" ", strip=True)
        if t:
            parts.append(t)
    text = "\n\n".join(parts) if parts else best.get_text("\n", strip=True)
    return text.strip()


CURLY = {"‘": "'", "’": "'", "“": '"', "”": '"', "′": "'"}


def normalize_text(s):
    s = unicodedata.normalize("NFKC", s)
    for k, v in CURLY.items():
        s = s.replace(k, v)
    s = re.sub(r"<[^>]+>", " ", s)            # leftover markup
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    if len(s) > 2 and s[0] == '"' and s[-1] == '"' and s.count('"') == 2:
        s = s[1:-1].strip()                    # wrapper quotes
    return s


def wayback_lookup(session, url, year):
    api = ("http://archive.org/wayback/available?url=" + quote(url, safe="") +
           f"&timestamp={year}0601")
    res = cached_fetch(session, api, "wbavail")
    if not res.get("html"):
        return None
    try:
        snap = json.loads(res["html"])["archived_snapshots"].get("closest")
    except (json.JSONDecodeError, KeyError):
        return None
    if snap and snap.get("available"):
        return snap["url"].replace("http://web.archive.org", "https://web.archive.org")
    return None


def wayback_cdx_lookup(session, url, year):
    """Fallback: the availability API misses many snapshots the CDX index has."""
    api = ("https://web.archive.org/cdx/search/cdx?url=" + quote(url, safe="") +
           "&output=json&limit=5&filter=statuscode:200&collapse=digest")
    res = cached_fetch(session, api, "wbcdx")
    if not res.get("html"):
        return None
    try:
        rows = json.loads(res["html"])
    except json.JSONDecodeError:
        return None
    if len(rows) < 2:
        return None
    # prefer snapshot closest to the list year
    target = int(f"{year}0601000000")
    best = min(rows[1:], key=lambda r: abs(int(r[1]) - target))
    return f"https://web.archive.org/web/{best[1]}/{best[2]}"


def main():
    os.makedirs(STORY_CACHE, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(STAGING, "wigleaf_labels.csv"))))
    pool = [r for r in rows if r["tier"] == "top50" and r["story_url"]]
    random.seed(SEED)
    years = sorted({r["year"] for r in pool})
    sample = []
    for y in years:
        sample += random.sample([r for r in pool if r["year"] == y], N_PER_YEAR)
    extra_pool = [r for r in pool if r not in sample]
    sample += random.sample(extra_pool, N_TOTAL - len(sample))
    print(f"Pilot sample: {len(sample)} stories across {len(years)} years (top50 tier; "
          f"longlist has no story URLs on wigleaf.com)")

    session = requests.Session()
    out_path = os.path.join(STAGING, "wigleaf_pilot_texts.jsonl")
    results = []
    for i, r in enumerate(sample):
        url = r["story_url"]
        domain = urlparse(url).netloc.lower().removeprefix("www.")
        status, text = "dead", ""
        res = cached_fetch(session, url, "live")
        live_ok = (res.get("status") == 200 and res.get("html")
                   and not looks_like_homepage_redirect(url, res["final_url"])
                   and not is_parked(res["html"], res["final_url"]))
        if live_ok:
            text = extract_main_text(res["html"])
            if len(text) >= 300:
                status = "live"
        if status != "live":
            for wb in filter(None, [wayback_lookup(session, url, r["year"]),
                                    wayback_cdx_lookup(session, url, r["year"])]):
                wres = cached_fetch(session, wb, "wayback")
                if wres.get("status") == 200 and wres.get("html"):
                    wtext = extract_main_text(wres["html"])
                    if len(wtext) >= 300:
                        status, text = "wayback", wtext
                        break
        rec = {"year": int(r["year"]), "tier": r["tier"], "title": r["title"],
               "author": r["author"], "magazine": r["magazine"],
               "story_url": url, "fetch_status": status, "domain": domain,
               "raw_text": text, "text": normalize_text(text)}
        results.append(rec)
        print(f"[{i+1}/{len(sample)}] {status:8s} {domain:35s} {r['year']} {r['title'][:40]}")

    with open(out_path, "w") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    from collections import Counter
    print("\nStatus counts:", dict(Counter(x["fetch_status"] for x in results)))
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
