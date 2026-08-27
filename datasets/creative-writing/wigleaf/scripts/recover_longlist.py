#!/usr/bin/env python3
"""Recover longlist story texts (the hard part: no stored URLs).

Strategy per (title, author, magazine, year):
  1. resolve the magazine's domain(s): from the Top50 known-URL map first, then
     a curated alias dictionary for high-count uncovered magazines.
  2. query the Wayback CDX index for ALL archived URLs on that domain, filter to
     200-status text pages, and slug-match the title against the URL path.
     (Most lit-mag CMSes put the title slug in the path, e.g.
     smokelong.com/.../the-off-season/ ; wordpress /2017/05/the-off-season/.)
  3. also try a DuckDuckGo HTML search ("title" author magazine) -> first result
     URL on the magazine domain.
  4. fetch the chosen URL (live, else its Wayback snapshot), run the SAME
     wig_textproc full_pipeline, and accept if >=MIN_TEXT chars and the title or
     a distinctive title-bigram appears OR the author surname appears (loose
     content check to reject wrong-page matches).

Recovery is expected to be PARTIAL; we report the honest rate. Everything is
cached on disk so re-runs resume. Politeness: <=1 req/sec per domain (archive.org
and the live magazine domains), academic UA, backoff on 429/503.

Output: longlist_texts.jsonl (status in {wayback,live,not_found}).
"""
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from urllib.parse import urlparse, quote, unquote

import requests
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wig_textproc import full_pipeline, looks_like_junk_page, normalize_magazine

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/creative-writing/wigleaf"
CACHE = os.path.join(BASE, "text_cache", "longlist")
CDX_CACHE = os.path.join(BASE, "text_cache", "cdx")
UA = ("norm-research-academic-crawler/0.2 "
      "(alex2awesome@gmail.com; Stanford research; respects robots, low-rate)")
SLEEP_S = 1.05
MIN_TEXT = 300

# curated magazine -> domain aliases for high-count magazines NOT in the Top50 map
CURATED = {
    "x-r-a-y literary magazine": "xraylitmag.com",
    "citron review": "citronreview.com",
    "milk candy": "milkcandyreview.com",
    "milk candy review": "milkcandyreview.com",
    "reckon review": "reckonreview.com",
    "pank magazine": "pankmagazine.com",
    "forge literary": "forgelitmag.com",
    "the forge literary magazine": "forgelitmag.com",
    "cafe irreal": "cafeirreal.com",
    "café irreal": "cafeirreal.com",
    "the cafe irreal": "cafeirreal.com",
    "the café irreal": "cafeirreal.com",
    "new world writing quarterly": "newworldwriting.net",
    "the mississippi review": "sites.google.com",
    "blip magazine": "blipmagazine.net",
    "the rupture": "therupturemag.com",
    "emerge literary journal": "emergeliteraryjournal.com",
    "knee jerk": "kneejerkmag.com",
    "fringe": "fringemagazine.org",
    "janus literary": "janusliterary.com",
    "molotov cocktail": "themolotovcocktail.com",
    "the molotov cocktail": "themolotovcocktail.com",
    "quarterly west": "quarterlywest.com",
    "noo journal": "noojournal.com",
    "heavy feather review": "heavyfeatherreview.org",
    "newfound": "newfound.org",
    "sunlight press": "thesunlightpress.com",
    "the sunlight press": "thesunlightpress.com",
    "los angeles review": "losangelesreview.org",
    "the los angeles review": "losangelesreview.org",
    "midway journal": "midwayjournal.com",
    "treehouse": "treehousemag.com",
    "visual verse": "visualverse.org",
    "moria": "moriaonline.com",
    "invisible city": "invisiblecitylit.com",
    "lightspeed": "lightspeedmagazine.com",
    "ligeia": "ligeiamagazine.com",
    "reflex press": "reflex.press",
    "reflex fiction": "reflex.press",
    "the phare": "thephare.com",
    "bull: men's fiction": "bullmensfiction.com",
    "bull": "bullmensfiction.com",
    "six little things": "sixlittlethings.com",
    "trampset": "trampset.org",
    "the offing": "theoffingmag.com",
    "wigleaf": "wigleaf.com",
    # no-domain tail additions (recognizable, multi-row magazines)
    "american literary review": "americanliteraryreview.com",
    "leon literary review": "leonliteraryreview.com",
    "stone of madness press": "stoneofmadnesspress.com",
    "verbsap": "verbsap.com",
    "flashquake": "flashquake.org",
    "eyeshot": "eyeshot.net",
    "the fiddleback": "thefiddleback.com",
    "tinhouse blog": "tinhouse.com",
    "tin house blog": "tinhouse.com",
    "hobart pulp": "hobartpulp.com",
    "hinchas de poesia": "hinchasdepoesia.com",
    "gingerbread house": "gingerbreadhouselitmag.com",
    "schuylkill valley journal": "svjlit.com",
    "untoward": "untowardmag.com",
    "banango street": "banangostreet.com",
    "staccato fiction": "staccatofiction.com",
    "cerise press": "cerisepress.com",
    "the fanzine": "thefanzine.com",
    "fanzine": "thefanzine.com",
    "wyvern lit": "wyvernlit.com",
    "barrelhouse": "barrelhousemag.com",
    "the boiler": "theboilerjournal.com",
    "the boiler journal": "theboilerjournal.com",
    "wherewithal": "wherewithallit.com",
    "ghost parachute": "ghostparachute.com",
    "the journal of compressed creative arts": "matterpress.com",
    "100 word story": "100wordstory.org",
    "no contact": "nocontactmag.com",
    "longleaf review": "longleafreview.com",
    "fractured lit": "fracturedlit.com",
    "bending genres": "bendinggenres.com",
    "cheap pop": "cheappoplit.com",
    "cleaver magazine": "cleavermagazine.com",
    "flash frog": "flashfrog.net",
    "the forge": "forgelitmag.com",
}


def build_mag2dom():
    df = pd.read_csv(os.path.join(BASE, "wigleaf_labels_fixed.csv"))
    t50 = df[df.tier == "top50"]
    mag2dom = defaultdict(Counter)
    for _, r in t50.iterrows():
        if isinstance(r.story_url, str) and r.story_url:
            dom = urlparse(r.story_url).netloc.lower().removeprefix("www.")
            mag2dom[str(r.magazine).strip().lower()][dom] += 1
    out = {}
    for m, c in mag2dom.items():
        out[m] = c.most_common(1)[0][0]
    # alias normalization: drop leading "the ", trailing " magazine/quarterly/review"
    def alias(m):
        m = m.lower().strip()
        m2 = re.sub(r"^the\s+", "", m)
        m2 = re.sub(r"\s+(magazine|quarterly|journal)$", "", m2)
        return m2
    alias_map = {}
    for m, d in out.items():
        alias_map[alias(m)] = d
    for m, d in CURATED.items():
        out[m] = d
        alias_map[alias(m)] = d
    return out, alias_map, alias


def domain_for(mag, mag2dom, alias_map, alias_fn):
    m = str(mag).strip().lower()
    if m in mag2dom:
        return mag2dom[m]
    a = alias_fn(m)
    if a in alias_map:
        return alias_map[a]
    return None


_last_fetch = {}


def polite_get(session, url, timeout=30):
    domain = urlparse(url).netloc
    el = time.monotonic() - _last_fetch.get(domain, 0)
    if el < SLEEP_S:
        time.sleep(SLEEP_S - el)
    for attempt in range(3):
        try:
            r = session.get(url, headers={"User-Agent": UA}, timeout=timeout,
                            allow_redirects=True)
            _last_fetch[domain] = time.monotonic()
            if r.status_code in (429, 503):
                time.sleep(2.0 * (attempt + 1))
                continue
            return r, None
        except requests.RequestException as e:
            _last_fetch[domain] = time.monotonic()
            if attempt == 2:
                return None, type(e).__name__
            time.sleep(1.0 * (attempt + 1))
    return None, "retries"


def cached_fetch(session, url, tag, cache_dir=CACHE):
    key = hashlib.sha1((tag + url).encode()).hexdigest()[:16]
    cpath = os.path.join(cache_dir, key + ".json")
    if os.path.exists(cpath):
        try:
            with open(cpath) as f:
                return json.load(f)
        except Exception:
            pass
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


def slugify(s):
    s = s.lower()
    s = re.sub(r"[''`]", "", s)
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s


STOP = {"the", "a", "an", "of", "to", "and", "in", "on", "for", "with", "is",
        "at", "by", "my", "i", "you", "it", "as", "or", "be"}


def title_tokens(title):
    toks = [t for t in re.split(r"[^a-z0-9]+", title.lower()) if len(t) > 2 and t not in STOP]
    return toks


def cdx_urls(session, domain, year):
    """All archived 200-status text/html URLs on a domain (cached per domain)."""
    key = hashlib.sha1(("cdx2" + domain).encode()).hexdigest()[:16]
    cpath = os.path.join(CDX_CACHE, key + ".json")
    if os.path.exists(cpath):
        try:
            return json.load(open(cpath))
        except Exception:
            pass
    api = ("https://web.archive.org/cdx/search/cdx?url=" + quote(domain, safe="") +
           "*&output=json&collapse=urlkey&fl=timestamp,original,statuscode,mimetype"
           "&filter=statuscode:200&filter=mimetype:text/html&limit=20000")
    res = cached_fetch(session, api, "cdxdom", CDX_CACHE)
    rows = []
    if res.get("html"):
        try:
            data = json.loads(res["html"])
            rows = data[1:] if len(data) > 1 else []
        except json.JSONDecodeError:
            rows = []
    json.dump(rows, open(cpath, "w"))
    return rows


def match_in_cdx(rows, title, year):
    """Find the archived URL whose path slug best matches the story title."""
    tslug = slugify(title)
    ttoks = set(title_tokens(title))
    if not ttoks:
        return None
    best, best_score = None, 0.0
    for ts, orig, sc, mime in rows:
        path = urlparse(orig).path.lower()
        pslug = re.sub(r"[^a-z0-9]+", "-", unquote(path)).strip("-")
        # exact slug substring = strong
        score = 0.0
        if tslug and tslug in pslug:
            score = 1.0
        else:
            ptoks = set(t for t in re.split(r"-+", pslug) if len(t) > 2)
            inter = ttoks & ptoks
            if ttoks:
                score = len(inter) / len(ttoks)
        if score > best_score or (score == best_score and best and
                                  abs(int(ts[:4]) - year) < abs(int(best[0][:4]) - year)):
            best, best_score = (ts, orig), score
    # require a solid match
    if best and best_score >= 0.6:
        ts, orig = best
        return f"https://web.archive.org/web/{ts}/{orig}", best_score
    return None


def content_ok(text, title, author):
    """Loose guard against wrong-page matches: title token OR author surname
    must appear in the recovered text."""
    tl = text.lower()
    ttoks = title_tokens(title)
    hit = sum(1 for t in ttoks if t in tl)
    if ttoks and hit >= max(1, len(ttoks) // 2):
        return True
    surname = str(author).strip().split()[-1].lower() if str(author).strip() else ""
    if len(surname) > 2 and surname in tl:
        return True
    # if title is very short (1-2 tokens), accept on any token hit
    if len(ttoks) <= 2 and hit >= 1:
        return True
    return False


def recover_one(session, title, author, magazine, year, domain):
    """-> (status, url, raw, clean) ; status in {wayback,live,not_found}."""
    if not domain:
        return "not_found", "", "", ""
    rows = cdx_urls(session, domain, year)
    m = match_in_cdx(rows, title, year)
    if m:
        wb_url, score = m
        res = cached_fetch(session, wb_url, "ll_wb")
        if res.get("status") == 200 and res.get("html"):
            raw, clean = full_pipeline(res["html"])
            if (len(clean) >= MIN_TEXT and not looks_like_junk_page(clean)
                    and content_ok(clean, title, author)):
                return "wayback", wb_url, raw, clean
    return "not_found", "", "", ""


def main():
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(CDX_CACHE, exist_ok=True)
    mag2dom, alias_map, alias_fn = build_mag2dom()
    df = pd.read_csv(os.path.join(BASE, "wigleaf_labels_fixed.csv"))
    ll = df[df.tier == "longlist"].copy()
    print(f"Longlist to recover: {len(ll)}", flush=True)
    session = requests.Session()
    out_path = os.path.join(BASE, "longlist_texts.jsonl")
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                d = json.loads(l)
                done.add((d["title"], d["author"], int(d["year"])))
            except Exception:
                pass
    fout = open(out_path, "a")
    counts = Counter()
    for i, (_, r) in enumerate(ll.iterrows()):
        key = (r["title"], r["author"], int(r["year"]))
        if key in done:
            continue
        mag = normalize_magazine(r["magazine"])
        domain = domain_for(r["magazine"], mag2dom, alias_map, alias_fn)
        status, url, raw, clean = recover_one(
            session, str(r["title"]), str(r["author"]), str(r["magazine"]),
            int(r["year"]), domain)
        counts[status] += 1
        rec = {"year": int(r["year"]), "tier": "longlist", "title": r["title"],
               "author": r["author"], "magazine": mag,
               "story_url": url, "fetch_status": status, "fetch_source": status,
               "domain": domain or "", "raw_text": raw, "text": clean}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        if (i + 1) % 25 == 0:
            print(f"[{i+1}/{len(ll)}] {dict(counts)}", flush=True)
    fout.close()
    print("DONE longlist:", dict(counts), flush=True)


if __name__ == "__main__":
    main()
