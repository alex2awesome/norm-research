#!/usr/bin/env python3
"""Semantic Scholar (S2) title-match join for the acad-C citation dataset.

Why S2: OpenAlex's free tier is now $1/day = 10,000 credits PER IP, and each
title.search costs 10 credits, which throttled the OpenAlex build (NeurIPS 2023
never finished). S2's match endpoint has no comparable budget wall (~1 req/sec
unauthenticated).

What this does: for each (key, title, year, venue) row needing citations, calls

  GET /graph/v1/paper/search/match?query={TITLE}
      &fields=title,citationCount,year,venue,externalIds,abstract

which returns the single best match plus a `matchScore`. We ACCEPT a match only
when ALL hold:
  * title similarity is high  -- exact normalized-title match OR difflib ratio
    >= ACCEPT_RATIO (matchScore is an absolute BM25-like score that scales with
    title length, so it is recorded for auditing but NOT used as the gate)
  * |matched_year - target_year| <= 1
  * venue token overlap (guards against same-title papers in other venues)

Output: s2_cache.jsonl, one record per key, RESUMABLE (keys already cached are
skipped). Each record:
  {key, title, target_year, target_venue, s2_paper_id, s2_corpus_id, s2_title,
   s2_year, s2_venue, citationCount, abstract, matchScore, doi, arxiv,
   accepted, reject_reason}

Prints accept / reject / year-mismatch / venue-mismatch / no-match counts.
"""
import argparse
import difflib
import json
import os
import re
import sys
import time
import unicodedata

import requests

MAILTO = "alex2awesome@gmail.com"
UA = f"norm-research acad-C build (mailto:{MAILTO})"
MATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/match"
FIELDS = "title,citationCount,year,venue,externalIds,abstract"
CACHE = "s2_cache.jsonl"

ACCEPT_RATIO = 0.92      # difflib ratio on normalized titles
YEAR_TOL = 1             # |s2_year - target_year| <= YEAR_TOL
PACE = 1.1               # seconds between requests
MAX_RETRY = 6

# venue normalization: map S2's free-text venue strings + our short codes onto
# a small set of canonical tokens so token overlap is meaningful
VENUE_TOKENS = {
    "ICLR": {"iclr", "learning", "representations"},
    "ICML": {"icml", "machine", "learning"},
    "NeurIPS": {"neurips", "nips", "neural", "information", "processing",
                "systems", "advances"},
}
# tokens too generic to count as venue evidence on their own
GENERIC = {"learning", "machine", "neural", "information", "processing",
           "systems", "advances", "international", "conference", "annual",
           "on", "of", "the", "in", "and", "proceedings"}

_STOP = {"a", "an", "the", "of", "for", "and", "to", "in", "on", "with",
         "via", "using", "by"}


def norm_text(s):
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = (s.replace("‘", "'").replace("’", "'")
          .replace("“", '"').replace("”", '"')
          .replace("–", "-").replace("—", "-"))
    return s


def norm_title(s):
    s = norm_text(s).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def venue_tokens(s):
    return {t for t in re.split(r"[^a-z0-9]+", (s or "").lower()) if t}


def venue_ok(target_venue, s2_venue):
    """True if the S2 venue plausibly matches the target venue.

    Strategy: the target is one of our 3 short codes. Build its canonical token
    set. The S2 venue string is free text. Require overlap on a NON-generic
    venue token (e.g. 'iclr', 'icml', 'neurips'/'nips') OR, failing that, a
    strong overlap on the descriptive tokens. Empty S2 venue -> treat as
    unknown (do not reject on venue alone; year+title still gate).
    """
    canon = VENUE_TOKENS.get(target_venue, set())
    s2 = venue_tokens(s2_venue)
    if not s2:
        return "empty"
    # hard venue tokens (acronyms / distinctive words)
    hard = {t for t in canon if t not in GENERIC}
    if hard & s2:
        return "hard"
    # descriptive-token overlap: need >= 2 shared canonical tokens
    shared = canon & s2
    if len(shared) >= 2:
        return "soft"
    return "miss"


def title_match(target_title, s2_title):
    a, b = norm_title(target_title), norm_title(s2_title)
    if not b:
        return 0.0, False
    exact = a == b
    if not exact:
        # tolerate trailing subtitle differences after a colon
        ap = norm_title(norm_text(target_title).split(":")[0])
        bp = norm_title(norm_text(s2_title).split(":")[0])
        if len(ap) >= 12 and ap == bp:
            exact = True
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return ratio, exact


def call_match(session, title):
    for attempt in range(MAX_RETRY):
        try:
            r = session.get(MATCH_URL, params={"query": title, "fields": FIELDS},
                            timeout=60)
        except requests.RequestException:
            time.sleep(min(2 ** attempt, 30))
            continue
        if r.status_code == 200:
            try:
                data = r.json().get("data") or []
            except ValueError:
                return None
            return data[0] if data else None
        if r.status_code == 404:
            # match endpoint returns 404 with {"error":"Title match not found"}
            return None
        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            wait = float(ra) if ra and ra.replace(".", "").isdigit() else min(2 ** attempt + 2, 30)
            time.sleep(wait)
            continue
        # 5xx etc.
        time.sleep(min(2 ** attempt, 30))
    return None


def evaluate(rec, target_title, target_year, target_venue):
    """Return (accepted: bool, reason: str). reason describes the gate result."""
    if rec is None:
        return False, "no_match"
    s2_title = rec.get("title")
    s2_year = rec.get("year")
    s2_venue = rec.get("venue")
    ratio, exact = title_match(target_title, s2_title)
    if not (exact or ratio >= ACCEPT_RATIO):
        return False, f"title_low({ratio:.2f})"
    if s2_year is None or abs(int(s2_year) - int(target_year)) > YEAR_TOL:
        return False, f"year_mismatch({s2_year})"
    v = venue_ok(target_venue, s2_venue)
    if v == "miss":
        return False, f"venue_mismatch({s2_venue!r})"
    return True, f"ok(title={'exact' if exact else f'{ratio:.2f}'},venue={v})"


def load_inputs(path, only_venue, only_year):
    """Read rows needing citations. Accepts CSV with venue,year,title[,dblp_key].

    Builds a stable `key` per row: prefers dblp_key, else venue|year|title.
    """
    import pandas as pd
    df = pd.read_csv(path)
    if only_venue:
        df = df[df.venue == only_venue]
    if only_year is not None:
        df = df[df.year == int(only_year)]
    rows = []
    for r in df.itertuples():
        key = getattr(r, "dblp_key", None)
        if key is None or (isinstance(key, float)):
            key = f"{r.venue}|{r.year}|{r.title}"
        rows.append({"key": str(key), "title": str(r.title),
                     "year": int(r.year), "venue": str(r.venue)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="dblp_papers.csv",
                    help="CSV with columns venue,year,title[,dblp_key]")
    ap.add_argument("--venue", default=None, help="restrict to one venue")
    ap.add_argument("--year", default=None, type=int, help="restrict to one year")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--limit", default=None, type=int,
                    help="stop after N new fetches (for validation passes)")
    ap.add_argument("--report-every", default=100, type=int)
    args = ap.parse_args()

    rows = load_inputs(args.input, args.venue, args.year)
    print(f"input rows: {len(rows)} "
          f"(venue={args.venue}, year={args.year})", flush=True)

    cached = set()
    if os.path.exists(args.cache):
        with open(args.cache) as f:
            for line in f:
                try:
                    cached.add(json.loads(line)["key"])
                except (ValueError, KeyError):
                    pass
    print(f"already cached: {len(cached)}", flush=True)

    todo = [r for r in rows if r["key"] not in cached]
    print(f"to fetch: {len(todo)}", flush=True)

    session = requests.Session()
    session.headers["User-Agent"] = UA

    counts = {"accept": 0, "no_match": 0, "title_low": 0,
              "year_mismatch": 0, "venue_mismatch": 0}
    out = open(args.cache, "a")
    n = 0
    last = 0.0
    for r in todo:
        if args.limit is not None and n >= args.limit:
            break
        wait = last + PACE - time.time()
        if wait > 0:
            time.sleep(wait)
        last = time.time()

        rec = call_match(session, r["title"])
        accepted, reason = evaluate(rec, r["title"], r["year"], r["venue"])

        ext = (rec or {}).get("externalIds") or {}
        corpus = ext.get("CorpusId")
        doi = ext.get("DOI")
        arxiv = ext.get("ArXiv")
        entry = {
            "key": r["key"],
            "title": r["title"],
            "target_year": r["year"],
            "target_venue": r["venue"],
            "s2_paper_id": (rec or {}).get("paperId"),
            "s2_corpus_id": corpus,
            "s2_title": (rec or {}).get("title"),
            "s2_year": (rec or {}).get("year"),
            "s2_venue": (rec or {}).get("venue"),
            "citationCount": (rec or {}).get("citationCount"),
            "abstract": (rec or {}).get("abstract") or "",
            "matchScore": (rec or {}).get("matchScore"),
            "doi": doi,
            "arxiv": arxiv,
            "accepted": accepted,
            "reject_reason": None if accepted else reason,
        }
        out.write(json.dumps(entry) + "\n")
        out.flush()

        if accepted:
            counts["accept"] += 1
        else:
            bucket = reason.split("(")[0]
            counts[bucket] = counts.get(bucket, 0) + 1
        n += 1
        if n % args.report_every == 0:
            print(f"  {n}/{len(todo)}  {counts}", flush=True)
    out.close()

    print(f"\nDONE: fetched {n} new this run", flush=True)
    print("counts (this run):", counts, flush=True)
    # full-cache tally
    acc = rej = 0
    by_reason = {}
    with open(args.cache) as f:
        for line in f:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if args.venue and e.get("target_venue") != args.venue:
                continue
            if args.year is not None and e.get("target_year") != args.year:
                continue
            if e.get("accepted"):
                acc += 1
            else:
                rej += 1
                rr = (e.get("reject_reason") or "?").split("(")[0]
                by_reason[rr] = by_reason.get(rr, 0) + 1
    total = acc + rej
    print(f"\nFULL CACHE (filtered to venue/year): accepted {acc}/{total} "
          f"({acc/total*100:.1f}%)" if total else "empty", flush=True)
    print("rejections by reason:", by_reason, flush=True)


if __name__ == "__main__":
    main()
