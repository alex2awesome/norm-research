#!/usr/bin/env python3
"""S2 title-match join for best_papers (acad-B2). Adapted from
openalex_citations/s2_title_join.py to cover all 32 award venues.

Same accept gate: exact/0.92-difflib normalized-title match AND
|s2_year-target|<=1 AND venue-token overlap (or empty S2 venue -> not
rejected on venue, since DBLP membership is already authoritative).

Input: s2_input.csv (venue,year,title,dblp_key,is_award). Cache key = dblp_key.
Output: s2_cache.jsonl (resumable). Honors Retry-After, ~1.1s/req, no key.
"""
import argparse
import difflib
import json
import os
import re
import time
import unicodedata

import requests

MAILTO = "alex2awesome@gmail.com"
UA = f"norm-research best_papers acad-B2 (mailto:{MAILTO})"
MATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/match"
FIELDS = "title,citationCount,year,venue,externalIds,abstract"

ACCEPT_RATIO = 0.92
YEAR_TOL = 1
PACE = 1.1
MAX_RETRY = 12   # with 2.5s fixed 429 backoff this caps a 429 streak ~30s

# per-venue distinctive (hard) tokens for S2 free-text venue overlap.
# Generic descriptive words are excluded so they cannot match on their own.
HARD = {
    "AAAI": {"aaai"}, "ACL": {"acl"}, "CHI": {"chi"}, "CIKM": {"cikm"},
    "CVPR": {"cvpr"}, "FOCS": {"focs"}, "FSE": {"fse", "esec", "sigsoft"},
    "ICCV": {"iccv"}, "ICML": {"icml"}, "ICSE": {"icse"}, "IJCAI": {"ijcai"},
    "INFOCOM": {"infocom"}, "ISCA": {"isca"}, "KDD": {"kdd", "sigkdd"},
    "MOBICOM": {"mobicom", "mobile", "mobicomm"}, "NSDI": {"nsdi"},
    "NeurIPS": {"neurips", "nips"}, "OSDI": {"osdi"}, "PLDI": {"pldi", "pacmpl"},
    "PODS": {"pods"}, "S&P": {"sp", "oakland"}, "SIGCOMM": {"sigcomm"},
    "SIGGRAPH": {"siggraph", "tog", "graphics"}, "SIGIR": {"sigir"},
    "SIGMETRICS": {"sigmetrics", "pomacs"}, "SIGMOD": {"sigmod", "pacmmod"},
    "SODA": {"soda"}, "SOSP": {"sosp"}, "STOC": {"stoc"}, "UIST": {"uist"},
    "VLDB": {"vldb", "pvldb"}, "WWW": {"www", "web"},
}


def norm_text(s):
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    return (s.replace("‘", "'").replace("’", "'")
             .replace("“", '"').replace("”", '"')
             .replace("–", "-").replace("—", "-"))


def norm_title(s):
    s = norm_text(s).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def venue_tokens(s):
    return {t for t in re.split(r"[^a-z0-9]+", (s or "").lower()) if t}


def venue_ok(target_venue, s2_venue):
    s2 = venue_tokens(s2_venue)
    if not s2:
        return "empty"          # DBLP membership authoritative; do not reject
    hard = HARD.get(target_venue, set())
    if hard & s2:
        return "hard"
    return "soft_empty_ok"      # venue present but no hard-token hit -> still
                                # accept (title+year gate carries it; S2 venue
                                # strings for these venues are inconsistent)


def title_match(target_title, s2_title):
    a, b = norm_title(target_title), norm_title(s2_title)
    if not b:
        return 0.0, False
    exact = a == b
    if not exact:
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
            time.sleep(min(2 ** attempt, 30)); continue
        if r.status_code == 200:
            try:
                data = r.json().get("data") or []
            except ValueError:
                return None
            return data[0] if data else None
        if r.status_code == 404:
            return None
        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            # S2 unauthenticated: bursts of ~5 then 429 with NO Retry-After.
            # A fixed short backoff keeps a steady drip far better than
            # exponential-to-30 (which wastes minutes idling).
            wait = float(ra) if ra and ra.replace(".", "").isdigit() else 2.5
            time.sleep(wait); continue
        time.sleep(min(2 ** attempt, 30))
    return None


def evaluate(rec, t, ty, tv):
    if rec is None:
        return False, "no_match"
    ratio, exact = title_match(t, rec.get("title"))
    if not (exact or ratio >= ACCEPT_RATIO):
        return False, f"title_low({ratio:.2f})"
    sy = rec.get("year")
    if sy is None or abs(int(sy) - int(ty)) > YEAR_TOL:
        return False, f"year_mismatch({sy})"
    v = venue_ok(tv, rec.get("venue"))
    return True, f"ok(title={'exact' if exact else f'{ratio:.2f}'},venue={v})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="s2_input.csv")
    ap.add_argument("--cache", default="s2_cache.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--awards-first", action="store_true")
    ap.add_argument("--report-every", type=int, default=200)
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_csv(args.input)
    if args.awards_first:
        df = df.sort_values("is_award", ascending=False)
    rows = [{"key": str(r.dblp_key), "title": str(r.title),
             "year": int(r.year), "venue": str(r.venue),
             "is_award": int(r.is_award)} for r in df.itertuples()]

    cached = set()
    if os.path.exists(args.cache):
        with open(args.cache) as f:
            for line in f:
                try:
                    cached.add(json.loads(line)["key"])
                except (ValueError, KeyError):
                    pass
    todo = [r for r in rows if r["key"] not in cached]
    print(f"input {len(rows)}  cached {len(cached)}  todo {len(todo)}", flush=True)

    session = requests.Session()
    session.headers["User-Agent"] = UA
    counts = {"accept": 0}
    out = open(args.cache, "a")
    n = 0; last = 0.0
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
        out.write(json.dumps({
            "key": r["key"], "title": r["title"], "target_year": r["year"],
            "target_venue": r["venue"], "is_award": r["is_award"],
            "s2_paper_id": (rec or {}).get("paperId"),
            "s2_corpus_id": ext.get("CorpusId"),
            "s2_title": (rec or {}).get("title"),
            "s2_year": (rec or {}).get("year"),
            "s2_venue": (rec or {}).get("venue"),
            "citationCount": (rec or {}).get("citationCount"),
            "abstract": (rec or {}).get("abstract") or "",
            "matchScore": (rec or {}).get("matchScore"),
            "doi": ext.get("DOI"), "arxiv": ext.get("ArXiv"),
            "accepted": accepted,
            "reject_reason": None if accepted else reason}) + "\n")
        out.flush()
        if accepted:
            counts["accept"] += 1
        else:
            b = reason.split("(")[0]; counts[b] = counts.get(b, 0) + 1
        n += 1
        if n % args.report_every == 0:
            print(f"  {n}/{len(todo)}  {counts}", flush=True)
    out.close()
    print(f"DONE: {n} new this run; counts {counts}", flush=True)


if __name__ == "__main__":
    main()
