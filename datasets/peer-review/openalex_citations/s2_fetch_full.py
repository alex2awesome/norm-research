#!/usr/bin/env python3
"""Full-corpus S2 title-match fetch for acad-C (ICLR/ICML/NeurIPS, all years).

Reads s2_todo.csv (venue,year,title,dblp_key) = the DBLP membership rows NOT
already covered by the merged reuse cache, and fetches each via the S2 match
endpoint. Resumable: keys already in --cache are skipped. Cache key = dblp_key.

Accept gate (same as s2_title_join.py / s2_join_best_papers.py):
  exact / >=0.92-difflib normalized-title match
  AND |s2_year - target_year| <= 1
  AND venue plausibility (empty S2 venue is NOT a reject; DBLP membership is
  authoritative). For ICLR/ICML/NeurIPS we require a hard acronym hit when a
  non-empty S2 venue is present, else accept (these venues' S2 venue strings
  are inconsistent; title+year carry the gate).

429 handling: honor Retry-After; else fixed 2.5s backoff (steady drip beats
exponential-to-30 for unauthenticated S2). ~2.5s/req effective. No API key.
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
UA = f"norm-research acad-C full S2 (mailto:{MAILTO})"
MATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/match"
FIELDS = "title,citationCount,year,venue,externalIds,abstract"

ACCEPT_RATIO = 0.92
YEAR_TOL = 1
PACE = 2.5
MAX_RETRY = 12

HARD = {
    "ICLR": {"iclr"},
    "ICML": {"icml"},
    "NeurIPS": {"neurips", "nips"},
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


def venue_ok(tv, s2v):
    s2 = venue_tokens(s2v)
    if not s2:
        return "empty"
    if HARD.get(tv, set()) & s2:
        return "hard"
    # guard against same-title collisions ACROSS our own three venues: if the
    # S2 venue clearly names a DIFFERENT one of ICLR/ICML/NeurIPS, reject.
    for other, toks in HARD.items():
        if other != tv and (toks & s2):
            return "wrong_target_venue"
    # otherwise the S2 venue is some unrelated string (or a workshop alias):
    # DBLP membership is authoritative, so title+year carry the accept.
    return "soft_empty_ok"


def title_match(t, s2t):
    a, b = norm_title(t), norm_title(s2t)
    if not b:
        return 0.0, False
    exact = a == b
    if not exact:
        ap = norm_title(norm_text(t).split(":")[0])
        bp = norm_title(norm_text(s2t).split(":")[0])
        if len(ap) >= 12 and ap == bp:
            exact = True
    return difflib.SequenceMatcher(None, a, b).ratio(), exact


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
    if v == "wrong_target_venue":
        return False, f"venue_mismatch({rec.get('venue')!r})"
    return True, f"ok(title={'exact' if exact else f'{ratio:.2f}'},venue={v})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="s2_todo.csv")
    ap.add_argument("--cache", default="s2_cache_full.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report-every", type=int, default=200)
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_csv(args.input)
    rows = [{"key": str(r.dblp_key), "title": str(r.title),
             "year": int(r.year), "venue": str(r.venue)}
            for r in df.itertuples()]

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
            "target_venue": r["venue"],
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
