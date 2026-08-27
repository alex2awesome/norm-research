#!/usr/bin/env python3
"""Join scraped best-paper awards to OpenAlex works by title search.

For each award (venue, year, title), queries the OpenAlex works endpoint
(polite pool) with title.search filtered to publication_year in [y-1, y+1],
scores candidates by normalized-title similarity, and keeps the best match.

Outputs best_papers_joined.csv with one row per award:
    venue, field, year, title, authors_raw,
    openalex_id, doi, oa_title, oa_year, cited_by_count, abstract,
    match_score, matched (bool)

Resumable: appends to a jsonl cache keyed by (venue, year, title).
"""
import argparse
import difflib
import json
import os
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

MAILTO = "alex2awesome@gmail.com"
API = "https://api.openalex.org/works"
SLEEP = 0.15  # global min interval between requests (~6 req/s)
ACCEPT_THRESHOLD = 0.85

_rate_lock = threading.Lock()
_last_req = [0.0]


def throttle():
    with _rate_lock:
        wait = _last_req[0] + SLEEP - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_req[0] = time.time()


def norm_text(s: str) -> str:
    """NFKC + curly->straight quotes. Same normalization used project-wide."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = (s.replace("‘", "'").replace("’", "'")
          .replace("“", '"').replace("”", '"')
          .replace("–", "-").replace("—", "-"))
    return s


def norm_title(s: str) -> str:
    s = norm_text(s).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def decode_abstract(inv):
    if not inv:
        return ""
    pos = []
    for word, idxs in inv.items():
        for i in idxs:
            pos.append((i, word))
    pos.sort()
    return " ".join(w for _, w in pos)


def query_openalex(session, title, year, mode="title"):
    # commas separate OpenAlex filters; strip them (and pipes) from the value
    q = norm_text(title).replace(",", " ").replace("|", " ").strip()
    years = f"{year - 1}|{year}|{year + 1}"
    if mode == "title":
        params = {"filter": f"title.search:{q},publication_year:{years}"}
    else:  # full search (title+abstract+fulltext) — recall fallback; OpenAlex
        # sometimes stores only the pre-colon part of a title (e.g. "Hike")
        params = {"search": q, "filter": f"publication_year:{years}"}
    params.update({
        "select": ("id,doi,title,publication_year,type,cited_by_count,"
                   "abstract_inverted_index,primary_location"),
        "per-page": "5",
        "mailto": MAILTO,
    })
    for attempt in range(5):
        throttle()
        try:
            r = session.get(API, params=params, timeout=60)
            if r.status_code == 200:
                return r.json().get("results", [])
            if r.status_code == 403:  # bad query (e.g. odd chars) -> give up
                return []
            if (r.status_code == 429
                    and r.headers.get("X-RateLimit-Remaining") == "0"):
                # daily credit budget exhausted (per-IP, resets midnight UTC).
                # Abort hard rather than poisoning the cache with fake
                # "no match" entries.
                raise SystemExit("OpenAlex daily budget exhausted -- rerun "
                                 "tomorrow or from another host (cache "
                                 "resumes).")
            time.sleep(2 ** attempt)
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return []


def title_score(award_title, oa_title):
    nt, no = norm_title(award_title), norm_title(oa_title or "")
    score = difflib.SequenceMatcher(None, nt, no).ratio()
    # OpenAlex sometimes drops the subtitle after a colon ("Hike: A Hybrid
    # ..." is stored as "Hike"). Credit exact pre-colon matches, both ways.
    award_pre = norm_title(norm_text(award_title).split(":")[0])
    oa_pre = norm_title(norm_text(oa_title or "").split(":")[0])
    if len(no) >= 4 and no == award_pre:
        score = max(score, 0.93)
    if len(nt) >= 12 and nt == oa_pre:
        score = max(score, 0.93)
    return score


def pick_best(title, results):
    """Best candidate by title similarity; prefer non-repository on ties."""
    best, best_key = None, (-1.0, -1, -1)
    for res in results:
        score = title_score(title, res.get("title"))
        src = (res.get("primary_location") or {}).get("source") or {}
        not_repo = 0 if src.get("type") == "repository" else 1
        key = (round(score, 4), not_repo, res.get("cited_by_count") or 0)
        if key > best_key:
            best_key, best = key, res
    return best, best_key[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--awards", default="best_papers_awards.csv")
    ap.add_argument("--cache", default="join_cache.jsonl")
    ap.add_argument("--out", default="best_papers_joined.csv")
    ap.add_argument("--no-query", action="store_true",
                    help="build from cache only; cache misses become "
                         "unmatched rows (no API spend)")
    args = ap.parse_args()

    df = pd.read_csv(args.awards)
    cache = {}
    if os.path.exists(args.cache):
        with open(args.cache) as f:
            for line in f:
                rec = json.loads(line)
                cache[rec["_key"]] = rec

    session = requests.Session()
    session.headers["User-Agent"] = f"norm-research (mailto:{MAILTO})"

    cache_f = open(args.cache, "a")
    write_lock = threading.Lock()
    n_query = [0]
    todo = [row for _, row in df.iterrows()
            if f"{row.venue}|{row.year}|{row.title}" not in cache]

    def work(row):
        key = f"{row.venue}|{row.year}|{row.title}"
        results = query_openalex(session, row.title, int(row.year))
        best, score = pick_best(row.title, results) if results else (None, 0.0)
        if score < ACCEPT_THRESHOLD:  # recall fallback: full search
            results2 = query_openalex(session, row.title, int(row.year),
                                      mode="search")
            best2, score2 = (pick_best(row.title, results2)
                             if results2 else (None, 0.0))
            if score2 > score:
                best, score = best2, score2
        rec = {"_key": key, "match_score": score}
        if best is not None:
            rec.update({
                "openalex_id": best["id"],
                "doi": best.get("doi"),
                "oa_title": best.get("title"),
                "oa_year": best.get("publication_year"),
                "oa_type": best.get("type"),
                "cited_by_count": best.get("cited_by_count"),
                "abstract": decode_abstract(best.get("abstract_inverted_index")),
                "oa_source": ((best.get("primary_location") or {})
                              .get("source") or {}).get("display_name"),
            })
        with write_lock:
            cache_f.write(json.dumps(rec) + "\n")
            cache_f.flush()
            cache[key] = rec
            n_query[0] += 1
            if n_query[0] % 100 == 0:
                print(f"  queried {n_query[0]}/{len(todo)}...", flush=True)

    if args.no_query:
        if todo:
            print(f"--no-query: {len(todo)} cache misses left unmatched")
    else:
        with ThreadPoolExecutor(max_workers=12) as ex:
            list(ex.map(work, todo))
    cache_f.close()

    out_rows = []
    for _, row in df.iterrows():
        rec = cache.get(f"{row.venue}|{row.year}|{row.title}",
                        {"match_score": None})
        out = dict(row)
        out.update({k: v for k, v in rec.items() if k != "_key"})
        out["matched"] = (rec.get("match_score", 0) or 0) >= ACCEPT_THRESHOLD \
            and rec.get("openalex_id") is not None
        out_rows.append(out)

    res = pd.DataFrame(out_rows)
    res.to_csv(args.out, index=False)
    n = len(res)
    m = int(res.matched.sum())
    print(f"\n{n} awards, {m} matched (score>={ACCEPT_THRESHOLD}) = {m/n:.1%}")
    print("match_score distribution:")
    print(res.match_score.describe().to_string())
    print("\nabstract coverage among matched:",
          f"{(res[res.matched].abstract.fillna('').str.len() > 0).mean():.1%}")
    print("\njoin rate by venue:")
    print(res.groupby("venue").matched.mean().sort_values().to_string())


if __name__ == "__main__":
    main()
