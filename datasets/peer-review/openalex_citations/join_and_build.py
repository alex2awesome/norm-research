#!/usr/bin/env python3
"""Join DBLP venue membership to OpenAlex records; build labeled dataset.

Pipeline:
  1. DBLP rows (venue x year membership, authoritative) are matched to the
     OpenAlex source-pull (openalex_source_works.parquet) by normalized title
     within venue.
  2. DBLP rows with no source-pull match are joined via the OpenAlex works
     API: title.search filtered to publication_year +-1, with a full-`search`
     fallback (OpenAlex sometimes truncates titles at a colon). Candidates
     scored by normalized-title similarity; accept >= 0.90. Resumable via
     title_join_cache.jsonl.
  3. Preprint/published dedup: among API candidates we prefer non-repository
     records; across the final table, duplicate normalized titles keep the
     venue-source record (else the higher-cited record).
  4. Citation percentile computed within venue x year over joined rows via
     average rank. judgement: 1 = top quartile (pct >= 0.75), 0 = bottom
     quartile (pct <= 0.25), blank otherwise.

Output: openalex_citations.csv.gz
  columns: id, text (TITLE\n\nABSTRACT, NFKC + straight quotes), judgement,
  percentile, venue, year, cited_by_count, record_kind, match_score, doi,
  title, has_abstract
"""
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
ACCEPT = 0.90
RATE_INTERVAL = 0.15  # global ~6.7 req/s
CACHE = "title_join_cache.jsonl"
SELECT = ("id,doi,title,publication_year,type,cited_by_count,"
          "abstract_inverted_index,primary_location")

_rate_lock = threading.Lock()
_last_req = [0.0]


def throttle():
    with _rate_lock:
        wait = _last_req[0] + RATE_INTERVAL - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_req[0] = time.time()


def norm_text(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
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


def decode_abstract(inv):
    if not inv:
        return ""
    pos = [(i, w) for w, idxs in inv.items() for i in idxs]
    pos.sort()
    return " ".join(w for _, w in pos)


def title_score(a, b):
    na, nb = norm_title(a), norm_title(b or "")
    score = difflib.SequenceMatcher(None, na, nb).ratio()
    a_pre = norm_title(norm_text(a).split(":")[0])
    b_pre = norm_title(norm_text(b or "").split(":")[0])
    if len(nb) >= 4 and nb == a_pre:
        score = max(score, 0.93)
    if len(na) >= 12 and na == b_pre:
        score = max(score, 0.93)
    return score


def api_get(session, params):
    for attempt in range(6):
        throttle()
        try:
            r = session.get(API, params=params, timeout=60)
            if r.status_code == 200:
                return r.json().get("results", [])
            if r.status_code == 403:
                return []
            if (r.status_code == 429
                    and r.headers.get("X-RateLimit-Remaining") == "0"):
                # per-IP daily credit budget gone -- abort instead of
                # poisoning the cache with fake "no match" records
                raise SystemExit("OpenAlex daily budget exhausted; rerun "
                                 "from another host (cache resumes)")
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return []


def search_one(session, title, year):
    q = norm_text(title).replace(",", " ").replace("|", " ").strip()
    years = f"{year - 1}|{year}|{year + 1}"
    base = {"select": SELECT, "per-page": "5", "mailto": MAILTO}
    results = api_get(session, {
        "filter": f"title.search:{q},publication_year:{years}", **base})
    best, score = pick_best(title, results)
    if score < ACCEPT:
        results2 = api_get(session, {
            "search": q, "filter": f"publication_year:{years}", **base})
        best2, score2 = pick_best(title, results2)
        if score2 > score:
            best, score = best2, score2
    return best, score


def pick_best(title, results):
    best, best_key = None, (-1.0, -1, -1)
    for res in results or []:
        score = title_score(title, res.get("title"))
        src = (res.get("primary_location") or {}).get("source") or {}
        not_repo = 0 if src.get("type") == "repository" else 1
        key = (round(score, 4), not_repo, res.get("cited_by_count") or 0)
        if key > best_key:
            best_key, best = key, res
    return best, max(best_key[0], 0.0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-query", action="store_true",
                    help="build from title_join_cache.jsonl only; cache "
                         "misses are dropped (reported), no API spend")
    args = ap.parse_args()

    dblp = pd.read_csv("dblp_papers.csv")
    src = pd.read_parquet("openalex_source_works.parquet")
    dblp["ntitle"] = dblp.title.map(norm_title)
    src["ntitle"] = src.title.map(norm_title)

    # within-venue-year DBLP dup titles: keep first
    dblp = dblp.drop_duplicates(["venue", "year", "ntitle"]).copy()
    print(f"DBLP membership: {len(dblp)} rows")

    # 1) exact normalized-title match against source pull, within venue
    src_dedup = (src.sort_values("cited_by_count", ascending=False)
                    .drop_duplicates(["venue", "ntitle"]))
    src_map = {(r.venue, r.ntitle): r for r in src_dedup.itertuples()}

    rows, todo = [], []
    for r in dblp.itertuples():
        hit = src_map.get((r.venue, r.ntitle))
        if hit is not None:
            rows.append({
                "venue": r.venue, "year": r.year, "dblp_title": r.title,
                "openalex_id": hit.openalex_id, "doi": hit.doi,
                "oa_title": hit.title, "cited_by_count": hit.cited_by_count,
                "abstract": hit.abstract, "record_kind": "venue_source",
                "match_score": 1.0,
            })
        else:
            todo.append(r)
    print(f"matched via venue source: {len(rows)}; need title join: {len(todo)}")

    # 2) API title join for the rest (cached, parallel, rate-limited)
    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            for line in f:
                rec = json.loads(line)
                cache[rec["_key"]] = rec

    session = requests.Session()
    session.headers["User-Agent"] = f"norm-research (mailto:{MAILTO})"
    cache_f = open(CACHE, "a")
    write_lock = threading.Lock()
    done = [0]

    def work(r):
        key = f"{r.venue}|{r.year}|{r.title}"
        if key in cache:
            return r, cache[key]
        if args.no_query:
            return r, {"match_score": 0.0}
        best, score = search_one(session, r.title, int(r.year))
        rec = {"_key": key, "match_score": score}
        if best is not None:
            src_ = (best.get("primary_location") or {}).get("source") or {}
            rec.update({
                "openalex_id": best["id"], "doi": best.get("doi"),
                "oa_title": best.get("title"),
                "cited_by_count": best.get("cited_by_count"),
                "abstract": decode_abstract(best.get("abstract_inverted_index")),
                "src_type": src_.get("type"),
                "src_name": src_.get("display_name"),
            })
        with write_lock:
            cache_f.write(json.dumps(rec) + "\n")
            cache_f.flush()
            cache[key] = rec
            done[0] += 1
            if done[0] % 250 == 0:
                print(f"  title-join {done[0]}/{len(todo)}", flush=True)
        return r, rec

    with ThreadPoolExecutor(max_workers=6) as ex:
        for r, rec in ex.map(work, todo):
            if rec.get("openalex_id") and rec.get("match_score", 0) >= ACCEPT:
                kind = ("titlejoin_preprint"
                        if rec.get("src_type") == "repository"
                        else "titlejoin_published")
                rows.append({
                    "venue": r.venue, "year": r.year, "dblp_title": r.title,
                    "openalex_id": rec["openalex_id"], "doi": rec.get("doi"),
                    "oa_title": rec.get("oa_title"),
                    "cited_by_count": rec.get("cited_by_count"),
                    "abstract": rec.get("abstract") or "",
                    "record_kind": kind,
                    "match_score": rec["match_score"],
                })
    cache_f.close()

    df = pd.DataFrame(rows)
    join_rate = df.groupby(["venue", "year"]).size()
    memb = dblp.groupby(["venue", "year"]).size()
    print("\njoin rate per venue-year:")
    print(pd.DataFrame({"dblp": memb, "joined": join_rate,
                        "rate": (join_rate / memb).round(3)}).to_string())

    # 3) cross-rows dedup on normalized title: prefer venue_source, then cites
    df["ntitle"] = df.dblp_title.map(norm_title)
    kind_rank = {"venue_source": 0, "titlejoin_published": 1,
                 "titlejoin_preprint": 2}
    df["kind_rank"] = df.record_kind.map(kind_rank)
    df = (df.sort_values(["kind_rank", "cited_by_count"],
                         ascending=[True, False])
            .drop_duplicates(["venue", "ntitle"])
            .drop_duplicates("openalex_id"))
    print(f"\nafter dedup: {len(df)} rows")

    # 4) percentiles within venue x year
    df["percentile"] = (df.groupby(["venue", "year"]).cited_by_count
                          .rank(pct=True, method="average"))
    df["judgement"] = ""
    df.loc[df.percentile >= 0.75, "judgement"] = "1"
    df.loc[df.percentile <= 0.25, "judgement"] = "0"

    # venue-years where less than half the DBLP membership joined get their
    # labels blanked: a percentile over a biased subsample (e.g. NeurIPS
    # 2023 pending the Curran-stub repair) is not trustworthy
    cov = (df.groupby(["venue", "year"]).size() / memb).fillna(0)
    low = cov[cov < 0.5].index
    if len(low):
        print(f"\nLOW-COVERAGE venue-years (labels blanked): "
              f"{[(v, y, round(cov[(v, y)], 2)) for v, y in low]}")
        mask = df.set_index(["venue", "year"]).index.isin(low)
        df.loc[mask, "judgement"] = ""
        df.loc[mask, "percentile"] = float("nan")

    df["title"] = df.dblp_title.map(norm_text)
    df["abstract"] = df.abstract.map(norm_text)
    df["has_abstract"] = df.abstract.str.len() > 0
    df["text"] = (df.title + "\n\n" + df.abstract).str.strip()

    out = df.rename(columns={"openalex_id": "id"})[
        ["id", "text", "judgement", "percentile", "venue", "year",
         "cited_by_count", "record_kind", "match_score", "doi", "title",
         "has_abstract"]]
    out.to_csv("openalex_citations.csv.gz", index=False, compression="gzip")
    print(f"\nwrote {len(out)} rows to openalex_citations.csv.gz")
    print("\nabstract availability per venue-year:")
    print(df.groupby(["venue", "year"]).has_abstract.mean().round(3).to_string())
    print("\nrecord_kind counts:")
    print(df.record_kind.value_counts().to_string())
    print("\nlabel counts:")
    print(out.judgement.value_counts().to_string())


if __name__ == "__main__":
    main()
