#!/usr/bin/env python3
"""Budget-aware batched OpenAlex title join.

OpenAlex moved to a credit system (2025/26): $1 = 10,000 credits per IP per
day, search-type requests cost 10 credits, plain filters 1. Joining tens of
thousands of titles via one title.search each is therefore infeasible.

This joiner exploits two verified facts (2026-06-12):
  * title.search accepts pipe-OR'd quoted phrases:
        filter=title.search:"t1"|"t2"|...,publication_year:2014|2015|2016
    and one request costs 10 credits REGARDLESS of how many titles are OR'd.
  * quoted phrases keep result counts small enough that a batch of ~15
    titles fits in 1-2 pages of 200.

Algorithm: group rows by year, batch B titles per request, score every
result against every batch title (normalized SequenceMatcher + pre-colon
credit), accept best >= threshold. Unmatched titles go to a singleton pass
(one title.search each, optional full-`search` fallback). Aborts cleanly on
429/budget-exhausted; output jsonl is append-only and resumable.

Input CSV columns: key, title, year. Output jsonl: one record per key.
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
API = "https://api.openalex.org/works"
SELECT = ("id,doi,title,publication_year,type,cited_by_count,"
          "abstract_inverted_index,primary_location")


class BudgetExhausted(SystemExit):
    pass


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


def sanitize(title):
    # strip filter-syntax chars (comma separates filters, pipe separates
    # OR terms, quotes delimit phrases)
    return (norm_text(title).replace(",", " ").replace("|", " ")
            .replace('"', " ").strip())


def api_get(session, params):
    for attempt in range(6):
        try:
            r = session.get(API, params=params, timeout=90)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                # Budget 429s carry an "Insufficient budget" JSON body, NOT
                # always the X-RateLimit-Remaining:0 header — treating them
                # as transient caches every key as a false no-match.
                if (r.headers.get("X-RateLimit-Remaining") == "0"
                        or "Insufficient budget" in r.text
                        or "Rate limit exceeded" in r.text):
                    raise BudgetExhausted(
                        "OpenAlex daily credit budget exhausted (per IP, "
                        "resets midnight UTC). Rerun later/elsewhere; "
                        "jsonl resumes.")
            if r.status_code == 403:
                return {"results": [], "meta": {}}
        except requests.RequestException:
            pass
        time.sleep(min(30, 2 ** attempt))
    return {"results": [], "meta": {}}


PREFER = "published"  # or "citations" (see --prefer)


def best_for(title, results):
    """Best candidate by title similarity.

    Tie-break: prefer="published" ranks non-repository records first (good
    default). prefer="citations" ranks by cited_by_count instead -- needed
    where the official proceedings record is a citation-dead stub (Curran
    NeurIPS 2022+ deposits carry ~0 citations; the arXiv twin holds them).
    """
    best, best_key = None, (-1.0, -1, -1)
    for res in results:
        score = title_score(title, res.get("title"))
        src = (res.get("primary_location") or {}).get("source") or {}
        not_repo = 0 if src.get("type") == "repository" else 1
        cites = res.get("cited_by_count") or 0
        if PREFER == "citations":
            key = (round(score, 4), cites, not_repo)
        else:
            key = (round(score, 4), not_repo, cites)
        if key > best_key:
            best_key, best = key, res
    return best, max(best_key[0], 0.0)


def to_record(key, best, score):
    rec = {"_key": key, "match_score": score}
    if best is not None:
        src = (best.get("primary_location") or {}).get("source") or {}
        rec.update({
            "openalex_id": best["id"], "doi": best.get("doi"),
            "oa_title": best.get("title"),
            "oa_year": best.get("publication_year"),
            "oa_type": best.get("type"),
            "cited_by_count": best.get("cited_by_count"),
            "abstract": decode_abstract(best.get("abstract_inverted_index")),
            "oa_source": src.get("display_name"),
            "src_type": src.get("type"),
            "src_name": src.get("display_name"),
        })
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="csv: key,title,year")
    ap.add_argument("--out", required=True, help="append-only jsonl")
    ap.add_argument("--batch", type=int, default=15)
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--fallback", choices=["none", "title", "full"],
                    default="title")
    ap.add_argument("--prefer", choices=["published", "citations"],
                    default="published")
    args = ap.parse_args()
    global PREFER
    PREFER = args.prefer

    import csv as _csv
    rows = list(_csv.DictReader(open(args.input)))
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(l)["_key"] for l in f}
    todo = [r for r in rows if r["key"] not in done]
    print(f"{len(rows)} rows, {len(todo)} to join", flush=True)

    session = requests.Session()
    session.headers["User-Agent"] = f"norm-research (mailto:{MAILTO})"
    out_f = open(args.out, "a")

    def emit(rec):
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()

    # group by year
    by_year = {}
    for r in todo:
        by_year.setdefault(int(r["year"]), []).append(r)

    singles, n_req, n_matched = [], 0, 0
    for year in sorted(by_year):
        group = by_year[year]
        for i in range(0, len(group), args.batch):
            batch = group[i:i + args.batch]
            terms = []
            for r in batch:
                t = sanitize(r["title"])
                terms.append(f'"{t}"')
                # OpenAlex sometimes stores only the pre-colon part of a
                # title; add that phrase too (same flat request cost)
                pre = t.split(":")[0].strip()
                if ":" in r["title"] and len(pre) >= 10:
                    terms.append(f'"{pre}"')
            phrases = "|".join(terms)
            years = f"{year - 1}|{year}|{year + 1}"
            results = []
            params = {"filter": f"title.search:{phrases},"
                                f"publication_year:{years}",
                      "select": SELECT, "per-page": "200", "mailto": MAILTO}
            j = api_get(session, params)
            n_req += 1
            results.extend(j.get("results", []))
            count = (j.get("meta") or {}).get("count") or 0
            if count > 200:  # crowded batch: grab one more page
                j2 = api_get(session, {**params, "page": "2"})
                n_req += 1
                results.extend(j2.get("results", []))
            for r in batch:
                best, score = best_for(r["title"], results)
                if best is not None and score >= args.threshold:
                    emit(to_record(r["key"], best, score))
                    n_matched += 1
                else:
                    singles.append(r)
            if n_req % 20 == 0:
                print(f"  batches: {n_req} requests, {n_matched} matched, "
                      f"{len(singles)} queued for singleton", flush=True)
            time.sleep(0.15)

    print(f"batch pass done: {n_matched} matched, {len(singles)} singletons",
          flush=True)

    if args.fallback != "none":
        import threading
        from concurrent.futures import ThreadPoolExecutor
        lock = threading.Lock()
        prog = [0]

        def single(r):
            years = (f"{int(r['year']) - 1}|{r['year']}|"
                     f"{int(r['year']) + 1}")
            q = sanitize(r["title"])
            # the batch pass already tried the quoted phrase -- go straight
            # to unquoted (term-level) search, which handles title variants
            j = api_get(session, {
                "filter": f"title.search:{q},publication_year:{years}",
                "select": SELECT, "per-page": "10", "mailto": MAILTO})
            best, score = best_for(r["title"], j.get("results", []))
            if score < args.threshold and args.fallback == "full":
                j = api_get(session, {
                    "search": q, "filter": f"publication_year:{years}",
                    "select": SELECT, "per-page": "10", "mailto": MAILTO})
                b2, s2 = best_for(r["title"], j.get("results", []))
                if s2 > score:
                    best, score = b2, s2
            ok = best is not None and score >= args.threshold
            with lock:
                emit(to_record(r["key"], best if ok else None, score))
                prog[0] += 1
                if prog[0] % 50 == 0:
                    print(f"  singletons: {prog[0]}/{len(singles)}",
                          flush=True)
            return ok

        with ThreadPoolExecutor(max_workers=8) as ex:
            n_matched += sum(ex.map(single, singles))

    out_f.close()
    print(f"done: {n_matched}/{len(todo)} matched", flush=True)


if __name__ == "__main__":
    main()
