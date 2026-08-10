#!/usr/bin/env python3
"""Resumable Semantic Scholar citation fetcher for the ICLR 2024-25 paper set.

For every paper in outputs/expand_v2/sample.csv, look up its title in the
peer_review_pdfs.db `papers` table (paper_id == forum there, no "iclr_"
prefix), query the Semantic Scholar paper/search/match endpoint, and write
one JSONL record per paper with citation info + a match_ok verdict.

Resumable: on start, reads the existing output JSONL and skips any
paper_id already present, so it can be killed/restarted freely.

Usage:
    python fetch_s2_citations.py \
        --csv outputs/expand_v2/sample.csv \
        --db datasets/peer-review/peer_review_pdfs.db \
        --out datasets/peer-review/s2_citations_2024_25.jsonl \
        --workers 2
"""

import argparse
import csv
import json
import os
import random
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

S2_MATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/match"
USER_AGENT = "research (alex2awesome@gmail.com)"
BACKOFF_SCHEDULE = [5, 15, 45]  # seconds, exponential-ish backoff on 429/errors
MIN_WORD_LEN = 4


def load_existing_ids(out_path):
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("paper_id")
            if pid:
                done.add(pid)
    return done


def load_papers(csv_path, db_path, done_ids):
    """Return list of dicts: paper_id, forum, year, title (from papers table)."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT paper_id, title FROM papers")
    title_by_forum = {row[0]: row[1] for row in cur.fetchall()}
    conn.close()

    papers = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["paper_id"]
            if pid in done_ids:
                continue
            forum = row["forum"]
            title = title_by_forum.get(forum)
            papers.append(
                {
                    "paper_id": pid,
                    "forum": forum,
                    "title": title,
                }
            )
    return papers


def word_set(s, min_len=MIN_WORD_LEN):
    words = []
    for tok in s.lower().split():
        tok = "".join(ch for ch in tok if ch.isalnum())
        if len(tok) >= min_len:
            words.append(tok)
    return set(words)


def titles_match(query_title, result_title):
    if not query_title or not result_title:
        return False
    q_words = word_set(query_title)
    r_words = word_set(result_title)
    if not q_words or not r_words:
        return False
    overlap = q_words & r_words
    frac_q = len(overlap) / len(q_words)
    frac_r = len(overlap) / len(r_words)
    return frac_q >= 0.6 or frac_r >= 0.6


def s2_search_match(title):
    """Query S2 match endpoint. Returns dict with match fields or None-filled dict.

    Raises on repeated failure so caller can decide how to record it.
    """
    params = {
        "query": title,
        "fields": "title,year,citationCount,paperId",
    }
    url = S2_MATCH_URL + "?" + urllib.parse.urlencode(params)
    headers = {"User-Agent": USER_AGENT}
    key_path = os.path.expanduser("~/.s2_api_key.txt")
    if os.path.exists(key_path):
        headers["x-api-key"] = open(key_path).read().strip()
    req = urllib.request.Request(url, headers=headers)

    last_err = None
    for attempt in range(len(BACKOFF_SCHEDULE) + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read()
                data = json.loads(body)
                results = data.get("data") or []
                if not results:
                    return {
                        "s2_title": None,
                        "s2_citationCount": None,
                        "s2_year": None,
                        "s2_paperId": None,
                    }
                top = results[0]
                return {
                    "s2_title": top.get("title"),
                    "s2_citationCount": top.get("citationCount"),
                    "s2_year": top.get("year"),
                    "s2_paperId": top.get("paperId"),
                }
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 404:
                # No match found for this title.
                return {
                    "s2_title": None,
                    "s2_citationCount": None,
                    "s2_year": None,
                    "s2_paperId": None,
                }
            if attempt < len(BACKOFF_SCHEDULE):
                sleep_s = BACKOFF_SCHEDULE[attempt] + random.uniform(0, 1.0)
                time.sleep(sleep_s)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            if attempt < len(BACKOFF_SCHEDULE):
                sleep_s = BACKOFF_SCHEDULE[attempt] + random.uniform(0, 1.0)
                time.sleep(sleep_s)
                continue
            raise
    raise last_err  # pragma: no cover


class Worker:
    def __init__(self, papers, out_path, sleep_min, sleep_max, lock, counter, total):
        self.papers = papers
        self.out_path = out_path
        self.sleep_min = sleep_min
        self.sleep_max = sleep_max
        self.lock = lock
        self.counter = counter
        self.total = total

    def run(self):
        for paper in self.papers:
            pid = paper["paper_id"]
            forum = paper["forum"]
            title = paper["title"]

            rec = {
                "paper_id": pid,
                "forum": forum,
                "title": title,
                "s2_title": None,
                "s2_citationCount": None,
                "s2_year": None,
                "s2_paperId": None,
                "match_ok": False,
            }

            if title:
                try:
                    s2_res = s2_search_match(title)
                    rec.update(s2_res)
                    rec["match_ok"] = titles_match(title, s2_res.get("s2_title"))
                except Exception as e:
                    rec["error"] = str(e)

            with self.lock:
                with open(self.out_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                self.counter[0] += 1
                n = self.counter[0]
                if n % 200 == 0:
                    print(f"[progress] {n}/{self.total} written", flush=True)

            time.sleep(random.uniform(self.sleep_min, self.sleep_max))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/expand_v2/sample.csv")
    ap.add_argument("--db", default="datasets/peer-review/peer_review_pdfs.db")
    ap.add_argument("--out", default="datasets/peer-review/s2_citations_2024_25.jsonl")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--sleep-min", type=float, default=1.0)
    ap.add_argument("--sleep-max", type=float, default=1.2)
    args = ap.parse_args()

    done_ids = load_existing_ids(args.out)
    print(f"[resume] {len(done_ids)} papers already done, skipping them", flush=True)

    papers = load_papers(args.csv, args.db, done_ids)
    total = len(papers)
    print(f"[start] {total} papers left to fetch", flush=True)

    no_title = sum(1 for p in papers if not p["title"])
    if no_title:
        print(f"[warn] {no_title} papers have no title in papers table", flush=True)

    if total == 0:
        print("S2_CITATIONS_DONE", flush=True)
        return

    lock = threading.Lock()
    counter = [0]

    # Round-robin split across workers so both threads make steady progress.
    n_workers = max(1, args.workers)
    shards = [papers[i::n_workers] for i in range(n_workers)]

    threads = []
    for shard in shards:
        w = Worker(shard, args.out, args.sleep_min, args.sleep_max, lock, counter, total)
        t = threading.Thread(target=w.run)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"[done] wrote {counter[0]} records this run", flush=True)
    print("S2_CITATIONS_DONE", flush=True)


if __name__ == "__main__":
    main()
