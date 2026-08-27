#!/usr/bin/env python3
"""
Full scrape of r/bestofWritingPrompts via Arctic Shift (curation signal:
curators repost standout r/WritingPrompts stories; each post's `url` links the
original WP comment/post — the join key back to our WritingPrompts pool).

Date-paginated with sort=asc like download_writingprompts.py. Small sub
(active since 2014-05), so this finishes in minutes.

Output:
    raw/bestofwp_posts.jsonl.gz      all submissions (full objects)
    raw/bestofwp_comments.jsonl.gz   all comments (full objects)
    scrape_state.json                resume cursor per kind

Usage:  python scrape_bestofwp.py
"""

import gzip
import json
import os
import time

import requests

API = "https://arctic-shift.photon-reddit.com/api"
SUB = "bestofWritingPrompts"
HEADERS = {"User-Agent": "norm-research/0.1 (academic; contact alex2awesome@gmail.com)"}
LIMIT = 100
RATE_LIMIT_SEC = 1.1

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
STATE = os.path.join(HERE, "scrape_state.json")


def crawl(kind, out_path, after):
    seen_last = after
    n = 0
    sess = requests.Session()
    sess.headers.update(HEADERS)
    mode = "ab" if after > 0 and os.path.exists(out_path) else "wb"
    with gzip.open(out_path, mode) as fh:
        while True:
            params = {"subreddit": SUB, "limit": LIMIT, "sort": "asc"}
            if seen_last > 0:
                params["after"] = seen_last
            for attempt in range(6):
                try:
                    r = sess.get(f"{API}/{kind}/search", params=params, timeout=120)
                    r.raise_for_status()
                    data = r.json()["data"]
                    break
                except Exception as e:
                    wait = 2 ** attempt * 5
                    print(f"{kind} after={seen_last} attempt {attempt}: {e} — sleep {wait}s", flush=True)
                    time.sleep(wait)
            else:
                raise RuntimeError(f"{kind} stalled at after={seen_last}")
            if not data:
                break
            for row in data:
                fh.write((json.dumps(row) + "\n").encode())
            n += len(data)
            new_last = max(row["created_utc"] for row in data)
            if new_last == seen_last:  # safety: same-second cluster; bump
                new_last += 1
            seen_last = new_last
            state = json.load(open(STATE)) if os.path.exists(STATE) else {}
            state[kind] = seen_last
            json.dump(state, open(STATE, "w"))
            print(f"{kind}: +{len(data)} (total {n}), cursor {seen_last}", flush=True)
            time.sleep(RATE_LIMIT_SEC)
    print(f"{kind}: done, {n} new rows")


def main():
    os.makedirs(RAW, exist_ok=True)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    crawl("posts", os.path.join(RAW, "bestofwp_posts.jsonl.gz"), state.get("posts", 0))
    crawl("comments", os.path.join(RAW, "bestofwp_comments.jsonl.gz"), state.get("comments", 0))


if __name__ == "__main__":
    main()
