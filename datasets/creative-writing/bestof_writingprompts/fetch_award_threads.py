#!/usr/bin/env python3
"""
Fetch the annual "Best of r/WritingPrompts" award threads (announcements,
nomination threads, winners posts) + their full comment trees via Arctic Shift.

Discovery = title search over r/WritingPrompts for best-of patterns, restricted
to meta flairs by title tag ([OT]/[MODPOST]/[CW]); then comments/search?link_id
per thread. Winners/nominees live in the post selftext and mod comments.

Output:
    award_threads/threads.jsonl.gz         post objects
    award_threads/thread_comments.jsonl.gz comments per thread

Usage:  python fetch_award_threads.py
"""

import gzip
import json
import os
import time

import requests

API = "https://arctic-shift.photon-reddit.com/api"
HEADERS = {"User-Agent": "norm-research/0.1 (academic; contact alex2awesome@gmail.com)"}
RATE = 1.1

TITLE_QUERIES = [
    "best of",           # catches announcement + winners + nomination threads
    "spotlight",         # mod spotlight features, if any
]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "award_threads")


def is_meta_award(post):
    t = post.get("title", "").lower()
    if "best of" not in t and "spotlight" not in t:
        return False
    return t.startswith(("[ot]", "[modpost]", "[cw]", "[meta]")) or (
        post.get("link_flair_text") or "").lower() in ("off topic", "mod post", "modpost")


def get(sess, path, **params):
    if params.get("after") == 0:
        del params["after"]
    for attempt in range(6):
        try:
            r = sess.get(f"{API}/{path}", params=params, timeout=120)
            r.raise_for_status()
            return r.json()["data"]
        except Exception as e:
            wait = 2 ** attempt * 5
            print(f"{path} {params} attempt {attempt}: {e} — sleep {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed: {path} {params}")


def main():
    os.makedirs(OUT, exist_ok=True)
    sess = requests.Session()
    sess.headers.update(HEADERS)

    threads = {}
    for q in TITLE_QUERIES:
        after = 0
        while True:
            data = get(sess, "posts/search", subreddit="WritingPrompts",
                       title=q, limit=100, sort="asc", after=after)
            if not data:
                break
            for p in data:
                if is_meta_award(p):
                    threads[p["id"]] = p
            after = max(p["created_utc"] for p in data) + 1
            time.sleep(RATE)
        print(f"query '{q}': {len(threads)} award threads so far", flush=True)

    with gzip.open(os.path.join(OUT, "threads.jsonl.gz"), "wb") as fh:
        for p in threads.values():
            fh.write((json.dumps(p) + "\n").encode())

    with gzip.open(os.path.join(OUT, "thread_comments.jsonl.gz"), "wb") as fh:
        for i, pid in enumerate(sorted(threads)):
            after = 0
            while True:
                data = get(sess, "comments/search", link_id=f"t3_{pid}",
                           limit=100, sort="asc", after=after)
                if not data:
                    break
                for c in data:
                    fh.write((json.dumps(c) + "\n").encode())
                after = max(c["created_utc"] for c in data) + 1
                time.sleep(RATE)
            if i % 10 == 0:
                print(f"comments: thread {i}/{len(threads)}", flush=True)
    print(f"done: {len(threads)} threads")


if __name__ == "__main__":
    main()
