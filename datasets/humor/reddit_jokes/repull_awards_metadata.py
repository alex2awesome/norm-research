#!/usr/bin/env python3
"""
Re-pull award/mod metadata for the 1M r/Jokes posts in reddit_jokes_1m.csv.gz.

Our original crawl kept only (id, permalink, title, selftext, score); this
fetches the full post objects from Arctic Shift by id (500/request) and keeps
the award-tier fields: gilded, gildings, total_awards_received, all_awardings,
plus mod fields (stickied, distinguished, link_flair_text) and a fresh score.

Output (append + dedup on resume; never overwrites):
    jokes_awards_metadata.jsonl.gz   one JSON object per post id
    repull_state.json                index of next batch

Usage:  python repull_awards_metadata.py
"""

import gzip
import json
import os
import time

import pandas as pd
import requests

API = "https://arctic-shift.photon-reddit.com/api/posts/ids"
HEADERS = {"User-Agent": "norm-research/0.1 (academic; contact alex2awesome@gmail.com)"}
BATCH = 500
RATE_LIMIT_SEC = 1.1
KEEP = [
    "id", "created_utc", "score", "ups", "upvote_ratio", "num_comments",
    "gilded", "gildings", "total_awards_received", "all_awardings",
    "stickied", "distinguished", "link_flair_text", "over_18",
    "retrieved_on", "retrieved_utc", "author",
]

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "reddit_jokes_1m.csv.gz")
OUT = os.path.join(HERE, "jokes_awards_metadata.jsonl.gz")
STATE = os.path.join(HERE, "repull_state.json")


def slim(post):
    row = {k: post.get(k) for k in KEEP}
    # all_awardings is bulky; keep only name/count/coin_price per award
    if row.get("all_awardings"):
        row["all_awardings"] = [
            {k: a.get(k) for k in ("name", "count", "coin_price")}
            for a in row["all_awardings"]
        ]
    return row


def main():
    # NOTE: file is named .csv.gz but is plain CSV (magic bytes 'ty')
    ids = pd.read_csv(SRC, usecols=["id"], compression=None)["id"].tolist()
    start = 0
    if os.path.exists(STATE):
        start = json.load(open(STATE))["next_batch"]
    n_batches = (len(ids) + BATCH - 1) // BATCH
    print(f"{len(ids)} ids, {n_batches} batches, resuming at batch {start}")

    sess = requests.Session()
    sess.headers.update(HEADERS)
    mode = "ab" if start > 0 and os.path.exists(OUT) else "wb"
    with gzip.open(OUT, mode) as fh:
        for b in range(start, n_batches):
            chunk = ids[b * BATCH:(b + 1) * BATCH]
            for attempt in range(6):
                try:
                    r = sess.get(API, params={"ids": ",".join(chunk)}, timeout=120)
                    r.raise_for_status()
                    data = r.json()["data"]
                    break
                except Exception as e:
                    wait = 2 ** attempt * 5
                    print(f"batch {b} attempt {attempt}: {e} — sleep {wait}s", flush=True)
                    time.sleep(wait)
            else:
                raise RuntimeError(f"batch {b} failed after retries")
            for post in data:
                fh.write((json.dumps(slim(post)) + "\n").encode())
            json.dump({"next_batch": b + 1}, open(STATE, "w"))
            if b % 50 == 0:
                print(f"batch {b}/{n_batches} ({len(data)} returned)", flush=True)
            time.sleep(RATE_LIMIT_SEC)
    print("done")


if __name__ == "__main__":
    main()
