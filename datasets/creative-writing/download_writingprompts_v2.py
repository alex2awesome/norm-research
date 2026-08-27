#!/usr/bin/env python3
"""
Download r/WritingPrompts stories from Arctic Shift API.
Paginates backwards from present using 'before' param on created_utc.
Filters to top-level story comments (parent_id starts with t3_).
"""
import csv
import gzip
import json
import time
import requests
from pathlib import Path

API_URL = "https://arctic-shift.photon-reddit.com/api/comments/search"
OUT_DIR = Path("/lfs/skampere3/0/alexspan/norm-research/datasets/creative-writing")
COMMENTS_FILE = OUT_DIR / "writingprompts_comments.jsonl.gz"
TARGET = 500_000  # download plenty, filter later
LIMIT = 100  # max per request
MIN_BODY_LEN = 200  # filter short non-story comments
SLEEP = 0.5  # rate limit

def download_comments():
    """Download comments paginating backwards from now."""
    before = int(time.time())
    total = 0
    kept = 0
    errors = 0

    with gzip.open(COMMENTS_FILE, "wt") as f:
        while kept < TARGET:
            try:
                r = requests.get(API_URL, params={
                    "subreddit": "WritingPrompts",
                    "limit": LIMIT,
                    "before": before,
                }, timeout=30)
                r.raise_for_status()
                data = r.json()["data"]

                if not data:
                    print(f"No more data at before={before}. Stopping.")
                    break

                for c in data:
                    total += 1
                    body = c.get("body", "")
                    author = c.get("author", "")
                    parent_id = c.get("parent_id", "")

                    # Filter: top-level comments only (parent is a post)
                    if not parent_id.startswith("t3_"):
                        continue
                    # Filter: skip bots
                    if author in ("AutoModerator", "[deleted]", "WritingPromptsRobot"):
                        continue
                    # Filter: skip very short (not real stories)
                    if len(body) < MIN_BODY_LEN:
                        continue
                    # Filter: skip deleted/removed
                    if body in ("[deleted]", "[removed]"):
                        continue

                    record = {
                        "body": body,
                        "score": c.get("score", 0),
                        "link_id": c.get("link_id", ""),
                        "created_utc": c.get("created_utc", 0),
                        "author": author,
                    }
                    f.write(json.dumps(record) + "\n")
                    kept += 1

                # Paginate: use earliest created_utc from this batch
                before = min(c["created_utc"] for c in data)
                errors = 0  # reset error counter on success

                if kept % 1000 < LIMIT:
                    print(f"  Downloaded {kept:,} stories ({total:,} total comments scanned), before={before}")

                time.sleep(SLEEP)

            except Exception as e:
                errors += 1
                print(f"  Error (attempt {errors}): {e}")
                if errors > 10:
                    print("Too many consecutive errors. Stopping.")
                    break
                time.sleep(2 ** errors)  # exponential backoff

    print(f"\nDone! Kept {kept:,} stories from {total:,} total comments")
    print(f"Saved to {COMMENTS_FILE}")
    return kept


def download_submissions():
    """Download submission posts (prompts) to pair with stories."""
    POSTS_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"
    POSTS_FILE = OUT_DIR / "writingprompts_submissions.jsonl.gz"

    before = int(time.time())
    total = 0
    errors = 0

    print("\nDownloading submissions (prompts)...")
    with gzip.open(POSTS_FILE, "wt") as f:
        while True:
            try:
                r = requests.get(POSTS_URL, params={
                    "subreddit": "WritingPrompts",
                    "limit": LIMIT,
                    "before": before,
                }, timeout=30)
                r.raise_for_status()
                data = r.json()["data"]

                if not data:
                    print(f"No more submissions at before={before}. Stopping.")
                    break

                for p in data:
                    record = {
                        "id": "t3_" + p.get("id", ""),
                        "title": p.get("title", ""),
                        "score": p.get("score", 0),
                        "created_utc": p.get("created_utc", 0),
                        "num_comments": p.get("num_comments", 0),
                    }
                    f.write(json.dumps(record) + "\n")
                    total += 1

                before = min(p["created_utc"] for p in data)
                errors = 0

                if total % 5000 < LIMIT:
                    print(f"  Downloaded {total:,} submissions, before={before}")

                time.sleep(SLEEP)

            except Exception as e:
                errors += 1
                print(f"  Error (attempt {errors}): {e}")
                if errors > 10:
                    break
                time.sleep(2 ** errors)

    print(f"Done! {total:,} submissions saved to {POSTS_FILE}")


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1: Download comments (stories)")
    print("=" * 60)
    n = download_comments()

    print("\n" + "=" * 60)
    print("STEP 2: Download submissions (prompts)")
    print("=" * 60)
    download_submissions()
