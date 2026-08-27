#!/usr/bin/env python3
"""
Download r/WritingPrompts data from Arctic Shift API to build a creative writing
quality dataset with ~200K+ story comments.

Arctic Shift API: https://arctic-shift.photon-reddit.com/api/
- comments/search?subreddit=WritingPrompts&limit=100
- posts/search?subreddit=WritingPrompts&limit=100

Pagination is date-based using after/before params with sort=asc.

Usage:
    python download_writingprompts.py [--resume]

Output:
    writingprompts_comments.jsonl.gz  — raw comments
    writingprompts_posts.jsonl.gz     — raw posts (prompts)
    writingprompts_raw.csv.gz         — joined dataset
"""

import argparse
import gzip
import json
import os
import sys
import time
from collections import defaultdict

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── Config ──────────────────────────────────────────────────────────────────
API_BASE = "https://arctic-shift.photon-reddit.com/api"
SUBREDDIT = "WritingPrompts"
LIMIT = 100  # max per request
MIN_BODY_LEN = 200  # filter short comments
RATE_LIMIT_SEC = 1.1  # seconds between requests
REQUEST_TIMEOUT = 120  # seconds per request

# Bot/AutoMod authors to exclude
BOT_AUTHORS = {
    "AutoModerator", "[deleted]", "WritingPromptsRobot",
    "RemindMeBot", "sneakpeekbot", "WikiTextBot",
    "HelperBot_", "CommonMisspellingBot", "FriendlySocietyBot",
}

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
COMMENTS_FILE = os.path.join(OUT_DIR, "writingprompts_comments.jsonl.gz")
POSTS_FILE = os.path.join(OUT_DIR, "writingprompts_posts.jsonl.gz")
OUTPUT_FILE = os.path.join(OUT_DIR, "writingprompts_raw.csv.gz")


# ── HTTP Session ────────────────────────────────────────────────────────────
def make_session():
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "WritingPromptsDatasetBuilder/1.0 (academic research)"
    })
    return session


# ── Download comments ───────────────────────────────────────────────────────
def download_comments(session, resume=False):
    """
    Download all top-level comments from r/WritingPrompts.
    Uses created_utc-based pagination: sort ascending, use last timestamp as 'after'.
    """
    # Resume from last timestamp if file exists
    last_utc = 0
    existing_count = 0
    if resume and os.path.exists(COMMENTS_FILE):
        print(f"Resuming from existing file: {COMMENTS_FILE}")
        with gzip.open(COMMENTS_FILE, "rt") as f:
            for line in f:
                existing_count += 1
                obj = json.loads(line)
                ts = obj.get("created_utc", 0)
                if ts > last_utc:
                    last_utc = ts
        print(f"  Found {existing_count} existing comments, last_utc={last_utc}")

    total = existing_count
    mode = "ab" if resume and os.path.exists(COMMENTS_FILE) else "wb"
    fout = gzip.open(COMMENTS_FILE, mode)

    after_utc = last_utc
    empty_pages = 0
    max_empty = 10  # stop after 10 consecutive empty pages

    print(f"Starting comment download from utc={after_utc}...")

    try:
        while True:
            params = {
                "subreddit": SUBREDDIT,
                "limit": LIMIT,
                "sort": "asc",
                "parent_id": "",  # top-level comments only (server-side filter)
            }
            # Only add 'after' if we have a valid start point
            if after_utc > 0:
                after_utc_str = time.strftime(
                    "%Y-%m-%dT%H:%M:%S", time.gmtime(after_utc)
                )
                params["after"] = after_utc_str

            try:
                resp = session.get(
                    f"{API_BASE}/comments/search",
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json().get("data", [])
            except Exception as e:
                print(f"  Error at utc={after_utc}: {e}")
                time.sleep(5)
                continue

            if not data:
                empty_pages += 1
                if empty_pages >= max_empty:
                    print(f"  {max_empty} consecutive empty pages, stopping.")
                    break
                # Jump forward by 1 day
                after_utc += 86400
                time.sleep(RATE_LIMIT_SEC)
                continue

            empty_pages = 0

            for comment in data:
                author = comment.get("author", "")
                body = comment.get("body", "")
                parent_id = comment.get("parent_id", "")
                created_utc = comment.get("created_utc", 0)

                # Filter: skip bots
                if author in BOT_AUTHORS:
                    continue

                # Filter: skip very short (not actual stories)
                if len(body) < MIN_BODY_LEN:
                    continue

                # Filter: skip deleted/removed
                if body in ("[deleted]", "[removed]"):
                    continue

                record = {
                    "id": comment.get("id", ""),
                    "author": author,
                    "body": body,
                    "score": comment.get("score", 0),
                    "link_id": comment.get("link_id", ""),
                    "parent_id": parent_id,
                    "created_utc": created_utc,
                    "permalink": comment.get("permalink", ""),
                }

                fout.write((json.dumps(record) + "\n").encode("utf-8"))
                total += 1

            # Update pagination cursor
            last_ts = max(c.get("created_utc", 0) for c in data)
            if last_ts <= after_utc:
                # No progress — nudge forward
                after_utc += 1
            else:
                after_utc = last_ts

            if total % 1000 < LIMIT:
                ts_str = time.strftime("%Y-%m-%d", time.gmtime(after_utc))
                print(f"  Comments: {total:,} | date: {ts_str} | utc: {after_utc}")

            time.sleep(RATE_LIMIT_SEC)

    finally:
        fout.close()

    print(f"\nTotal comments saved: {total:,}")
    return total


# ── Download posts (prompts) ───────────────────────────────────────────────
def download_posts(session, needed_post_ids=None, resume=False):
    """
    Download posts from r/WritingPrompts.
    If needed_post_ids is given, only fetch those (by batched ID lookup).
    Otherwise, download all posts via date pagination.
    """
    if needed_post_ids:
        return download_posts_by_ids(session, needed_post_ids)
    else:
        return download_posts_paginated(session, resume=resume)


def download_posts_paginated(session, resume=False):
    """Download all posts via date-based pagination."""
    last_utc = 0
    existing_count = 0
    if resume and os.path.exists(POSTS_FILE):
        print(f"Resuming from existing posts file: {POSTS_FILE}")
        with gzip.open(POSTS_FILE, "rt") as f:
            for line in f:
                existing_count += 1
                obj = json.loads(line)
                ts = obj.get("created_utc", 0)
                if ts > last_utc:
                    last_utc = ts
        print(f"  Found {existing_count} existing posts, last_utc={last_utc}")

    total = existing_count
    mode = "ab" if resume and os.path.exists(POSTS_FILE) else "wb"
    fout = gzip.open(POSTS_FILE, mode)

    after_utc = last_utc
    empty_pages = 0
    max_empty = 10

    print(f"Starting post download from utc={after_utc}...")

    try:
        while True:
            params = {
                "subreddit": SUBREDDIT,
                "limit": LIMIT,
                "sort": "asc",
            }
            if after_utc > 0:
                after_utc_str = time.strftime(
                    "%Y-%m-%dT%H:%M:%S", time.gmtime(after_utc)
                )
                params["after"] = after_utc_str

            try:
                resp = session.get(
                    f"{API_BASE}/posts/search",
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json().get("data", [])
            except Exception as e:
                print(f"  Error at utc={after_utc}: {e}")
                time.sleep(5)
                continue

            if not data:
                empty_pages += 1
                if empty_pages >= max_empty:
                    print(f"  {max_empty} consecutive empty pages, stopping.")
                    break
                after_utc += 86400
                time.sleep(RATE_LIMIT_SEC)
                continue

            empty_pages = 0

            for post in data:
                record = {
                    "id": post.get("id", ""),
                    "title": post.get("title", ""),
                    "selftext": post.get("selftext", ""),
                    "author": post.get("author", ""),
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "created_utc": post.get("created_utc", 0),
                    "permalink": post.get("permalink", ""),
                }
                fout.write((json.dumps(record) + "\n").encode("utf-8"))
                total += 1

            last_ts = max(p.get("created_utc", 0) for p in data)
            if last_ts <= after_utc:
                after_utc += 1
            else:
                after_utc = last_ts

            if total % 1000 < LIMIT:
                ts_str = time.strftime("%Y-%m-%d", time.gmtime(after_utc))
                print(f"  Posts: {total:,} | date: {ts_str} | utc: {after_utc}")

            time.sleep(RATE_LIMIT_SEC)

    finally:
        fout.close()

    print(f"\nTotal posts saved: {total:,}")
    return total


def download_posts_by_ids(session, post_ids):
    """Download posts by batched ID lookup."""
    print(f"Downloading {len(post_ids):,} posts by ID...")

    # Load existing posts to avoid re-downloading
    existing_ids = set()
    if os.path.exists(POSTS_FILE):
        with gzip.open(POSTS_FILE, "rt") as f:
            for line in f:
                obj = json.loads(line)
                existing_ids.add(obj.get("id", ""))
        print(f"  {len(existing_ids):,} posts already downloaded")

    needed = [pid for pid in post_ids if pid not in existing_ids]
    print(f"  {len(needed):,} posts still needed")

    if not needed:
        return len(existing_ids)

    fout = gzip.open(POSTS_FILE, "ab" if existing_ids else "wb")
    total = len(existing_ids)

    try:
        # Batch by 100 IDs at a time
        batch_size = 100
        for i in range(0, len(needed), batch_size):
            batch = needed[i:i + batch_size]
            ids_str = ",".join(batch)

            try:
                resp = session.get(
                    f"{API_BASE}/posts/ids",
                    params={"ids": ids_str},
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json().get("data", [])
            except Exception as e:
                print(f"  Error fetching batch {i//batch_size}: {e}")
                time.sleep(5)
                continue

            for post in data:
                record = {
                    "id": post.get("id", ""),
                    "title": post.get("title", ""),
                    "selftext": post.get("selftext", ""),
                    "author": post.get("author", ""),
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "created_utc": post.get("created_utc", 0),
                    "permalink": post.get("permalink", ""),
                }
                fout.write((json.dumps(record) + "\n").encode("utf-8"))
                total += 1

            if (i // batch_size) % 10 == 0:
                print(f"  Posts: {total:,} / {len(post_ids):,}")

            time.sleep(RATE_LIMIT_SEC)

    finally:
        fout.close()

    print(f"\nTotal posts saved: {total:,}")
    return total


# ── Build joined dataset ───────────────────────────────────────────────────
def build_dataset():
    """Join comments with posts to create the final dataset."""
    print("\n=== Building joined dataset ===")

    # Load posts
    print("Loading posts...")
    posts = {}
    with gzip.open(POSTS_FILE, "rt") as f:
        for line in f:
            p = json.loads(line)
            posts[p["id"]] = p
    print(f"  Loaded {len(posts):,} posts")

    # Load comments and join
    print("Loading comments and joining with posts...")
    records = []
    missing_posts = 0
    with gzip.open(COMMENTS_FILE, "rt") as f:
        for line in f:
            c = json.loads(line)
            # link_id is "t3_XXXXX" — strip prefix to get post id
            post_id = c["link_id"].replace("t3_", "")
            post = posts.get(post_id)

            if post is None:
                missing_posts += 1
                prompt_title = ""
            else:
                prompt_title = post.get("title", "")

            # Build the modeling text
            text = f"PROMPT: {prompt_title}\n\nSTORY: {c['body']}"

            records.append({
                "comment_id": c["id"],
                "post_id": post_id,
                "prompt": prompt_title,
                "story": c["body"],
                "text": text,
                "score": c["score"],
                "author": c["author"],
                "created_utc": c["created_utc"],
                "story_len": len(c["body"]),
                "post_score": post.get("score", 0) if post else 0,
            })

    print(f"  {len(records):,} joined records ({missing_posts:,} missing posts)")

    # Deduplicate by comment_id
    df = pd.DataFrame(records)
    before = len(df)
    df = df.drop_duplicates(subset="comment_id", keep="first")
    print(f"  Deduplicated: {before:,} → {len(df):,}")

    # Save
    df.to_csv(OUTPUT_FILE, index=False, compression="gzip")
    print(f"\nSaved to: {OUTPUT_FILE}")
    print(f"Total examples: {len(df):,}")

    # Print stats
    print("\n=== Dataset Statistics ===")
    print(f"  Score distribution:")
    print(f"    min={df['score'].min()}, median={df['score'].median():.0f}, "
          f"mean={df['score'].mean():.1f}, max={df['score'].max()}")
    print(f"  Story length (chars):")
    print(f"    min={df['story_len'].min()}, median={df['story_len'].median():.0f}, "
          f"mean={df['story_len'].mean():.0f}, max={df['story_len'].max()}")
    print(f"  Unique authors: {df['author'].nunique():,}")
    print(f"  Unique prompts: {df['post_id'].nunique():,}")
    print(f"  Date range: {pd.to_datetime(df['created_utc'].min(), unit='s').date()} "
          f"to {pd.to_datetime(df['created_utc'].max(), unit='s').date()}")

    return df


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download r/WritingPrompts data from Arctic Shift"
    )
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--comments-only", action="store_true",
                        help="Only download comments")
    parser.add_argument("--posts-only", action="store_true",
                        help="Only download posts")
    parser.add_argument("--build-only", action="store_true",
                        help="Only build dataset from existing downloads")
    parser.add_argument("--posts-by-id", action="store_true",
                        help="Download posts by ID (from comments link_ids)")
    args = parser.parse_args()

    session = make_session()

    if args.build_only:
        build_dataset()
        return

    if not args.posts_only:
        print("=" * 60)
        print("STEP 1: Download comments (stories)")
        print("=" * 60)
        download_comments(session, resume=args.resume)

    if not args.comments_only:
        if args.posts_by_id:
            # Collect needed post IDs from comments
            print("\n" + "=" * 60)
            print("STEP 2: Download posts by ID (from comments)")
            print("=" * 60)
            needed_ids = set()
            with gzip.open(COMMENTS_FILE, "rt") as f:
                for line in f:
                    c = json.loads(line)
                    pid = c["link_id"].replace("t3_", "")
                    needed_ids.add(pid)
            download_posts(session, needed_post_ids=list(needed_ids))
        else:
            print("\n" + "=" * 60)
            print("STEP 2: Download posts (prompts)")
            print("=" * 60)
            download_posts(session, resume=args.resume)

    print("\n" + "=" * 60)
    print("STEP 3: Build joined dataset")
    print("=" * 60)
    build_dataset()


if __name__ == "__main__":
    main()
