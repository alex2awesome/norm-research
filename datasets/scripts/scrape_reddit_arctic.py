#!/usr/bin/env python3
"""
Generic Arctic Shift Reddit scraper.

- Preserves ALL fields including parent_id, link_id (so comment trees are reconstructable).
- No top-level-only filter: pulls comments AND replies-to-replies.
- Per-subreddit pagination via 'after' on created_utc, sort=asc.
- Output: gzipped JSONL sharded by month: {output_dir}/raw/{sub}_{kind}_{YYYYMM}.jsonl.gz
- Resume via {output_dir}/raw/.cursor_{sub}_{kind} (stores last successful created_utc).
- Periodic .done.{YYYYMM} sentinel for resume.
- Honors 429 with exponential backoff.

Usage:
    python scrape_reddit_arctic.py \
        --subreddit WritingPrompts \
        --output-dir /lfs/skampere3/0/alexspan/data/creative_writing/comments \
        --start-ts 2010-01-01 \
        --end-ts now \
        --kind comments \
        --delay 1.0
"""
import argparse
import gzip
import json
import os
import sys
import time
import random
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_BASE = "https://arctic-shift.photon-reddit.com/api"
LIMIT = 100
REQUEST_TIMEOUT = 120


def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def make_session():
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504],  # 429 handled manually
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({
        "User-Agent": "RedditArcticScraper/1.0 (academic research; spangher@usc.edu)"
    })
    return s


def parse_ts(s):
    """Parse a timestamp argument. Accepts 'now', int seconds, or ISO date YYYY-MM-DD[THH:MM:SS]."""
    if s is None or s == "now":
        return int(time.time())
    try:
        return int(s)
    except ValueError:
        pass
    fmts = ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d")
    for f in fmts:
        try:
            dt = datetime.strptime(s, f).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {s!r}")


def shard_path(output_dir, subreddit, kind, created_utc):
    yyyymm = datetime.fromtimestamp(created_utc, tz=timezone.utc).strftime("%Y%m")
    return Path(output_dir) / "raw" / f"{subreddit}_{kind}_{yyyymm}.jsonl.gz"


def cursor_path(output_dir, subreddit, kind):
    return Path(output_dir) / "raw" / f".cursor_{subreddit}_{kind}"


def done_sentinel(output_dir, subreddit, kind, yyyymm):
    return Path(output_dir) / "raw" / f".done.{subreddit}_{kind}_{yyyymm}"


def load_cursor(output_dir, subreddit, kind, default):
    p = cursor_path(output_dir, subreddit, kind)
    if p.exists():
        try:
            v = int(p.read_text().strip())
            log(f"  resume: cursor={v} ({datetime.fromtimestamp(v, tz=timezone.utc).isoformat()})")
            return v
        except Exception as e:
            log(f"  cursor unreadable ({e}); starting from default={default}")
    return default


def save_cursor(output_dir, subreddit, kind, ts):
    p = cursor_path(output_dir, subreddit, kind)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(str(int(ts)))
    tmp.replace(p)


class ShardWriter:
    """Append-mode gzip writer with per-month sharding."""
    def __init__(self, output_dir, subreddit, kind):
        self.output_dir = output_dir
        self.subreddit = subreddit
        self.kind = kind
        self.current_yyyymm = None
        self.handle = None
        self.path = None

    def write(self, record):
        ts = record.get("created_utc", 0)
        yyyymm = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m")
        if yyyymm != self.current_yyyymm:
            self._rotate(yyyymm)
        self.handle.write((json.dumps(record) + "\n").encode("utf-8"))

    def _rotate(self, yyyymm):
        if self.handle:
            # mark prior month done
            prev = self.current_yyyymm
            self.close()
            if prev:
                done_sentinel(self.output_dir, self.subreddit, self.kind, prev).touch()
        self.current_yyyymm = yyyymm
        self.path = shard_path(self.output_dir, self.subreddit, self.kind, int(datetime.strptime(yyyymm, "%Y%m").replace(tzinfo=timezone.utc).timestamp()))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = gzip.open(self.path, "ab")
        log(f"  opened shard {self.path.name}")

    def flush(self):
        if self.handle:
            self.handle.flush()

    def close(self):
        if self.handle:
            self.handle.close()
            self.handle = None


def fetch_with_backoff(session, url, params, delay):
    """GET with manual 429 handling + exponential backoff. Returns parsed JSON dict."""
    attempt = 0
    while True:
        try:
            resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                attempt += 1
                wait = min(60 * 5, (2 ** attempt) + random.uniform(0, 1))
                log(f"  429 rate-limited, sleeping {wait:.1f}s (attempt {attempt})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            attempt += 1
            wait = min(60 * 5, (2 ** attempt) + random.uniform(0, 1))
            log(f"  request error: {e} — sleeping {wait:.1f}s (attempt {attempt})")
            if attempt > 10:
                log(f"  too many failures, returning empty")
                return {"data": []}
            time.sleep(wait)


def fetch_kind(session, subreddit, kind, start_ts, end_ts, output_dir, delay, flush_every):
    """Fetch one kind ('comments' or 'posts') for one subreddit."""
    assert kind in ("comments", "posts")
    endpoint = f"{API_BASE}/{kind}/search"

    after_utc = load_cursor(output_dir, subreddit, kind, default=start_ts)
    writer = ShardWriter(output_dir, subreddit, kind)
    total = 0
    empty_pages = 0
    max_empty = 30  # nudge forward up to 30 days before stopping if no data
    last_cursor_save = 0

    log(f"=== {subreddit}/{kind}: starting from {after_utc} ({datetime.fromtimestamp(after_utc, tz=timezone.utc).isoformat()}) → end {end_ts}")

    try:
        while after_utc < end_ts:
            params = {
                "subreddit": subreddit,
                "limit": LIMIT,
                "sort": "asc",
                "after": after_utc,
            }
            data = fetch_with_backoff(session, endpoint, params, delay).get("data", [])

            if not data:
                empty_pages += 1
                if empty_pages >= max_empty:
                    log(f"  {max_empty} consecutive empty pages at {after_utc}, stopping {subreddit}/{kind}.")
                    break
                # jump forward 1 day
                after_utc += 86400
                time.sleep(delay)
                continue

            empty_pages = 0

            max_ts = after_utc
            for rec in data:
                ts = rec.get("created_utc", 0)
                if ts > end_ts:
                    continue
                # Preserve ALL fields. (We persist the whole record verbatim.)
                writer.write(rec)
                total += 1
                if ts > max_ts:
                    max_ts = ts

            # Advance cursor strictly
            if max_ts <= after_utc:
                after_utc += 1
            else:
                after_utc = max_ts

            # Periodic flush + cursor save
            if total - last_cursor_save >= flush_every:
                writer.flush()
                save_cursor(output_dir, subreddit, kind, after_utc)
                last_cursor_save = total
                ts_str = datetime.fromtimestamp(after_utc, tz=timezone.utc).strftime("%Y-%m-%d")
                log(f"  {subreddit}/{kind}: fetched {total:,} records | cursor {ts_str}")

            time.sleep(delay)
    finally:
        writer.close()
        save_cursor(output_dir, subreddit, kind, after_utc)

    log(f"=== {subreddit}/{kind}: DONE. total fetched this run: {total:,}")
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subreddit", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--start-ts", default="2010-01-01",
                   help="UTC start (YYYY-MM-DD, ISO, int seconds, or 'now')")
    p.add_argument("--end-ts", default="now",
                   help="UTC end (YYYY-MM-DD, ISO, int seconds, or 'now')")
    p.add_argument("--workers", type=int, default=1,
                   help="Reserved; current impl is single-threaded per process.")
    p.add_argument("--kind", choices=["comments", "posts", "both"], default="comments")
    p.add_argument("--delay", type=float, default=1.0,
                   help="Seconds between requests.")
    p.add_argument("--flush-every", type=int, default=200,
                   help="Flush + cursor save every N records.")
    args = p.parse_args()

    start_ts = parse_ts(args.start_ts)
    end_ts = parse_ts(args.end_ts)
    Path(args.output_dir, "raw").mkdir(parents=True, exist_ok=True)

    session = make_session()

    kinds = ["comments", "posts"] if args.kind == "both" else [args.kind]
    grand_total = 0
    for kind in kinds:
        grand_total += fetch_kind(
            session=session,
            subreddit=args.subreddit,
            kind=kind,
            start_ts=start_ts,
            end_ts=end_ts,
            output_dir=args.output_dir,
            delay=args.delay,
            flush_every=args.flush_every,
        )

    log(f"ALL DONE: {grand_total:,} total records across {kinds}")


if __name__ == "__main__":
    main()
