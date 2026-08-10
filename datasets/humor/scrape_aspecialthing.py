#!/usr/bin/env python3
"""
Wayback Machine scraper for A Special Thing comedy forum (aspecialthing.com/forum).

Three stages:
  --stage cdx    : enumerate Wayback CDX captures, keep earliest 200-OK per URL
  --stage fetch  : bulk fetch raw HTML (with `if_` suffix) at <=2 req/s, resumable
  --stage parse  : extract posts via vBulletin post_message regex

Targets:
  threads:        showthread.php?t=NNN
  posts:          NNNN-postN.html
  forum sections: forumdisplay.php?f=N
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests

# ----- paths -----
BASE = Path(os.environ.get(
    "ASPT_BASE",
    "/lfs/skampere3/0/alexspan/data/humor/aspecialthing",
))
RAW_DIR = BASE / "raw"
LOG_DIR = BASE / "logs"
PARSED_DIR = BASE / "parsed"
CDX_FILE = RAW_DIR / "cdx_targets.jsonl"
CDX_MISS_FILE = LOG_DIR / "cdx_missing.jsonl"

for d in (RAW_DIR, LOG_DIR, PARSED_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ----- constants -----
CDX_BASE = "http://web.archive.org/cdx/search/cdx"
WB_FETCH = "https://web.archive.org/web/{ts}if_/{url}"
USER_AGENT = "Mozilla/5.0 (research-archive-fetcher; spangher@usc.edu)"
RATE_LIMIT_SLEEP = 0.5   # seconds between Wayback fetches (~2/s)
CDX_PAGE_LIMIT = 50000   # rows per CDX request

THREAD_RE = re.compile(r"showthread\.php\?t=\d+", re.I)
POST_RE = re.compile(r"\d+-post\d+\.html", re.I)
FORUMDISP_RE = re.compile(r"forumdisplay\.php\?f=\d+", re.I)
POST_MSG_RE = re.compile(
    r'<div\s+id="post_message_(\d+)"[^>]*>(.*?)</div>',
    re.IGNORECASE | re.DOTALL,
)
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# =========================
# Stage 1: CDX enumeration
# =========================
def cdx_fetch_page(offset: int, session: requests.Session,
                   url_prefix: str = "aspecialthing.com/forum/") -> list[list[str]]:
    """One CDX page. Returns list of rows (excluding header)."""
    params = {
        "url": url_prefix,
        "matchType": "prefix",
        "filter": ["statuscode:200", "mimetype:text/html"],
        "collapse": "urlkey",  # dedup server-side: one row per unique URL
        "output": "json",
        "fl": "urlkey,timestamp,original,statuscode",
        "limit": str(CDX_PAGE_LIMIT),
        "offset": str(offset),
    }
    for attempt in range(6):
        try:
            r = session.get(CDX_BASE, params=params, timeout=180)
            if r.status_code == 200:
                data = r.json()
                if not data:
                    return []
                return data[1:]  # drop header
            if r.status_code in (429, 503, 504):
                wait = 2 ** attempt + random.random()
                log(f"CDX got {r.status_code}, backoff {wait:.1f}s")
                time.sleep(wait)
                continue
            log(f"CDX unexpected status {r.status_code}: {r.text[:200]}")
            time.sleep(5)
        except (requests.RequestException, ValueError) as e:
            wait = 2 ** attempt + random.random()
            log(f"CDX error: {e}; backoff {wait:.1f}s")
            time.sleep(wait)
    return []


def is_target(url: str) -> str | None:
    """Return the category if the URL is a target, else None."""
    if THREAD_RE.search(url):
        return "thread"
    if POST_RE.search(url):
        return "post"
    if FORUMDISP_RE.search(url):
        return "forumdisplay"
    return None


def _enumerate_prefix(prefix: str, session: requests.Session,
                      earliest: dict[str, tuple[str, str, str]]) -> int:
    """Walk CDX pages for a given URL prefix, merging into `earliest`. Returns rows seen."""
    total_seen = 0
    offset = 0
    last_kept = len(earliest)
    no_new_pages = 0
    NO_NEW_STOP = 6
    HARD_OFFSET_CAP = 2_000_000
    while True:
        log(f"CDX prefix={prefix!r} offset={offset}")
        rows = cdx_fetch_page(offset, session, url_prefix=prefix)
        if not rows:
            log("  0 rows; done with prefix")
            break
        for row in rows:
            if len(row) < 4:
                continue
            _urlkey, ts, original, _status = row[0], row[1], row[2], row[3]
            total_seen += 1
            cat = is_target(original)
            if cat is None:
                continue
            prev = earliest.get(original)
            if prev is None or ts < prev[0]:
                earliest[original] = (ts, original, cat)
        new_this_page = len(earliest) - last_kept
        last_kept = len(earliest)
        log(f"  rows={len(rows)} cum_seen={total_seen} kept={len(earliest)} "
            f"new={new_this_page}")
        if new_this_page == 0:
            no_new_pages += 1
            if no_new_pages >= NO_NEW_STOP:
                log(f"  no new targets in {NO_NEW_STOP} pages; stopping prefix")
                break
        else:
            no_new_pages = 0
        if len(rows) < CDX_PAGE_LIMIT:
            break
        offset += CDX_PAGE_LIMIT
        if offset >= HARD_OFFSET_CAP:
            log(f"  hit HARD_OFFSET_CAP={HARD_OFFSET_CAP}; stopping prefix")
            break
        time.sleep(0.5)
    return total_seen


def stage_cdx() -> None:
    log("Stage 1: CDX enumeration")
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    earliest: dict[str, tuple[str, str, str]] = {}
    grand_total = 0
    # Hit three prefixes — query-string URLs sort lexically after non-query URLs,
    # and `aspecialthing.com/forum/` prefix alone never reaches them within
    # reasonable offsets, so we run them explicitly.
    for prefix in (
        "aspecialthing.com/forum/",                  # post permalinks NNNN-postN.html
        "aspecialthing.com/forum/showthread.php",    # threads
        "aspecialthing.com/forum/forumdisplay.php",  # section listings
    ):
        grand_total += _enumerate_prefix(prefix, session, earliest)

    # write jsonl
    by_cat: dict[str, int] = {}
    with open(CDX_FILE, "w") as fh:
        for url, (ts, orig, cat) in earliest.items():
            by_cat[cat] = by_cat.get(cat, 0) + 1
            fh.write(json.dumps({
                "url": orig,
                "timestamp": ts,
                "category": cat,
            }) + "\n")
    log(f"CDX done: grand_total_seen={grand_total} kept={len(earliest)} "
        f"by_category={by_cat}")
    log(f"Wrote {CDX_FILE}")


# =========================
# Stage 2: bulk fetch
# =========================
def target_path(ts: str, url: str) -> Path:
    h = hashlib.sha1(url.encode()).hexdigest()[:16]
    return RAW_DIR / f"{ts}_{h}.html.gz"


def fetch_one(ts: str, url: str, session: requests.Session) -> tuple[str, int, int]:
    """Returns (status_label, http_status, bytes_written). status_label in {ok, skip, http_err, exc}."""
    out = target_path(ts, url)
    if out.exists() and out.stat().st_size > 0:
        return ("skip", 0, 0)
    wb_url = WB_FETCH.format(ts=ts, url=url)
    for attempt in range(5):
        try:
            r = session.get(wb_url, timeout=60, allow_redirects=True)
            if r.status_code == 200:
                data = r.content
                with gzip.open(out, "wb") as fh:
                    fh.write(data)
                return ("ok", 200, len(data))
            if r.status_code in (429, 503, 504):
                wait = 2 ** attempt + random.random() * 0.5
                time.sleep(wait)
                continue
            # 404, redirect-to-non-html etc. give up cleanly
            return ("http_err", r.status_code, 0)
        except requests.RequestException as e:
            wait = 1.5 ** attempt + random.random() * 0.5
            time.sleep(wait)
    return ("exc", 0, 0)


def stage_fetch(limit: int | None = None) -> None:
    if not CDX_FILE.exists():
        log(f"ERROR: CDX file missing: {CDX_FILE}. Run --stage cdx first.")
        sys.exit(1)
    log("Stage 2: bulk fetch")
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    miss_fh = open(CDX_MISS_FILE, "a")
    n = 0
    n_ok = 0
    n_skip = 0
    n_err = 0
    t0 = time.time()
    with open(CDX_FILE) as fh:
        for line in fh:
            if limit is not None and n >= limit:
                break
            row = json.loads(line)
            ts, url = row["timestamp"], row["url"]
            n += 1
            status, http_status, nbytes = fetch_one(ts, url, session)
            if status == "ok":
                n_ok += 1
                time.sleep(RATE_LIMIT_SLEEP)  # rate limit only on real fetches
            elif status == "skip":
                n_skip += 1
            else:
                n_err += 1
                miss_fh.write(json.dumps({
                    "url": url, "timestamp": ts,
                    "status": status, "http": http_status,
                }) + "\n")
                miss_fh.flush()
                time.sleep(RATE_LIMIT_SLEEP)
            if n % 100 == 0:
                rate = n / max(1.0, time.time() - t0)
                log(f"progress n={n} ok={n_ok} skip={n_skip} err={n_err} "
                    f"rate={rate:.2f}/s last_url={url[:90]}")
    log(f"Fetch done: n={n} ok={n_ok} skip={n_skip} err={n_err}")
    miss_fh.close()


# =========================
# Stage 3: parse
# =========================
def html_strip(html: str) -> str:
    text = TAG_RE.sub(" ", html)
    text = (text.replace("&nbsp;", " ")
                .replace("&amp;", "&")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&quot;", '"')
                .replace("&#39;", "'"))
    return WS_RE.sub(" ", text).strip()


def parse_thread_id_from_url(url: str) -> str | None:
    m = re.search(r"showthread\.php\?t=(\d+)", url)
    if m:
        return m.group(1)
    m = re.search(r"/(\d+)-post(\d+)\.html", url)
    if m:
        return m.group(1)
    return None


def stage_parse() -> None:
    log("Stage 3: parse")
    if not CDX_FILE.exists():
        log("ERROR: CDX file missing.")
        sys.exit(1)
    # index by ts+hash -> meta
    meta: dict[Path, dict] = {}
    with open(CDX_FILE) as fh:
        for line in fh:
            r = json.loads(line)
            meta[target_path(r["timestamp"], r["url"])] = r

    out_path = PARSED_DIR / "threads_normalized.jsonl"
    n_files = 0
    n_posts = 0
    with open(out_path, "w") as out:
        for fpath in RAW_DIR.glob("*.html.gz"):
            n_files += 1
            try:
                with gzip.open(fpath, "rb") as fh:
                    raw = fh.read().decode("utf-8", errors="replace")
            except OSError:
                continue
            m = meta.get(fpath, {})
            source_url = m.get("url", "")
            ts = m.get("timestamp", "")
            cat = m.get("category", "")
            thread_id = parse_thread_id_from_url(source_url)
            title_m = TITLE_RE.search(raw)
            title = html_strip(title_m.group(1)) if title_m else ""
            for pm in POST_MSG_RE.finditer(raw):
                post_id = pm.group(1)
                body = html_strip(pm.group(2))
                if not body:
                    continue
                out.write(json.dumps({
                    "thread_id": thread_id,
                    "post_id": post_id,
                    "title": title,
                    "body": body,
                    "ts": ts,
                    "category": cat,
                    "source_url": source_url,
                }) + "\n")
                n_posts += 1
            if n_files % 500 == 0:
                log(f"parsed files={n_files} posts={n_posts}")
    log(f"Parse done: files={n_files} posts={n_posts} out={out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["cdx", "fetch", "parse"], required=True)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap fetches (for smoke testing)")
    args = p.parse_args()
    if args.stage == "cdx":
        stage_cdx()
    elif args.stage == "fetch":
        stage_fetch(limit=args.limit)
    elif args.stage == "parse":
        stage_parse()


if __name__ == "__main__":
    main()
