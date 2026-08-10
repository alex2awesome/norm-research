"""Polite cached fetcher shared by all contest_corpus scrapers.

Hard rules:
- max 1 request per 2 seconds per domain (sleep 2s between requests to same domain)
- fixed academic User-Agent
- every fetched page cached to HTML_CACHE; re-runs never re-fetch
"""
import hashlib
import json
import os
import time
import urllib.parse

import requests

STAGING = "/Users/spangher/Projects/stanford-research/norm-research/.staging/norm-datasets/humor/contest_corpus"
HTML_CACHE = os.path.join(STAGING, "html_cache")
USER_AGENT = "norm-research-academic-crawler/0.1 (alex2awesome@gmail.com; Stanford research)"

os.makedirs(HTML_CACHE, exist_ok=True)

_last_fetch_per_domain = {}
_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT})


def cache_path(url: str) -> str:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    # keep a readable slug for debugging
    slug = urllib.parse.quote_plus(url)[:120]
    return os.path.join(HTML_CACHE, f"{h}__{slug}.html")


def fetch(url: str, force: bool = False, timeout: int = 30) -> str | None:
    """Fetch a URL with per-domain 2s rate limit and on-disk cache.

    Returns HTML text, or None on hard failure (recorded in cache as sentinel
    so we don't hammer dead URLs on re-runs).
    """
    cp = cache_path(url)
    if not force and os.path.exists(cp):
        with open(cp, "r", encoding="utf-8") as f:
            body = f.read()
        if body.startswith("__FETCH_ERROR__"):
            return None
        return body

    domain = urllib.parse.urlparse(url).netloc
    last = _last_fetch_per_domain.get(domain, 0.0)
    wait = 2.0 - (time.time() - last)
    if wait > 0:
        time.sleep(wait)

    try:
        resp = _session.get(url, timeout=timeout)
        _last_fetch_per_domain[domain] = time.time()
        if resp.status_code != 200:
            body = f"__FETCH_ERROR__ status={resp.status_code}"
            with open(cp, "w", encoding="utf-8") as f:
                f.write(body)
            print(f"  [fetch] {resp.status_code} {url}")
            return None
        text = resp.text
        with open(cp, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  [fetch] 200 {url} ({len(text)} bytes)")
        return text
    except requests.RequestException as e:
        _last_fetch_per_domain[domain] = time.time()
        with open(cp, "w", encoding="utf-8") as f:
            f.write(f"__FETCH_ERROR__ exception={type(e).__name__}: {e}")
        print(f"  [fetch] EXC {url}: {e}")
        return None


def fetch_binary(url: str, timeout: int = 30) -> bytes | None:
    """Like fetch() but for binary assets (PDFs). Same rate limit + cache."""
    cp = cache_path(url) + ".bin"
    if os.path.exists(cp):
        with open(cp, "rb") as f:
            return f.read()

    domain = urllib.parse.urlparse(url).netloc
    last = _last_fetch_per_domain.get(domain, 0.0)
    wait = 2.0 - (time.time() - last)
    if wait > 0:
        time.sleep(wait)
    try:
        resp = _session.get(url, timeout=timeout)
        _last_fetch_per_domain[domain] = time.time()
        if resp.status_code != 200:
            print(f"  [fetch-bin] {resp.status_code} {url}")
            return None
        with open(cp, "wb") as f:
            f.write(resp.content)
        print(f"  [fetch-bin] 200 {url} ({len(resp.content)} bytes)")
        return resp.content
    except requests.RequestException as e:
        print(f"  [fetch-bin] EXC {url}: {e}")
        return None
