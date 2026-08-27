#!/usr/bin/env python3
"""Concurrent, per-domain-throttled fetch engine shared by both fetchers.

Politeness is preserved PER DOMAIN (<=1 req / SLEEP_S to any single host) while
many DIFFERENT domains are fetched in parallel by a thread pool. archive.org is
the one shared host (every Wayback/CDX call hits it); it tolerates parallel reads
so we throttle it to a small fixed spacing rather than serializing everything.

Disk-cached: a 200 body is written once; re-runs and duplicate URLs are free.
"""
import hashlib
import json
import os
import threading
import time
from urllib.parse import urlparse

import requests

UA = ("norm-research-academic-crawler/0.2 "
      "(alex2awesome@gmail.com; Stanford research; respects robots, low-rate)")
SLEEP_S = 1.05          # per non-archive domain
ARCHIVE_SLEEP_S = 0.7   # archive.org: gentle spacing (it refuses under heavy load)


class DomainThrottle:
    """Per-domain last-fetch clock guarded by per-domain locks."""

    def __init__(self):
        self._locks = {}
        self._last = {}
        self._guard = threading.Lock()

    def _lock_for(self, domain):
        with self._guard:
            if domain not in self._locks:
                self._locks[domain] = threading.Lock()
            return self._locks[domain]

    def wait(self, domain):
        sleep_s = ARCHIVE_SLEEP_S if "archive.org" in domain else SLEEP_S
        lk = self._lock_for(domain)
        lk.acquire()
        try:
            el = time.monotonic() - self._last.get(domain, 0)
            if el < sleep_s:
                time.sleep(sleep_s - el)
            self._last[domain] = time.monotonic()
        finally:
            lk.release()


_throttle = DomainThrottle()
_thread_local = threading.local()


def _session():
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        _thread_local.session = s
    return s


FAST_FAIL = os.environ.get("WIG_FAST_FAIL") == "1"


def polite_get(url, timeout=30):
    domain = urlparse(url).netloc
    is_archive = "archive.org" in domain
    if FAST_FAIL:
        timeout = min(timeout, 15)
        tries = 2
    else:
        tries = 5 if is_archive else 3
    for attempt in range(tries):
        _throttle.wait(domain)
        try:
            r = _session().get(url, headers={"User-Agent": UA}, timeout=timeout,
                               allow_redirects=True)
            if r.status_code in (429, 503):
                time.sleep((3.0 if is_archive else 2.0) * (attempt + 1))
                continue
            return r, None
        except requests.RequestException as e:
            if attempt == tries - 1:
                return None, type(e).__name__
            # connection-refused / reset: archive.org is overloaded -> longer wait
            time.sleep((3.0 if is_archive else 1.0) * (attempt + 1))
    return None, "retries"


_cache_locks = {}
_cache_guard = threading.Lock()


def cached_fetch(url, tag, cache_dir):
    key = hashlib.sha1((tag + url).encode()).hexdigest()[:16]
    cpath = os.path.join(cache_dir, key + ".json")
    if os.path.exists(cpath):
        try:
            with open(cpath) as f:
                return json.load(f)
        except Exception:
            pass
    r, err = polite_get(url)
    out = {"url": url}
    if r is None:
        out.update(status=None, final_url=None, html=None, error=err)
    else:
        enc = r.apparent_encoding if r.apparent_encoding else "utf-8"
        try:
            r.encoding = enc
        except Exception:
            pass
        out.update(status=r.status_code, final_url=r.url,
                   html=r.text if r.status_code == 200 else None, error=None)
    tmp = cpath + f".tmp{os.getpid()}.{threading.get_ident()}"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, cpath)
    return out
