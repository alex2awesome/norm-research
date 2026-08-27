"""Shared utilities for the RoyalRoad stub dataset pipeline.

Politeness: <=1 req/sec per host, real UA with contact email, exponential
backoff on 429/5xx/connection errors. Never evades blocks.
"""
import hashlib
import json
import re
import time
import unicodedata

import requests
from bs4 import BeautifulSoup

USER_AGENT = (
    "norm-research-dataset-builder/1.0 "
    "(academic research on creative-writing quality signals; "
    "contact: alex2awesome@gmail.com)"
)

MIN_INTERVAL = 1.05  # seconds between requests, per host
MAX_RETRIES = 6


class PoliteSession:
    """requests.Session with per-host rate limiting and retry/backoff."""

    def __init__(self):
        self.sess = requests.Session()
        self.sess.headers.update({"User-Agent": USER_AGENT})
        self._last_req = {}  # host -> monotonic time

    def get(self, url, timeout=60, **kw):
        host = url.split("/")[2]
        last_err = None
        for attempt in range(MAX_RETRIES):
            wait = self._last_req.get(host, 0) + MIN_INTERVAL - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_req[host] = time.monotonic()
            try:
                r = self.sess.get(url, timeout=timeout, **kw)
            except (requests.ConnectionError, requests.Timeout,
                    requests.exceptions.ContentDecodingError,
                    requests.exceptions.ChunkedEncodingError) as e:
                last_err = e
                time.sleep(min(5 * 2 ** attempt, 90))
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(f"HTTP {r.status_code}")
                time.sleep(min(5 * 2 ** attempt, 90))
                continue
            return r
        raise RuntimeError(f"giving up on {url}: {last_err}")


# ---------------------------------------------------------------------------
# Search-page parsing
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"([\d,]+)\s+(Followers|Pages|Views|Chapters)")


def parse_search_page(html):
    """Parse a /fictions/search results page into a list of card dicts.

    Per-card fault tolerance: a malformed card is skipped, never fatal."""
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for item in soup.select("div.fiction-list-item"):
        try:
            card = _parse_card(item)
        except Exception:
            continue
        if card:
            out.append(card)
    return out


def _parse_card(item):
        a = item.select_one("h2.fiction-title a[href]")
        if not a:
            return None
        m = re.match(r"/fiction/(\d+)/?([^/?#]*)", a["href"])
        if not m:
            return None
        card = {
            "fiction_id": int(m.group(1)),
            "slug": m.group(2),
            "title": a.get_text(strip=True),
            "labels": [],
            "tags": [t.get_text(strip=True) for t in item.select("a.fiction-tag")],
            "followers": None,
            "pages": None,
            "views": None,
            "chapters_listed": None,
            "rating": None,
            "last_update_ts": None,
            "description": None,
        }
        for lab in item.select("span.label"):
            txt = lab.get_text(strip=True)
            # strip popover icon text artifacts
            txt = txt.split("\n")[0].strip()
            if txt:
                card["labels"].append(txt)
        stats_text = item.get_text(" ", strip=True)
        for num, what in _NUM_RE.findall(stats_text):
            key = {
                "Followers": "followers",
                "Pages": "pages",
                "Views": "views",
                "Chapters": "chapters_listed",
            }[what]
            if card[key] is None:
                card[key] = int(num.replace(",", ""))
        rdiv = item.select_one("div[aria-label^='Rating:']")
        if rdiv:
            rm = re.search(r"Rating:\s*(\d+(?:\.\d+)?)", rdiv.get("aria-label", ""))
            if rm:
                card["rating"] = float(rm.group(1))
        t = item.select_one("time[unixtime]")
        if t:
            u = (t.get("unixtime") or "").strip()
            if u.lstrip("-").isdigit():
                card["last_update_ts"] = int(u)
        desc = item.select_one("div[id^=description-]")
        if desc:
            card["description"] = desc.get_text("\n", strip=True)[:5000]
        return card


# ---------------------------------------------------------------------------
# Chapter HTML extraction (works on live and Wayback `id_` captures)
# ---------------------------------------------------------------------------

_HIDDEN_CLASS_RE = re.compile(
    r"\.([\w-]+)\s*\{[^}]*?(?:display\s*:\s*none|speak\s*:\s*never)[^}]*\}",
    re.I,
)


def extract_chapter_text(html):
    """Return (chapter_title, text) from a RoyalRoad chapter page.

    Strips RoyalRoad's hidden anti-scrape watermark elements (random CSS
    classes styled display:none / speak:never in inline <style> blocks).
    Returns (None, None) if no chapter-content div is found.
    """
    hidden = set()
    for style in re.findall(r"<style[^>]*>(.*?)</style>", html, re.S):
        hidden.update(_HIDDEN_CLASS_RE.findall(style))

    soup = BeautifulSoup(html, "html.parser")
    div = soup.select_one("div.chapter-inner.chapter-content") or soup.select_one(
        "div.chapter-content"
    )
    if div is None:
        return None, None
    for el in div.find_all(True):
        # bs4 can yield Tags with attrs=None on malformed old captures
        cls = set((getattr(el, "attrs", None) or {}).get("class") or [])
        if cls & hidden:
            el.decompose()
    parts = []
    blocks = div.find_all(["p", "h1", "h2", "h3", "h4", "li", "blockquote"])
    if blocks:
        for p in blocks:
            txt = p.get_text(" ", strip=True)
            if txt:
                parts.append(txt)
        text = "\n\n".join(parts)
    else:
        text = div.get_text("\n", strip=True)

    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else None
    if not title:
        tm = re.search(r"<title>([^<|]*)", html)
        title = tm.group(1).strip() if tm else None
    return title, text


# ---------------------------------------------------------------------------
# Presentation normalization (mandatory: cross-source formatting leak guard)
# ---------------------------------------------------------------------------

_QUOTES = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "′": "'", "″": '"',
}


def display(text):
    """NFKC; curly->straight quotes; em/en dash -> --/-; ellipsis -> ...;
    strip wrapper quotes; collapse whitespace (keep paragraph breaks)."""
    if text is None:
        return None
    t = unicodedata.normalize("NFKC", text)
    for k, v in _QUOTES.items():
        t = t.replace(k, v)
    t = t.replace("—", "--").replace("–", "-").replace("…", "...")
    t = t.replace(" ", " ").replace("​", "")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" ?\n ?", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if len(t) >= 2 and t[0] in "\"'" and t[-1] == t[0]:
        inner = t[1:-1]
        if t[0] not in inner:
            t = inner.strip()
    return t


# ---------------------------------------------------------------------------
# Stable splits — NEVER seeded shuffle
# ---------------------------------------------------------------------------

def stable_split(fiction_id):
    h = int(hashlib.md5(str(fiction_id).encode()).hexdigest(), 16) % 10
    return "train" if h <= 7 else ("eval" if h == 8 else "test")


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path):
    out = []
    bad = 0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    bad += 1
    except FileNotFoundError:
        pass
    if bad:
        import sys as _sys
        print(f"[read_jsonl] {path}: skipped {bad} corrupt lines", file=_sys.stderr)
    return out
