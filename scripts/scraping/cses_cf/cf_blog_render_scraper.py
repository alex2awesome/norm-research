"""
cf_blog_render_scraper.py
=========================

Re-scrape Codeforces editorial blog entries with a real JavaScript-executing
browser so that `div.spoiler` code blocks (which Codeforces renders client-side)
are actually present in the saved HTML.

Background / why this exists
----------------------------
The earlier static-HTML scrape in `cf_delta_scraper.py` got real pages (median
312KB, zero Cloudflare challenges) but 0/3,688 extracted texts contain ANY code
marker. A saved raw page (datasets/cses/cf_blog_83343.html, 221KB) has 8
`div.spoiler` shells, ZERO `<pre>`/`<code>` tags, and only 729 chars of
`div.ttypography` text — i.e. the code is injected by JS after page load.

What this script does
---------------------
1) Loads contest_blog_map.parquet → unique blog_entry_ids (~853). Skips any
   blog already successfully captured to disk (resumable).
2) For each blog id, opens https://codeforces.com/blog/entry/{id} in Camoufox
   (the same anti-detect Firefox + Webshare rotating proxy stack used for the
   AtCoder scrape — see `stealth_fetch.py`), waits for the DOM to settle,
   tries to click `.spoiler-title` toggles to force lazy code blocks to mount,
   and saves the full rendered DOM to:
       datasets/codeforces_delta/rendered_html/blog_{id}.html
3) Appends one JSONL record per blog to
       datasets/codeforces_delta/render_scrape_log.jsonl
   with: blog_id, status, html_len, n_spoilers, n_pre, n_code_tags,
         ttypography_text_len, attempt, proxy_host, error.
4) Politeness: ~1 req per 3–5 s with jitter; 1 fetch at a time per blog;
   retries x3 with backoff; rotates proxies; detects Cloudflare "Just a moment"
   challenge pages and retries.

APPEND-ONLY: never modifies editorials.parquet or contest_blog_map.parquet.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup

# Reuse proxy pool + Proxy dataclass from stealth_fetch.
sys.path.insert(0, "/lfs/skampere3/0/alexspan/norm-research/scripts/scraping")
from stealth_fetch import get_proxy_pool, pick_proxy  # noqa: E402

from camoufox.async_api import AsyncCamoufox  # noqa: E402

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research/datasets/codeforces_delta")
RENDERED_DIR = ROOT / "rendered_html"
LOG_JSONL = ROOT / "render_scrape_log.jsonl"
RUN_LOG = ROOT / "render_scrape.out"  # mostly for nohup; this file just writes
PROGRESS = ROOT / "render_scrape_progress.json"

CONTEST_BLOG_MAP = ROOT / "contest_blog_map.parquet"

DEFAULT_TIMEOUT_MS = 60_000
DEFAULT_RETRIES = 3
DEFAULT_JITTER = (3.0, 5.0)  # per task: ~1 req per 3-5s with jitter
SPOILER_EXPAND_MAX = 30  # cap on clicks per page to avoid runaway
POST_EXPAND_WAIT_MS = 1500
POST_GOTO_WAIT_MS = 1500

log = logging.getLogger("cf_blog_render")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)

RENDERED_DIR.mkdir(parents=True, exist_ok=True)
ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.utcnow().isoformat()


def jsonl_append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def looks_like_challenge(html: str) -> bool:
    if not html:
        return True
    low = html[:4000].lower()
    if "just a moment" in low:
        return True
    if "cf-chl-bypass" in low:
        return True
    if "challenge-platform" in low and "<title" in low and "just a moment" in low:
        return True
    return False


@dataclass
class RenderResult:
    blog_id: int
    status: int
    html: str
    proxy_host: str
    proxy_country: str
    attempt: int
    error: Optional[str]


# ---------------------------------------------------------------------------
# Core fetch (Camoufox + click-to-expand spoilers + full HTML save)
# ---------------------------------------------------------------------------


async def render_blog_once(
    blog_id: int,
    proxy_index: Optional[int] = None,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> RenderResult:
    """One attempt at rendering a single blog id. Returns full DOM HTML."""
    url = f"https://codeforces.com/blog/entry/{blog_id}"
    proxy = pick_proxy(index=proxy_index, strategy="round_robin")

    status = 0
    html = ""
    err: Optional[str] = None

    try:
        async with AsyncCamoufox(
            headless=True,
            humanize=True,
            proxy=proxy.for_playwright(),
            geoip=True,
        ) as browser:
            page = await browser.new_page()
            resp = await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            status = resp.status if resp else 0

            # Let any client-side scripts mount.
            await page.wait_for_timeout(POST_GOTO_WAIT_MS)

            # If a Cloudflare interstitial sneaks in, give it a moment.
            try:
                title = (await page.title()) or ""
                if "just a moment" in title.lower():
                    await page.wait_for_timeout(5000)
            except Exception:
                pass

            # Some CF templates lazy-render spoiler bodies only on click.
            # Click every spoiler-title we can find (capped), then wait so the
            # injected <pre>/<code> blocks materialize.
            try:
                # Wait for at least one spoiler shell to appear if present.
                try:
                    await page.wait_for_selector("div.spoiler, .spoiler-title", timeout=4000)
                except Exception:
                    pass
                titles = await page.query_selector_all(".spoiler-title")
                clicked = 0
                for t in titles[:SPOILER_EXPAND_MAX]:
                    try:
                        await t.click(timeout=2000)
                        clicked += 1
                    except Exception:
                        # Skip non-clickable / detached nodes; keep going.
                        continue
                if clicked:
                    await page.wait_for_timeout(POST_EXPAND_WAIT_MS)
                    # Sometimes spoiler bodies are .spoiler-content > <pre>;
                    # wait specifically for code-ish content if we can.
                    try:
                        await page.wait_for_function(
                            "document.querySelectorAll('div.spoiler pre, div.spoiler code').length > 0",
                            timeout=4000,
                        )
                    except Exception:
                        pass
            except Exception as e:
                # Spoiler-expansion is best-effort; don't fail the whole fetch.
                log.warning("blog %s: spoiler-expand best-effort failed: %s", blog_id, e)

            html = await page.content()
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"

    return RenderResult(
        blog_id=blog_id,
        status=status,
        html=html,
        proxy_host=proxy.host,
        proxy_country=proxy.country,
        attempt=0,  # caller fills in
        error=err,
    )


async def render_blog(
    blog_id: int,
    retries: int = DEFAULT_RETRIES,
    jitter: tuple[float, float] = DEFAULT_JITTER,
) -> RenderResult:
    last: Optional[RenderResult] = None
    for attempt in range(1, retries + 1):
        # Rotate proxy by attempt index.
        result = await render_blog_once(blog_id, proxy_index=None)
        result.attempt = attempt

        good = (
            result.error is None
            and 200 <= result.status < 400
            and result.html
            and not looks_like_challenge(result.html)
        )
        if good:
            # polite jitter after success
            await asyncio.sleep(random.uniform(*jitter))
            return result

        last = result
        log.warning(
            "blog %s attempt %d/%d failed (status=%s, err=%s, len=%d, proxy=%s)",
            blog_id,
            attempt,
            retries,
            result.status,
            result.error,
            len(result.html or ""),
            result.proxy_host,
        )
        # exponential backoff
        await asyncio.sleep((2 ** (attempt - 1)) + random.uniform(0, 1.5))

    assert last is not None
    return last


# ---------------------------------------------------------------------------
# Parse counts (for the JSONL log + validation gate)
# ---------------------------------------------------------------------------


def parse_counts(html: str) -> dict:
    if not html:
        return {
            "html_len": 0,
            "n_spoilers": 0,
            "n_pre": 0,
            "n_code_tags": 0,
            "n_pre_in_spoiler": 0,
            "n_code_in_spoiler": 0,
            "ttypography_text_len": 0,
            "has_include": False,
            "has_python_def": False,
        }
    soup = BeautifulSoup(html, "html.parser")
    spoilers = soup.select("div.spoiler")
    pres = soup.find_all("pre")
    codes = soup.find_all("code")
    n_pre_in_spoiler = 0
    n_code_in_spoiler = 0
    for s in spoilers:
        n_pre_in_spoiler += len(s.find_all("pre"))
        n_code_in_spoiler += len(s.find_all("code"))
    tt = soup.select_one("div.ttypography")
    tt_text = tt.get_text("\n", strip=True) if tt else ""
    return {
        "html_len": len(html),
        "n_spoilers": len(spoilers),
        "n_pre": len(pres),
        "n_code_tags": len(codes),
        "n_pre_in_spoiler": n_pre_in_spoiler,
        "n_code_in_spoiler": n_code_in_spoiler,
        "ttypography_text_len": len(tt_text),
        "has_include": "#include" in html,
        "has_python_def": bool(re.search(r"\bdef\s+\w+\s*\(", html)),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def load_blog_ids(map_path: Path) -> list[int]:
    df = pd.read_parquet(map_path)
    if "blog_entry_id" not in df.columns:
        raise SystemExit(f"contest_blog_map missing blog_entry_id column: {df.columns.tolist()}")
    df = df.dropna(subset=["blog_entry_id"]).copy()
    df["blog_entry_id"] = df["blog_entry_id"].astype(int)
    # Highest contest_id first (modern contests are more useful for the
    # validation gate; we want at least a few high-id ones up front).
    if "contest_id" in df.columns:
        # contest_id can be string; sort numerically when possible.
        def _to_int(x):
            try:
                return int(x)
            except Exception:
                return -1
        df["_cid_int"] = df["contest_id"].map(_to_int)
        df = df.sort_values("_cid_int", ascending=False)
    return df["blog_entry_id"].drop_duplicates().tolist()


def already_done(blog_id: int) -> bool:
    p = RENDERED_DIR / f"blog_{blog_id}.html"
    if not p.exists():
        return False
    try:
        return p.stat().st_size > 5000  # sanity floor
    except Exception:
        return False


async def scrape_all(
    limit: Optional[int],
    only_ids: Optional[list[int]],
    dry_validate: bool,
) -> None:
    pool = get_proxy_pool()
    log.info("proxy pool size: %d", len(pool))

    if only_ids:
        ids = only_ids
        log.info("running on explicit id list, %d ids", len(ids))
    else:
        ids = load_blog_ids(CONTEST_BLOG_MAP)
        log.info("loaded %d unique blog_entry_ids from %s", len(ids), CONTEST_BLOG_MAP)

    # resume: skip ones we've already saved (unless caller forces).
    todo = [b for b in ids if not already_done(b)]
    log.info("after resume filter: %d blogs to fetch", len(todo))

    if limit:
        todo = todo[:limit]
        log.info("limit applied: %d blogs", len(todo))

    if not todo:
        log.info("nothing to do")
        return

    started = time.time()
    n_ok = 0
    n_with_code = 0

    for i, blog_id in enumerate(todo, 1):
        t0 = time.time()
        try:
            result = await render_blog(blog_id)
        except Exception as e:  # noqa: BLE001
            jsonl_append(
                LOG_JSONL,
                {
                    "ts": now_iso(),
                    "blog_id": blog_id,
                    "status": 0,
                    "error": f"driver_exception: {type(e).__name__}: {e}",
                    "html_len": 0,
                },
            )
            log.error("blog %s: driver exception: %s", blog_id, e)
            continue

        # Save raw HTML — never want to re-scrape.
        out_path = RENDERED_DIR / f"blog_{blog_id}.html"
        if result.html:
            try:
                out_path.write_text(result.html, encoding="utf-8")
            except Exception as e:
                log.error("blog %s: failed writing %s: %s", blog_id, out_path, e)

        counts = parse_counts(result.html)
        rec = {
            "ts": now_iso(),
            "blog_id": blog_id,
            "status": result.status,
            "attempt": result.attempt,
            "proxy_host": result.proxy_host,
            "proxy_country": result.proxy_country,
            "error": result.error,
            "elapsed_s": round(time.time() - t0, 2),
            **counts,
        }
        jsonl_append(LOG_JSONL, rec)

        if result.status >= 200 and result.status < 400 and result.html and not result.error:
            n_ok += 1
            if counts["has_include"] or counts["n_pre_in_spoiler"] > 0 or counts["n_code_in_spoiler"] > 0:
                n_with_code += 1

        if i % 10 == 0 or dry_validate:
            elapsed = time.time() - started
            rate = i / elapsed if elapsed else 0.0
            remaining = (len(todo) - i) / rate if rate else 0.0
            log.info(
                "[%d/%d] ok=%d with_code=%d  blog_id=%s len=%d spoilers=%d pre=%d "
                "code=%d pre_in_spoil=%d  rate=%.2f/s  eta=%.1f min",
                i,
                len(todo),
                n_ok,
                n_with_code,
                blog_id,
                counts["html_len"],
                counts["n_spoilers"],
                counts["n_pre"],
                counts["n_code_tags"],
                counts["n_pre_in_spoiler"],
                rate,
                remaining / 60.0,
            )

        if i % 25 == 0:
            PROGRESS.write_text(
                json.dumps(
                    {
                        "ts": now_iso(),
                        "done": i,
                        "total": len(todo),
                        "ok": n_ok,
                        "with_code": n_with_code,
                    }
                )
            )

    PROGRESS.write_text(
        json.dumps(
            {
                "ts": now_iso(),
                "done": len(todo),
                "total": len(todo),
                "ok": n_ok,
                "with_code": n_with_code,
                "final": True,
            }
        )
    )
    log.info("DONE: %d/%d ok, %d with code", n_ok, len(todo), n_with_code)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Only render this many blogs (after resume filter).")
    ap.add_argument("--ids", type=str, default=None,
                    help="Comma-separated blog_entry_ids to render (overrides map).")
    ap.add_argument("--validate", action="store_true",
                    help="Print extra-detailed stats every blog (use for the 5-blog gate).")
    args = ap.parse_args()

    only_ids = None
    if args.ids:
        only_ids = [int(x.strip()) for x in args.ids.split(",") if x.strip()]

    asyncio.run(scrape_all(limit=args.limit, only_ids=only_ids, dry_validate=args.validate))


if __name__ == "__main__":
    main()
