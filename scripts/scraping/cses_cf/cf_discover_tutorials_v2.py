"""
cf_discover_tutorials_v2.py
===========================

Phase 1: DISCOVERY pass that finds Codeforces "Tutorial" blog links for ~917
modern contest_ids (mostly 2019+) that are present in
``problems_registry.parquet`` but NOT in the existing
``contest_blog_map.parquet`` (v1 — only covers pre-2018 contests with
editorial-MISSING problems in the open-r1 bundle).

For each contest_id:
  1. Fetch ``https://codeforces.com/contest/{id}``
       - try a plain HTTP fetch first (very cheap)
       - fall back to Camoufox (JS-rendering anti-detect Firefox + Webshare
         rotating proxies — same plumbing as cf_blog_render_scraper.py) if
         we see a Cloudflare "Just a moment" challenge or any non-200 status.
  2. Parse the contest page with the same ``extract_tutorial_entry`` logic
     used by ``cf_delta_scraper.py``: prefer anchors whose text matches
     "tutorial" / "editorial" / Russian equivalents; otherwise fall back to
     the first ``/blog/entry/N`` link on the page (sidebar "Contest
     materials").
  3. Append one row to ``contest_blog_map_v2.parquet`` with
     {contest_id, blog_entry_id (or None), status, scraped_at} — append-only,
     checkpointed every 20 contests, fully resumable.
  4. Append one JSONL record per fetch attempt to
     ``discover_v2_log.jsonl``.

This script NEVER touches the v1 ``contest_blog_map.parquet`` or the
running render-scrape (PID 3634835). It writes to disjoint files only.

CLI:
    python cf_discover_tutorials_v2.py [--limit N] [--validate] [--start-from K]

``--validate`` runs ONLY the first 5 contests and prints anchor text per
hit; intended as the mandatory pre-flight gate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
from bs4 import BeautifulSoup

# Reuse proxy pool from stealth_fetch (same module the render-scraper uses).
sys.path.insert(0, "/lfs/skampere3/0/alexspan/norm-research/scripts/scraping")
from stealth_fetch import get_proxy_pool, pick_proxy  # noqa: E402

from camoufox.async_api import AsyncCamoufox  # noqa: E402

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research/datasets/codeforces_delta")
REGISTRY = ROOT / "problems_registry.parquet"
MAP_V1 = ROOT / "contest_blog_map.parquet"  # READ-ONLY — used for dedup target
MAP_V2 = ROOT / "contest_blog_map_v2.parquet"  # new, append-only
LOG_JSONL = ROOT / "discover_v2_log.jsonl"
RENDERED_DIR = ROOT / "rendered_html"

DEFAULT_TIMEOUT_MS = 45_000
DEFAULT_RETRIES = 3
DEFAULT_JITTER = (3.0, 5.0)  # ~1 contest per 3-5s with jitter
CHECKPOINT_EVERY = 20

log = logging.getLogger("cf_discover_v2")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)

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
    return ("just a moment" in low) or ("cf-chl-bypass" in low)


# ---------------------------------------------------------------------------
# Tutorial-link extraction (same logic as cf_delta_scraper.py)
# ---------------------------------------------------------------------------

TUTORIAL_RE = re.compile(r"/blog/entry/(\d+)", re.IGNORECASE)


def extract_tutorial_entry(html: str, verbose: bool = False) -> tuple[Optional[int], Optional[str]]:
    """Return (blog_entry_id, matched_anchor_text) or (None, None)."""
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[tuple[int, str]] = []
    for a in soup.find_all("a", href=True):
        m = TUTORIAL_RE.search(a["href"])
        if not m:
            continue
        eid = int(m.group(1))
        txt = a.get_text(" ", strip=True)
        candidates.append((eid, txt))
    if verbose:
        for eid, txt in candidates[:8]:
            log.info("  candidate: blog/entry/%s — %r", eid, txt[:80])
    # prefer "tutorial"/"editorial" anchors
    for keyword in ("tutorial", "editorial", "разбор", "анализ"):
        for eid, txt in candidates:
            if keyword in txt.lower():
                return eid, txt
    if candidates:
        return candidates[0][0], candidates[0][1]
    return None, None


# ---------------------------------------------------------------------------
# Fetch: static-first, Camoufox fallback
# ---------------------------------------------------------------------------


@dataclass
class FetchResult:
    contest_id: str
    status: int
    html: str
    mode: str            # "static" or "camoufox"
    proxy_host: str
    attempt: int
    error: Optional[str]
    elapsed_s: float


async def fetch_static(contest_id: str, timeout_s: float = 20.0) -> FetchResult:
    url = f"https://codeforces.com/contest/{contest_id}"
    t0 = time.time()
    status = 0
    html = ""
    err: Optional[str] = None
    try:
        # Plain unproxied httpx with a real-ish UA. Cheap because if it
        # works we save a Camoufox spin-up.
        async with httpx.AsyncClient(
            timeout=timeout_s,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            },
        ) as client:
            r = await client.get(url)
            status = r.status_code
            html = r.text or ""
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    return FetchResult(
        contest_id=contest_id,
        status=status,
        html=html,
        mode="static",
        proxy_host="",
        attempt=1,
        error=err,
        elapsed_s=round(time.time() - t0, 2),
    )


async def fetch_camoufox_once(
    contest_id: str,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> FetchResult:
    url = f"https://codeforces.com/contest/{contest_id}"
    proxy = pick_proxy(index=None, strategy="round_robin")
    t0 = time.time()
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
            # Settle a moment for any CF challenge to clear / sidebar render.
            await page.wait_for_timeout(1200)
            try:
                title = (await page.title()) or ""
                if "just a moment" in title.lower():
                    await page.wait_for_timeout(5000)
            except Exception:
                pass
            html = await page.content()
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    return FetchResult(
        contest_id=contest_id,
        status=status,
        html=html,
        mode="camoufox",
        proxy_host=proxy.host,
        attempt=0,  # caller fills
        error=err,
        elapsed_s=round(time.time() - t0, 2),
    )


async def fetch_contest_page(
    contest_id: str,
    retries: int = DEFAULT_RETRIES,
) -> FetchResult:
    """Try cheap static once; fall back to Camoufox with retries."""
    # 1) Cheap static attempt — saves a browser spin-up when it works.
    sres = await fetch_static(contest_id)
    if (
        sres.error is None
        and 200 <= sres.status < 400
        and sres.html
        and not looks_like_challenge(sres.html)
    ):
        return sres

    jsonl_append(
        LOG_JSONL,
        {
            "ts": now_iso(),
            "contest_id": contest_id,
            "step": "static_fallback",
            "status": sres.status,
            "html_len": len(sres.html or ""),
            "error": sres.error,
        },
    )

    # 2) Camoufox attempts with proxy rotation + backoff.
    last: Optional[FetchResult] = None
    for attempt in range(1, retries + 1):
        result = await fetch_camoufox_once(contest_id)
        result.attempt = attempt
        good = (
            result.error is None
            and 200 <= result.status < 400
            and result.html
            and not looks_like_challenge(result.html)
        )
        if good:
            return result
        last = result
        log.warning(
            "contest %s attempt %d/%d failed (status=%s, err=%s, len=%d, proxy=%s)",
            contest_id,
            attempt,
            retries,
            result.status,
            result.error,
            len(result.html or ""),
            result.proxy_host,
        )
        await asyncio.sleep((2 ** (attempt - 1)) + random.uniform(0, 1.5))
    assert last is not None
    return last


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def load_target_contests(start_from: Optional[int] = None) -> list[str]:
    reg = pd.read_parquet(REGISTRY)
    map_v1 = pd.read_parquet(MAP_V1) if MAP_V1.exists() else pd.DataFrame(columns=["contest_id"])
    v1_cids = set(map_v1["contest_id"].dropna().astype(str).tolist())

    # Also dedup against anything already in v2.
    v2_cids: set[str] = set()
    if MAP_V2.exists():
        try:
            v2 = pd.read_parquet(MAP_V2)
            v2_cids = set(v2["contest_id"].dropna().astype(str).tolist())
        except Exception as e:  # noqa: BLE001
            log.warning("could not read existing v2 map: %s", e)

    reg_cids = sorted(
        {str(c) for c in reg["contest_id"].dropna().unique().tolist()},
        key=lambda s: int(s) if s.isdigit() else -1,
        reverse=True,
    )
    pending = [c for c in reg_cids if c not in v1_cids and c not in v2_cids]
    if start_from is not None:
        pending = [c for c in pending if c.isdigit() and int(c) <= start_from]
    log.info(
        "registry=%d unique contests, v1 covers %d, v2 already done %d → pending=%d",
        len(reg_cids),
        len(v1_cids),
        len(v2_cids),
        len(pending),
    )
    return pending


def load_already_rendered_blog_ids() -> set[int]:
    """Blog ids already present as rendered_html/blog_{id}.html (>5KB) — we
    don't need to re-render these, but for discovery we still record the
    blog mapping.
    """
    if not RENDERED_DIR.exists():
        return set()
    ids = set()
    for p in RENDERED_DIR.glob("blog_*.html"):
        try:
            if p.stat().st_size > 5000:
                m = re.match(r"blog_(\d+)\.html$", p.name)
                if m:
                    ids.add(int(m.group(1)))
        except Exception:
            continue
    return ids


def append_checkpoint(rows: list[dict]) -> None:
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    if MAP_V2.exists():
        try:
            df_old = pd.read_parquet(MAP_V2)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except Exception as e:  # noqa: BLE001
            log.error("failed reading existing v2 map (%s) — writing fresh", e)
            df_all = df_new
    else:
        df_all = df_new
    # Dedup on contest_id (keep newest = last).
    df_all = df_all.drop_duplicates(subset=["contest_id"], keep="last")
    tmp = MAP_V2.with_suffix(".parquet.tmp")
    df_all.to_parquet(tmp, index=False)
    tmp.replace(MAP_V2)


async def discover(
    limit: Optional[int],
    validate: bool,
    start_from: Optional[int],
    jitter: tuple[float, float] = DEFAULT_JITTER,
) -> None:
    pool = get_proxy_pool()
    log.info("proxy pool size: %d", len(pool))

    pending = load_target_contests(start_from=start_from)
    if validate:
        # Pick a spread of modern ids for the pre-flight gate.
        modern = [c for c in pending if c.isdigit() and int(c) >= 1800]
        # Sample 5 across the modern range.
        if len(modern) >= 5:
            step = max(1, len(modern) // 5)
            pending = modern[:5]
        else:
            pending = modern[:5]
        log.info("VALIDATE mode: using 5 modern contests: %s", pending)
    if limit:
        pending = pending[:limit]

    if not pending:
        log.info("nothing to do")
        return

    rendered_ids = load_already_rendered_blog_ids()
    log.info("%d blog ids already on disk in rendered_html/", len(rendered_ids))

    rows_buf: list[dict] = []
    started = time.time()
    n_ok = 0
    n_hit = 0
    n_already_rendered = 0

    for i, contest_id in enumerate(pending, 1):
        t0 = time.time()
        try:
            fetch = await fetch_contest_page(contest_id)
        except Exception as e:  # noqa: BLE001
            jsonl_append(
                LOG_JSONL,
                {
                    "ts": now_iso(),
                    "contest_id": contest_id,
                    "step": "driver_exception",
                    "error": f"{type(e).__name__}: {e}",
                },
            )
            log.error("contest %s: driver exception: %s", contest_id, e)
            continue

        good = (
            fetch.error is None
            and 200 <= fetch.status < 400
            and fetch.html
            and not looks_like_challenge(fetch.html)
        )

        blog_id: Optional[int] = None
        anchor_txt: Optional[str] = None
        if good:
            n_ok += 1
            blog_id, anchor_txt = extract_tutorial_entry(fetch.html, verbose=validate)
            if blog_id is not None:
                n_hit += 1
                if blog_id in rendered_ids:
                    n_already_rendered += 1

        status_label = (
            "blog_found" if blog_id is not None
            else ("no_blog_link" if good else f"fetch_fail_{fetch.status}")
        )

        rec = {
            "contest_id": contest_id,
            "blog_entry_id": blog_id,
            "status": status_label,
            "anchor_text": anchor_txt,
            "mode": fetch.mode,
            "proxy_host": fetch.proxy_host,
            "fetch_status": fetch.status,
            "html_len": len(fetch.html or ""),
            "elapsed_s": round(time.time() - t0, 2),
            "error": fetch.error,
            "scraped_at": now_iso(),
        }
        rows_buf.append(rec)
        jsonl_append(LOG_JSONL, {"ts": now_iso(), "step": "discover", **rec})

        if validate or i % 5 == 0:
            elapsed = time.time() - started
            rate = i / elapsed if elapsed else 0.0
            remaining = (len(pending) - i) / rate if rate else 0.0
            log.info(
                "[%d/%d] contest=%s blog=%s mode=%s status=%s anchor=%r hit_rate=%.0f%% "
                "rate=%.2f/s eta=%.1f min",
                i,
                len(pending),
                contest_id,
                blog_id,
                fetch.mode,
                status_label,
                (anchor_txt or "")[:60],
                100.0 * n_hit / max(1, n_ok),
                rate,
                remaining / 60.0,
            )

        # checkpoint
        if i % CHECKPOINT_EVERY == 0:
            append_checkpoint(rows_buf)
            log.info("checkpoint: flushed %d rows to %s", len(rows_buf), MAP_V2)
            rows_buf = []

        # Polite jitter only on successful fetches; failed ones already
        # backed off inside fetch_contest_page.
        if good:
            await asyncio.sleep(random.uniform(*jitter))

    if rows_buf:
        append_checkpoint(rows_buf)
        log.info("final flush: %d rows", len(rows_buf))

    log.info(
        "DONE: %d/%d processed, %d successful fetches, %d tutorial hits "
        "(%d already in rendered_html/)",
        len(pending),
        len(pending),
        n_ok,
        n_hit,
        n_already_rendered,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--validate", action="store_true",
                    help="Only the first 5 modern contests; print anchor candidates.")
    ap.add_argument("--start-from", type=int, default=None,
                    help="Only process contest_ids <= this value (descending traversal).")
    args = ap.parse_args()
    asyncio.run(discover(limit=args.limit, validate=args.validate,
                          start_from=args.start_from))


if __name__ == "__main__":
    main()
