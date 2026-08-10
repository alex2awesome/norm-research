"""
Fetch AoPS forum topics via the /m/community/ajax.php endpoint by piggy-backing
on a real Chromium browser context. The browser warms Cloudflare cookies on
/community first, then we POST through `page.evaluate("fetch(...)")` so the
request inherits cf_clearance, the same TLS fingerprint, and all headers
Cloudflare expects.

Usage:
    python scripts/aops_fetch_playwright.py 495607 2 3
    # outputs raw/topic_<id>.json for each id
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

from playwright.sync_api import sync_playwright, BrowserContext, Page
from playwright_stealth import Stealth


REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "raw"
LOG_DIR = REPO_ROOT / "logs"
RAW_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

AJAX_URL = "https://artofproblemsolving.com/m/community/ajax.php"
COMMUNITY_URL = "https://artofproblemsolving.com/community"
DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def _extract_session_id(html: str) -> Optional[str]:
    """Pull the 32-char AoPS.session id out of the bootstrap HTML."""
    m = re.search(r"AoPS\.session\s*=\s*['\"]([0-9a-f]{32})['\"]", html)
    if m:
        return m.group(1)
    # Sometimes the assignment is `AoPS.session = {id: '...'}` style; try a
    # broader fallback.
    m = re.search(r"AoPS\.session\s*=\s*\{[^}]*?id\s*:\s*['\"]([0-9a-f]{32})['\"]", html)
    if m:
        return m.group(1)
    # As a last resort, look at all 32-hex blobs in the immediate window.
    idx = html.find("AoPS.session")
    if idx >= 0:
        m = re.search(r"([0-9a-f]{32})", html[idx : idx + 400])
        if m:
            return m.group(1)
    return None


def warm_session(page: Page) -> str:
    """Navigate to /community, solve any CF challenge by waiting, and return aops_session_id."""
    page.goto(COMMUNITY_URL, wait_until="domcontentloaded", timeout=60_000)

    # Cloudflare challenge can take a few seconds.  Wait until the
    # AoPS.session token shows up in the document.
    deadline = time.time() + 45
    session_id: Optional[str] = None
    while time.time() < deadline:
        html = page.content()
        session_id = _extract_session_id(html)
        if session_id:
            break
        # If page still says "Just a moment" wait a bit longer for CF JS.
        time.sleep(1.5)

    if not session_id:
        raise RuntimeError("Could not extract AoPS.session after warming /community")
    return session_id


def fetch_topic(page: Page, topic_id: int, session_id: str) -> dict:
    """Run the POST via fetch() inside the page so CF cookies are auto-attached."""
    payload = {
        "topic_fetch": "initial",
        "new_topic_id": str(topic_id),
        "fetch_first": "1",
        "fetch_all": "1",
        "a": "change_focus_topic",
        "aops_logged_in": "false",
        "aops_user_id": "1",
        "aops_session_id": session_id,
    }

    js = """
    async (args) => {
        const form = new URLSearchParams();
        for (const [k, v] of Object.entries(args.payload)) form.append(k, v);
        const res = await fetch(args.url, {
            method: 'POST',
            credentials: 'include',
            headers: {
                'Accept': 'application/json, text/javascript, */*; q=0.01',
                'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
                'X-Requested-With': 'XMLHttpRequest',
            },
            body: form.toString(),
        });
        const text = await res.text();
        return { status: res.status, text: text };
    }
    """
    result = page.evaluate(js, {"url": AJAX_URL, "payload": payload})
    status = result["status"]
    text = result["text"]
    if status != 200:
        raise RuntimeError(
            f"Non-200 ({status}) for topic {topic_id}; first 300 chars: {text[:300]!r}"
        )
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Non-JSON response for topic {topic_id}: {text[:300]!r}"
        ) from e


def fetch_topics(topic_ids: Iterable[int], headless: bool = True, delay_s: float = 2.0) -> List[Path]:
    """Iterate over topic_ids and persist each raw response."""
    saved: List[Path] = []
    with Stealth().use_sync(sync_playwright()) as p:
        browser = p.chromium.launch(headless=headless)
        context: BrowserContext = browser.new_context(
            user_agent=DEFAULT_UA,
            viewport={"width": 1366, "height": 900},
            locale="en-US",
        )
        page = context.new_page()
        session_id = warm_session(page)
        print(f"[info] aops_session_id={session_id}")

        last = 0.0
        for topic_id in topic_ids:
            # Be polite: at most one request per `delay_s` seconds.
            wait = delay_s - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.time()
            try:
                data = fetch_topic(page, int(topic_id), session_id)
            except RuntimeError as e:
                print(f"[warn] topic {topic_id} failed: {e}")
                # Try refreshing the session once.
                session_id = warm_session(page)
                data = fetch_topic(page, int(topic_id), session_id)

            out_path = RAW_DIR / f"topic_{topic_id}.json"
            out_path.write_text(json.dumps(data, ensure_ascii=False))
            saved.append(out_path)
            print(f"[ok] saved {out_path}")

        context.close()
        browser.close()
    return saved


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("topic_ids", nargs="+", type=int, help="Topic ids to fetch")
    parser.add_argument("--headed", action="store_true", help="Show the browser window")
    parser.add_argument("--delay", type=float, default=2.0, help="Min seconds between requests")
    args = parser.parse_args(argv)

    saved = fetch_topics(args.topic_ids, headless=not args.headed, delay_s=args.delay)

    # Sanity-check the first one: print post[0] keys + the metadata fields the
    # caller specifically cares about.
    if saved:
        first = json.loads(saved[0].read_text())
        try:
            posts = first["response"]["posts"]
            p0 = posts[0]
            keys = sorted(p0.keys())
            print(f"\n[schema] post[0] has {len(keys)} keys:")
            for k in keys:
                print(f"  - {k}")
            interesting = [
                "thanks_received", "vote_score", "username",
                "post_time", "post_number", "post_canonical",
                "post_rendered",
            ]
            print("\n[schema] requested metadata presence:")
            for k in interesting:
                marker = "y" if k in p0 else "MISSING"
                print(f"  {marker}: {k}")
        except (KeyError, IndexError) as e:
            print(f"[warn] could not introspect first response: {e}")
            print(json.dumps(first, indent=2)[:500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
