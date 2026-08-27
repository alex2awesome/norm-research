"""
Scaled AoPS crawler — Playwright stealth, resume-capable, gzip-compressed.

One process = one browser context. Iterate over a topic-id range, save each raw
response as a gzipped jsonl shard so we can compress 5–10x over flat .json files.

Usage:
    # Pilot (200 topics, ~7 min at 2s/req):
    python scripts/aops_bulk_crawl.py --start 495000 --end 495200 --shard pilot

    # Resume same shard:
    python scripts/aops_bulk_crawl.py --start 495000 --end 495200 --shard pilot

    # Run multiple workers in parallel: shard topic_ids modulo N
    python scripts/aops_bulk_crawl.py --start 1 --end 1000000 --worker 0 --num-workers 4 &
    python scripts/aops_bulk_crawl.py --start 1 --end 1000000 --worker 1 --num-workers 4 &
    ...

Each worker writes to its own shard file so writes don't conflict.

Output layout:
    raw/shards/<shard>__w<worker>.jsonl.gz   — one topic per line, raw json blob
    raw/shards/<shard>__w<worker>.done       — set of completed topic_ids (text)
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Set

from playwright.sync_api import sync_playwright, BrowserContext, Page
from playwright_stealth import Stealth

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "raw"
SHARDS = RAW / "shards"
LOGS = REPO / "logs"
for d in (RAW, SHARDS, LOGS):
    d.mkdir(parents=True, exist_ok=True)

AJAX_URL = "https://artofproblemsolving.com/m/community/ajax.php"
COMMUNITY_URL = "https://artofproblemsolving.com/community"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


def _extract_session_id(html: str) -> Optional[str]:
    for pat in (
        r"AoPS\.session\s*=\s*['\"]([0-9a-f]{32})['\"]",
        r"AoPS\.session\s*=\s*\{[^}]*?id\s*[:=]\s*['\"]([0-9a-f]{32})['\"]",
        r"AoPS\.session\s*=\s*\{[^}]*?\"id\"\s*:\s*\"([0-9a-f]{32})\"",
    ):
        m = re.search(pat, html)
        if m:
            return m.group(1)
    idx = html.find("AoPS.session")
    if idx >= 0:
        m = re.search(r"([0-9a-f]{32})", html[idx:idx + 400])
        if m:
            return m.group(1)
    return None


def warm_session(page: Page, max_wait_s: float = 45.0) -> str:
    page.goto(COMMUNITY_URL, wait_until="domcontentloaded", timeout=60_000)
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        html = page.content()
        sid = _extract_session_id(html)
        if sid:
            return sid
        time.sleep(1.5)
    raise RuntimeError("Could not extract AoPS.session after warming /community")


def fetch_topic(page: Page, topic_id: int, session_id: str) -> dict:
    payload = {
        "topic_fetch": "initial",
        "new_topic_id": str(topic_id),
        "fetch_first": "1", "fetch_all": "1",
        "a": "change_focus_topic",
        "aops_logged_in": "false", "aops_user_id": "1",
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
    status, text = result["status"], result["text"]
    if status != 200:
        raise RuntimeError(f"HTTP {status} t={topic_id}: {text[:200]!r}")
    return json.loads(text)


def load_done(done_path: Path) -> Set[int]:
    if not done_path.exists():
        return set()
    return {int(x) for x in done_path.read_text().split() if x.strip().isdigit()}


def iter_remaining(start: int, end: int, worker: int, num_workers: int, done: Set[int]) -> Iterable[int]:
    for tid in range(start, end):
        if tid % num_workers != worker:
            continue
        if tid in done:
            continue
        yield tid


def crawl(start: int, end: int, shard: str, worker: int, num_workers: int,
          delay_s: float = 2.0, headless: bool = True,
          refresh_every: int = 500) -> None:
    out_path = SHARDS / f"{shard}__w{worker}.jsonl.gz"
    done_path = SHARDS / f"{shard}__w{worker}.done"
    log_path = LOGS / f"{shard}__w{worker}.log"

    done = load_done(done_path)
    total_seen = 0
    total_ok = 0
    total_err = 0
    log_f = log_path.open("a")
    log_f.write(f"\n[{time.ctime()}] start range=[{start},{end}) shard={shard} worker={worker}/{num_workers} delay={delay_s}s done_already={len(done)}\n")
    log_f.flush()

    with Stealth().use_sync(sync_playwright()) as p:
        browser = p.chromium.launch(headless=headless)
        ctx: BrowserContext = browser.new_context(
            user_agent=UA,
            viewport={"width": 1366, "height": 900},
            locale="en-US",
        )
        page = ctx.new_page()
        session_id = warm_session(page)
        print(f"[w{worker}] session={session_id[:8]}…")
        log_f.write(f"[{time.ctime()}] session_id={session_id}\n")
        log_f.flush()

        last_t = 0.0
        # Append-mode gzip
        out_f = gzip.open(out_path, "at", encoding="utf-8")
        try:
            for tid in iter_remaining(start, end, worker, num_workers, done):
                wait = delay_s - (time.time() - last_t)
                if wait > 0:
                    time.sleep(wait)
                last_t = time.time()
                total_seen += 1

                try:
                    data = fetch_topic(page, tid, session_id)
                except Exception as e:
                    msg = str(e)[:200]
                    log_f.write(f"[{time.ctime()}] t={tid} ERR: {msg}\n")
                    # Try one refresh if the session may have died
                    try:
                        session_id = warm_session(page)
                        data = fetch_topic(page, tid, session_id)
                        log_f.write(f"[{time.ctime()}] t={tid} recovered after re-warm\n")
                    except Exception as e2:
                        log_f.write(f"[{time.ctime()}] t={tid} ERR after re-warm: {str(e2)[:200]}\n")
                        log_f.flush()
                        total_err += 1
                        continue

                rec = {"topic_id": tid, "response": data}
                out_f.write(json.dumps(rec, ensure_ascii=False))
                out_f.write("\n")
                done_path.open("a").write(f"{tid}\n")
                total_ok += 1

                if total_seen % 50 == 0:
                    out_f.flush()
                    err = data.get("error_code") if isinstance(data, dict) else None
                    n_posts = 0
                    try:
                        n_posts = len(data["response"]["posts"])
                    except Exception:
                        pass
                    log_f.write(f"[{time.ctime()}] progress: seen={total_seen} ok={total_ok} err={total_err} last_t={tid} last_err={err} last_posts={n_posts}\n")
                    log_f.flush()
                    print(f"[w{worker}] seen={total_seen} ok={total_ok} err={total_err} last={tid}")

                if total_seen % refresh_every == 0:
                    # Refresh session to avoid expiry
                    try:
                        session_id = warm_session(page)
                        log_f.write(f"[{time.ctime()}] refreshed session after {total_seen} reqs\n")
                    except Exception as e:
                        log_f.write(f"[{time.ctime()}] session refresh failed: {e}\n")
        finally:
            out_f.flush()
            out_f.close()
            log_f.write(f"[{time.ctime()}] done shard={shard} worker={worker} seen={total_seen} ok={total_ok} err={total_err}\n")
            log_f.close()
            ctx.close()
            browser.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end",   type=int, required=True)
    ap.add_argument("--shard", required=True, help="Shard name (used in output filenames)")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--refresh-every", type=int, default=500)
    args = ap.parse_args()

    crawl(args.start, args.end, args.shard, args.worker, args.num_workers,
          delay_s=args.delay, headless=not args.headed, refresh_every=args.refresh_every)
    return 0


if __name__ == "__main__":
    sys.exit(main())
