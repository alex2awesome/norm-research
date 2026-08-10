"""
AoPS crawler v2 — Camoufox (anti-detect Firefox) + Webshare rotating proxies.

Drop-in replacement for aops_bulk_crawl.py (same CLI, same output contract).
The old Chromium+stealth stack has been 100% Cloudflare-blocked since
2026-05-30; Camoufox passes the CF browser checks and proxies give us fresh
IPs when one gets burned.

Usage:
    # Pilot through a proxy:
    python scripts/aops_camoufox_crawl.py --start 495500 --end 495530 --shard camoufox_pilot

    # Resume same shard (re-running skips ids in the .done ledger):
    python scripts/aops_camoufox_crawl.py --start 495500 --end 495530 --shard camoufox_pilot

    # Parallel workers (modulo sharding; each starts on a different proxy):
    python scripts/aops_camoufox_crawl.py --start 1 --end 1000000 --worker 0 --num-workers 4 --shard full &
    python scripts/aops_camoufox_crawl.py --start 1 --end 1000000 --worker 1 --num-workers 4 --shard full &

    # No proxy (direct connection):
    python scripts/aops_camoufox_crawl.py ... --no-proxy

Proxy rotation: on --cf-threshold consecutive Cloudflare failures (HTTP
403/429/503 or a "Just a moment" challenge page), the browser is closed, the
next proxy in the Webshare list is selected, and a fresh browser is launched
and re-warmed. Every rotation is logged. Default starting proxy is
(worker % n_proxies) so parallel workers spread across IPs; override with
--proxy-index.

Output layout:
    raw/shards/<shard>__w<worker>.jsonl      — one topic per line, raw json blob
                                               (plain text, append-only; compress
                                               offline — gzip "at" streams proved
                                               crash-unsafe and corrupted ~46% of
                                               records when workers were killed)
    raw/shards/<shard>__w<worker>.done       — set of completed topic_ids (text)

Priority queue: pass --priority-file <path> (one topic_id per line). Each
worker first crawls the priority ids matching its modulo (skipping ids already
in its .done ledger), then falls back to its normal --start/--end sweep.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from camoufox.sync_api import Camoufox
from playwright.sync_api import Error as PlaywrightError, Page

sys.path.insert(0, str(Path(__file__).resolve().parent))
from webshare_proxies import fetch_proxy_list  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "raw"
SHARDS = RAW / "shards"
LOGS = REPO / "logs"
for d in (RAW, SHARDS, LOGS):
    d.mkdir(parents=True, exist_ok=True)

AJAX_URL = "https://artofproblemsolving.com/m/community/ajax.php"
COMMUNITY_URL = "https://artofproblemsolving.com/community"


class CloudflareBlocked(Exception):
    """Request or page-warm was intercepted by a Cloudflare challenge/block."""


def _looks_like_cloudflare(text: str) -> bool:
    low = text[:4000].lower()
    return (
        "just a moment" in low
        or "cf-chl" in low
        or "challenge-platform" in low
        or "attention required! | cloudflare" in low
    )


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


def _has_cf_clearance(page: Page) -> bool:
    try:
        return any(c["name"] == "cf_clearance" for c in page.context.cookies())
    except Exception:
        return False


def warm_session(page: Page, max_wait_s: float = 60.0,
                 clearance_grace_s: float = 8.0) -> str:
    """Load /community and pull AoPS.session out of the page source.

    Camoufox usually clears the Cloudflare JS challenge on its own within a
    few seconds, so we poll until the session id appears. We then give CF a
    short grace window to drop the cf_clearance cookie — the ajax.php POST
    403s until that cookie exists (the HTML page itself can render first).
    """
    page.goto(COMMUNITY_URL, wait_until="domcontentloaded", timeout=90_000)
    deadline = time.time() + max_wait_s
    html = ""
    sid: Optional[str] = None
    sid_found_at: Optional[float] = None
    while time.time() < deadline:
        html = page.content()
        if sid is None:
            sid = _extract_session_id(html)
            if sid:
                sid_found_at = time.time()
        if sid and _has_cf_clearance(page):
            return sid
        if sid and sid_found_at and time.time() - sid_found_at > clearance_grace_s:
            # sid found and grace window for cf_clearance elapsed; proceed —
            # CF may simply not have issued a challenge on this IP.
            return sid
        time.sleep(1.5)
    if sid:
        return sid
    if _looks_like_cloudflare(html):
        raise CloudflareBlocked("warm_session stuck on Cloudflare challenge")
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
    if status in (403, 429, 503):
        raise CloudflareBlocked(f"HTTP {status} t={topic_id}: {text[:200]!r}")
    if status != 200:
        raise RuntimeError(f"HTTP {status} t={topic_id}: {text[:200]!r}")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if _looks_like_cloudflare(text):
            raise CloudflareBlocked(f"CF challenge body t={topic_id}: {text[:120]!r}")
        raise RuntimeError(f"non-JSON 200 t={topic_id}: {text[:200]!r}")


def load_done(done_path: Path) -> Set[int]:
    if not done_path.exists():
        return set()
    return {int(x) for x in done_path.read_text().split() if x.strip().isdigit()}


def _proxy_label(proxy: Optional[Dict[str, str]]) -> str:
    if proxy is None:
        return "DIRECT"
    return f"{proxy['server']} ({proxy.get('country', '?')})"


def crawl(start: int, end: int, shard: str, worker: int, num_workers: int,
          delay_s: float = 2.0, headless: bool = True, refresh_every: int = 500,
          proxies: Optional[List[Dict[str, str]]] = None,
          proxy_index: Optional[int] = None,
          cf_threshold: int = 3, max_rotations: int = 200,
          priority_ids: Optional[Set[int]] = None) -> None:
    out_path = SHARDS / f"{shard}__w{worker}.jsonl"
    done_path = SHARDS / f"{shard}__w{worker}.done"
    log_path = LOGS / f"{shard}__w{worker}.log"

    done = load_done(done_path)
    priority = []
    if priority_ids:
        priority = sorted(t for t in priority_ids
                          if t % num_workers == worker and t not in done)
    prio_set = set(priority)
    remaining = priority + [t for t in range(start, end)
                            if t % num_workers == worker and t not in done
                            and t not in prio_set]

    pool: List[Optional[Dict[str, str]]] = list(proxies) if proxies else [None]
    if proxy_index is None:
        proxy_index = worker % len(pool)
    p_idx = proxy_index % len(pool)

    total_seen = total_ok = total_err = rotations = 0
    log_f = log_path.open("a")

    def log(msg: str) -> None:
        log_f.write(f"[{time.ctime()}] {msg}\n")
        log_f.flush()

    log(f"start range=[{start},{end}) shard={shard} worker={worker}/{num_workers} "
        f"delay={delay_s}s done_already={len(done)} remaining={len(remaining)} "
        f"priority={len(priority)} "
        f"proxies={len(pool) if pool != [None] else 0} proxy_index={p_idx} "
        f"cf_threshold={cf_threshold}")

    i = 0
    last_t = 0.0
    # Plain-text append: each record is written and flushed atomically enough
    # that a kill/crash loses at most the final partial line (recoverable),
    # unlike a long-lived gzip "at" stream whose tail deflate block is lost.
    out_f = out_path.open("a", encoding="utf-8")
    try:
        while i < len(remaining):
            proxy = pool[p_idx]
            log(f"launching camoufox via {_proxy_label(proxy)}")
            print(f"[w{worker}] launching camoufox via {_proxy_label(proxy)}")
            launch_kw: dict = {"headless": headless}
            if proxy is not None:
                launch_kw["proxy"] = {k: proxy[k] for k in ("server", "username", "password")}
                launch_kw["geoip"] = True
            try:
                with Camoufox(**launch_kw) as browser:
                    page = browser.new_page()
                    session_id = warm_session(page)
                    cf_strikes = 0
                    print(f"[w{worker}] session={session_id[:8]}…")
                    log(f"session_id={session_id}")

                    while i < len(remaining):
                        tid = remaining[i]
                        wait = delay_s - (time.time() - last_t)
                        if wait > 0:
                            time.sleep(wait)
                        last_t = time.time()
                        total_seen += 1

                        try:
                            data = fetch_topic(page, tid, session_id)
                        except CloudflareBlocked as e:
                            cf_strikes += 1
                            total_seen -= 1  # retried, not consumed
                            log(f"t={tid} CF strike {cf_strikes}/{cf_threshold}: {str(e)[:200]}")
                            if cf_strikes >= cf_threshold:
                                raise  # → rotate proxy below
                            session_id = warm_session(page)  # CF here also rotates
                            continue  # retry same tid
                        except Exception as e:
                            log(f"t={tid} ERR: {str(e)[:200]}")
                            try:  # one re-warm + retry, as in v1
                                session_id = warm_session(page)
                                data = fetch_topic(page, tid, session_id)
                                log(f"t={tid} recovered after re-warm")
                            except CloudflareBlocked:
                                raise
                            except Exception as e2:
                                log(f"t={tid} ERR after re-warm: {str(e2)[:200]}")
                                total_err += 1
                                i += 1
                                continue

                        cf_strikes = 0
                        out_f.write(json.dumps({"topic_id": tid, "response": data},
                                               ensure_ascii=False))
                        out_f.write("\n")
                        out_f.flush()
                        done_path.open("a").write(f"{tid}\n")
                        total_ok += 1
                        i += 1

                        if total_seen % 50 == 0:
                            err = data.get("error_code") if isinstance(data, dict) else None
                            try:
                                n_posts = len(data["response"]["posts"])
                            except Exception:
                                n_posts = 0
                            log(f"progress: seen={total_seen} ok={total_ok} err={total_err} "
                                f"last_t={tid} prio={tid in prio_set} "
                                f"last_err={err} last_posts={n_posts}")
                            print(f"[w{worker}] seen={total_seen} ok={total_ok} "
                                  f"err={total_err} last={tid}")

                        if total_seen % refresh_every == 0:
                            try:
                                session_id = warm_session(page)
                                log(f"refreshed session after {total_seen} reqs")
                            except CloudflareBlocked:
                                raise
                            except Exception as e:
                                log(f"session refresh failed: {e}")
            except (CloudflareBlocked, PlaywrightError) as e:
                # PlaywrightError here = browser/page crash or nav failure that
                # escaped the per-topic handlers (e.g. proxy died mid-warm).
                # Treat it like a burned proxy: advance and relaunch.
                rotations += 1
                if rotations > max_rotations:
                    log(f"ABORT: exceeded max_rotations={max_rotations}; all proxies burned?")
                    raise RuntimeError("exceeded --max-rotations; aborting") from e
                old = _proxy_label(proxy)
                p_idx = (p_idx + 1) % len(pool)
                kind = "CF" if isinstance(e, CloudflareBlocked) else "BROWSER"
                log(f"ROTATE #{rotations} [{kind}]: {old} failed ({str(e)[:150]}) "
                    f"→ {_proxy_label(pool[p_idx])}")
                print(f"[w{worker}] ROTATE #{rotations} [{kind}]: {old} → {_proxy_label(pool[p_idx])}")
                out_f.flush()
    finally:
        out_f.flush()
        out_f.close()
        log(f"done shard={shard} worker={worker} seen={total_seen} ok={total_ok} "
            f"err={total_err} rotations={rotations}")
        log_f.close()
        print(f"[w{worker}] DONE ok={total_ok} err={total_err} rotations={rotations}")


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
    ap.add_argument("--no-proxy", action="store_true",
                    help="Connect directly (no Webshare proxy)")
    ap.add_argument("--proxy-index", type=int, default=None,
                    help="Starting index into the Webshare proxy list "
                         "(default: worker %% n_proxies)")
    ap.add_argument("--key-path", default=None,
                    help="Webshare API key file (default ~/.proxies-webshare-key.txt)")
    ap.add_argument("--cf-threshold", type=int, default=3,
                    help="Consecutive Cloudflare failures before rotating proxy")
    ap.add_argument("--max-rotations", type=int, default=200)
    ap.add_argument("--priority-file", default=None,
                    help="File of topic_ids (one per line) to crawl first; "
                         "each worker takes ids matching its modulo")
    args = ap.parse_args()

    priority_ids: Optional[Set[int]] = None
    if args.priority_file:
        priority_ids = {int(x) for x in Path(args.priority_file).read_text().split()
                        if x.strip().isdigit()}
        print(f"loaded {len(priority_ids)} priority topic ids from {args.priority_file}")

    proxies = None
    if not args.no_proxy:
        proxies = fetch_proxy_list(args.key_path)
        print(f"loaded {len(proxies)} webshare proxies")

    crawl(args.start, args.end, args.shard, args.worker, args.num_workers,
          delay_s=args.delay, headless=not args.headed,
          refresh_every=args.refresh_every, proxies=proxies,
          proxy_index=args.proxy_index, cf_threshold=args.cf_threshold,
          max_rotations=args.max_rotations, priority_ids=priority_ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
