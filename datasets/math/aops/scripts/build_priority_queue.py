"""
Build the contest-thread priority queue for the forum crawler.

Scans the wiki scrape (wiki/*.jsonl.gz, wikitext per page) for links into the
AoPS community and emits one topic_id per line, for use with
aops_camoufox_crawl.py --priority-file.

Link forms actually present in the wikitext (census 2026-06-11):
  - artofproblemsolving.com/community/c<digits>h<TOPIC>p<post>   (direct)
  - .../community/q2h<TOPIC>p<post>                              (direct)
  - .../Forum/viewtopic.php?t=<TOPIC>                            (direct)
  - .../community/p<POST>            -> needs redirect resolution
  - .../Forum/viewtopic.php?p=<POST> -> needs redirect resolution

Post-id links are resolved to topic ids by loading the URL in headless
Playwright (laptop clears Cloudflare; plain curl 403s) and reading the
final redirected URL, which has the canonical c<cat>h<TOPIC>p<post> form.

Usage:
    python scripts/build_priority_queue.py --out priority_contest_topics.txt \
        [--skip-resolve]
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WIKI = REPO / "wiki"

TOPIC_PATS = [
    re.compile(r"/community/[a-z]?\d*h(\d+)", re.I),
    re.compile(r"viewtopic\.php\?[^\s\]\|<>\"'\)}]*?[?&]?t=(\d+)", re.I),
]
POST_PATS = [
    re.compile(r"/community/p(\d+)", re.I),
    re.compile(r"viewtopic\.php\?[^\s\]\|<>\"'\)}]*?[?&]p=(\d+)", re.I),
]


def extract() -> tuple[set[int], set[int]]:
    topic_ids: set[int] = set()
    post_ids: set[int] = set()
    for shard in sorted(WIKI.glob("*.jsonl.gz")):
        for line in gzip.open(shard, "rt"):
            wt = json.loads(line).get("wikitext") or ""
            for pat in TOPIC_PATS:
                topic_ids.update(int(m) for m in pat.findall(wt))
            for pat in POST_PATS:
                post_ids.update(int(m) for m in pat.findall(wt))
    return topic_ids, post_ids


def resolve_posts(post_ids: set[int], delay_s: float = 1.5) -> set[int]:
    from playwright.sync_api import sync_playwright
    from playwright_stealth import Stealth

    resolved: set[int] = set()
    h_pat = re.compile(r"/community/[a-z]?\d*h(\d+)", re.I)
    with Stealth().use_sync(sync_playwright()) as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        for n, pid in enumerate(sorted(post_ids), 1):
            url = f"https://artofproblemsolving.com/community/p{pid}"
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                # AoPS rewrites the URL client-side to /community/cXhYpZ.
                # NB: must pump the Playwright event loop (wait_for_timeout),
                # NOT time.sleep — page.url only updates during API calls.
                deadline = time.time() + 12
                m = None
                while time.time() < deadline:
                    m = h_pat.search(page.url)
                    if m:
                        break
                    page.wait_for_timeout(500)
                if m:
                    resolved.add(int(m.group(1)))
                    print(f"[{n}/{len(post_ids)}] p{pid} -> h{m.group(1)}")
                else:
                    print(f"[{n}/{len(post_ids)}] p{pid} UNRESOLVED url={page.url[:90]}")
            except Exception as e:
                print(f"[{n}/{len(post_ids)}] p{pid} ERR {str(e)[:90]}")
            time.sleep(delay_s)
        browser.close()
    return resolved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "priority_contest_topics.txt"))
    ap.add_argument("--skip-resolve", action="store_true",
                    help="Skip Playwright post-id resolution (direct ids only)")
    args = ap.parse_args()

    topic_ids, post_ids = extract()
    print(f"direct topic ids: {len(topic_ids)}  post ids to resolve: {len(post_ids)}")
    if not args.skip_resolve and post_ids:
        topic_ids |= resolve_posts(post_ids)
    out = Path(args.out)
    out.write_text("".join(f"{t}\n" for t in sorted(topic_ids)))
    print(f"wrote {len(topic_ids)} unique topic ids -> {out}")
    return 0


if __name__ == "__main__":
    return_code = main()
    raise SystemExit(return_code)
