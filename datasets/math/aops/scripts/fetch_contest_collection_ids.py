"""
Enumerate AoPS community Contest Collections -> topic_ids for the crawler's
priority queue.

The wiki scrape's wikitext has almost no forum links (~113 resolvable topic
ids total), so the canonical source for contest-thread ids is the community
category tree rooted at c13 "Contest Collections": folders -> contest folders
-> per-year "view_posts" leaf categories whose items carry post_data.topic_id.

We scope to the contests the wiki scrape covers (IMO, AMC 8/10/12, AIME,
USAMO, USAJMO) so the run stays ~400 requests. Fetches go through headless
Playwright from inside an artofproblemsolving.com page (same-origin ajax,
inherits Cloudflare clearance — plain curl 403s).

Pagination: fetch_more_items requires echoing BOTH last_item_score and
last_item_text from the previous page; fetch_before alone is silently ignored.

Usage:
    python scripts/fetch_contest_collection_ids.py --out contest_topic_ids.txt
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth

ROOT_CATEGORY = 13  # Contest Collections

# Depth-1/2 folders we are willing to descend into looking for target contests
DESCEND_PAT = re.compile(
    r"^(International Contests|National and Regional Contests|National Olympiads"
    r"|Junior Olympiads|Team Selection Tests|Undergraduate Contests"
    r"|USA Contests|United States.*|MAA AMC)$", re.I)
# Contest folders whose year-leaves we harvest
TARGET_PAT = re.compile(
    r"^(IMO|AMC\s*8|AMC\s*10|AMC\s*12(/AHSME)?|AMC\s*10/12"
    r"|AIME( Problems)?|USAMO|USAJMO)$", re.I)

AJAX_JS = """
async (args) => {
    const form = new URLSearchParams();
    for (const [k, v] of Object.entries(args.payload)) form.append(k, v);
    const res = await fetch('https://artofproblemsolving.com/m/community/ajax.php', {
        method: 'POST', credentials: 'include',
        headers: {'Accept':'application/json, text/javascript, */*; q=0.01',
                  'Content-Type':'application/x-www-form-urlencoded; charset=UTF-8',
                  'X-Requested-With':'XMLHttpRequest'},
        body: form.toString()});
    return {status: res.status, text: await res.text()};
}
"""


class Fetcher:
    def __init__(self, page, session_id: str, delay_s: float = 1.0):
        self.page, self.sid, self.delay = page, session_id, delay_s
        self.n_requests = 0

    def _ajax(self, payload: dict) -> dict:
        time.sleep(self.delay)
        self.n_requests += 1
        base = {"aops_logged_in": "false", "aops_user_id": "1",
                "aops_session_id": self.sid, "user_id": "0",
                "fetch_archived": "0", "fetch_announcements": "0"}
        r = self.page.evaluate(AJAX_JS, {"payload": {**base, **payload}})
        if r["status"] != 200:
            raise RuntimeError(f"HTTP {r['status']}: {r['text'][:120]!r}")
        return json.loads(r["text"])

    def category_items(self, cat_id: int) -> tuple[dict, list[dict]]:
        d = self._ajax({"a": "fetch_category_data", "category_id": str(cat_id),
                        "category_type": "folder", "log_visit": "0",
                        "fetch_before": "0"})
        cat = d["response"]["category"]
        items = list(cat.get("items") or [])
        no_more = cat.get("no_more_items", True)
        lis, lit = cat.get("last_item_score"), cat.get("last_item_text")
        seen = {it.get("item_id") for it in items}
        while not no_more and lis is not None:
            d2 = self._ajax({"a": "fetch_more_items", "category_id": str(cat_id),
                             "fetch_before": str(lis), "last_item_score": str(lis),
                             "last_item_text": str(lit or "")})
            resp = d2["response"]
            new = [it for it in (resp.get("items") or [])
                   if it.get("item_id") not in seen]
            if not new:
                break
            items.extend(new)
            seen.update(it.get("item_id") for it in new)
            no_more = resp.get("no_more_items", True)
            lis, lit = resp.get("last_item_score"), resp.get("last_item_text")
        return cat, items


def warm(page) -> str:
    page.goto("https://artofproblemsolving.com/community",
              wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_timeout(4_000)
    html = page.content()
    i = html.find("AoPS.session")
    m = re.search(r"([0-9a-f]{32})", html[i:i + 400]) if i >= 0 else None
    if not m:
        raise RuntimeError("no AoPS.session — Cloudflare challenge?")
    return m.group(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--max-requests", type=int, default=1500)
    ap.add_argument("--folders", default=None,
                    help="Comma-separated category ids: skip the BFS and "
                         "harvest these contest folders directly")
    args = ap.parse_args()

    topic_ids: set[int] = set()
    with Stealth().use_sync(sync_playwright()) as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        sid = warm(page)
        print(f"session={sid[:8]}…", flush=True)
        fx = Fetcher(page, sid, args.delay)

        # BFS folders; harvest year-leaves under TARGET contest folders.
        target_folders: list[tuple[int, str]] = []
        if args.folders:
            target_folders = [(int(c), f"c{c}") for c in args.folders.split(",")]
        queue: list[tuple[int, str, int]] = (
            [] if args.folders else [(ROOT_CATEGORY, "Contest Collections", 0)])
        while queue:
            cid, name, depth = queue.pop(0)
            _, items = fx.category_items(cid)
            folders = [it for it in items if it.get("item_type") == "folder"]
            print(f"folder c{cid} {name!r} depth={depth}: {len(items)} items, "
                  f"{len(folders)} subfolders", flush=True)
            for it in folders:
                t = str(it.get("item_text") or "").strip()
                if TARGET_PAT.match(t):
                    target_folders.append((it["item_id"], t))
                elif depth < 3 and DESCEND_PAT.match(t):
                    queue.append((it["item_id"], t, depth + 1))

        print(f"target contest folders: {[(c, n) for c, n in target_folders]}", flush=True)

        leaves: list[tuple[int, str]] = []
        for cid, name in target_folders:
            _, items = fx.category_items(cid)
            for it in items:
                if it.get("item_type") == "view_posts":
                    leaves.append((it["item_id"], str(it.get("item_text"))))
                elif it.get("item_type") == "folder":
                    # e.g. AIME -> "AIME I"/"AIME II" subfolders
                    _, sub = fx.category_items(it["item_id"])
                    leaves.extend((s["item_id"], str(s.get("item_text")))
                                  for s in sub if s.get("item_type") == "view_posts")
            print(f"contest {name!r}: cumulative leaves={len(leaves)} "
                  f"reqs={fx.n_requests}", flush=True)

        for n, (leaf_id, leaf_name) in enumerate(leaves, 1):
            if fx.n_requests >= args.max_requests:
                print(f"STOP: hit --max-requests={args.max_requests}", flush=True)
                break
            try:
                _, items = fx.category_items(leaf_id)
            except Exception as e:
                print(f"leaf c{leaf_id} {leaf_name!r} ERR {str(e)[:90]}", flush=True)
                continue
            got = {it["post_data"]["topic_id"] for it in items
                   if it.get("item_type") == "post" and it.get("post_data")}
            topic_ids.update(int(t) for t in got)
            print(f"[{n}/{len(leaves)}] c{leaf_id} {leaf_name!r}: +{len(got)} "
                  f"(total {len(topic_ids)}) reqs={fx.n_requests}", flush=True)
        browser.close()

    Path(args.out).write_text("".join(f"{t}\n" for t in sorted(topic_ids)))
    print(f"wrote {len(topic_ids)} unique topic ids -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
