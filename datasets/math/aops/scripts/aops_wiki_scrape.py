"""
AoPS Wiki scraper — contest problem pages, solutions, and answer keys.

Purpose (2026-06-10, see README §8 "Editorial-similarity parallel"): the AoPS
Wiki hosts community-canonical solutions and official answer keys for
AMC / AIME / USAMO / USAJMO / IMO. These give us:
  1. V via answer keys (AMC choice letters, AIME integers 0-999) — checkable
     correctness for forum solutions, no autoformalization.
  2. y via editorial similarity (per forum solution, max embedding similarity
     to the problem's wiki solutions) — the LeetCode editorial recipe.

The wiki sits behind the same Cloudflare as the forum, so we reuse the
Playwright + stealth pattern from aops_bulk_crawl.py and fetch the MediaWiki
API from inside the page context (same-origin, inherits cf_clearance).

Cloudflare notes (2026-06-10): fresh headless contexts from sk3's IP are
hard-challenged and never clear ("Just a moment..." forever) — the long-lived
forum workers only survive on clearance cookies from May 29. Laptop headless
clears instantly. If laptop runs start erroring too, the fallback is
Camoufox (hardened-Firefox anti-detect browser) + Webshare rotating proxies —
API key at ~/.proxies-webshare-key.txt (paid account; pull a proxy batch via
https://proxy.webshare.io/api/v2/proxy/list/).
We pull *wikitext* (action=parse&prop=wikitext) — much cleaner than rendered
HTML: problem statement and '== Solution N ==' sections with raw LaTeX.

Usage (pilot):
    python aops_wiki_scrape.py --contests AIME --years 2015-2024 \
        --out-dir /lfs/skampere3/0/alexspan/aops/wiki --shard pilot
Full:
    python aops_wiki_scrape.py --contests AIME,AMC10,AMC12,AMC8,USAMO,USAJMO,IMO \
        --years 1950-2025 --out-dir /lfs/skampere3/0/alexspan/aops/wiki --shard full

Output: <out-dir>/<shard>.jsonl.gz  — {"title", "kind", "contest", "year",
        "problem", "status", "wikitext" | "error"} per line
        <out-dir>/<shard>.done      — completed titles ledger
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

from playwright.sync_api import sync_playwright, BrowserContext, Page
from playwright_stealth import Stealth

WIKI_API = "https://artofproblemsolving.com/wiki/api.php"
WIKI_HOME = "https://artofproblemsolving.com/wiki/index.php/AoPS_Wiki"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


def title_plan(contests: list[str], y0: int, y1: int) -> Iterable[Tuple[str, dict]]:
    """Yield (page_title, meta) for problem pages and answer keys.

    Title conventions observed on the wiki:
      {Y}_AIME_Problems/Problem_{n}        (1983-1999, n 1-15)
      {Y}_AIME_{I,II}_Problems/Problem_{n} (2000+,    n 1-15)
      {Y}_AMC_{10,12}{A,B}_Problems/Problem_{n} (2000+, n 1-25; 2021 adds Fall variants — fetched, may 404)
      {Y}_AMC_8_Problems/Problem_{n}       (1999+,   n 1-25)
      {Y}_USAMO_Problems/Problem_{n}       (1972+,   n 1-6)
      {Y}_USAJMO_Problems/Problem_{n}      (2010+,   n 1-6)
      {Y}_IMO_Problems/Problem_{n}         (1959+,   n 1-6)
    Answer keys: {prefix}_Answer_Key for AIME/AMC variants.
    Missing pages return a MediaWiki 'missingtitle' error — recorded, cheap.
    """
    for year in range(y0, y1 + 1):
        for c in contests:
            variants: list[Tuple[str, int, bool]] = []  # (prefix, n_problems, has_key)
            if c == "AIME":
                if year < 1983:
                    continue
                if year < 2000:
                    variants = [(f"{year}_AIME", 15, True)]
                else:
                    variants = [(f"{year}_AIME_I", 15, True),
                                (f"{year}_AIME_II", 15, True)]
            elif c in ("AMC10", "AMC12"):
                if year < 2000:
                    continue
                base = c.replace("AMC", "AMC_")
                variants = [(f"{year}_{base}A", 25, True),
                            (f"{year}_{base}B", 25, True)]
            elif c == "AMC8":
                if year < 1999:
                    continue
                variants = [(f"{year}_AMC_8", 25, True)]
            elif c == "USAMO":
                if year < 1972:
                    continue
                variants = [(f"{year}_USAMO", 6, False)]
            elif c == "USAJMO":
                if year < 2010:
                    continue
                variants = [(f"{year}_USAJMO", 6, False)]
            elif c == "IMO":
                if year < 1959:
                    continue
                variants = [(f"{year}_IMO", 6, False)]
            else:
                raise ValueError(f"unknown contest {c}")

            for prefix, n_problems, has_key in variants:
                for n in range(1, n_problems + 1):
                    yield (f"{prefix}_Problems/Problem_{n}",
                           {"kind": "problem", "contest": c, "year": year,
                            "problem": n, "variant": prefix})
                if has_key:
                    yield (f"{prefix}_Answer_Key",
                           {"kind": "answer_key", "contest": c, "year": year,
                            "problem": None, "variant": prefix})


def fetch_wikitext(page: Page, title: str) -> dict:
    url = (f"{WIKI_API}?action=parse&prop=wikitext&format=json&page="
           + urllib.parse.quote(title, safe=""))
    js = """
    async (url) => {
        const res = await fetch(url, {credentials: 'include',
            headers: {'Accept': 'application/json'}});
        const text = await res.text();
        return {status: res.status, text: text};
    }
    """
    result = page.evaluate(js, url)
    if result["status"] != 200:
        raise RuntimeError(f"HTTP {result['status']}: {result['text'][:200]!r}")
    return json.loads(result["text"])


COMMUNITY_URL = "https://artofproblemsolving.com/community"


def warm(page: Page, max_wait_s: float = 75.0) -> None:
    """Clear Cloudflare via /community (known to pass headless+stealth — the
    forum crawler lives on it); cf_clearance is domain-wide, so the wiki API
    XHR inherits it. The wiki home itself does NOT clear headless (tested
    2026-06-10)."""
    page.goto(COMMUNITY_URL, wait_until="domcontentloaded", timeout=60_000)
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        html = page.content()
        if "Just a moment" not in html and "AoPS" in html:
            return
        time.sleep(2.5)
    raise RuntimeError("Cloudflare challenge did not clear on /community")


def load_done(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    return {x.strip() for x in path.read_text().splitlines() if x.strip()}


def scrape(contests: list[str], y0: int, y1: int, out_dir: Path, shard: str,
           delay_s: float, headless: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{shard}.jsonl.gz"
    done_path = out_dir / f"{shard}.done"
    done = load_done(done_path)
    plan = [(t, m) for t, m in title_plan(contests, y0, y1) if t not in done]
    print(f"[{time.ctime()}] {len(plan)} pages to fetch ({len(done)} already done)",
          flush=True)

    n_ok = n_missing = n_err = 0
    with Stealth().use_sync(sync_playwright()) as p:
        browser = p.chromium.launch(headless=headless)
        ctx: BrowserContext = browser.new_context(
            user_agent=UA, viewport={"width": 1366, "height": 900}, locale="en-US")
        page = ctx.new_page()
        warm(page)
        out_f = gzip.open(out_path, "at", encoding="utf-8")
        last_t = 0.0
        try:
            for i, (title, meta) in enumerate(plan):
                wait = delay_s - (time.time() - last_t)
                if wait > 0:
                    time.sleep(wait)
                last_t = time.time()
                rec = {"title": title, **meta}
                try:
                    data = fetch_wikitext(page, title)
                    if "error" in data:
                        rec["status"] = "missing"
                        rec["error"] = data["error"].get("code")
                        n_missing += 1
                    else:
                        rec["status"] = "ok"
                        rec["wikitext"] = data["parse"]["wikitext"]["*"]
                        n_ok += 1
                except Exception as e:
                    try:  # one re-warm attempt (CF challenge / session decay)
                        warm(page)
                        data = fetch_wikitext(page, title)
                        rec["status"] = "ok"
                        rec["wikitext"] = data["parse"]["wikitext"]["*"]
                        n_ok += 1
                    except Exception as e2:
                        rec["status"] = "error"
                        rec["error"] = str(e2)[:200]
                        n_err += 1
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done_path.open("a").write(title + "\n")
                if (i + 1) % 50 == 0:
                    out_f.flush()
                    print(f"[{time.ctime()}] {i+1}/{len(plan)} ok={n_ok} "
                          f"missing={n_missing} err={n_err}", flush=True)
        finally:
            out_f.close()
            ctx.close()
            browser.close()
    print(f"[{time.ctime()}] DONE ok={n_ok} missing={n_missing} err={n_err}",
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contests", default="AIME",
                    help="comma list: AIME,AMC10,AMC12,AMC8,USAMO,USAJMO,IMO")
    ap.add_argument("--years", default="2015-2024", help="Y0-Y1 inclusive")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()
    y0, y1 = (int(x) for x in args.years.split("-"))
    scrape([c.strip() for c in args.contests.split(",")], y0, y1,
           Path(args.out_dir), args.shard, args.delay, not args.headed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
