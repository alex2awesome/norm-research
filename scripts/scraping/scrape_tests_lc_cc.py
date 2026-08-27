"""
scrape_tests_lc_cc.py
=====================

Scrape visible test cases (Input/Output pairs) from LeetCode and CodeChef
problem pages for the Phase 1b sample.

Inputs
------
- outputs/v2_analysis/comp_qwen_phase1b_sample.parquet   (pids per platform)
- datasets/codechef/editorials.parquet                   (slug -> problem_code)

Outputs (one JSON per problem)
------------------------------
- datasets/leetcode_editorials/tests_scraped/{slug}.json
- datasets/codechef/tests_scraped/{slug}.json

Schema
------
{
  "platform": "lc" | "cc",
  "slug": "...",
  "url": "...",
  "tests": [{"input": "...", "output": "...", "explanation": "..."}],
  "scraped_at": ISO ts,
  "scrape_status": "ok" | "no_tests" | "blocked" | "404" | "error",
  "error_message": "..."
}

Usage
-----
    python scripts/scraping/scrape_tests_lc_cc.py --platform lc --limit 20
    python scripts/scraping/scrape_tests_lc_cc.py --platform cc --limit 20
    python scripts/scraping/scrape_tests_lc_cc.py --platform both

Resume-resilient: skips slugs that already have an output JSON file.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")
sys.path.insert(0, str(ROOT / "scripts" / "scraping"))

from stealth_fetch import fetch_url  # noqa: E402

PHASE1B_PARQUET = ROOT / "outputs" / "v2_analysis" / "comp_qwen_phase1b_sample.parquet"
CC_EDITORIALS_PARQUET = ROOT / "datasets" / "codechef" / "editorials.parquet"

LC_OUT_DIR = ROOT / "datasets" / "leetcode_editorials" / "tests_scraped"
CC_OUT_DIR = ROOT / "datasets" / "codechef" / "tests_scraped"
LOG_DIR = ROOT / "outputs" / "scraping" / "test_scrape_logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)
LC_OUT_DIR.mkdir(parents=True, exist_ok=True)
CC_OUT_DIR.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.handlers = []
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    log.addHandler(sh)
    fh = logging.FileHandler(LOG_DIR / f"{name}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    log.addHandler(fh)
    return log


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# Strip HTML tags conservatively (keep content)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")


def strip_tags(s: str) -> str:
    s = re.sub(r"<script.*?</script>", " ", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<style.*?</style>", " ", s, flags=re.DOTALL | re.IGNORECASE)
    s = _TAG_RE.sub(" ", s)
    s = re.sub(r"&nbsp;", " ", s)
    s = re.sub(r"&lt;", "<", s)
    s = re.sub(r"&gt;", ">", s)
    s = re.sub(r"&amp;", "&", s)
    s = re.sub(r"&quot;", '"', s)
    s = re.sub(r"&#39;", "'", s)
    return s


# ---- LeetCode ----------------------------------------------------------

# Match Example sections of the form:
#   Example N:
#       Input: ...
#       Output: ...
#       (optional) Explanation: ...
# Stop at the next "Example", "Constraints", or "Follow-up" header.
LC_EXAMPLE_RE = re.compile(
    r"Example\s*\d+\s*:?\s*"
    r"(?:.*?Input\s*:?\s*(?P<input>.*?))?"
    r"\n+\s*Output\s*:?\s*(?P<output>.*?)"
    r"(?:\n+\s*Explanation\s*:?\s*(?P<explanation>.*?))?"
    r"(?=\n\s*(?:Example\s*\d+|Constraints|Follow[- ]up|Note\s*:|&nbsp;)|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def parse_leetcode_html(html: str) -> Tuple[List[Dict[str, str]], str]:
    """
    Returns (tests, status). status in {"ok", "no_tests", "blocked"}.
    """
    if "Sign in" in html and "translation-progress" in html and len(html) < 30000:
        return [], "blocked"

    text = strip_tags(html)
    # Collapse repeated newlines but keep one
    text = re.sub(r"\r\n", "\n", text)
    # Normalize multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    tests: List[Dict[str, str]] = []
    for m in LC_EXAMPLE_RE.finditer(text):
        inp = (m.group("input") or "").strip()
        out = (m.group("output") or "").strip()
        expl = (m.group("explanation") or "").strip()
        # Truncate runaway captures (>4000 chars suggests we crossed boundaries)
        if len(inp) > 4000:
            inp = inp[:4000]
        if len(out) > 4000:
            out = out[:4000]
        if len(expl) > 4000:
            expl = expl[:4000]
        if inp or out:
            tests.append({"input": inp, "output": out, "explanation": expl})

    if not tests:
        return [], "no_tests"
    return tests, "ok"


# ---- CodeChef -------------------------------------------------------------

# After hydration, CodeChef problem pages have one or more blocks like:
#   <h3 ...>Sample N:</h3>
#   <div ..._input_output__table_...>
#       <div ..._text_copy__container_...>
#           <div ..._text_copy__..._input_top__box_...><span>Input</span> ...</div>
#           <div ..._text_copy__..._ouput_top__box_...><span>Output</span> ...</div>
#       </div>
#       <div ..._values__container_...>
#           <div ..._values_...><pre>...input data...</pre></div>
#           <div ..._values_...><pre>...output data...</pre></div>
#       </div>
#   </div>

CC_SAMPLE_RE = re.compile(
    r"Sample\s*\d+\s*:.*?"
    r"_values__container[^>]*>\s*"
    r"<div[^>]*_values_[^>]*>\s*<pre[^>]*>(?P<input>.*?)</pre>\s*</div>\s*"
    r"<div[^>]*_values_[^>]*>\s*<pre[^>]*>(?P<output>.*?)</pre>\s*</div>",
    re.DOTALL | re.IGNORECASE,
)


def _decode_pre(content: str) -> str:
    # Strip any inner tags, decode entities.
    content = strip_tags(content)
    # CC pre blocks preserve newlines; do not collapse them.
    content = content.replace(" ", " ")
    return content.strip("\n")


# Legacy CC format: <h3>Example</h3>...<pre>...<b>Input:</b> ... <b>Output:</b> ...</pre>
# Multiple Examples possible. Each Example is a single <pre> block, but Input/Output
# pairs may be repeated inside.
CC_LEGACY_EXAMPLE_RE = re.compile(
    r"<h[1-4][^>]*>\s*(?:Example|Sample)[^<]*</h[1-4]>\s*"
    r"<pre[^>]*>(?P<block>.*?)</pre>",
    re.DOTALL | re.IGNORECASE,
)


def _parse_legacy_block(block: str) -> List[Dict[str, str]]:
    """Parse a <pre>...Input:...Output:...</pre> block; multiple I/O pairs possible."""
    text = _decode_pre(block)
    pat = re.compile(
        r"Input\s*:?\s*(?P<input>.*?)\n\s*Output\s*:?\s*(?P<output>.*?)(?=\n\s*Input\s*:|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    tests: List[Dict[str, str]] = []
    for m in pat.finditer(text):
        inp = m.group("input").strip()
        out = m.group("output").strip()
        if inp or out:
            tests.append({"input": inp, "output": out, "explanation": ""})
    return tests


def parse_codechef_html(html: str) -> Tuple[List[Dict[str, str]], str]:
    if 'You need to enable JavaScript' in html and len(html) < 25000:
        return [], "blocked"
    tests: List[Dict[str, str]] = []

    # Modern (post-redesign) layout
    for m in CC_SAMPLE_RE.finditer(html):
        inp = _decode_pre(m.group("input"))
        out = _decode_pre(m.group("output"))
        if inp or out:
            tests.append({"input": inp, "output": out, "explanation": ""})

    if tests:
        return tests, "ok"

    # Legacy layout
    for m in CC_LEGACY_EXAMPLE_RE.finditer(html):
        tests.extend(_parse_legacy_block(m.group("block")))

    if not tests:
        return [], "no_tests"
    return tests, "ok"


# ---------------------------------------------------------------------------
# Job loaders
# ---------------------------------------------------------------------------


def load_lc_jobs() -> List[Tuple[str, str]]:
    df = pd.read_parquet(PHASE1B_PARQUET)
    pids = df[df.platform == "lc"]["canonical_pid"].drop_duplicates().tolist()
    jobs = []
    for pid in pids:
        slug = pid.split(":", 1)[1] if ":" in pid else pid
        url = f"https://leetcode.com/problems/{slug}/"
        jobs.append((slug, url))
    return jobs


def load_cc_jobs() -> List[Tuple[str, str]]:
    df = pd.read_parquet(PHASE1B_PARQUET)
    ed = pd.read_parquet(CC_EDITORIALS_PARQUET)
    ed = ed[["topic_slug", "problem_code", "problem_url"]].copy()
    ed["bare_slug"] = ed["topic_slug"].str.replace(r"-editorial$", "", regex=True)
    # Map slug -> problem_code (uppercase) and URL
    code_by_slug: Dict[str, Tuple[str, str]] = {}
    for _, r in ed.iterrows():
        slug = (r["bare_slug"] or "").lower()
        code = r["problem_code"]
        url = r["problem_url"]
        if slug and isinstance(code, str) and code:
            if slug not in code_by_slug:
                code_by_slug[slug] = (code, url or f"https://www.codechef.com/problems/{code}")
    pids = df[df.platform == "cc"]["canonical_pid"].drop_duplicates().tolist()
    jobs: List[Tuple[str, str]] = []
    missing = 0
    for pid in pids:
        slug = pid.split(":", 1)[1] if ":" in pid else pid
        slug_l = slug.lower()
        if slug_l in code_by_slug:
            code, url = code_by_slug[slug_l]
        else:
            # Fallback: try uppercase slug as problem code
            if not slug:
                missing += 1
                continue
            code = slug.upper()
            url = f"https://www.codechef.com/problems/{code}"
            missing += 1
        jobs.append((slug, url))
    print(f"[cc] {len(jobs)} jobs; {missing} via fallback (no editorial-parquet match)")
    return jobs


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


async def scrape_one_lc(slug: str, url: str, log: logging.Logger) -> Dict:
    out_path = LC_OUT_DIR / f"{slug}.json"
    if out_path.exists():
        return {"slug": slug, "status": "skipped"}
    rec: Dict = {
        "platform": "lc",
        "slug": slug,
        "url": url,
        "tests": [],
        "scraped_at": dt.datetime.utcnow().isoformat() + "Z",
        "scrape_status": "error",
        "error_message": "",
    }
    try:
        status, html = await fetch_url(
            url,
            wait_until="domcontentloaded",
            extra_wait_ms=2500,
            timeout_ms=45000,
            jitter=(2.0, 5.0),
            respect_robots=True,  # LC robots.txt allows /problems/
        )
        if status == 404:
            rec["scrape_status"] = "404"
        elif status < 200 or status >= 400 or not html:
            rec["scrape_status"] = "error"
            rec["error_message"] = f"http_status={status}"
        else:
            tests, parse_status = parse_leetcode_html(html)
            rec["tests"] = tests
            rec["scrape_status"] = parse_status
    except Exception as e:  # noqa: BLE001
        rec["scrape_status"] = "error"
        rec["error_message"] = f"{type(e).__name__}: {e}"
        log.warning("LC %s: %s", slug, rec["error_message"])
    out_path.write_text(json.dumps(rec, ensure_ascii=False))
    return {"slug": slug, "status": rec["scrape_status"], "n_tests": len(rec["tests"])}


async def scrape_one_cc(slug: str, url: str, log: logging.Logger) -> Dict:
    out_path = CC_OUT_DIR / f"{slug}.json"
    if out_path.exists():
        return {"slug": slug, "status": "skipped"}
    rec: Dict = {
        "platform": "cc",
        "slug": slug,
        "url": url,
        "tests": [],
        "scraped_at": dt.datetime.utcnow().isoformat() + "Z",
        "scrape_status": "error",
        "error_message": "",
    }
    try:
        status, html = await fetch_url(
            url,
            wait_until="networkidle",
            extra_wait_ms=6000,
            timeout_ms=60000,
            jitter=(2.0, 5.0),
            respect_robots=False,  # CC robots disallows /problems/* — research scrape
        )
        if status == 404:
            rec["scrape_status"] = "404"
        elif status < 200 or status >= 400 or not html:
            rec["scrape_status"] = "error"
            rec["error_message"] = f"http_status={status}"
        else:
            tests, parse_status = parse_codechef_html(html)
            rec["tests"] = tests
            rec["scrape_status"] = parse_status
    except Exception as e:  # noqa: BLE001
        rec["scrape_status"] = "error"
        rec["error_message"] = f"{type(e).__name__}: {e}"
        log.warning("CC %s: %s", slug, rec["error_message"])
    out_path.write_text(json.dumps(rec, ensure_ascii=False))
    return {"slug": slug, "status": rec["scrape_status"], "n_tests": len(rec["tests"])}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def run_platform(
    platform: str,
    jobs: List[Tuple[str, str]],
    concurrency: int,
    log: logging.Logger,
) -> Dict[str, int]:
    sem = asyncio.Semaphore(concurrency)
    counts = {"ok": 0, "no_tests": 0, "blocked": 0, "404": 0, "error": 0, "skipped": 0}
    counts_lock = asyncio.Lock()

    async def worker(slug: str, url: str, idx: int):
        async with sem:
            if platform == "lc":
                res = await scrape_one_lc(slug, url, log)
            else:
                res = await scrape_one_cc(slug, url, log)
            async with counts_lock:
                counts[res["status"]] = counts.get(res["status"], 0) + 1
                if idx % 25 == 0 or idx < 25:
                    n_tests = res.get("n_tests", 0)
                    total_done = sum(counts.values())
                    log.info(
                        "[%s %d/%d] %s -> %s (tests=%s)  cumulative=%s",
                        platform,
                        total_done,
                        len(jobs),
                        slug,
                        res["status"],
                        n_tests,
                        counts,
                    )

    tasks = [worker(slug, url, i) for i, (slug, url) in enumerate(jobs)]
    await asyncio.gather(*tasks)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=["lc", "cc", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0, help="limit jobs per platform (0=all)")
    ap.add_argument("--lc-concurrency", type=int, default=2)
    ap.add_argument("--cc-concurrency", type=int, default=2)
    ap.add_argument("--shuffle-seed", type=int, default=0)
    args = ap.parse_args()

    log = setup_logger("scrape_tests")
    log.info("Args: %s", args)

    rnd = random.Random(args.shuffle_seed)

    summary: Dict[str, Dict[str, int]] = {}

    if args.platform in ("lc", "both"):
        jobs = load_lc_jobs()
        rnd.shuffle(jobs)
        # Skip already-done
        jobs_filtered = [(s, u) for (s, u) in jobs if not (LC_OUT_DIR / f"{s}.json").exists()]
        log.info("LC: %d total jobs, %d remaining after dedup", len(jobs), len(jobs_filtered))
        if args.limit > 0:
            jobs_filtered = jobs_filtered[: args.limit]
            log.info("LC: limited to %d jobs", len(jobs_filtered))
        if jobs_filtered:
            t0 = time.time()
            counts = asyncio.run(
                run_platform("lc", jobs_filtered, args.lc_concurrency, log)
            )
            log.info(
                "LC done in %.0fs. counts=%s", time.time() - t0, counts
            )
            summary["lc"] = counts

    if args.platform in ("cc", "both"):
        jobs = load_cc_jobs()
        rnd.shuffle(jobs)
        jobs_filtered = [(s, u) for (s, u) in jobs if not (CC_OUT_DIR / f"{s}.json").exists()]
        log.info("CC: %d total jobs, %d remaining after dedup", len(jobs), len(jobs_filtered))
        if args.limit > 0:
            jobs_filtered = jobs_filtered[: args.limit]
            log.info("CC: limited to %d jobs", len(jobs_filtered))
        if jobs_filtered:
            t0 = time.time()
            counts = asyncio.run(
                run_platform("cc", jobs_filtered, args.cc_concurrency, log)
            )
            log.info(
                "CC done in %.0fs. counts=%s", time.time() - t0, counts
            )
            summary["cc"] = counts

    log.info("ALL DONE summary=%s", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
