"""
scrape_luogu_cc_votes.py
========================

Background scraper for two community-preference data sources:

Track 1 -- Luogu problem discussions / solutions
    For each Luogu pid in the editorial corpus, hit the solution page
    (https://www.luogu.com.cn/problem/solution/{pid}) and the discussion
    list (https://www.luogu.com.cn/discuss/lists?forumname={pid}) and
    extract comment text + per-comment like counts.

    Output: datasets/luogu/comments_scraped/{pid}.json

Track 2 -- CodeChef user profiles
    For each unique username harvested from datasets/codechef/editorials.parquet
    (post_username + ".../users/<name>" links inside author_url / tester_url /
    editorialist_url / editorial_md / editorial_cooked_html), fetch the user
    profile page and extract rating / stars / problems-solved / contests-rated.

    Output: datasets/codechef/user_votes_scraped/{username}.json

Stack
-----
Reuses stealth_fetch (Camoufox + Webshare proxy rotation), same as the
existing scrape_tests_lc_cc.py. Resume-resilient (skip if output exists).
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import random
import re
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")
sys.path.insert(0, str(ROOT / "scripts" / "scraping"))

from stealth_fetch import fetch_url  # noqa: E402

UNIFIED_EDITORIALS = ROOT / "datasets" / "competition_unified" / "editorials.parquet"
CC_EDITORIALS = ROOT / "datasets" / "codechef" / "editorials.parquet"

LU_OUT_DIR = ROOT / "datasets" / "luogu" / "comments_scraped"
CC_OUT_DIR = ROOT / "datasets" / "codechef" / "user_votes_scraped"
LOG_DIR = ROOT / "outputs" / "scraping" / "votes_scrape_logs"

LU_OUT_DIR.mkdir(parents=True, exist_ok=True)
CC_OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


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
# Generic HTML helpers
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_ENTITIES = {
    "&nbsp;": " ",
    "&lt;": "<",
    "&gt;": ">",
    "&amp;": "&",
    "&quot;": '"',
    "&#39;": "'",
}


def strip_tags(s: str) -> str:
    s = re.sub(r"<script.*?</script>", " ", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<style.*?</style>", " ", s, flags=re.DOTALL | re.IGNORECASE)
    s = _TAG_RE.sub(" ", s)
    for k, v in _ENTITIES.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _safe_int(x) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(str(x).replace(",", "").strip())
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Luogu parsing
# ---------------------------------------------------------------------------
#
# Luogu's modern site is a Nuxt SPA. The first <script> tag with
# id="lentille-context" carries a JSON blob with the page's data; this is the
# most reliable source for solutions + their like counts. Schema observed in
# practice (mid-2025):
#   { "data": { "solutions": { "result": [
#         { "content": "...", "user": {"name": "..."},
#           "thumbUp": 123, "thumbDown": 4, "postTime": 1700000000 }, ... ]}}}
# Discussion list page emits:
#   { "data": { "threads": { "result": [
#         { "title": "...", "author": {"name": "..."},
#           "replies": N, "updateTime": ... }, ... ]}}}
# Real keys vary by build; we walk the JSON tree to be resilient.

_LENTILLE_RE = re.compile(
    r'<script[^>]*id="lentille-context"[^>]*>(?P<json>.*?)</script>',
    re.DOTALL | re.IGNORECASE,
)
# Older builds used JSON.parse(decodeURIComponent("...")) in _feInjection
_FEINJECT_RE = re.compile(
    r'window\._feInjection\s*=\s*JSON\.parse\(decodeURIComponent\("(?P<enc>[^"]+)"\)\)',
    re.DOTALL,
)


def _extract_lentille_json(html: str) -> Optional[dict]:
    m = _LENTILLE_RE.search(html)
    if m:
        raw = m.group("json").strip()
        try:
            return json.loads(raw)
        except Exception:  # noqa: BLE001
            pass
    m = _FEINJECT_RE.search(html)
    if m:
        try:
            decoded = urllib.parse.unquote(m.group("enc"))
            return json.loads(decoded)
        except Exception:  # noqa: BLE001
            pass
    return None


def _walk(obj, predicate):
    """Yield (path, value) for every dict/value where predicate(path, value) is True."""
    stack = [([], obj)]
    while stack:
        path, cur = stack.pop()
        if isinstance(cur, dict):
            if predicate(path, cur):
                yield path, cur
            for k, v in cur.items():
                stack.append((path + [k], v))
        elif isinstance(cur, list):
            for i, v in enumerate(cur):
                stack.append((path + [i], v))


def _find_solution_records(blob: dict) -> List[dict]:
    """Find dicts that look like solution / comment records."""
    out: List[dict] = []
    for _, node in _walk(
        blob,
        lambda p, n: (
            isinstance(n, dict)
            and ("content" in n or "rawContent" in n or "text" in n)
            and ("thumbUp" in n or "upvote" in n or "likes" in n or "praises" in n)
        ),
    ):
        out.append(node)
    return out


def _find_thread_records(blob: dict) -> List[dict]:
    out: List[dict] = []
    for _, node in _walk(
        blob,
        lambda p, n: (
            isinstance(n, dict)
            and ("title" in n or "subject" in n)
            and ("replies" in n or "reply" in n or "replyCount" in n)
            and ("author" in n or "user" in n or "poster" in n)
        ),
    ):
        out.append(node)
    return out


def _author_name(node: dict) -> str:
    for k in ("user", "author", "poster"):
        v = node.get(k)
        if isinstance(v, dict):
            for kk in ("name", "username", "uname"):
                if v.get(kk):
                    return str(v[kk])
        elif isinstance(v, str) and v:
            return v
    return ""


def _ts(node: dict) -> str:
    for k in ("postTime", "createTime", "updateTime", "time", "createdAt"):
        if k in node:
            v = node[k]
            try:
                # epoch seconds or ms
                vv = int(v)
                if vv > 10_000_000_000:
                    vv //= 1000
                return dt.datetime.utcfromtimestamp(vv).isoformat() + "Z"
            except Exception:  # noqa: BLE001
                return str(v)
    return ""


def _likes(node: dict) -> int:
    for k in ("thumbUp", "upvote", "likes", "praises", "like"):
        if k in node:
            n = _safe_int(node[k])
            if n is not None:
                return n
    return 0


def _content(node: dict) -> str:
    for k in ("content", "rawContent", "text", "title"):
        v = node.get(k)
        if isinstance(v, str) and v:
            return strip_tags(v)[:8000]
    return ""


def parse_luogu_html(html: str) -> Tuple[List[Dict], Dict, str]:
    """
    Parse a Luogu /problem/{pid} page.

    Returns (items, problem_stats, status).

    `items` are discussion-thread records visible on the problem page (title,
    author, time). NB: per-comment LIKE counts are gated behind login on
    Luogu and are not in this blob; we record what's publicly visible only.

    `problem_stats` carries totalSubmit / totalAccepted / difficulty / tags.
    """
    if not html:
        return [], {}, "error"
    if "Just a moment" in html and len(html) < 8000:
        return [], {}, "blocked"
    if "页面未找到" in html or "页面没找到" in html:
        return [], {}, "404"

    blob = _extract_lentille_json(html)
    if not blob:
        return [], {}, "no_data"

    data = blob.get("data") or {}
    problem = data.get("problem") or {}
    problem_stats: Dict = {}
    for k in ("totalSubmit", "totalAccepted", "difficulty", "name", "type", "tags"):
        if k in problem:
            v = problem[k]
            if k == "tags" and isinstance(v, list):
                v = [t.get("name") if isinstance(t, dict) else str(t) for t in v]
            problem_stats[k] = v

    items: List[Dict] = []
    for t in data.get("discussions") or []:
        if not isinstance(t, dict):
            continue
        items.append(
            {
                "kind": "thread",
                "thread_id": t.get("id"),
                "text": (t.get("title") or "").strip()[:2000],
                "likes": None,  # not exposed to unauthenticated viewers
                "author": _author_name(t),
                "timestamp": _ts(t),
            }
        )

    forum_obj = data.get("forum") or {}
    if isinstance(forum_obj, dict):
        problem_stats["forum_name"] = forum_obj.get("name")
        problem_stats["forum_slug"] = forum_obj.get("slug")

    # Fallback: also scan the whole blob for anything that looks like a
    # solution/thread record exposing likes (in case Luogu ships them
    # unauthenticated for some problems).
    for s in _find_solution_records(blob):
        items.append(
            {
                "kind": "solution",
                "text": _content(s),
                "likes": _likes(s),
                "author": _author_name(s),
                "timestamp": _ts(s),
            }
        )

    if not items and not problem_stats:
        return items, problem_stats, "no_data"
    return items, problem_stats, "ok"


# ---------------------------------------------------------------------------
# CodeChef profile parsing
# ---------------------------------------------------------------------------
#
# CC user pages embed the visible profile in HTML (server-rendered) and also
# include a __NUXT__ blob. We pick a few well-known fields.

_CC_RATING_RE = re.compile(
    r'<div\s+class="rating-number[^"]*"[^>]*>\s*(?P<n>\d+\??)\s*</div>',
    re.IGNORECASE,
)
# rating-star block contains one ★ <span> per star tier (1..7)
_CC_STAR_BLOCK_RE = re.compile(
    r'<div\s+class="rating-star"[^>]*>(?P<inner>.*?)</div>',
    re.DOTALL | re.IGNORECASE,
)
_CC_TOTAL_SOLVED_RE = re.compile(
    r"Total Problems Solved\s*:\s*(\d+)",
    re.IGNORECASE,
)
_CC_CONTESTS_PARTICIPATED_RE = re.compile(
    r"No\.\s*of\s*Contests?\s*Participated\s*:\s*<b>(\d+)</b>",
    re.IGNORECASE,
)
_CC_DIVISION_RE = re.compile(r"\(Div\s*(\d)\)", re.IGNORECASE)
_CC_HIGHEST_RATING_RE = re.compile(
    r"Highest\s*Rating\s*(\d+)",
    re.IGNORECASE,
)
_CC_USER_NOT_FOUND = re.compile(
    r"user[\s_]*(does not|not found|n\W+t exist)", re.IGNORECASE
)


def parse_codechef_user_html(html: str) -> Tuple[Dict, str]:
    if not html:
        return {}, "error"
    if _CC_USER_NOT_FOUND.search(html):
        return {}, "404"
    if 'You need to enable JavaScript' in html and len(html) < 25000:
        return {}, "blocked"

    out: Dict = {
        "rating": None,
        "highest_rating": None,
        "stars": None,
        "problems_solved": None,
        "contests_participated": None,
        "division": None,
    }

    # Rating
    m = _CC_RATING_RE.search(html)
    if m:
        # Strip trailing "?" (used for unrated)
        out["rating"] = _safe_int(re.sub(r"\D", "", m.group("n")))

    # Highest rating
    m = _CC_HIGHEST_RATING_RE.search(html)
    if m:
        out["highest_rating"] = _safe_int(m.group(1))

    # Stars: count ★ inside the rating-star div
    m = _CC_STAR_BLOCK_RE.search(html)
    if m:
        inner = m.group("inner")
        n_stars = inner.count("★")
        if n_stars > 0:
            out["stars"] = n_stars

    # Total problems solved
    m = _CC_TOTAL_SOLVED_RE.search(html)
    if m:
        out["problems_solved"] = _safe_int(m.group(1))

    # Contests participated (rated)
    m = _CC_CONTESTS_PARTICIPATED_RE.search(html)
    if m:
        out["contests_participated"] = _safe_int(m.group(1))

    # Division
    m = _CC_DIVISION_RE.search(html)
    if m:
        out["division"] = _safe_int(m.group(1))

    nonempty = [v for v in out.values() if v is not None]
    if not nonempty:
        return out, "no_data"
    return out, "ok"


# ---------------------------------------------------------------------------
# Job loaders
# ---------------------------------------------------------------------------


def load_luogu_jobs() -> List[Tuple[str, str]]:
    """Return list of (pid, problem_url).

    Only the bare problem page (/problem/{pid}) is publicly accessible
    without login; /problem/solution/{pid} and /discuss/{id} return 401.
    The problem page exposes data.discussions (thread metadata) and
    data.problem.totalSubmit/totalAccepted as the community signal.
    """
    df = pd.read_parquet(UNIFIED_EDITORIALS)
    pids = (
        df[df.platform == "luogu"]["canonical_pid"].dropna().drop_duplicates().tolist()
    )
    jobs = []
    for cpid in pids:
        pid = cpid.split(":", 1)[1] if ":" in cpid else cpid
        # Luogu pids in the upstream parquet are lowercase (e.g. "p1001");
        # the URL accepts either case but we standardize to uppercase first
        # letter (matches the on-site canonical form).
        url_pid = pid[:1].upper() + pid[1:] if pid else pid
        url = f"https://www.luogu.com.cn/problem/{url_pid}"
        jobs.append((pid, url))
    return jobs


def load_cc_user_jobs() -> List[Tuple[str, str]]:
    """Return list of (username, url) harvested from the CC editorials parquet."""
    ed = pd.read_parquet(CC_EDITORIALS)
    users = set()
    for v in ed.get("post_username", pd.Series([], dtype=object)).dropna().unique():
        v = str(v).strip()
        if v and v.lower() not in ("none", "nan"):
            users.add(v)
    url_pat = re.compile(r"codechef\.com/users/([A-Za-z0-9_\-\.]+)")
    for col in [
        "author_url",
        "tester_url",
        "editorialist_url",
        "editorial_md",
        "editorial_cooked_html",
    ]:
        if col not in ed.columns:
            continue
        for v in ed[col].dropna():
            for u in url_pat.findall(str(v)):
                users.add(u)
    # Strip stray punctuation
    users = {u.strip(".-_") for u in users if u and len(u) >= 2}
    jobs = []
    for u in sorted(users):
        jobs.append((u, f"https://www.codechef.com/users/{u}"))
    return jobs


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


async def _fetch(url: str, **kwargs) -> Tuple[int, str]:
    return await fetch_url(
        url,
        respect_robots=False,  # explicit per user
        retries=2,
        jitter=(2.0, 5.0),
        timeout_ms=60_000,
        **kwargs,
    )


async def scrape_one_luogu(pid: str, url: str, log: logging.Logger) -> Dict:
    out_path = LU_OUT_DIR / f"{pid}.json"
    if out_path.exists():
        return {"pid": pid, "status": "skipped"}
    rec: Dict = {
        "platform": "luogu",
        "pid": pid,
        "url": url,
        "comments": [],
        "problem_stats": {},
        "scraped_at": dt.datetime.utcnow().isoformat() + "Z",
        "scrape_status": "error",
        "error_message": "",
    }
    try:
        http_status, html = await _fetch(
            url, wait_until="domcontentloaded", extra_wait_ms=3000
        )
        if http_status == 404:
            rec["scrape_status"] = "404"
        elif http_status < 200 or http_status >= 400 or not html:
            rec["scrape_status"] = "error"
            rec["error_message"] = f"http={http_status}"
        else:
            items, problem_stats, parse_status = parse_luogu_html(html)
            rec["comments"] = items
            rec["problem_stats"] = problem_stats
            rec["scrape_status"] = parse_status
    except Exception as e:  # noqa: BLE001
        rec["scrape_status"] = "error"
        rec["error_message"] = f"{type(e).__name__}: {e}"
        log.warning("luogu %s: %s", pid, e)

    out_path.write_text(json.dumps(rec, ensure_ascii=False))
    return {
        "pid": pid,
        "status": rec["scrape_status"],
        "n": len(rec.get("comments", [])),
    }


async def scrape_one_cc_user(username: str, url: str, log: logging.Logger) -> Dict:
    out_path = CC_OUT_DIR / f"{username}.json"
    if out_path.exists():
        return {"username": username, "status": "skipped"}
    rec: Dict = {
        "platform": "cc",
        "username": username,
        "url": url,
        "rating": None,
        "highest_rating": None,
        "stars": None,
        "problems_solved": None,
        "contests_participated": None,
        "division": None,
        "scraped_at": dt.datetime.utcnow().isoformat() + "Z",
        "scrape_status": "error",
        "error_message": "",
    }
    try:
        http_status, html = await _fetch(
            url, wait_until="networkidle", extra_wait_ms=4000
        )
        if http_status == 404:
            rec["scrape_status"] = "404"
        elif http_status < 200 or http_status >= 400 or not html:
            rec["scrape_status"] = "error"
            rec["error_message"] = f"http={http_status}"
        else:
            parsed, parse_status = parse_codechef_user_html(html)
            for k, v in parsed.items():
                if k in rec:
                    rec[k] = v
            rec["scrape_status"] = parse_status
    except Exception as e:  # noqa: BLE001
        rec["scrape_status"] = "error"
        rec["error_message"] = f"{type(e).__name__}: {e}"
        log.warning("cc user %s: %s", username, e)
    out_path.write_text(json.dumps(rec, ensure_ascii=False))
    return {"username": username, "status": rec["scrape_status"]}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def run_platform(
    platform: str,
    jobs: List[Tuple],
    concurrency: int,
    log: logging.Logger,
) -> Dict[str, int]:
    sem = asyncio.Semaphore(concurrency)
    counts: Dict[str, int] = {}
    counts_lock = asyncio.Lock()

    async def worker(idx: int, job: Tuple):
        async with sem:
            if platform == "luogu":
                pid, url = job
                res = await scrape_one_luogu(pid, url, log)
                key = pid
            else:
                u, url = job
                res = await scrape_one_cc_user(u, url, log)
                key = u
            async with counts_lock:
                counts[res["status"]] = counts.get(res["status"], 0) + 1
                total_done = sum(counts.values())
                if idx % 25 == 0 or idx < 10:
                    log.info(
                        "[%s %d/%d] %s -> %s  cumulative=%s",
                        platform,
                        total_done,
                        len(jobs),
                        key,
                        res["status"],
                        counts,
                    )

    await asyncio.gather(*[worker(i, j) for i, j in enumerate(jobs)])
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=["luogu", "cc", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0, help="limit per platform (0=all)")
    ap.add_argument("--lu-concurrency", type=int, default=3)
    ap.add_argument("--cc-concurrency", type=int, default=3)
    ap.add_argument("--shuffle-seed", type=int, default=0)
    args = ap.parse_args()

    log = setup_logger("scrape_votes")
    log.info("Args: %s", args)

    rnd = random.Random(args.shuffle_seed)
    summary: Dict[str, Dict[str, int]] = {}

    if args.platform in ("luogu", "both"):
        jobs = load_luogu_jobs()
        rnd.shuffle(jobs)
        remaining = [j for j in jobs if not (LU_OUT_DIR / f"{j[0]}.json").exists()]
        log.info(
            "luogu: %d total, %d remaining after dedup",
            len(jobs),
            len(remaining),
        )
        if args.limit > 0:
            remaining = remaining[: args.limit]
            log.info("luogu: limited to %d jobs", len(remaining))
        if remaining:
            t0 = time.time()
            counts = asyncio.run(
                run_platform("luogu", remaining, args.lu_concurrency, log)
            )
            log.info("luogu done in %.0fs. counts=%s", time.time() - t0, counts)
            summary["luogu"] = counts

    if args.platform in ("cc", "both"):
        jobs = load_cc_user_jobs()
        rnd.shuffle(jobs)
        remaining = [j for j in jobs if not (CC_OUT_DIR / f"{j[0]}.json").exists()]
        log.info("cc: %d total, %d remaining after dedup", len(jobs), len(remaining))
        if args.limit > 0:
            remaining = remaining[: args.limit]
            log.info("cc: limited to %d jobs", len(remaining))
        if remaining:
            t0 = time.time()
            counts = asyncio.run(
                run_platform("cc", remaining, args.cc_concurrency, log)
            )
            log.info("cc done in %.0fs. counts=%s", time.time() - t0, counts)
            summary["cc"] = counts

    log.info("ALL DONE summary=%s", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
