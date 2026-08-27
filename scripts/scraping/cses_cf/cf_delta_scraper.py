"""
Codeforces delta editorial scraper.

For each contest_id with at least one editorial-missing problem in the open-r1
bundle:
  1) fetch codeforces.com/contest/{contest_id} → parse "Tutorial" link
  2) fetch that blog/entry/{N} page → store full text + per-problem chunks
     (we slice by problem index letters A/B/C... and by problem titles).

Concurrency=2, save every 50 contests, log every fetch.
"""
import asyncio
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, "/lfs/skampere3/0/alexspan/norm-research/scripts/scraping")
from stealth_fetch import fetch_url  # noqa: E402

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research/datasets/codeforces_delta")
ROOT.mkdir(parents=True, exist_ok=True)
LOG = ROOT / "scrape_log.jsonl"
CHECKPOINT = ROOT / "contest_blog_map.parquet"
EDITORIALS = ROOT / "editorials.parquet"


def log_jsonl(path: Path, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def now_iso() -> str:
    return datetime.utcnow().isoformat()


# ---------------------------------------------------------------------------
# Loading registry & identifying gap
# ---------------------------------------------------------------------------
def load_gap_contests() -> tuple[pd.DataFrame, list[str]]:
    reg = pd.read_parquet(ROOT / "problems_registry.parquet")
    reg["has_ed"] = reg["editorial"].apply(
        lambda x: isinstance(x, str) and len(x.strip()) > 30
    )
    # We want all contests where at least one problem lacks an editorial; pull the
    # blog and we get all problems in that round.
    missing = reg[~reg["has_ed"]]
    gap_contests = sorted(missing["contest_id"].dropna().unique().tolist())
    print(f"[delta] {len(reg)} problems, {missing.shape[0]} missing editorials "
          f"across {len(gap_contests)} contests", flush=True)
    return reg, gap_contests


# ---------------------------------------------------------------------------
# Step 1: contest page → Tutorial blog/entry/{N}
# ---------------------------------------------------------------------------
TUTORIAL_RE = re.compile(r"/blog/entry/(\d+)", re.IGNORECASE)


def extract_tutorial_entry(html: str) -> int | None:
    """Find first blog/entry URL whose anchor text contains 'tutorial' or
    'editorial' (case-insensitive). Otherwise, return the most prominent
    blog/entry on the page (the contest right-sidebar 'Contest materials' list)."""
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[tuple[int, str]] = []  # (entry_id, anchor_text_lower)
    for a in soup.find_all("a", href=True):
        m = TUTORIAL_RE.search(a["href"])
        if not m:
            continue
        eid = int(m.group(1))
        txt = a.get_text(" ", strip=True).lower()
        candidates.append((eid, txt))
    # prefer "tutorial"/"editorial"
    for keyword in ("tutorial", "editorial", "разбор", "анализ"):
        for eid, txt in candidates:
            if keyword in txt:
                return eid
    # otherwise return first sidebar entry
    if candidates:
        return candidates[0][0]
    return None


async def get_blog_id_for_contest(contest_id: str) -> int | None:
    url = f"https://codeforces.com/contest/{contest_id}"
    status, html = await fetch_url(url, respect_robots=False, retries=3, timeout_ms=45000)
    log_jsonl(LOG, {"step": "contest_page", "contest_id": contest_id, "status": status, "len": len(html), "ts": now_iso()})
    if status < 200 or status >= 400 or not html:
        return None
    return extract_tutorial_entry(html)


# ---------------------------------------------------------------------------
# Step 2: fetch blog entry
# ---------------------------------------------------------------------------
async def fetch_blog_entry(entry_id: int) -> tuple[int, str]:
    url = f"https://codeforces.com/blog/entry/{entry_id}"
    status, html = await fetch_url(url, respect_robots=False, retries=3, timeout_ms=45000)
    log_jsonl(LOG, {"step": "blog_entry", "entry_id": entry_id, "status": status, "len": len(html), "ts": now_iso()})
    return status, html


def blog_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("div", class_="ttypography") or soup.body or soup
    return body.get_text("\n", strip=True)


def slice_blog_by_index(text: str, problem_idxs: list[str]) -> dict[str, str]:
    """For a CF round editorial, the chunks are usually demarcated by:
      "Problem A. ...", "1788A - Title", "A. Title", etc.
    Try several patterns per index letter and return chunks.
    """
    chunks: dict[str, str] = {}
    # collect anchor positions
    anchors: list[tuple[int, str]] = []
    for idx in problem_idxs:
        idx_esc = re.escape(idx)
        patterns = [
            rf"(?m)^[ \t]*[Pp]roblem\s+{idx_esc}[\.\s\-:]",
            rf"(?m)^[ \t]*{idx_esc}\s*[\-\.\:]\s*[A-Z]",   # "A - Title" or "A. Title"
            rf"(?m)^[ \t]*[Dd]iv\.?\s*\d+\s*{idx_esc}[\.\s\-:]",
            rf"(?m)^[ \t]*\d+{idx_esc}\s*[\-\.\:]\s*[A-Z]",  # "1788A - Title"
        ]
        best_pos = None
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                if best_pos is None or m.start() < best_pos:
                    best_pos = m.start()
        if best_pos is not None:
            anchors.append((best_pos, idx))
    anchors.sort()
    for i, (offset, idx) in enumerate(anchors):
        end = anchors[i + 1][0] if i + 1 < len(anchors) else len(text)
        chunk = text[offset:end].strip()
        if len(chunk) >= 50:
            chunks[idx] = chunk
    return chunks


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
async def main():
    reg, gap_contests = load_gap_contests()

    # resume from checkpoint if exists
    if CHECKPOINT.exists():
        prev_map = pd.read_parquet(CHECKPOINT)
        done = set(prev_map["contest_id"].astype(str).tolist())
        print(f"[delta] resuming: {len(done)} contests already processed", flush=True)
    else:
        prev_map = pd.DataFrame(columns=["contest_id", "blog_entry_id", "status", "scraped_at"])
        done = set()

    if EDITORIALS.exists():
        prev_eds = pd.read_parquet(EDITORIALS)
        print(f"[delta] resuming: {len(prev_eds)} editorials already saved", flush=True)
    else:
        prev_eds = pd.DataFrame(columns=["problem_id", "contest_id", "problem_index",
                                          "editorial_text", "editorial_source_url",
                                          "scraped_at"])

    pending = [c for c in gap_contests if c not in done]
    print(f"[delta] {len(pending)} contests to process", flush=True)

    map_rows = prev_map.to_dict("records")
    ed_rows = prev_eds.to_dict("records")

    # group problem indexes per contest
    contest_indexes = (
        reg[~reg["editorial"].apply(lambda x: isinstance(x, str) and len(x.strip()) > 30)]
        .groupby("contest_id")["index"]
        .apply(list)
        .to_dict()
    )
    contest_index_to_pid = dict(
        zip(
            zip(reg["contest_id"].astype(str), reg["index"]),
            reg["id"],
        )
    )

    save_every = 30
    processed_in_session = 0

    for i, contest_id in enumerate(pending, 1):
        try:
            blog_id = await get_blog_id_for_contest(str(contest_id))
            entry = {
                "contest_id": str(contest_id),
                "blog_entry_id": blog_id,
                "status": "blog_found" if blog_id else "no_blog_link",
                "scraped_at": now_iso(),
            }
            map_rows.append(entry)
            if blog_id is None:
                print(f"[delta] {i}/{len(pending)} contest={contest_id}: no tutorial link", flush=True)
            else:
                status, html = await fetch_blog_entry(blog_id)
                if status >= 200 and status < 400 and html:
                    text = blog_text(html)
                    idxs = contest_indexes.get(contest_id, [])
                    chunks = slice_blog_by_index(text, idxs)
                    url = f"https://codeforces.com/blog/entry/{blog_id}"
                    if chunks:
                        for idx, chunk in chunks.items():
                            ed_rows.append(
                                {
                                    "problem_id": contest_index_to_pid.get((str(contest_id), idx)),
                                    "contest_id": str(contest_id),
                                    "problem_index": idx,
                                    "editorial_text": chunk,
                                    "editorial_source_url": url,
                                    "scraped_at": now_iso(),
                                }
                            )
                    # always also keep a full-blog blob keyed by contest with index="ALL"
                    ed_rows.append(
                        {
                            "problem_id": None,
                            "contest_id": str(contest_id),
                            "problem_index": "ALL",
                            "editorial_text": text,
                            "editorial_source_url": url,
                            "scraped_at": now_iso(),
                        }
                    )
                    entry["status"] = "blog_fetched"
                    print(
                        f"[delta] {i}/{len(pending)} contest={contest_id} blog={blog_id} "
                        f"text={len(text)} chunks={len(chunks)}",
                        flush=True,
                    )
                else:
                    entry["status"] = f"blog_fetch_fail_{status}"
                    print(
                        f"[delta] {i}/{len(pending)} contest={contest_id} blog={blog_id} "
                        f"FAILED status={status}",
                        flush=True,
                    )
        except Exception as e:  # noqa: BLE001
            log_jsonl(LOG, {"step": "exception", "contest_id": str(contest_id), "err": str(e), "ts": now_iso()})
            print(f"[delta] {i}/{len(pending)} contest={contest_id} EXCEPTION: {e}", flush=True)

        processed_in_session += 1
        if processed_in_session % save_every == 0:
            pd.DataFrame(map_rows).to_parquet(CHECKPOINT, index=False)
            pd.DataFrame(ed_rows).to_parquet(EDITORIALS, index=False)
            print(f"[delta] checkpoint: {len(map_rows)} contests, {len(ed_rows)} editorial rows", flush=True)

    # final save
    pd.DataFrame(map_rows).to_parquet(CHECKPOINT, index=False)
    pd.DataFrame(ed_rows).to_parquet(EDITORIALS, index=False)
    print("[delta] DONE", flush=True)

    df_ed = pd.DataFrame(ed_rows)
    df_perp = df_ed[df_ed["problem_id"].notna()]
    print(
        f"[delta] FINAL: {len(df_perp)} per-problem editorials covering "
        f"{df_perp['problem_id'].nunique()} unique CF problem IDs",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
