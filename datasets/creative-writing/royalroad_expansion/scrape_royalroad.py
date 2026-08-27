#!/usr/bin/env python3
"""
RoyalRoad expansion scrape — multiple y-variables on one platform.

Modes (run in this order):
  magazine  — Community Magazine Contest editions: every entry chapter of the
              10 magazine fictions (Jan 2022 - Jun 2026). Raw HTML per chapter
              (entry first-chapter text + any embedded source-fiction link).
              This is the FULL judged-contest pool (~2,153 entries).
  blogs     — RoyalRoad blog archive (winner announcements live here).
  listings  — full search-index enumeration, pages 1..N of /fictions/search:
              per-card metadata for every fiction on the site (~135.6K):
              id, title, tags, status (incl. STUB = commercial pickup),
              followers, pages, views, chapters, rating. ~6.8K requests.

Politeness: 1 req/s, browser UA with contact email, exp backoff on 429/5xx.
All outputs append-only with resume state; raw HTML kept for magazine/blogs.

  python scrape_royalroad.py --mode magazine
  python scrape_royalroad.py --mode blogs
  python scrape_royalroad.py --mode listings [--max-pages 7000]
"""

import argparse
import gzip
import json
import os
import re
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 norm-research-academic "
      "(contact: alex2awesome@gmail.com)")
BASE = "https://www.royalroad.com"
RATE = 1.05

MAGAZINE_EDITIONS = [  # (fiction_id, label)
    (173505, "2026-06"), (147817, "2026-01"), (120257, "2025-06"),
    (103100, "2025-01"), (88331, "2024-06"), (79296, "2024-01"),
    (69303, "2023-06"), (63158, "2023-01"), (55346, "2022-06"),
    (50199, "2022-01"),
]


def sess_get(sess, url, params=None):
    for attempt in range(7):
        try:
            r = sess.get(url, params=params, timeout=60)
            if r.status_code in (429, 500, 502, 503, 520, 522):
                raise RuntimeError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.text
        except Exception as e:
            wait = min(300, 2 ** attempt * 5)
            print(f"{url} attempt {attempt}: {e} — sleep {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"giving up: {url}")


def load_state(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def save_state(path, st):
    json.dump(st, open(path, "w"))


def mode_magazine(sess):
    out_dir = os.path.join(HERE, "magazine_raw")
    os.makedirs(out_dir, exist_ok=True)
    state_p = os.path.join(HERE, "magazine_state.json")
    st = load_state(state_p)
    for fid, label in MAGAZINE_EDITIONS:
        toc_f = os.path.join(out_dir, f"edition_{label}_toc.html")
        if not os.path.exists(toc_f):
            html = sess_get(sess, f"{BASE}/fiction/{fid}")
            open(toc_f, "w").write(html)
            time.sleep(RATE)
        html = open(toc_f).read()
        chaps = sorted(set(re.findall(rf'href="(/fiction/{fid}/[^"]*/chapter/[^"]+)"', html)))
        done = set(st.get(label, []))
        print(f"edition {label}: {len(chaps)} chapters, {len(done)} done", flush=True)
        for ch in chaps:
            if ch in done:
                continue
            cid = ch.rstrip("/").split("/chapter/")[1].split("/")[0]
            chtml = sess_get(sess, BASE + ch)
            with gzip.open(os.path.join(out_dir, f"edition_{label}_ch{cid}.html.gz"), "wt") as fh:
                fh.write(f"<!-- url: {ch} -->\n" + chtml)
            done.add(ch)
            st[label] = sorted(done)
            save_state(state_p, st)
            time.sleep(RATE)
    print("magazine done", flush=True)


def mode_blogs(sess):
    out_dir = os.path.join(HERE, "blog_raw")
    os.makedirs(out_dir, exist_ok=True)
    # blog ids are small sequential ints; sweep 1..120, keep 200s
    for bid in range(1, 121):
        out_f = os.path.join(out_dir, f"blog_{bid}.html.gz")
        if os.path.exists(out_f):
            continue
        r = sess.get(f"{BASE}/blog/{bid}", timeout=60)
        if r.status_code == 200 and "StatusNotFound" not in r.text:
            with gzip.open(out_f, "wt") as fh:
                fh.write(r.text)
            print(f"blog {bid}: saved", flush=True)
        time.sleep(RATE)
    print("blogs done", flush=True)


CARD_RE = re.compile(r'fiction-list-item', re.I)


def parse_cards(html):
    """Split a search page into per-fiction card dicts."""
    rows = []
    blocks = re.split(r'(?=<div[^>]*fiction-list-item)', html)
    for b in blocks[1:]:
        m = re.search(r'href="/fiction/(\d+)/([^"]+)"', b)
        if not m:
            continue
        row = {"fiction_id": int(m.group(1)), "slug": m.group(2)}
        t = re.search(r'class="fiction-title"[^>]*>\s*<a[^>]*>([^<]+)', b)
        row["title"] = (t.group(1).strip() if t else None)
        row["tags"] = re.findall(r'fictions/search\?tagsAdd=([^"&]+)', b)
        lab = re.findall(r'<span class="label[^"]*"[^>]*>\s*([A-Z][A-Z ]+?)\s*<', b)
        row["labels"] = [x.strip() for x in lab]
        for key, pat in [("followers", r'([\d,]+)\s*Followers'),
                         ("pages", r'([\d,]+)\s*Pages'),
                         ("views", r'([\d,]+)\s*Views'),
                         ("chapters", r'([\d,]+)\s*Chapters')]:
            mm = re.search(pat, b)
            row[key] = int(mm.group(1).replace(",", "")) if mm else None
        mm = re.search(r'width:\s*([\d.]+)%', b)
        row["rating_pct"] = float(mm.group(1)) if mm else None
        mm = re.search(r'Last Updated:?\s*</?[^>]*>?\s*([^<]{4,40})', b)
        row["last_update_raw"] = mm.group(1).strip() if mm else None
        rows.append(row)
    return rows


def mode_listings(sess, max_pages, status=None):
    suffix = f"_{status.lower()}" if status else "_all"
    out_f = os.path.join(HERE, f"listings{suffix}.jsonl.gz")
    state_p = os.path.join(HERE, f"listings{suffix}_state.json")
    st = load_state(state_p)
    page = st.get("next_page", 1)
    mode = "ab" if page > 1 and os.path.exists(out_f) else "wb"
    empty_streak = 0
    with gzip.open(out_f, mode) as fh:
        while page <= max_pages:
            params = {"page": page}
            if status:
                params["status"] = status
            html = sess_get(sess, f"{BASE}/fictions/search", params=params)
            rows = parse_cards(html)
            if not rows:
                empty_streak += 1
                if empty_streak >= 3:
                    print(f"3 empty pages ending at {page}; stopping", flush=True)
                    break
            else:
                empty_streak = 0
                for r in rows:
                    r["search_page"] = page
                    fh.write((json.dumps(r) + "\n").encode())
            if page % 50 == 0:
                print(f"page {page}: +{len(rows)} cards", flush=True)
                fh.flush()
            page += 1
            st["next_page"] = page
            save_state(state_p, st)
            time.sleep(RATE)
    print("listings done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["magazine", "blogs", "listings"])
    ap.add_argument("--max-pages", type=int, default=7100)
    ap.add_argument("--status", default=None, help="e.g. STUB")
    a = ap.parse_args()
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA})
    if a.mode == "magazine":
        mode_magazine(sess)
    elif a.mode == "blogs":
        mode_blogs(sess)
    else:
        mode_listings(sess, a.max_pages, a.status)


if __name__ == "__main__":
    main()
