"""Enumerate RoyalRoad fictions from search listings.

Modes:
  stubs    — /fictions/search?status=STUB (y=1 candidates)
  controls — /fictions/search?orderBy=followers&dir=desc (y=0 candidate pool)

Resumable: checkpoint file records last completed page per mode; output is
append-only jsonl, deduped at read time on fiction_id (keep last).

Usage: python scrape_search_listings.py --mode stubs --data-dir DIR [--max-pages N]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rr_common import PoliteSession, append_jsonl, parse_search_page

URLS = {
    "stubs": "https://www.royalroad.com/fictions/search?status=STUB&page={page}",
    "controls": "https://www.royalroad.com/fictions/search?orderBy=followers&dir=desc&page={page}",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["stubs", "controls"], required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--max-pages", type=int, default=100000)
    args = ap.parse_args()

    raw = os.path.join(args.data_dir, "raw")
    os.makedirs(raw, exist_ok=True)
    out_path = os.path.join(raw, f"{args.mode}_listing.jsonl")
    ckpt_path = os.path.join(raw, f"{args.mode}_listing.ckpt")

    start_page = 1
    if os.path.exists(ckpt_path):
        start_page = int(open(ckpt_path).read().strip()) + 1
    print(f"[{args.mode}] starting at page {start_page}", flush=True)

    sess = PoliteSession()
    seen_first_fid = None
    empty_streak = 0
    for page in range(start_page, args.max_pages + 1):
        r = sess.get(URLS[args.mode].format(page=page))
        cards = parse_search_page(r.text)
        if not cards:
            empty_streak += 1
            if empty_streak >= 2:
                print(f"[{args.mode}] page {page}: empty, stopping", flush=True)
                break
            continue
        empty_streak = 0
        # detect page-overflow (RR repeats last page content past the end)
        if cards[0]["fiction_id"] == seen_first_fid:
            print(f"[{args.mode}] page {page}: repeat of previous page, stopping", flush=True)
            break
        seen_first_fid = cards[0]["fiction_id"]
        for c in cards:
            c["listing_mode"] = args.mode
            c["listing_page"] = page
            append_jsonl(out_path, c)
        with open(ckpt_path, "w") as f:
            f.write(str(page))
        if page % 25 == 0 or page == start_page:
            print(f"[{args.mode}] page {page}: {len(cards)} cards "
                  f"(last followers={cards[-1].get('followers')})", flush=True)
    print(f"[{args.mode}] done.", flush=True)


if __name__ == "__main__":
    main()
