"""Recover early chapter text for RoyalRoad fictions from the Wayback Machine.

For each fiction id:
  1. CDX query  https://web.archive.org/cdx/search/cdx?url=royalroad.com/fiction/{fid}/*
     (cached to raw/cdx/{fid}.json — never re-fetched if present and non-empty)
  2. Collect chapter-page captures (no query string), group by chapter id.
     Chapter ids are platform-globally monotonic -> lowest ids = earliest
     chapters. Take the N_CHAPTERS lowest chapter ids.
  3. For each chapter, fetch the earliest 200 capture via the `id_` raw view;
     if extraction yields < MIN_WORDS words, try up to 2 later captures.
  4. Append one jsonl row per recovered chapter to raw/{side}_chapters.jsonl
     and a per-fiction summary row to raw/{side}_fiction_status.jsonl.

Both y=1 (stubs) and y=0 (controls) go through THIS SAME pipeline — the
archive-availability filter applies symmetrically (x_source='wayback' always).

Resumable: raw/{side}_done.txt holds one fiction id per line. Re-running skips
those. Appends only; build_dataset.py dedups on (fiction_id, chapter_id).

Usage:
  python recover_wayback.py --side stub --data-dir DIR [--limit N]
      [--order-file FILE] [--print-sample]
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rr_common import PoliteSession, append_jsonl, display, extract_chapter_text, read_jsonl

N_CHAPTERS = 3
MIN_WORDS = 200
MAX_SNAPSHOT_TRIES = 3

CDX_URL = (
    "https://web.archive.org/cdx/search/cdx?url=royalroad.com/fiction/{fid}/*"
    "&output=json&filter=statuscode:200&fl=timestamp,original,statuscode,length"
    "&limit=60000"
)
SNAP_URL = "https://web.archive.org/web/{ts}id_/{orig}"


def get_cdx(sess, fid, cdx_dir):
    path = os.path.join(cdx_dir, f"{fid}.json")
    if os.path.exists(path) and os.path.getsize(path) > 2:
        try:
            return json.load(open(path))
        except json.JSONDecodeError:
            pass
    r = sess.get(CDX_URL.format(fid=fid), timeout=90)
    try:
        rows = r.json() if r.text.strip() else []
    except json.JSONDecodeError:
        rows = []
    with open(path, "w") as f:
        json.dump(rows, f)
    return rows


def chapter_captures(rows, fid):
    """rows -> {chapter_id: [(ts, original), ...] sorted by ts}."""
    pat = re.compile(
        rf"^https?://(?:www\.)?royalroad\.com/fiction/{fid}/[^/?]+/chapter/(\d+)/[^?]*$"
    )
    chaps = {}
    data = rows[1:] if rows and rows[0] and rows[0][0] == "timestamp" else rows
    for row in data:
        if len(row) < 2:
            continue
        ts, orig = row[0], row[1]
        m = pat.match(orig)
        if m:
            chaps.setdefault(int(m.group(1)), []).append((ts, orig))
    for v in chaps.values():
        v.sort()
    return chaps


def recover_fiction(sess, fid, cdx_dir):
    rows = get_cdx(sess, fid, cdx_dir)
    chaps = chapter_captures(rows, fid)
    picked = sorted(chaps)[:N_CHAPTERS]
    recovered = []
    for rank, cid in enumerate(picked, 1):
        snaps = chaps[cid][:MAX_SNAPSHOT_TRIES]
        for ts, orig in snaps:
            try:
                r = sess.get(SNAP_URL.format(ts=ts, orig=orig), timeout=90)
            except RuntimeError:
                continue
            if r.status_code != 200:
                continue
            title, text = extract_chapter_text(r.text)
            if text and len(text.split()) >= MIN_WORDS:
                recovered.append({
                    "fiction_id": fid,
                    "chapter_id": cid,
                    "chapter_rank": rank,
                    "chapter_title": title,
                    "wayback_ts": ts,
                    "source_url": orig,
                    "n_words": len(text.split()),
                    "raw_text": text,
                })
                break
    return len(chaps), picked, recovered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["stub", "control"], required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="max NEW fictions to process this run")
    ap.add_argument("--order-file", default=None,
                    help="file with one fiction id per line, processed in order")
    ap.add_argument("--print-sample", action="store_true",
                    help="print metadata + first 300 chars per recovered chapter")
    args = ap.parse_args()

    raw = os.path.join(args.data_dir, "raw")
    cdx_dir = os.path.join(raw, "cdx")
    os.makedirs(cdx_dir, exist_ok=True)
    listing_file = os.path.join(
        raw, "stubs_listing.jsonl" if args.side == "stub" else "controls_listing.jsonl")
    chapters_out = os.path.join(raw, f"{args.side}_chapters.jsonl")
    status_out = os.path.join(raw, f"{args.side}_fiction_status.jsonl")
    done_path = os.path.join(raw, f"{args.side}_done.txt")

    cards = {}
    for c in read_jsonl(listing_file):
        cards[c["fiction_id"]] = c  # keep last
    if args.side == "control":
        # exclude anything labeled STUB that slipped into the control listing
        cards = {fid: c for fid, c in cards.items()
                 if not any("STUB" in l.upper() for l in c.get("labels", []))}

    if args.order_file:
        order = [int(l.strip()) for l in open(args.order_file) if l.strip()]
        order = [fid for fid in order if fid in cards]
    else:
        order = sorted(cards)

    done = set()
    if os.path.exists(done_path):
        done = {int(l.strip()) for l in open(done_path) if l.strip()}
    # self-heal: retry fictions whose only status rows are errors
    # (an older revision marked errored fictions done; never had a success pass)
    status_rows = read_jsonl(status_out)
    ok_status = {r["fiction_id"] for r in status_rows if "n_recovered" in r}
    errored = {r["fiction_id"] for r in status_rows if "error" in r}
    retry = (errored - ok_status) & done
    if retry:
        print(f"[{args.side}] retrying {len(retry)} previously-errored fictions",
              flush=True)
        done -= retry
    todo = [fid for fid in order if fid not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[{args.side}] {len(cards)} in pool, {len(done)} done, "
          f"{len(todo)} to process this run", flush=True)

    sess = PoliteSession()
    t0 = time.time()
    n_ok = 0
    for i, fid in enumerate(todo, 1):
        card = cards[fid]
        try:
            n_in_cdx, picked, recovered = recover_fiction(sess, fid, cdx_dir)
        except Exception as e:  # noqa: BLE001 — log and move on, job must survive
            # NOT marked done: errored fictions are retried on the next run
            print(f"[{args.side}] fid={fid} ERROR {type(e).__name__}: {e}", flush=True)
            append_jsonl(status_out, {"fiction_id": fid, "error": str(e)[:300]})
            continue
        for row in recovered:
            row["title"] = card.get("title")
            append_jsonl(chapters_out, row)
        append_jsonl(status_out, {
            "fiction_id": fid,
            "n_chapter_urls_in_cdx": n_in_cdx,
            "picked_chapter_ids": picked,
            "n_recovered": len(recovered),
        })
        with open(done_path, "a") as f:
            f.write(f"{fid}\n")
        if recovered:
            n_ok += 1
        if args.print_sample:
            print("=" * 78)
            print(f"FICTION {fid} | {card.get('title')!r} | followers={card.get('followers')} "
                  f"rating={card.get('rating')} pages={card.get('pages')} "
                  f"tags={card.get('tags')}")
            print(f"  labels={card.get('labels')} chapter_urls_in_cdx={n_in_cdx} "
                  f"recovered={len(recovered)}/{len(picked)}")
            for row in recovered:
                norm = display(row["raw_text"])
                print(f"  -- ch_rank={row['chapter_rank']} cid={row['chapter_id']} "
                      f"ts={row['wayback_ts']} words={row['n_words']} "
                      f"title={row['chapter_title']!r}")
                print("     " + norm[:300].replace("\n", " | "))
        if i % 20 == 0:
            rate = i / (time.time() - t0)
            print(f"[{args.side}] {i}/{len(todo)} processed, {n_ok} with >=1 chapter "
                  f"({rate*3600:.0f} fictions/hr)", flush=True)
    print(f"[{args.side}] run complete: {len(todo)} processed, {n_ok} with >=1 chapter",
          flush=True)


if __name__ == "__main__":
    main()
