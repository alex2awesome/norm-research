#!/usr/bin/env python3
"""Reedsy Prompts full scrape — weekly contest pools with editorial winner.

Structure (verified 2026-08-13):
  index:   /creative-writing-prompts/contests/page/N/   (SSR; "Won by" -> /short-story/<id>/)
  contest: /creative-writing-prompts/contests/<cid>/?page=K   (15 stories/page, SSR)
  story:   /short-story/<sid>/                            (SSR full text, likes, points)

Stages (each resumable via state.json):
  1. index  -> contests.jsonl (cid, title-line raw html slice, winner sid)
  2. lists  -> contest_stories.jsonl (cid -> [sid, ...])
  3. story  -> stories_raw/<sid>.html.gz

Politeness 1.05s/req, browser UA + contact.
"""
import gzip
import json
import os
import re
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://reedsy.com"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 norm-research-academic "
      "(contact: alex2awesome@gmail.com)")
RATE = 4.0
STATE = os.path.join(HERE, "state.json")


def st_load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def st_save(s):
    json.dump(s, open(STATE, "w"))


def get(sess, url):
    for a in range(7):
        try:
            r = sess.get(url, timeout=60)
            if r.status_code in (429, 500, 502, 503):
                raise RuntimeError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.text
        except Exception as e:
            w = min(300, 2 ** a * 5)
            print(f"{url} attempt {a}: {e}; sleep {w}", flush=True)
            time.sleep(w)
    raise RuntimeError(url)


def stage_index(sess, st):
    if st.get("index_done"):
        return
    out = open(os.path.join(HERE, "contests_index_raw.jsonl"), "w")
    page = 1
    while page < 60:
        h = get(sess, f"{BASE}/creative-writing-prompts/contests/page/{page}/")
        blocks = re.split(r'(?=<div class="panel panel-thin)', h)[1:]
        found = 0
        for b in blocks:
            m = re.search(r'href="/creative-writing-prompts/contests/(\d+)/"', b)
            if not m:
                continue
            found += 1
            win = re.search(r'Won by <a[^>]*href="/short-story/([a-z0-9]+)/"[^>]*>([^<]*)', b)
            cnt = re.search(r'([\d,]+) contest entries /\s*([\d,]+) stories', b)
            out.write(json.dumps({
                "contest_id": int(m.group(1)),
                "winner_story": win.group(1) if win else None,
                "winner_author": win.group(2).strip() if win else None,
                "entries": int(cnt.group(1).replace(",", "")) if cnt else None,
                "stories": int(cnt.group(2).replace(",", "")) if cnt else None,
                "raw": b[:4000],
            }) + "\n")
        print(f"index page {page}: {found} contests", flush=True)
        if found == 0:
            break
        page += 1
        time.sleep(RATE)
    out.close()
    st["index_done"] = True
    st_save(st)


def stage_lists(sess, st):
    contests = [json.loads(l) for l in open(os.path.join(HERE, "contests_index_raw.jsonl"))]
    done = set(st.get("lists_done", []))
    out = open(os.path.join(HERE, "contest_stories.jsonl"), "a")
    for c in contests:
        cid = c["contest_id"]
        if cid in done:
            continue
        sids, page = [], 1
        while page < 40:
            h = get(sess, f"{BASE}/creative-writing-prompts/contests/{cid}/?page={page}")
            ids = re.findall(r'href="/short-story/([a-z0-9]+)/"', h)
            new = [i for i in dict.fromkeys(ids) if i not in sids]
            if not new:
                break
            sids.extend(new)
            page += 1
            time.sleep(RATE)
        out.write(json.dumps({"contest_id": cid, "story_ids": sids}) + "\n")
        out.flush()
        done.add(cid)
        st["lists_done"] = sorted(done)
        st_save(st)
        print(f"contest {cid}: {len(sids)} stories ({len(done)}/{len(contests)})", flush=True)
        time.sleep(RATE)
    out.close()


def stage_stories(sess, st):
    raw_dir = os.path.join(HERE, "stories_raw")
    os.makedirs(raw_dir, exist_ok=True)
    sids = []
    for l in open(os.path.join(HERE, "contest_stories.jsonl")):
        sids.extend(json.loads(l)["story_ids"])
    sids = list(dict.fromkeys(sids))
    print(f"{len(sids)} unique stories", flush=True)
    for i, sid in enumerate(sids):
        f = os.path.join(raw_dir, f"{sid}.html.gz")
        if os.path.exists(f):
            continue
        h = get(sess, f"{BASE}/short-story/{sid}/")
        with gzip.open(f, "wt") as fh:
            fh.write(h)
        if i % 200 == 0:
            print(f"story {i}/{len(sids)}", flush=True)
        time.sleep(RATE)


def main():
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA})
    st = st_load()
    stage_index(sess, st)
    stage_lists(sess, st)
    stage_stories(sess, st)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
