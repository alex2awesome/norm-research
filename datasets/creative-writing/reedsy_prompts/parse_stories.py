#!/usr/bin/env python3
"""Parse fetched Reedsy story HTML -> stories_parsed.jsonl.gz (incremental).

Per story: sid, title, author_slug, contest_id, likes, winner/shortlist badge,
full text. Winner labels are additionally joined from contests_index_raw.jsonl
(authoritative "Won by" on the contest card).
"""
import glob
import gzip
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def clean(x):
    x = re.sub(r"<[^>]+>", " ", x)
    x = x.replace("&nbsp;", " ").replace("&amp;", "&").replace("&#39;", "'")
    return re.sub(r"[ \t]+", " ", x).strip()


def main():
    winners = {}
    for l in open(os.path.join(HERE, "contests_index_raw.jsonl")):
        d = json.loads(l)
        if d.get("winner_story"):
            winners[d["winner_story"]] = d["contest_id"]
    story_contest = {}
    for l in open(os.path.join(HERE, "contest_stories.jsonl")):
        d = json.loads(l)
        for sid in d["story_ids"]:
            story_contest.setdefault(sid, d["contest_id"])

    done = set()
    outp = os.path.join(HERE, "stories_parsed.jsonl.gz")
    if os.path.exists(outp):
        try:
            fh = gzip.open(outp, "rt")
            while True:
                l = fh.readline()
                if not l:
                    break
                try:
                    done.add(json.loads(l)["sid"])
                except Exception:
                    break
        except Exception:
            pass
    out = gzip.open(outp, "ab")
    n = 0
    for f in sorted(glob.glob(os.path.join(HERE, "stories_raw", "*.html.gz"))):
        sid = os.path.basename(f)[:-8]
        if sid in done:
            continue
        try:
            h = gzip.open(f, "rt").read()
        except Exception:
            continue
        m = re.search(r"<h1[^>]*>\s*([^<]{1,200}?)\s*</h1>", h)
        title = clean(m.group(1)) if m else None
        m = re.search(r"author/([a-z0-9-]+)/", h)
        author = m.group(1) if m else None
        m = re.search(r"/creative-writing-prompts/contests/(\d+)/", h)
        contest = int(m.group(1)) if m else story_contest.get(sid)
        m = re.search(r"(\d[\d,]*)\s*likes?", h)
        likes = int(m.group(1).replace(",", "")) if m else None
        badge = None
        m = re.search(r"Contest #(\d+) (Winner|Shortlist)", h)
        if m and (contest is None or int(m.group(1)) == contest):
            badge = m.group(2).lower()
        text = None
        m = re.search(r'<article class="article story-typeset[^"]*"[^>]*>(.*?)</article>', h, re.S)
        if m:
            paras = [clean(p) for p in re.findall(r"<p[^>]*>(.*?)</p>", m.group(1), re.S)]
            text = "\n\n".join(p for p in paras if p)
        rec = {"sid": sid, "title": title, "author_slug": author,
               "contest_id": contest, "likes": likes,
               "badge": badge, "is_contest_winner": sid in winners,
               "text_chars": len(text) if text else 0, "text": text}
        out.write((json.dumps(rec) + "\n").encode())
        n += 1
        if n % 2000 == 0:
            out.flush()
            print(f"parsed {n}", flush=True)
    out.close()
    print(f"parsed {n} new stories ({len(done)} already done)")


if __name__ == "__main__":
    main()
