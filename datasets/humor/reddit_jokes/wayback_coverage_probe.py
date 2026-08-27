#!/usr/bin/env python3
"""Wayback coverage probe for mod-removed r/Jokes posts.

Per era: sample mod-removed post ids (Arctic Shift), CDX-check each for a
capture within CAPTURE_WINDOW_DAYS of posting, and verify body extraction
(selftext in the SSR hydration blob) on a subsample of hits.
"""
import json
import re
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "norm-research/0.1 (academic; contact alex2awesome@gmail.com)"}
AS_API = "https://arctic-shift.photon-reddit.com/api/posts/search"
CDX = "http://web.archive.org/cdx/search/cdx"
ERAS = ["2021-06-01", "2022-06-01", "2023-06-01"]
PER_ERA = 25
CAPTURE_WINDOW_DAYS = 7
VERIFY_PER_ERA = 25


def get_json(url):
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers=UA)
            return json.load(urllib.request.urlopen(req, timeout=90))
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt * 5)


def get_text(url):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=120).read().decode(errors="ignore")


def removed_sample(era):
    out = []
    after = era
    pages = 0
    while len(out) < PER_ERA and pages < 30:
        pages += 1
        url = f"{AS_API}?subreddit=Jokes&after={after}&limit=100&sort=asc"
        data = get_json(url)["data"]
        if not data:
            break
        for p in data:
            if p.get("removed_by_category") == "moderator" and not p.get("over_18"):
                out.append(p)
                if len(out) >= PER_ERA:
                    break
        after = str(int(max(p["created_utc"] for p in data)) + 1)
        time.sleep(1.1)
    return out


def cdx_first_capture(post):
    path = urllib.parse.quote(f"reddit.com/r/Jokes/comments/{post['id']}*", safe="*/")
    url = f"{CDX}?url={path}&output=text&fl=timestamp,original,statuscode&limit=40"
    try:
        rows = [l.split() for l in get_text(url).strip().splitlines() if l.strip()]
    except Exception:
        return None
    def in_window(ts):
        t = time.mktime(time.strptime(ts, "%Y%m%d%H%M%S"))
        return t - post["created_utc"] <= CAPTURE_WINDOW_DAYS * 86400
    ok = [(ts, orig) for ts, orig, code in rows if code == "200" and in_window(ts)]
    # prefer thread-page captures over comment permalinks
    thread = [r for r in ok if "/comment/" not in r[1]]
    pool = thread or ok
    return min(pool) if pool else None


def body_from_capture(ts, orig):
    try:
        h = get_text(f"http://web.archive.org/web/{ts}/{orig}")
    except Exception:
        return None
    # 1) rendered post-body div (new-reddit SSR thread pages)
    m = re.search(r'data-click-id="text"[^>]*>(.*?)</div></div>', h, re.S)
    if m:
        t = re.sub(r'<[^>]+>', ' ', m.group(1))
        t = re.sub(r'\s+', ' ', t).strip()
        if t and t not in ("[removed]", "[deleted]") and "Click to see nsfw" not in t:
            return t[:120]
    # 2) selftext field (older SSR / json-ish pages)
    m = re.search(r'"selftext":\s*"((?:[^"\\]|\\.)*)"', h)
    if m and m.group(1) not in ("", "[removed]", "[deleted]"):
        return m.group(1)[:120]
    return None


def main():
    for era in ERAS:
        posts = removed_sample(era)
        hits = []
        for p in posts:
            cap = cdx_first_capture(p)
            if cap:
                hits.append((p, cap))
            time.sleep(1.0)
        verified = 0
        tried = 0
        samples = []
        for p, (ts, orig) in hits[:VERIFY_PER_ERA]:
            tried += 1
            body = body_from_capture(ts, orig)
            if body:
                verified += 1
                samples.append((p["title"][:50], body[:60]))
            time.sleep(1.5)
        print(f"{era[:7]}: {len(posts)} removed sampled | CDX capture<= {CAPTURE_WINDOW_DAYS}d: "
              f"{len(hits)} ({len(hits)/max(1,len(posts)):.0%}) | body extracted {verified}/{tried}",
              flush=True)
        for t, b in samples:
            print(f"    {t!r} -> {b!r}", flush=True)


if __name__ == "__main__":
    main()
