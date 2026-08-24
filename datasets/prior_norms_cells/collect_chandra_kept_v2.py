#!/usr/bin/env python3
"""Kept-side controls v2 for the Chandrasekharan pooled removal cells.

v1 flaw (leak audit notes/2026-08-24__chandra_leak_audit.md): the backward crawl
from the window end stopped at target, so kept rows covered only the last 1-6
DAYS of the era window for 8/9 subs (nosleep 29d) while removals span 11 months
-> class|date confound ("dragonstone"/"ronaldo" were top kept features).

v2 (coordinator 2026-08-24): sample UNIFORMLY over 2016-05-01..2017-03-31 via
~22 evenly spaced fortnightly strata; per-window quota = target/22; same subs,
same fields, same pacing as v1. Resume-state per (sub, window). tifu re-pulled
from scratch (v1 gzip had a torn tail). Writes kept_v2_<sub>.jsonl.gz ALONGSIDE
v1 files — v1 outputs untouched (never-delete-data rule).

Coordinator addendum (2026-08-24): also persist the Arctic Shift `author` field
(kept side) for author-disjoint readouts. Removal side has NO author anywhere
(reddit-removal-log.csv = body+subreddit; the released macro-norm CSVs are bare
text) — recorded as untestable in the v2 build meta."""
import gzip, json, math, time
from pathlib import Path
import requests

API = "https://arctic-shift.photon-reddit.com/api/comments/search"
OUT = Path(__file__).resolve().parent
WINDOW = (1462060800, 1490918400)  # 2016-05-01 .. 2017-03-31 UTC
N_STRATA = 22                      # ~15.2-day windows
TARGETS = {
    "funny": 21000, "Showerthoughts": 15000, "tifu": 10500, "nottheonion": 9700,
    "me_irl": 7900, "nosleep": 24000, "books": 12400, "gameofthrones": 9300,
    "asoiaf": 7600}
STATE = OUT / "collect_state_v2.json"
state = json.loads(STATE.read_text()) if STATE.exists() else {}

span = WINDOW[1] - WINDOW[0]
edges = [WINDOW[0] + round(i * span / N_STRATA) for i in range(N_STRATA + 1)]

for sub, target in TARGETS.items():
    quota = math.ceil(target / N_STRATA)
    sst = state.setdefault(sub, {})
    for w in range(N_STRATA):
        wkey = str(w)
        w_lo, w_hi = edges[w], edges[w + 1]
        wst = sst.setdefault(wkey, {"got": 0, "before": w_hi})
        if wst["got"] >= quota or wst.get("exhausted"):
            continue
        got, before = wst["got"], wst["before"]
        fh = gzip.open(OUT / f"kept_v2_{sub}.jsonl.gz", "at")
        empties = wst.get("empties", 0)
        while got < quota and before > w_lo:
            try:
                r = requests.get(API, params={"subreddit": sub, "limit": 100,
                                              "before": before, "after": w_lo},
                                 timeout=30)
                rows = r.json().get("data", [])
            except Exception as e:
                print(f"[{sub} w{w}] {type(e).__name__}, sleep 60", flush=True)
                time.sleep(60)
                continue
            if not rows:
                empties += 1
                wst.update({"got": got, "before": before, "empties": empties,
                            "exhausted": empties >= 2})
                STATE.write_text(json.dumps(state))
                if empties >= 2:
                    break
                time.sleep(30)
                continue
            for c in rows:
                body = c.get("body") or ""
                if body in ("[deleted]", "[removed]") or len(body) < 5:
                    continue
                fh.write(json.dumps({"id": c.get("id"), "subreddit": sub,
                                     "body": body,
                                     "created_utc": c.get("created_utc"),
                                     "score": c.get("score"),
                                     "author": c.get("author")}) + "\n")
                got += 1
            before = min(int(c.get("created_utc", before)) for c in rows)
            wst.update({"got": got, "before": before, "empties": empties})
            STATE.write_text(json.dumps(state))
            time.sleep(1.2)
        fh.close()  # close per window: a crash tears at most one window's tail
        print(f"[{sub}] window {w+1}/{N_STRATA} done: {got}/{quota}", flush=True)
    total = sum(sst[str(i)]["got"] for i in range(N_STRATA) if str(i) in sst)
    print(f"[{sub}] FINISHED v2 total={total}/{target}", flush=True)
print("CHANDRA_KEPT_COLLECT_V2_DONE")
