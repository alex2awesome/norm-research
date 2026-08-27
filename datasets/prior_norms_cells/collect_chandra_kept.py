#!/usr/bin/env python3
"""Kept-side controls for the Chandrasekharan pooled removal cells (user
2026-08-20): era-bounded Arctic Shift comment sampling for the 9 humor/CW
study subreddits. Window = study period (2016-05-01 .. 2017-03-31); no per-row
era matching possible (removal log carries body+subreddit only — DECLARED).
Kept rows text-matching a removal body are excluded downstream (archive may
predate removal). Resume: per-sub 'before' cursor persisted."""
import gzip, json, time
from pathlib import Path
import requests

API = "https://arctic-shift.photon-reddit.com/api/comments/search"
OUT = Path(__file__).resolve().parent
WINDOW = (1462060800, 1490918400)  # 2016-05-01 .. 2017-03-31 UTC
# HUMOR + CW SUBS (user 2026-08-20 final: both sets)
TARGETS = {
    "funny": 21000, "Showerthoughts": 15000, "tifu": 10500, "nottheonion": 9700,
    "me_irl": 7900, "nosleep": 24000, "books": 12400, "gameofthrones": 9300,
    "asoiaf": 7600}
STATE = OUT / "collect_state.json"
state = json.loads(STATE.read_text()) if STATE.exists() else {}

for sub, target in TARGETS.items():
    got = state.get(sub, {}).get("got", 0)
    before = state.get(sub, {}).get("before", WINDOW[1])
    if got >= target or state.get(sub, {}).get("exhausted"):
        print(f"[{sub}] done ({got})", flush=True)
        continue
    fh = gzip.open(OUT / f"kept_{sub}.jsonl.gz", "at")
    while got < target and before > WINDOW[0]:
        try:
            r = requests.get(API, params={"subreddit": sub, "limit": 100,
                                          "before": before, "after": WINDOW[0]},
                             timeout=30)
            rows = r.json().get("data", [])
        except Exception as e:
            print(f"[{sub}] {type(e).__name__}, sleep 60", flush=True)
            time.sleep(60)
            continue
        if not rows:
            empties = state.get(sub, {}).get("empties", 0) + 1
            state[sub] = {"got": got, "before": before, "empties": empties,
                          "exhausted": empties >= 2}
            if empties >= 2:
                break
            time.sleep(30)
            continue
        for c in rows:
            body = c.get("body") or ""
            if body in ("[deleted]", "[removed]") or len(body) < 5:
                continue
            fh.write(json.dumps({"id": c.get("id"), "subreddit": sub,
                                 "body": body, "created_utc": c.get("created_utc"),
                                 "score": c.get("score")}) + "\n")
            got += 1
        before = min(int(c.get("created_utc", before)) for c in rows)
        state[sub] = {"got": got, "before": before}
        STATE.write_text(json.dumps(state))
        if got % 2000 < 100:
            print(f"[{sub}] {got}/{target} before={before}", flush=True)
        time.sleep(1.2)
    fh.close()
    print(f"[{sub}] FINISHED {got}", flush=True)
print("CHANDRA_KEPT_COLLECT_DONE")
