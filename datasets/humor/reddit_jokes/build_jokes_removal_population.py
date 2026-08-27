#!/usr/bin/env python3
"""U2 — r/Jokes REMOVAL (verdict) population from the ARRIVED wayback texts.

y = 1: joke removed by moderators (id in the removal universe, pre-removal text
recovered via Wayback); y = 0: kept joke from the kept-side universe.
APPEND-friendly: rebuilt from whatever texts have arrived (the stageC fetch is
still running); rerun extends the population, never deletes (row_id = reddit id).

Controls: kept jokes sampled 2:1 per created-MONTH stratum (era confound pinned
at design time).  Declared covariates carried, never features: created_utc,
over_18 flag where present, score (kept side; removed jokes' scores are
frozen-at-removal and NOT comparable — carried as score_at_capture, flagged).
Spot-check gates printed.  Run on sk3.
"""
import gzip
import hashlib
import zlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

RD = Path("/lfs/skampere3/0/alexspan/data/reddit_dumps")
OUT = Path("/lfs/skampere3/0/alexspan/norm-research/datasets/humor/reddit_jokes/removal_cell")
OUT.mkdir(parents=True, exist_ok=True)
CONTROL_RATIO = 2
MIN_CHARS, MAX_CHARS = 20, 12000

removed = {}
for f in ("jokes_wayback_text.jsonl.gz", "jokes_wayback_text2.jsonl.gz"):
    p = RD / f
    if not p.exists():
        continue
    fh = gzip.open(p, "rt")
    while True:
        # the stageC fetch appends to this file LIVE — the final gzip member can be
        # mid-write. Read line-at-a-time and stop cleanly at the torn tail.
        try:
            line = fh.readline()
        except (EOFError, OSError, zlib.error):
            print(f"  [{f}] torn tail reached (fetch still writing) — keeping parsed rows")
            break
        if not line:
            break
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = str(r.get("id") or r.get("post_id") or "")
        title = str(r.get("title") or "")
        body = str(r.get("selftext") or r.get("body") or r.get("text") or "")
        text = (title + "\n\n" + body).strip()
        if rid and MIN_CHARS <= len(text) <= MAX_CHARS:
            removed[rid] = {"row_id": rid, "text": text, "judgement": 1,
                            "created_utc": r.get("created_utc"),
                            "over_18": r.get("over_18"),
                            "score_at_capture": r.get("score"),
                            "source": f}
    fh.close()
print(f"removed-with-text: {len(removed)}")

kept = []
for line in gzip.open(RD / "jokes_kept_universe.jsonl.gz", "rt"):
    r = json.loads(line)
    rid = str(r.get("id") or "")
    title = str(r.get("title") or "")
    body = str(r.get("selftext") or r.get("body") or r.get("text") or "")
    text = (title + "\n\n" + body).strip()
    if rid and rid not in removed and MIN_CHARS <= len(text) <= MAX_CHARS:
        kept.append({"row_id": rid, "text": text, "judgement": 0,
                     "created_utc": r.get("created_utc"),
                     "over_18": r.get("over_18"),
                     "score_at_capture": r.get("score"), "source": "kept_universe"})
print(f"kept pool (gated): {len(kept)}")


def month(u):
    try:
        import datetime
        return datetime.datetime.utcfromtimestamp(int(float(u))).strftime("%Y-%m")
    except Exception:
        return "unknown"


rem_by_m = Counter(month(r["created_utc"]) for r in removed.values())
kept_by_m = defaultdict(list)
for r in kept:
    kept_by_m[month(r["created_utc"])].append(r)

rows = list(removed.values())
short = {}
for m, n_rem in sorted(rem_by_m.items()):
    pool = kept_by_m.get(m, [])
    want = n_rem * CONTROL_RATIO
    rng = random.Random(int(hashlib.sha256(f"jokes-removal|{m}".encode()).hexdigest()[:12], 16))
    take = pool if len(pool) <= want else rng.sample(pool, want)
    if len(take) < want:
        short[m] = (len(take), want)
    rows.extend(take)

pos = sum(r["judgement"] for r in rows)
with gzip.open(OUT / "population.jsonl.gz", "wt") as fh:
    for r in sorted(rows, key=lambda x: x["row_id"]):
        fh.write(json.dumps(r) + "\n")
man = {
    "cell": "jokes_removal (VERDICT: moderator removal)",
    "n": len(rows), "n_removed": pos, "pos_rate": round(pos / len(rows), 4),
    "controls": f"kept 2:1 per created-month stratum, stable-hash month seeds",
    "months_short_of_ratio": short,
    "partial_corpus_note": "stageC wayback fetch still running; rerun APPENDS "
                           "(row_id-keyed, never deletes)",
    "confounds_declared": ["created era (pinned by stratified controls)",
                           "over_18", "removal-reason mix (reposts vs rules vs "
                           "quality — NOT separable yet, flagged)",
                           "score_at_capture NOT comparable across sides"],
}
(OUT / "manifest.json").write_text(json.dumps(man, indent=1))
print(json.dumps(man, indent=1))
print("JOKES_REMOVAL_POP_DONE")
