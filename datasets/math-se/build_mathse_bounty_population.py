#!/usr/bin/env python3
"""math.SE BOUNTY (curated) population — step 1 of the U3 unified-X cell.

y = bounty-close (VoteTypeId=9): the asker/bounty-setter AWARDED the bounty to this
answer — a deliberate, costly curation act.  Design mirrors the vote_score cell:
WITHIN-QUESTION — bounty-winning answer (y=1) vs the other answers on the SAME
bountied question (y=0); questions with <2 answers dropped (no contrast).

Inputs (sk3): data/se_dumps/mathse_extracted/Votes.xml (1.2GB), Posts.xml (5.8GB).
Streaming parse, no DOM.  Output: mathse_bounty_population.jsonl.gz with one row
per answer on a bountied question: {qid, aid, y, answer_body, answer_score,
answer_created, q_title, q_body(2k), n_answers_on_q, bounty_close_date}.

Spot-check gates printed at the end (dataset-first protocol): n bountied questions,
n with >=2 answers, per-question positive count (should be ~1), y rate.
"""
import gzip
import json
import re
from pathlib import Path
from xml.sax.saxutils import unescape

D = Path("/lfs/skampere3/0/alexspan/data/se_dumps")
OUT = D / "mathse_bounty_population.jsonl.gz"

ROW = re.compile(r'<row (.+?)/>')
ATTR = re.compile(r'(\w+)="([^"]*)"')


def attrs(line):
    m = ROW.search(line)
    return dict(ATTR.findall(m.group(1))) if m else None


# ---- pass 1: bounty-close votes -> winning answer ids + dates ----------------
win = {}          # answer PostId -> close date
print("[1/3] scanning Votes.xml for VoteTypeId=9 ...", flush=True)
with open(D / "mathse_extracted" / "Votes.xml", encoding="utf-8", errors="ignore") as fh:
    for line in fh:
        if 'VoteTypeId="9"' not in line:
            continue
        a = attrs(line)
        if a:
            win[a["PostId"]] = a.get("CreationDate", "")
print(f"      bounty-close votes: {len(win)}", flush=True)

# ---- pass 2: Posts.xml — find winners' questions, then all answers -----------
print("[2/3] Posts.xml pass A: map winning answers -> questions ...", flush=True)
win_q = {}        # question id -> set of winning answer ids
with open(D / "mathse_extracted" / "Posts.xml", encoding="utf-8", errors="ignore") as fh:
    for line in fh:
        if 'PostTypeId="2"' not in line:
            continue
        a = attrs(line)
        if a and a["Id"] in win:
            win_q.setdefault(a["ParentId"], set()).add(a["Id"])
print(f"      bountied questions with a recorded winner: {len(win_q)}", flush=True)

print("[3/3] Posts.xml pass B: collect all answers on those questions + q text ...", flush=True)
qmeta, rows = {}, []
with open(D / "mathse_extracted" / "Posts.xml", encoding="utf-8", errors="ignore") as fh:
    for line in fh:
        a = None
        if 'PostTypeId="1"' in line:
            a = attrs(line)
            if a and a["Id"] in win_q:
                qmeta[a["Id"]] = {"title": unescape(a.get("Title", "")),
                                  "body": unescape(a.get("Body", ""))[:2000]}
        elif 'PostTypeId="2"' in line:
            a = attrs(line)
            if a and a.get("ParentId") in win_q:
                rows.append({"qid": a["ParentId"], "aid": a["Id"],
                             "y": int(a["Id"] in win_q[a["ParentId"]]),
                             "answer_body": unescape(a.get("Body", "")),
                             "answer_score": int(a.get("Score", 0)),
                             "answer_created": a.get("CreationDate", ""),
                             "bounty_close_date": win.get(a["Id"], "")})

from collections import Counter
per_q = Counter(r["qid"] for r in rows)
keep_q = {q for q, n in per_q.items() if n >= 2}
kept = [r for r in rows if r["qid"] in keep_q]
for r in kept:
    qm = qmeta.get(r["qid"], {})
    r["q_title"] = qm.get("title", "")
    r["q_body"] = qm.get("body", "")
    r["n_answers_on_q"] = per_q[r["qid"]]

with gzip.open(OUT, "wt", encoding="utf-8") as fh:
    for r in kept:
        fh.write(json.dumps(r) + "\n")

pos_per_q = Counter()
for r in kept:
    if r["y"]:
        pos_per_q[r["qid"]] += 1
print(json.dumps({
    "bounty_close_votes": len(win),
    "bountied_questions_with_winner": len(win_q),
    "questions_with_>=2_answers": len(keep_q),
    "rows": len(kept),
    "pos_rate": round(sum(r["y"] for r in kept) / max(1, len(kept)), 4),
    "questions_with_multiple_winners": sum(1 for v in pos_per_q.values() if v > 1),
    "out": str(OUT),
}, indent=1), flush=True)
print("MATHSE_BOUNTY_POP_DONE", flush=True)
