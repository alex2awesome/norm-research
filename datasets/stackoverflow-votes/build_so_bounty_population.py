#!/usr/bin/env python3
"""U4 — StackOverflow BOUNTY (curated) population, python corpus.

Same design + same precision discipline as the math.SE bounty cell (U3):
  * y = MANUAL bounty award (deliberate curation).  AUTO half-awards at expiry
    (system gives floor(amount/2) to the top-scored eligible answer) are NOT
    curation — questions whose only award is AUTO are dropped entirely.
  * WITHIN-QUESTION contrast: awarded answer (y=1) vs the other answers on the
    same bountied question (y=0); questions with <2 answers in-corpus dropped.
  * X-space = the SAME python answer corpus as the V6 so_votes cell
    (so_python_answers.parquet), so verdict/curated/community share X.

Input: se_dumps/so_bounty_votes_raw.txt (grep-prefiltered VoteTypeId 8/9 lines
from the 23GB Votes.xml — produced by the launcher), so_python parquets.
Output: so_bounty_manual_population.jsonl.gz + audit JSON.
"""
import gzip
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

D = Path("/lfs/skampere3/0/alexspan/data/se_dumps")
SOP = Path("/lfs/skampere3/0/alexspan/norm-research/datasets/stackoverflow_python")
ATTR = re.compile(r'(\w+)="([^"]*)"')

print("[1/4] loading python answers parquet ...", flush=True)
ans = pd.read_parquet(SOP / "so_python_answers.parquet",
                      columns=["Id", "ParentId", "Score", "CreationDate", "Body"])
ans["Id"] = ans.Id.astype("int64").astype(str)
# ParentId is float64 in the parquet (pandas NaN-capable dtype); a bare astype(str)
# yields "337.0" and silently breaks every join against the XML's integer strings —
# the exact bug that produced an all-OTHER classification on the first run.
assert ans.ParentId.notna().all(), "answers with null ParentId"
ans["ParentId"] = ans.ParentId.astype("int64").astype(str)
aid_set = set(ans.Id)
print(f"      answers: {len(ans)}", flush=True)

print("[2/4] parsing prefiltered bounty votes ...", flush=True)
starts = defaultdict(list)   # qid -> [(amount, date)]
closes = []                  # (aid, amount_or_None, date)
with open(D / "so_bounty_votes_raw.txt", encoding="utf-8", errors="ignore") as fh:
    for line in fh:
        a = dict(ATTR.findall(line))
        if a.get("VoteTypeId") == "8":
            starts[a["PostId"]].append((int(a.get("BountyAmount", 0)),
                                        a.get("CreationDate", "")))
        elif a.get("VoteTypeId") == "9":
            amt = a.get("BountyAmount")
            closes.append((a["PostId"], int(amt) if amt else None,
                           a.get("CreationDate", "")))
print(f"      starts {sum(len(v) for v in starts.values())} closes {len(closes)}", flush=True)

print("[3/4] classifying closes on python answers (manual vs auto) ...", flush=True)
a2q = dict(zip(ans.Id, ans.ParentId))
cls_count = Counter()
winners = {}                 # aid -> ("MANUAL"|"AUTO_HALF"|"OTHER", close_date)
for aid, amt, cdate in closes:
    if aid not in aid_set:
        continue
    qid = a2q[aid]
    st = starts.get(qid, [])
    if amt is None:
        cls = "NOAMT"
    elif any(s[0] == amt for s in st):
        cls = "MANUAL"
    elif any(s[0] // 2 == amt and s[0] != amt for s in st):
        cls = "AUTO_HALF"
    else:
        cls = "OTHER"
    cls_count[cls] += 1
    prev = winners.get(aid)
    winners[aid] = (cls, cdate) if prev is None else prev
print(f"      classes: {dict(cls_count)}", flush=True)

# questions whose winners are ALL manual
qwin = defaultdict(set)
for aid, (cls, _) in winners.items():
    qwin[a2q[aid]].add(cls)
keep_q = {q for q, cs in qwin.items() if cs == {"MANUAL"}}

print("[4/4] assembling within-question population ...", flush=True)
sub = ans[ans.ParentId.isin(keep_q)].copy()
nq = sub.ParentId.value_counts()
sub = sub[sub.ParentId.isin(nq[nq >= 2].index)]
sub["y"] = sub.Id.map(lambda i: int(i in winners and winners[i][0] == "MANUAL"))

qs = pd.read_parquet(SOP / "so_python_questions.parquet")
qs["Id"] = qs.Id.astype("int64").astype(str)
qcols = {c.lower(): c for c in qs.columns}
title_c = qcols.get("title", "Title" if "Title" in qs.columns else None)
body_c = qcols.get("body", "Body" if "Body" in qs.columns else None)
qm = qs.set_index("Id")
rows = []
for r in sub.itertuples():
    qrow = qm.loc[r.ParentId] if r.ParentId in qm.index else None
    rows.append({"qid": r.ParentId, "aid": r.Id, "y": int(r.y),
                 "answer_body": str(r.Body), "answer_score": int(r.Score),
                 "answer_created": str(r.CreationDate),
                 "q_title": (str(qrow[title_c]) if qrow is not None and title_c else ""),
                 "q_body": (str(qrow[body_c])[:2000] if qrow is not None and body_c else ""),
                 "n_answers_on_q": int(nq[r.ParentId])})
with gzip.open(D / "so_bounty_manual_population.jsonl.gz", "wt") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")

out = {"close_classes_on_python_answers": dict(cls_count),
       "questions_all_manual": len(keep_q),
       "questions_with_>=2_answers": int((nq >= 2).sum()),
       "rows": len(rows),
       "pos_rate": round(sum(r["y"] for r in rows) / max(1, len(rows)), 4)}
(D / "so_bounty_population_audit.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1), flush=True)
print("SO_BOUNTY_POP_DONE", flush=True)
