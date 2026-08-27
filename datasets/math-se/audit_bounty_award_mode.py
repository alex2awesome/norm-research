#!/usr/bin/env python3
"""U3 PRECISION AUDIT — manual vs auto bounty awards (before any scoring).

SE bounty mechanics: the bounty-setter can AWARD manually (full amount, a
deliberate curation act) OR the system AUTO-AWARDS at expiry (half the amount,
to the highest-scored eligible answer).  Auto-awards are NOT curation — they are
a deterministic function of the community vote signal, and would contaminate the
curated y with the community y.

Classification per (question, close) pair, joining VoteTypeId=8 (start: question
PostId, UserId, amount, date) to VoteTypeId=9 (close: answer PostId, amount?, date)
through the population's answer->question map:
  MANUAL  close amount == a start amount on that question
  AUTO    close amount == floor(start/2) (system half-award)
  NOAMT   close row carries no BountyAmount (classify separately; count)
  OTHER   amount matches neither (multi-bounty ambiguity etc.)
Also: winner-is-top-scored fraction per class (the community-mixing check), and
the close-minus-start day gap distribution per class (auto should sit at ~7d).
"""
import gzip
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

D = Path("/lfs/skampere3/0/alexspan/data/se_dumps")
ATTR = re.compile(r'(\w+)="([^"]*)"')

starts = defaultdict(list)   # qid -> [(amount, date, userid)]
closes = []                  # (aid, amount_or_None, date)
print("[audit] scanning Votes.xml types 8/9 ...", flush=True)
with open(D / "mathse_extracted" / "Votes.xml", encoding="utf-8", errors="ignore") as fh:
    for line in fh:
        if 'VoteTypeId="8"' in line:
            a = dict(ATTR.findall(line))
            starts[a["PostId"]].append((int(a.get("BountyAmount", 0)),
                                        a.get("CreationDate", ""), a.get("UserId", "")))
        elif 'VoteTypeId="9"' in line:
            a = dict(ATTR.findall(line))
            amt = a.get("BountyAmount")
            closes.append((a["PostId"], int(amt) if amt else None, a.get("CreationDate", "")))

a2q, top_by_q, score_of = {}, {}, {}
for line in gzip.open(D / "mathse_bounty_population.jsonl.gz", "rt"):
    r = json.loads(line)
    a2q[r["aid"]] = r["qid"]
    score_of[r["aid"]] = r["answer_score"]
    cur = top_by_q.get(r["qid"])
    if cur is None or r["answer_score"] > cur[1]:
        top_by_q[r["qid"]] = (r["aid"], r["answer_score"])

def days(d1, d0):
    try:
        return (datetime.fromisoformat(d1[:19]) - datetime.fromisoformat(d0[:19])).days
    except Exception:
        return None

cls_count, cls_top, cls_gap = Counter(), Counter(), defaultdict(Counter)
per_aid_class = {}
for aid, amt, cdate in closes:
    qid = a2q.get(aid)
    if qid is None:
        continue                      # not in the contrastable population
    st = starts.get(qid, [])
    if amt is None:
        cls = "NOAMT"
        gap = None
    else:
        full = [s for s in st if s[0] == amt]
        half = [s for s in st if s[0] // 2 == amt and s[0] != amt]
        if full:
            cls, ref = "MANUAL", full[0]
        elif half:
            cls, ref = "AUTO_HALF", half[0]
        else:
            cls, ref = "OTHER", (st[0] if st else None)
        gap = days(cdate, ref[1]) if ref else None
    cls_count[cls] += 1
    per_aid_class[aid] = cls
    if gap is not None:
        cls_gap[cls][min(max(gap, 0), 14)] += 1
    if top_by_q.get(qid, (None,))[0] == aid:
        cls_top[cls] += 1

out = {"n_closes_in_population": sum(cls_count.values()),
       "class_counts": dict(cls_count),
       "winner_is_top_scored_frac_by_class":
           {c: round(cls_top[c] / n, 4) for c, n in cls_count.items()},
       "close_minus_start_days_hist_by_class":
           {c: dict(sorted(h.items())) for c, h in cls_gap.items()}}
Path(D / "mathse_bounty_award_mode_audit.json").write_text(json.dumps(out, indent=1))
with gzip.open(D / "mathse_bounty_award_class.jsonl.gz", "wt") as fh:
    for aid, c in per_aid_class.items():
        fh.write(json.dumps({"aid": aid, "award_class": c}) + "\n")
print(json.dumps(out, indent=1)[:2000], flush=True)
print("AWARD_MODE_AUDIT_DONE", flush=True)
