#!/usr/bin/env python3
"""U3 — mathse_bounty CURATED cell build (population.csv.gz + dense_standard).

INSTRUMENT IDENTITY with the sibling math.SE cells (accepted verdict / vote
community) is the point of the unified-X program, so nothing is reimplemented:
  * text prep    = build_multiy_v2.clean_body (imported) + the same
                   "QUESTION: {title}\\n\\nANSWER:\\n{body}" template + the same
                   MIN 50 / MAX 12,000 char gates
  * row identity = sha1(f"{{qid}}|{{aid}}")[:20], the sibling row_id convention
  * splits       = build_v2_va_population.stable_hash_bucket_map (imported),
                   80/10/10 by question group, pos-rate balanced
y = MANUAL bounty award (the award-mode audit's filter is upstream in
mathse_bounty_manual_population.jsonl.gz).  After char gates, questions must
re-satisfy >=2 answers AND exactly->=1 winner or they drop (gates recounted).
"""
import gzip
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

NR = Path("/lfs/skampere3/0/alexspan/norm-research")
MSE = NR / "datasets/math-stackexchange"
D = Path("/lfs/skampere3/0/alexspan/data/se_dumps")
OUT = NR / "datasets/math-se/mathse_bounty"
OUT.mkdir(parents=True, exist_ok=True)


def _mod(path, alias):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m
    spec.loader.exec_module(m)
    return m


MV2 = _mod(MSE / "build_multiy_v2.py", "mb_mv2")
V2P = _mod(MSE / "build_v2_va_population.py", "mb_v2p")

rows = [json.loads(l) for l in gzip.open(D / "mathse_bounty_manual_population.jsonl.gz", "rt")]
prepped, n_gate = [], 0
for r in rows:
    body = MV2.clean_body(r["answer_body"])
    if not (MV2.MIN_CHARS <= len(body) <= MV2.MAX_CHARS):
        n_gate += 1
        continue
    prepped.append({
        "row_id": hashlib.sha1(f"{r['qid']}|{r['aid']}".encode()).hexdigest()[:20],
        "question_id": r["qid"], "answer_id": r["aid"], "group": r["qid"],
        "text": f"QUESTION: {r['q_title']}\n\nANSWER:\n{body}",
        "judgement": int(r["y"]), "answer_score": r["answer_score"],
        "n_answers_on_q": r["n_answers_on_q"],
    })
df = pd.DataFrame(prepped)
per_q = Counter(df.question_id)
winners = df[df.judgement == 1].groupby("question_id").size()
keep = {q for q, n in per_q.items() if n >= 2 and q in winners.index}
df = df[df.question_id.isin(keep)].reset_index(drop=True)
print(f"char-gate dropped {n_gate}; final rows {len(df)} questions {df.question_id.nunique()} "
      f"pos {df.judgement.mean():.4f}")

ybg = {g: d.judgement.tolist() for g, d in df.groupby("group")}
bmap = V2P.stable_hash_bucket_map(ybg)
df["split"] = df.group.map(bmap)
for s in ("train", "eval", "test"):
    sub = df[df.split == s]
    print(f"  {s}: n={len(sub)} pos={sub.judgement.mean():.4f} groups={sub.group.nunique()}")

df.to_csv(OUT / "population.csv.gz", index=False, compression="gzip")
DS = OUT / "dense_standard_mathse_bounty"
(DS / "split").mkdir(parents=True, exist_ok=True)
cols = ["text", "judgement", "group", "row_id"]
df[cols].to_csv(DS / "data.csv", index=False)
for s in ("train", "eval", "test"):
    df[df.split == s][cols].to_csv(DS / "split" / f"{s}.csv", index=False)
man = {
    "cell": "mathse_bounty (CURATED: manual bounty award, within-question)",
    "y_definition": ("1 = this answer received a MANUAL bounty award (deliberate act "
                     "by the bounty-setter; AUTO half-awards at expiry excluded "
                     "upstream, see mathse_bounty_award_mode_audit.json); 0 = other "
                     "answer on the same bountied question; questions need >=2 "
                     "gated answers and >=1 winner"),
    "x_convention": "build_multiy_v2.clean_body + 'QUESTION:/ANSWER:' template, "
                    "IDENTICAL to the accepted/vote sibling cells (unified X)",
    "n": int(len(df)), "n_questions": int(df.question_id.nunique()),
    "pos_rate": float(df.judgement.mean()),
    "splits": {s: int((df.split == s).sum()) for s in ("train", "eval", "test")},
    "char_gate_dropped": n_gate,
}
(OUT / "manifest.json").write_text(json.dumps(man, indent=1))
print("MATHSE_BOUNTY_CELL_DONE")
