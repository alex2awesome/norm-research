#!/usr/bin/env python3
"""U4a — so_bounty CURATED cell build (population.csv.gz + dense_standard).

INSTRUMENT IDENTITY with the V6 so_votes community cell (same python corpus):
  * text prep    = V6's convention verbatim: body = Body.astype(str).str.strip(),
                   MIN_CHARS 50 gate, text = "QUESTION: {title}\\n\\nANSWER:\\n{body}"
  * row identity = answer Id string (V6's row_id convention)
  * splits       = the same frozen stable_hash_bucket_map (imported from the
                   math builder, itself verbatim from hashtagwars), 80/10/10 by
                   question group
y = MANUAL bounty award (award-mode filter upstream in
so_bounty_manual_population.jsonl.gz).  MAX char gate follows the math sibling
(12,000) — V6 has no max, recorded as the one deviation (bounty answers can be
long; unbounded rows break the dense token budget identically across cells).
"""
import gzip
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

NR = Path("/lfs/skampere3/0/alexspan/norm-research")
D = Path("/lfs/skampere3/0/alexspan/data/se_dumps")
OUT = NR / "datasets/stackoverflow-votes/so_bounty"
OUT.mkdir(parents=True, exist_ok=True)
MIN_CHARS, MAX_CHARS = 50, 12000


def _mod(path, alias):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m
    spec.loader.exec_module(m)
    return m


V2P = _mod(NR / "datasets/math-stackexchange/build_v2_va_population.py", "sb_v2p")

rows = [json.loads(l) for l in gzip.open(D / "so_bounty_manual_population.jsonl.gz", "rt")]
prepped, n_gate = [], 0
for r in rows:
    body = str(r["answer_body"]).strip()
    if not (MIN_CHARS <= len(body) <= MAX_CHARS):
        n_gate += 1
        continue
    prepped.append({
        "row_id": str(r["aid"]), "question_id": r["qid"], "group": r["qid"],
        "text": f"QUESTION: {str(r['q_title']).strip()}\n\nANSWER:\n{body}",
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
DS = OUT / "dense_standard_so_bounty"
(DS / "split").mkdir(parents=True, exist_ok=True)
cols = ["text", "judgement", "group", "row_id"]
df[cols].to_csv(DS / "data.csv", index=False)
for s in ("train", "eval", "test"):
    df[df.split == s][cols].to_csv(DS / "split" / f"{s}.csv", index=False)
man = {
    "cell": "so_bounty (CURATED: manual bounty award, within-question, python corpus)",
    "y_definition": ("1 = MANUAL bounty award (auto half-awards excluded upstream, "
                     "so_bounty_population_audit.json); 0 = other answer on the same "
                     "bountied question; >=2 gated answers and >=1 winner per question"),
    "x_convention": ("V6 so_votes convention verbatim (body.strip, QUESTION:/ANSWER: "
                     "template, answer-Id row ids); DEVIATION: MAX 12,000-char gate "
                     "added (math-sibling parity; V6 had no max)"),
    "n": int(len(df)), "n_questions": int(df.question_id.nunique()),
    "pos_rate": float(df.judgement.mean()),
    "splits": {s: int((df.split == s).sum()) for s in ("train", "eval", "test")},
    "char_gate_dropped": n_gate,
}
(OUT / "manifest.json").write_text(json.dumps(man, indent=1))
print("SO_BOUNTY_CELL_DONE")
