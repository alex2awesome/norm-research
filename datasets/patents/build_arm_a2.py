#!/usr/bin/env python3
"""Patents arm_a_v2 (2026-08-13): the V3 fused arm WITH the judged bank_v1 scores
in the prompt block — the user's directive ("make sure V3 sees VA+VA_new scores").

Block per claim = 8 content features + the 26 non-collapsed judged criteria
("<name>: <score|NA>", Gemma-4-31B, scored corpus-wide: train npz + eval/test npz).
Confound channels (claim ordinal etc.) stay OUT of every input per the standing
rule — they live in the harvest-side nuisance block only.

Run ON sk3:  envs/ai_usage python build_arm_a2.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

NR = Path("/lfs/skampere3/0/alexspan/norm-research")
D = NR / "datasets/patents/v3_claimonly"
SC = NR / "methods/taste_decomposition/closure/patents_claimonly"

bank = json.load(open(D / "bank_v1.json"))["bank"]
rep = json.load(open(SC / "patents_claimonly_r0_score_report.json"))
collapsed = {k for k, v in rep["per_criterion"].items() if v["collapsed"]}
live = [c for c in bank if c["id"] not in collapsed]
print(f"[bank] {len(live)} non-collapsed criteria in block", flush=True)

score_by_id = {}
for f in ("patents_claimonly_r0_scores.npz", "patents_claimonly_train_r0_scores.npz"):
    z = np.load(SC / f, allow_pickle=True)
    cid = [str(c) for c in z["crit_ids"]]
    col = {u: i for i, u in enumerate(cid)}
    for i, rid in enumerate([str(x) for x in z["row_id"]]):
        score_by_id[rid] = {u: z["X"][i, col[u]] for u in cid}
print(f"[scores] {len(score_by_id)} rows with judged scores", flush=True)

DEP = re.compile(r"\bof claim (\d+)\b", re.I)
def fmt(v):
    if v is None or v != v:
        return "NA"
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return f"{v:.3f}".rstrip("0").rstrip(".")

def content_lines(el):
    words = el.split()
    dep = DEP.search(el)
    feats = {"is dependent claim": 1.0 if dep else 0.0,
             "character length": float(len(el)), "word count": float(len(words)),
             "mean word length": float(np.mean([len(w) for w in words])) if words else 0.0,
             "comma count": float(el.count(",")), "semicolon count": float(el.count(";")),
             "wherein-clause count": float(len(re.findall(r"\bwherein\b", el, re.I))),
             "numeric token count": float(len(re.findall(r"\d+(?:\.\d+)?", el)))}
    return [f"    {k}: {fmt(v)}" for k, v in feats.items()]

(D / "arm_a2" / "split").mkdir(parents=True, exist_ok=True)
parts = []
for nm in ("train", "eval", "test"):
    d = pd.read_csv(D / f"arm_t/split/{nm}.csv")
    el = d.text.str.replace("CLAIM ELEMENT:\n", "", regex=False)
    texts = []
    n_missing = 0
    for e, rid in zip(el, d.row_id):
        lines = ["VA metrics:"] + content_lines(e)
        sc = score_by_id.get(str(rid))
        if sc is None:
            n_missing += 1
        for c in live:
            v = sc.get(c["id"]) if sc else float("nan")
            lines.append(f"    {c['name']}: {fmt(v)}")
        texts.append("\n".join(lines) + "\n\nCLAIM ELEMENT:\n" + e)
    out = pd.DataFrame({"text": texts, "judgement": d.judgement.astype(int),
                        "group": d.group.astype(str), "row_id": d.row_id})
    out.to_csv(D / "arm_a2" / "split" / f"{nm}.csv", index=False)
    parts.append(out)
    print(f"[{nm}] {len(out)} rows, {n_missing} without judged scores", flush=True)
pd.concat(parts).to_csv(D / "arm_a2" / "data.csv", index=False)
print("ARM_A2_BUILD_DONE", flush=True)
