#!/usr/bin/env python3
"""Patents REBUILD, phase 1 (user request 2026-08-13: "get me a trustworthy VAT
number, preferably V3"): claim-text-ONLY construct.

CONSTRUCT RENAME (licensed by the post-mortem's placebo audit): y = the examiner
REJECTED this claim element (any statutory ground) — a decision-maker verdict on
the claim's own text. The eight candidate references are DROPPED from every input:
that removes the positives-carry-the-gold-reference construction asymmetry and
makes the placebo criticism moot (no reference-reading is claimed).

Arms (both reuse dense_standard's exact rows and app_id-grouped splits — same-rows
with the recorded V .5925/.6265 and VA .6214/.6434 ladder):
  arm_t    claim element text only                     -> honest T for THIS construct
  arm_a    "VA metrics:" block (V_claim + STRUCT, incl. claim ordinal as a DECLARED
           feature per RUNBOOK revival condition 4) + claim text -> V3 fused arm,
           max-of-variants VAT column only
The judged-A bank (revival condition 3, from online-rubrics) upgrades arm_a later;
until then the articulated block is V_claim + STRUCT and is labeled as such.

rejection_type is the label sidecar (alone-AUC .988) — it is READ here only to be
EXCLUDED from outputs; it never enters text, block, or data.csv. Kept only in a
separate harvest-side strata file for the §102/§103 replicate readout.

Run on sk3:  envs/ai_usage python build_patents_v3_claimonly.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

NR = Path("/lfs/skampere3/0/alexspan/norm-research")
SRC = NR / "datasets/patents/dense_standard/split"
OUT = NR / "datasets/patents/v3_claimonly"

REF_MARK = re.compile(r"\n\s*REFERENCE 1 \(", re.S)
DEP = re.compile(r"\bof claim (\d+)\b", re.I)


def claim_only(t):
    m = REF_MARK.search(t)
    body = t[:m.start()] if m else t
    return body.replace("CLAIM ELEMENT:", "").strip()


def feats(el, claim_num):
    """BLOCK = text-derived content features ONLY (user ruling 2026-08-13 + the
    RUNBOOK prereg: claim ordinal position is a DECLARED TRACK-B NUISANCE, handled
    by the deconfounding machinery at harvest — quote Δ over V+A+STRUCT — and own
    claim_num is metadata invisible in the text, so feeding it to the fused arm
    would inject the nuisance channel into the VAT bar). The dependency flag stays:
    it is computed from a substring the model already sees."""
    words = el.split()
    dep = DEP.search(el)
    return {
        "is dependent claim": 1.0 if dep else 0.0,
        "character length": float(len(el)),
        "word count": float(len(words)),
        "mean word length": float(np.mean([len(w) for w in words])) if words else 0.0,
        "comma count": float(el.count(",")),
        "semicolon count": float(el.count(";")),
        "wherein-clause count": float(len(re.findall(r"\bwherein\b", el, re.I))),
        "numeric token count": float(len(re.findall(r"\d+(?:\.\d+)?", el))),
    }


def struct_feats(el, claim_num):
    """The declared Track-B nuisance block for the harvest-side deconfounding
    (F2-style stacked arms): own ordinal (metadata), parent-claim number, plus the
    same text-derived structure. NEVER enters any model input."""
    dep = DEP.search(el)
    return {
        "claim_num": float(claim_num) if claim_num == claim_num else np.nan,
        "parent_claim_num": float(dep.group(1)) if dep else 0.0,
        "is_dependent": 1.0 if dep else 0.0,
        "char_len": float(len(el)),
        "word_len": float(len(el.split())),
    }


def fmt(v):
    if v != v:
        return "NA"
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return f"{v:.3f}".rstrip("0").rstrip(".")


(OUT / "arm_t" / "split").mkdir(parents=True, exist_ok=True)
(OUT / "arm_a" / "split").mkdir(parents=True, exist_ok=True)
man = {"design_id": "patents_v3_claimonly", "construct":
       "examiner rejected this claim element (any ground) — verdict on claim text; "
       "references DROPPED (construction asymmetry removed; placebo moot)",
       "estimand_arm_a": "V3 fused arm (text-derived V_claim block + text), "
                         "max-of-variants VAT column only; claim ordinal/parent-num = "
                         "TRACK-B NUISANCE handled at harvest by the F2 machinery "
                         "(quote Δ over V+A+STRUCT); judged-A bank pending (cond. 3)",
       "estimand_arm_t": "honest T for the claim-only construct",
       "rows_identical_to": "dense_standard splits (app_id-grouped)", "splits": {}}

strata_rows = []
for nm in ("train", "eval", "test"):
    d = pd.read_csv(SRC / f"{nm}.csv")
    el = d.text.map(claim_only)
    F = pd.DataFrame([feats(e, c) for e, c in zip(el, d.claim_num)])
    block = ["VA metrics:\n" + "\n".join(f"    {k}: {fmt(v)}" for k, v in row.items())
             for row in F.to_dict("records")]
    base = pd.DataFrame({"judgement": d.judgement.astype(int),
                         "group": d.group.astype(str),
                         "row_id": [f"{nm}_{i}" for i in range(len(d))]})
    ta = base.assign(text=[b + "\n\nCLAIM ELEMENT:\n" + e for b, e in zip(block, el)])
    tt = base.assign(text=["CLAIM ELEMENT:\n" + e for e in el])
    ta.to_csv(OUT / "arm_a" / "split" / f"{nm}.csv", index=False)
    tt.to_csv(OUT / "arm_t" / "split" / f"{nm}.csv", index=False)
    S = pd.DataFrame([struct_feats(e, c) for e, c in zip(el, d.claim_num)])
    strata_rows.append(pd.concat([pd.DataFrame({"row_id": base.row_id,
                                                "rejection_type": d.rejection_type,
                                                "split": nm,
                                                "group": base.group,
                                                "judgement": base.judgement}),
                                  S], axis=1))
    man["splits"][nm] = {"n": int(len(d)), "pos_rate": float(d.judgement.mean()),
                         "median_claim_chars": int(el.str.len().median()),
                         "p99_claim_chars": int(el.str.len().quantile(.99))}
    print(nm, man["splits"][nm], flush=True)

for arm in ("arm_a", "arm_t"):
    parts = [pd.read_csv(OUT / arm / "split" / f"{nm}.csv") for nm in ("train", "eval", "test")]
    pd.concat(parts).to_csv(OUT / arm / "data.csv", index=False)

pd.concat(strata_rows).to_csv(OUT / "harvest_strata_NEVER_AN_INPUT.csv", index=False)
(OUT / "manifest.json").write_text(json.dumps(man, indent=1))
print("BUILD_DONE", OUT, flush=True)
