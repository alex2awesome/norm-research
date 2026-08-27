#!/usr/bin/env python3
"""Patents phase 2, step 2: curate the claim-text criteria bank with GLM-5.2
(judge-family recorded; codex is quota-dead until Aug 18). LABEL-BLIND: prompts
carry criterion texts only — never the y channel, never example claims with labels.

Two stages:
  A. SWEEP: 10 random batches of 150 prefiltered criteria -> GLM extracts distinct
     CLAIM-TEXT-JUDGEABLE concepts (judgeable from ONE claim element's text alone —
     no prosecution history, no references, no drawings).
  B. MERGE: all sweep concepts -> GLM merges/dedupes into exactly 30 bank criteria,
     each with a name + a one-sentence 0-10 judging instruction (GEPA-style phrasing).
Output: v3_claimonly/bank_v1.json  (30 criteria, provenance glm-5.2).
"""
import json
import random
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
D = HERE / "v3_claimonly"
KEYS = [open(Path.home() / f).read().strip()
        for f in (".z-ai-api-key.txt", ".z-ai-api-key-spangher.txt")
        if (Path.home() / f).exists()]
URL = "https://api.z.ai/api/anthropic/v1/messages"
MODEL = "glm-5.2"

def ask(prompt, max_tokens=4000, tries=6):
    for i in range(tries):
        key = KEYS[i % len(KEYS)]
        try:
            r = requests.post(URL, timeout=180,
                              headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                       "content-type": "application/json"},
                              json={"model": MODEL, "max_tokens": max_tokens,
                                    "messages": [{"role": "user", "content": prompt}]})
            if r.status_code == 200:
                return "".join(b.get("text", "") for b in r.json().get("content", []))
        except requests.RequestException:
            pass
        time.sleep(8 * (i + 1))
    raise RuntimeError("GLM unreachable after retries")

def extract_json(text):
    s, e = text.find("["), text.rfind("]")
    if s < 0:
        s, e = text.find("{"), text.rfind("}")
    return json.loads(text[s:e + 1])

pool = json.load(open(D / "bank_prefilter.json"))
rng = random.Random(20260813)

SWEEP = """You are curating evaluation criteria for judging the DRAFTING QUALITY of a
single patent claim element from its text alone. Below are candidate criteria mined
from patent-law writing. Extract the DISTINCT concepts that are genuinely judgeable
from ONE claim element's text in isolation — no prosecution history, no cited
references, no drawings, no other claims. Discard procedural/legal-process criteria.
Return a JSON list of objects: {"concept": "<short name>", "what_it_measures": "<one sentence>"}.
Aim for quality over quantity; typically 10-25 concepts per batch.

CANDIDATES:
%s"""

concepts = []
state = D / "bank_sweep_state.json"
done = json.load(open(state)) if state.exists() else {"batches": 0, "concepts": []}
concepts = done["concepts"]
for b in range(done["batches"], 10):
    batch = rng.sample(pool, 150)
    txt = "\n".join(f"- {c['name']}: {str(c['description'])[:200]}" for c in batch)
    out = ask(SWEEP % txt)
    got = extract_json(out)
    concepts += got
    done = {"batches": b + 1, "concepts": concepts}
    json.dump(done, open(state, "w"))
    print(f"[sweep {b+1}/10] +{len(got)} concepts (total {len(concepts)})", flush=True)

MERGE = """Below are %d candidate concepts for judging the drafting quality of a single
patent claim element from its text alone. Merge duplicates and near-duplicates, drop
anything not judgeable from one claim element's isolated text, and produce EXACTLY 30
final bank criteria. Each must have:
  "name": a short distinctive title,
  "instruction": one or two sentences telling a judge how to score the claim element
                 0-10 on this criterion, judging the criterion as literally described
                 (not overall quality).
Cover diverse families (clarity/definiteness, scope/breadth, structure, support-style
language, precision of terms, functional vs structural language, formalism, etc.).
Return a JSON list of 30 objects.

CONCEPTS:
%s"""

txt = "\n".join(f"- {c.get('concept')}: {c.get('what_it_measures','')}" for c in concepts)
out = ask(MERGE % (len(concepts), txt), max_tokens=6000)
bank = extract_json(out)
assert 25 <= len(bank) <= 35, f"merge returned {len(bank)}"
for i, c in enumerate(bank):
    c["id"] = f"pb{i+1:02d}"
json.dump({"bank": bank, "provenance": "glm-5.2 curation over 25,051 prefiltered "
           "online-rubrics criteria (10x150 sweep + merge), label-blind, 2026-08-13",
           "n": len(bank)}, open(D / "bank_v1.json", "w"), indent=1)
print(f"BANK_V1_DONE {len(bank)} criteria", flush=True)
