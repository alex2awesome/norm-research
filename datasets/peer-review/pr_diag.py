#!/usr/bin/env python3
"""Diagnostic: print RAW Gemma-4 outputs for a few (abstract x rubric) pairs to
diagnose the high NA rate — is it genuine 'no evidence' or a formatting/parse failure?"""
import csv, gzip, json, re
from pathlib import Path
import numpy as np
from vllm import LLM, SamplingParams
from collections import Counter

BASE = Path("/lfs/skampere3/0/alexspan/norm-research/datasets/peer-review")
RUBRICS = Path("/lfs/skampere3/0/alexspan/data/peer_review/rubrics.jsonl")
GEMMA4 = "/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/3548789868c5356dbf307c98e6f609007b82b3eb"

SYS = ("You are an expert academic peer reviewer. You are given a paper's ABSTRACT and ONE "
       "quality criterion. Decide how strongly the abstract, on its own evidence, satisfies "
       "that criterion. Answer with EXACTLY ONE token:\n"
       "  1.0 = clearly satisfies the criterion\n  0.5 = partially / weakly / borderline\n"
       "  0.0 = fails / cuts against the criterion\n"
       "  NA = the abstract gives no evidence bearing on this criterion\n"
       "Judge the paper's quality, not whether it will be accepted. Output only the token.")

rubrics = [json.loads(l) for l in open(RUBRICS) if l.strip()][:20]
rows = []
with gzip.open(BASE/"splits/eval.csv.gz","rt",errors="ignore") as fh:
    for d in csv.DictReader(fh):
        if len((d.get("text") or ""))>200:
            rows.append(d["text"]);
        if len(rows)>=5: break

def block(m): return f"CRITERION: {m['name']}\nDESCRIPTION: {m.get('description','')}\n\nAnswer with one token:"

llm = LLM(model=GEMMA4, dtype="bfloat16", gpu_memory_utilization=0.85,
          max_model_len=4096, enable_prefix_caching=True, trust_remote_code=True)
# Try TWO settings: max_tokens=6 (current) and max_tokens=64 (verbose allowed)
for mt in (6, 64):
    convs=[]
    for txt in rows:
        for m in rubrics:
            convs.append([{"role":"user","content":f"{SYS}\n\nABSTRACT:\n{txt[:5000]}\n\n{block(m)}"}])
    outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=mt))
    texts = [o.outputs[0].text.strip() for o in outs]
    print(f"\n===== max_tokens={mt}: first-token histogram =====")
    firsts = Counter((t[:8].replace("\n"," ")) for t in texts)
    for k,v in firsts.most_common(15): print(f"  {v:3d}  {k!r}")
    print(f"===== max_tokens={mt}: 12 sample raw outputs =====")
    for t in texts[:12]: print("   ", repr(t[:80]))
print("DIAG_DONE")
