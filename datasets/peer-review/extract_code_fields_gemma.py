#!/usr/bin/env python3
"""Gemma-4 extraction pass for the 5 peer-review code-metric aspects.

For each abstract x aspect, prompt Gemma-4 to return JSON of that aspect's LLM_FIELDS
(loaded directly from the metric_seam modules — single source of truth). Output jsonl:
{id: {aspect: {field: value}}}. The downstream scorer computes score(text, extracted, ops).

Usage (gemma4 env, 1 GPU):
  CUDA_VISIBLE_DEVICES=2 python extract_code_fields_gemma.py --sample 2400 \
      --venue-substr ICLR --source peerread_openreview --split train
"""
import argparse, csv, gzip, importlib.util, json, os, pathlib, random, re, sys
import numpy as np
from vllm import LLM, SamplingParams

BASE = pathlib.Path("/lfs/skampere3/0/alexspan/norm-research")
PROG = BASE / "methods/metric_seam/hybrids/programs_peer_review"
GEMMA4 = "/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/3548789868c5356dbf307c98e6f609007b82b3eb"

ASPECT_LABELS = {
    "a163": "positioning of the paper relative to prior work and baselines",
    "a130": "novelty and significance of the contribution",
    "a214": "reproducibility and data/code transparency",
    "a25":  "alignment between claims and their supporting evidence",
    "a45":  "dataset provenance and documentation",
}

def load_module(aid):
    spec = importlib.util.spec_from_file_location(f"pr_{aid}", PROG / f"{aid}_h0.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

SYS = ("You extract structured fields from a research-paper ABSTRACT. For each question, "
       "give the shortest faithful answer, or NONE if the abstract gives no evidence. "
       "Reply with ONE compact JSON object mapping each field name to its value. No prose.")

def build_prompt(text, aid, mod):
    fields = mod.LLM_FIELDS
    qs = "\n".join(f'"{k}": {v}' for k, v in fields.items())
    return (f"ABSTRACT:\n{text[:5000]}\n\n"
            f"We are assessing {ASPECT_LABELS[aid]}. Extract these fields:\n{qs}\n\n"
            f'Reply with JSON only, e.g. {{"field1": "...", "field2": "NONE"}}.')

_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
def parse_json(raw):
    if not raw:
        return {}
    m = _OBJ_RE.search(raw)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        # last-resort: strip trailing commas
        try:
            return json.loads(re.sub(r",\s*}", "}", re.sub(r",\s*]", "]", m.group(0))))
        except Exception:
            return {}

def load_rows(split, sample, venue_substr, source):
    p = BASE / "datasets/peer-review/splits" / f"{split}.csv.gz"
    rows = []
    with gzip.open(p, "rt", errors="ignore") as fh:
        for d in csv.DictReader(fh):
            txt = (d.get("text") or "").strip()
            try: y = int(d.get("judgement"))
            except: continue
            if y not in (0, 1) or len(txt) < 50: continue
            if venue_substr and venue_substr.lower() not in (d.get("venue") or "").lower(): continue
            if source and source != d.get("source"): continue
            rows.append((d.get("paper_id") or d.get("id"), txt, y))
    if sample and len(rows) > sample:
        random.seed(0)
        pos = [r for r in rows if r[2] == 1]; neg = [r for r in rows if r[2] == 0]
        k = sample // 2
        rows = random.sample(pos, min(k, len(pos))) + random.sample(neg, min(k, len(neg)))
        random.shuffle(rows)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--sample", type=int, default=2400)
    ap.add_argument("--venue-substr", default="ICLR")
    ap.add_argument("--source", default="peerread_openreview")
    ap.add_argument("--util", type=float, default=0.85)
    ap.add_argument("--out", default=str(BASE / "datasets/peer-review/peer_review_fields.jsonl"))
    a = ap.parse_args()

    aids = ["a163", "a130", "a214", "a25", "a45"]
    mods = {aid: load_module(aid) for aid in aids}
    rows = load_rows(a.split, a.sample, a.venue_substr, a.source)
    print(f"[fields] {len(rows)} abstracts x {len(aids)} aspects = {len(rows)*len(aids)} prompts", flush=True)

    llm = LLM(model=GEMMA4, dtype="bfloat16", gpu_memory_utilization=a.util,
              max_model_len=4096, enable_prefix_caching=True, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=140)

    convs, key = [], []
    for pid, txt, _ in rows:
        for aid in aids:
            convs.append([{"role": "user", "content": f"{SYS}\n\n{build_prompt(txt, aid, mods[aid])}"}])
            key.append((pid, aid))
    print(f"[fields] extracting {len(convs)} on Gemma-4-31B ...", flush=True)
    outs = llm.chat(convs, sp)

    by_id = {}
    for (pid, aid), o in zip(key, outs):
        obj = parse_json(o.outputs[0].text)
        # normalize: ensure every declared field exists (default NONE)
        for f in mods[aid].LLM_FIELDS:
            obj.setdefault(f, "NONE")
        by_id.setdefault(pid, {})[aid] = obj

    with open(a.out, "w") as fh:
        for pid, txt, y in rows:
            fh.write(json.dumps({"id": pid, "y": y, "text": txt, "fields": by_id.get(pid, {})}) + "\n")
    print(f"[fields] saved -> {a.out}  ({len(by_id)} ids)", flush=True)
    print("FIELDS_DONE", flush=True)

if __name__ == "__main__":
    main()
