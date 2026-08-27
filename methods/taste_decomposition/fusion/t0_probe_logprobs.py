#!/usr/bin/env python3
"""DIFFERENTIAL PROBE for the BBC T0 binary-collapse (2026-08-14).

BBC t0 p_yes came back one-hot ({0,1} exactly, 2 distinct over 10,147 rows) while
every original-battery cell is continuous (homepage 487 distinct).  Hypothesis: the
env/version scoring THIS run returns processed/masked logprobs (clamped -inf mass),
i.e. a different instrument from the one that scored the 16-cell battery.

Probe: rescore the FIRST 100 prompts of (a) homepage (known-continuous under the
original run) and (b) bbc, under the CURRENT env, with logprobs_mode passed
explicitly where supported.  If homepage reproduces ~continuous p_yes here, the
data explains BBC's collapse (genuine near-one-hot prior on short headlines); if
homepage is ALSO binary here, this env is a different instrument and the BBC scores
must be regenerated under the original semantics (raw logprobs).

Usage: python t0_probe_logprobs.py [--mode raw_logprobs|processed_logprobs|default]
"""
import argparse
import gzip
import json
import math
import os
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

HERE = Path(__file__).resolve().parent
T = json.loads((HERE / "t0_templates.json").read_text())
TPL = T["prompt_format"]["template"]

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="default")
ap.add_argument("--n", type=int, default=100)
a = ap.parse_args()

MODEL = ("/lfs/skampere3/0/alexspan/.cache/huggingface/hub/"
         "models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b")

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(MODEL)
variants = T["score"]["yes_no_variants"]
pos_ids, neg_ids = [], []
for s in variants["pos"]:
    ids = tok.encode(s, add_special_tokens=False)
    if len(ids) == 1:
        pos_ids.append(ids[0])
for s in variants["neg"]:
    ids = tok.encode(s, add_special_tokens=False)
    if len(ids) == 1:
        neg_ids.append(ids[0])
allowed = sorted(set(pos_ids) | set(neg_ids))

prompts, tags = [], []
for cell in ("homepage_curation_storygrouped", "bbc_mostread"):
    q = T["cells"][cell]["question"]
    k = 0
    for line in gzip.open(HERE / "t0_rows" / f"{cell}.texts.jsonl.gz", "rt"):
        r = json.loads(line)
        doc = tok.decode(tok.encode(r["text"], add_special_tokens=False)[:1024])
        prompts.append(TPL.format(question=q, document=doc))
        tags.append(cell)
        k += 1
        if k >= a.n:
            break

from vllm import LLM, SamplingParams  # noqa: E402
kw = dict(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.85,
          max_model_len=1280, tensor_parallel_size=1)
if a.mode != "default":
    kw["logprobs_mode"] = a.mode
llm = LLM(**kw)
sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=len(allowed),
                    allowed_token_ids=allowed)
outs = llm.generate(prompts, sp)

res = {}
for cell in ("homepage_curation_storygrouped", "bbc_mostread"):
    ps = []
    for tg, o in zip(tags, outs):
        if tg != cell:
            continue
        lp = o.outputs[0].logprobs[0]
        s_p = s_n = 0.0
        for tid, obj in lp.items():
            v = getattr(obj, "logprob", obj)
            if not math.isfinite(v):
                continue
            if int(tid) in pos_ids:
                s_p += math.exp(v)
            elif int(tid) in neg_ids:
                s_n += math.exp(v)
        ps.append(s_p / (s_p + s_n) if (s_p + s_n) > 0 else float("nan"))
    dis = len(set(round(p, 6) for p in ps))
    med = sorted(ps)[len(ps) // 2]
    res[cell] = {"n": len(ps), "n_distinct": dis, "median": med,
                 "min": min(ps), "max": max(ps)}
    print(f"[probe mode={a.mode}] {cell}: distinct={dis} median={med:.4f} "
          f"min={min(ps):.4f} max={max(ps):.4f}", flush=True)

out = HERE / f"t0_probe_{a.mode}.json"
out.write_text(json.dumps({"mode": a.mode, "results": res}, indent=1))
print("T0_PROBE_DONE", flush=True)
