#!/usr/bin/env python3
"""Full-paper version: extract LLM_FIELDS from SECTION-TARGETED evidence (not abstract).

Same as extract_code_fields_gemma.py but each (paper,aspect) prompt uses the per-aspect
section-targeted evidence from peer_review_fullpaper_evidence.jsonl (related_work+experiments
for a163, repro windows for a214, etc.) instead of the bare abstract. Isolates the input lever.

Usage (gemma4 env, 1 GPU):
  CUDA_VISIBLE_DEVICES=2 python extract_code_fields_fullpaper.py \
      --evidence peer_review_fullpaper_evidence.jsonl --out peer_review_fields_fullpaper.jsonl
"""
import argparse, importlib.util, json, pathlib, re
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

SYS = ("You extract structured fields from a research PAPER (sections provided). For each question, "
       "give the shortest faithful answer from the provided text, or NONE if the text gives no evidence. "
       "Reply with ONE compact JSON object mapping each field name to its value. No prose.")

def build_prompt(evidence, aid, mod):
    fields = mod.LLM_FIELDS
    qs = "\n".join(f'"{k}": {v}' for k, v in fields.items())
    return (f"PAPER TEXT:\n{evidence[:5000]}\n\n"
            f"We are assessing {ASPECT_LABELS[aid]}. Extract these fields:\n{qs}\n\n"
            f'Reply with JSON only, e.g. {{"field1": "...", "field2": "NONE"}}.')

_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
def parse_json(raw):
    if not raw: return {}
    m = _OBJ_RE.search(raw)
    if not m: return {}
    try: return json.loads(m.group(0))
    except Exception:
        try: return json.loads(re.sub(r",\s*}", "}", re.sub(r",\s*]", "]", m.group(0))))
        except Exception: return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default="peer_review_fullpaper_evidence.jsonl")
    ap.add_argument("--util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--out", default=str(BASE / "datasets/peer-review/peer_review_fields_fullpaper.jsonl"))
    a = ap.parse_args()

    aids = ["a163", "a130", "a214", "a25", "a45"]
    mods = {aid: load_module(aid) for aid in aids}
    rows = [json.loads(l) for l in open(a.evidence) if l.strip()]
    print(f"[fp-fields] {len(rows)} papers x {len(aids)} aspects = {len(rows)*len(aids)} prompts", flush=True)

    llm = LLM(model=GEMMA4, dtype="bfloat16", gpu_memory_utilization=a.util,
              max_model_len=a.max_model_len, enable_prefix_caching=True, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=160)

    convs, key = [], []
    for r in rows:
        for aid in aids:
            ev = (r.get("ev") or {}).get(aid) or r.get("abstract") or ""
            convs.append([{"role": "user", "content": f"{SYS}\n\n{build_prompt(ev, aid, mods[aid])}"}])
            key.append((r["paper_id"], aid))
    print(f"[fp-fields] extracting {len(convs)} on Gemma-4-31B ...", flush=True)
    outs = llm.chat(convs, sp)

    by_id = {}
    for (pid, aid), o in zip(key, outs):
        obj = parse_json(o.outputs[0].text)
        for f in mods[aid].LLM_FIELDS:
            obj.setdefault(f, "NONE")
        by_id.setdefault(pid, {})[aid] = obj

    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps({"id": r["paper_id"], "y": r["y"],
                                 "text": r.get("abstract", ""), "fields": by_id.get(r["paper_id"], {})}) + "\n")
    print(f"[fp-fields] saved -> {a.out} ({len(by_id)} ids)", flush=True)
    print("FP_FIELDS_DONE", flush=True)

if __name__ == "__main__":
    main()
