#!/usr/bin/env python3
"""Abstract-extraction on the SAME 2400 paper_ids as the full-paper evidence jsonl,
so the abstract-vs-fullpaper comparison is on matched samples (no population confound).
Reads peer_review_fullpaper_evidence.jsonl, uses the 'abstract' field for every aspect."""
import argparse, importlib.util, json, pathlib, re
from vllm import LLM, SamplingParams

BASE = pathlib.Path("/lfs/skampere3/0/alexspan/norm-research")
PROG = BASE / "methods/metric_seam/hybrids/programs_peer_review"
GEMMA4 = "/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/3548789868c5356dbf307c98e6f609007b82b3eb"
ASPECT_LABELS = {"a163":"positioning relative to prior work and baselines","a130":"novelty and significance",
                 "a214":"reproducibility and data/code transparency","a25":"claim-evidence alignment","a45":"dataset provenance"}

def load_module(aid):
    spec = importlib.util.spec_from_file_location(f"pr_{aid}", PROG/f"{aid}_h0.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

SYS = ("You extract structured fields from a research-paper ABSTRACT. For each question, give the "
       "shortest faithful answer, or NONE if the abstract gives no evidence. Reply with ONE compact JSON.")
_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
def parse_json(raw):
    if not raw: return {}
    m=_OBJ_RE.search(raw)
    if not m: return {}
    try: return json.loads(m.group(0))
    except:
        try: return json.loads(re.sub(r",\s*}", "}", re.sub(r",\s*]", "]", m.group(0))))
        except: return {}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--evidence",default="peer_review_fullpaper_evidence.jsonl")
    ap.add_argument("--util",type=float,default=0.85)
    ap.add_argument("--out",default=str(BASE/"datasets/peer-review/peer_review_fields_abstract_matched.jsonl"))
    a=ap.parse_args()
    aids=["a163","a130","a214","a25","a45"]; mods={aid:load_module(aid) for aid in aids}
    rows=[json.loads(l) for l in open(a.evidence) if l.strip()]
    print(f"[abs-match] {len(rows)} papers x {len(aids)} aspects (ABSTRACT, matched)",flush=True)
    llm=LLM(model=GEMMA4,dtype="bfloat16",gpu_memory_utilization=a.util,max_model_len=4096,
            enable_prefix_caching=True,trust_remote_code=True)
    sp=SamplingParams(temperature=0.0,max_tokens=140)
    convs,key=[],[]
    for r in rows:
        ab=r.get("abstract","")
        for aid in aids:
            qs="\n".join(f'"{k}": {v}' for k,v in mods[aid].LLM_FIELDS.items())
            convs.append([{"role":"user","content":f"{SYS}\n\nABSTRACT:\n{ab[:5000]}\n\nAssessing {ASPECT_LABELS[aid]}. Extract:\n{qs}\n\nReply JSON only."}])
            key.append((r["paper_id"],aid))
    print(f"[abs-match] extracting {len(convs)} ...",flush=True)
    outs=llm.chat(convs,sp)
    by_id={}
    for (pid,aid),o in zip(key,outs):
        obj=parse_json(o.outputs[0].text)
        for f in mods[aid].LLM_FIELDS: obj.setdefault(f,"NONE")
        by_id.setdefault(pid,{})[aid]=obj
    with open(a.out,"w") as fh:
        for r in rows:
            fh.write(json.dumps({"id":r["paper_id"],"y":r["y"],"text":r.get("abstract",""),"fields":by_id.get(r["paper_id"],{})})+"\n")
    print(f"[abs-match] saved -> {a.out}",flush=True); print("ABS_MATCH_DONE",flush=True)

if __name__=="__main__": main()
