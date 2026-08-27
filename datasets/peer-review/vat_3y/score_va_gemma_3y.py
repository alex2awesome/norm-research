#!/usr/bin/env python3
"""Academia 3-y VAT scorer (Gemma-4-31B). Scores union_toscore.jsonl abstracts x
154-rubric A-bank + programmatic V-features. LABEL-INDEPENDENT: y attached later per
rung (verdict/curation/revealed). Same bank/SYS/V-feats as score_va_gemma.py (verbatim)."""
import argparse, json, os, re
from pathlib import Path
import numpy as np
from vllm import LLM, SamplingParams

BASE=Path("/lfs/skampere3/0/alexspan/norm-research/datasets/peer-review")
RUBRICS=Path("/lfs/skampere3/0/alexspan/data/peer_review/rubrics.jsonl")
GEMMA4="/lfs/skampere3/0/shared_hf_cache/models--google--gemma-4-31b-it/snapshots/3548789868c5356dbf307c98e6f609007b82b3eb"
SYS=("You are an expert academic peer reviewer. You are given a paper's ABSTRACT and ONE "
     "quality criterion. Decide how strongly the abstract, on its own evidence, satisfies "
     "that criterion. Answer with EXACTLY ONE token:\n"
     "  1.0 = clearly satisfies the criterion\n  0.5 = partially / weakly / borderline\n"
     "  0.0 = fails / cuts against the criterion\n"
     "  NA = the abstract gives no evidence bearing on this criterion\n"
     "Judge the paper's quality, not whether it will be accepted. Output only the token.")
def load_rubrics():
    ms=[]
    for line in open(RUBRICS):
        line=line.strip()
        if line:
            r=json.loads(line)
            if r.get("name"): ms.append(r)
    return ms
def metric_block(m): return f"CRITERION: {m['name']}\nDESCRIPTION: {m.get('description','')}\n\nAnswer with one token:"
def parse_tok(t):
    t=(t or "").strip().lower()
    if t.startswith("na") or "n/a" in t or t=="na": return np.nan
    if "0.5" in t or t.startswith("0.5"): return 0.5
    if re.search(r"\b1(\.0)?\b",t) or t.startswith("1"): return 1.0
    if re.search(r"\b0(\.0)?\b",t) or t.startswith("0"): return 0.0
    return np.nan
NUMTOK=re.compile(r"\b\d[\d,\.]*\b"); SENT_RE=re.compile(r"[.!?]+")
KW={"v_kw_baseline":re.compile(r"baseline|state[- ]of[- ]the[- ]art|\bsota\b|outperform|benchmark|compared? with|compared? to",re.I),
"v_kw_ablation":re.compile(r"ablation|ablate",re.I),"v_kw_dataset":re.compile(r"dataset|corpus|benchmark",re.I),
"v_kw_novel":re.compile(r"\bnovel\b|first\b|new\b|propose|introduce|present a",re.I),
"v_kw_theory":re.compile(r"theorem|proof|prove|\bbound\b|guarantee|convergence|optimal",re.I),
"v_kw_code":re.compile(r"github|open[- ]source|code (?:is )?(?:available|released)|release our",re.I),
"v_kw_hedge":re.compile(r"\bmay\b|\bmight\b|\bcould\b|suggest|potential|possibly|appears? to",re.I),
"v_kw_superlative":re.compile(r"\bbest\b|superior|significant|substantial|dramatic|remarkable|state[- ]of[- ]the[- ]art",re.I),
"v_kw_cite":re.compile(r"\bSection\b|\bFigure\b|\bTable\b|\bEq\.|\bAppendix\b",re.I)}
def v_features(text):
    t=text or ""; words=t.split(); nw=max(len(words),1)
    sents=[s for s in SENT_RE.split(t) if s.strip()]; ns=max(len(sents),1)
    feats={"v_char_len":float(len(t)),"v_word_len":float(nw),"v_sent_count":float(ns),
    "v_avg_word_len":float(sum(len(w) for w in words)/nw),"v_avg_sent_len":float(nw/ns),
    "v_num_density":float(100.0*len(NUMTOK.findall(t))/nw),"v_pct_count":float(t.count("%")),"v_question":float(t.count("?"))}
    for n,rgx in KW.items(): feats[n]=float(len(rgx.findall(t)))
    return feats
V_NAMES=["v_char_len","v_word_len","v_sent_count","v_avg_word_len","v_avg_sent_len","v_num_density","v_pct_count","v_question"]+list(KW.keys())
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default=str(BASE/"vat_3y/union_toscore.jsonl"))
    ap.add_argument("--util",type=float,default=0.85); ap.add_argument("--max-model-len",type=int,default=4096)
    ap.add_argument("--out",default=str(BASE/"vat_3y/union_scores.npz"))
    a=ap.parse_args()
    rows=[json.loads(l) for l in open(a.input) if l.strip()]
    metrics=load_rubrics(); blocks=[metric_block(m) for m in metrics]; a_names=[m["name"] for m in metrics]
    nt=np.array([r["ntitle"] for r in rows],dtype=object)
    Vf=np.array([[v_features(r["text"])[n] for n in V_NAMES] for r in rows],dtype=float)
    print(f"[acad3y] {len(rows)} abstracts x {len(metrics)} rubrics = {len(rows)*len(metrics)} prompts",flush=True)
    llm=LLM(model=GEMMA4,dtype="bfloat16",gpu_memory_utilization=a.util,max_model_len=a.max_model_len,enable_prefix_caching=True,trust_remote_code=True)
    sp=SamplingParams(temperature=0.0,max_tokens=6)
    convs=[]
    for r in rows:
        f=r["text"][:5000]
        for b in blocks: convs.append([{"role":"user","content":f"{SYS}\n\nABSTRACT:\n{f}\n\n{b}"}])
    print(f"[acad3y] scoring {len(convs)} prompts ...",flush=True)
    outs=llm.chat(convs,sp)
    vals=[parse_tok(o.outputs[0].text) for o in outs]
    X=np.array(vals,dtype=float).reshape(len(rows),len(metrics))
    na=float(np.isnan(X).mean()); print(f"[acad3y] A NA rate {na:.3f}",flush=True)
    np.savez_compressed(a.out,X=X,V=Vf,ntitle=nt,a_names=np.array(a_names,dtype=object),v_names=np.array(V_NAMES,dtype=object),na_rate=na)
    print(f"[acad3y] saved -> {a.out}\nSCORE_DONE",flush=True)
if __name__=="__main__": main()
