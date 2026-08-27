#!/usr/bin/env python3
"""GEPA-reconstruction on (a) EXISTING-4 signal-carrier prompts + (b) CODE-KIND metrics for the new
linguistic/structural dimensions. judge=Gemma-4-32B(8006), strong=GLM-5.2(z.ai). Code body =
Python module exposing score(text)->float|None. Saves optimized prompts+code + fidelity gains."""
import os,sys,json,time,warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,"methods")
import numpy as np,pandas as pd
from metric_implementer.config import ImplementerConfig
from metric_implementer.backends import make_roles_mixed, LLMBackend, BACKENDS
from metric_implementer.optimizer import improve
from metric_implementer.artifact import MetricArtifact
from metric_implementer.measures import compute_scorecard
from metric_implementer.registry import Registry
DS="/lfs/skampere3/0/alexspan/norm-research/datasets/news-homepages"
open("/tmp/gemma_dummy_key.txt","w").write("dummy")
BACKENDS["local"]={"url":"http://127.0.0.1:8006/v1/chat/completions","key":"/tmp/gemma_dummy_key.txt","format":"openai"}
# --- EXISTING-4 prompt-kind seeds (the signal carriers) ---
EXISTING4=[
("hard_vs_soft","Score 1 if HARD news (politics/war/economy/crime/disaster), 0 if SOFT (lifestyle/entertainment/sports/service). Return only the score."),
("elite_political_actor","Score 1 if the headline names a head of state, top government official, SCOTUS justice, legislature/party leader, or major political figure as a central subject; else 0."),
("ongoing_top_story","Score 1 if clearly part of a top-tier ongoing national/international story (war, election, major trial, crisis); else 0."),
("breaking_developing","Score 1 if a just-happened or actively-unfolding event (breaking, live, developing, imminent); else 0."),
]
# --- CODE-KIND seeds for new linguistic/structural dimensions (score(text)->float|None) ---
CODE_METRICS=[
("code_concrete_numbers",'import re\ndef score(text):\n    t=text or ""\n    return float(len(re.findall(r"\\$\\s?\\d+|\\d+(?:\\.\\d+)?\\s?%|\\b\\d[\\d,]{2,}\\b", t)))\n'),
("code_vivid_verbs",'import re\ndef score(text):\n    t=(text or "").lower()\n    return float(len(re.findall(r"\\b(slam|torch|explode|collapse|seize|blast|erupt|sweep|deadlock|bury|crack|storm|crush|strike|blow)\\b", t)))\n'),
("code_curiosity_gap",'import re\ndef score(text):\n    t=text or ""\n    return 1.0 if re.search(r"\\?|\\b(why|how|what|here|secret|reveals?)\\b", t, re.I) else 0.0\n'),
("code_moral_outrage",'import re\ndef score(text):\n    t=(text or "").lower()\n    return float(len(re.findall(r"\\b(cover-?up|scandal|betray|exploit|fraud|abuse|kill|murder|corrupt|weaponiz|coverup|lie)\\b", t)))\n'),
("code_reader_stakes",'import re\ndef score(text):\n    t=(text or "").lower()\n    return float(len(re.findall(r"\\b(price|prices|jobs|layoff|tariff|tax|mortgage|rent|gas|grocer|evacuat|safe|self-deport|visa|deport)\\b", t)))\n'),
("code_named_numbers",'import re\ndef score(text):\n    t=text or ""\n    # specificity: proper nouns AND numbers together\n    prop=len(re.findall(r"\\b[A-Z][a-z]{2,}\\b", t)); num=len(re.findall(r"\\b\\d[\\d,]*\\b", t))\n    return float(prop*num) if prop and num else 0.0\n'),
]
cfg=ImplementerConfig()
cfg.task="news-homepages"; cfg.backend="local"; cfg.judge_model="gemma"; cfg.reviser_model="glm-5.2"
cfg.domain_plural="news homepage articles (headline + summary + sibling-headline context)"
cfg.item_label="ARTICLE"; cfg.item_noun="article"; cfg.cf_keep_clause="keep the story topic, named actors, and stated facts"
cfg.cf_hold_default="length, section, and promotional tone"; cfg.recon_item_phrase="headline excerpts"
cfg.text_column="text"; cfg.id_column="snapshot_id"; cfg.random_seed=9
cfg.llm_concurrency=8; cfg.n_oracle_items=0; cfg.max_text_chars=2000; cfg.request_timeout_s=120
print("[gepa-e4c] judge=Gemma-4-32B(8006), strong=GLM-5.2(z.ai)",flush=True)
judge=LLMBackend("gemma","judge",cfg,cfg.judge_temperature)
roles=make_roles_mixed(judge,strong_model="glm-5.2",strong_backend="zai_anthropic",base_cfg=cfg)
d=pd.read_csv(f"{DS}/homepage_newsworthiness_clean_v9.csv.gz",compression="gzip"); d["text"]=d.text.fillna("")
texts=d.text.sample(n=min(40,len(d)),random_state=9).tolist()
registry=Registry(cfg.registry_dir()); out=[]
targets=[(n,g,"prompt") for n,g in EXISTING4]+[(n,c,"code") for n,c in CODE_METRICS]
print(f"[gepa-e4c] {len(targets)} targets ({len(EXISTING4)} prompt + {len(CODE_METRICS)} code), {len(texts)} probe texts",flush=True)
for i,(name,body,kind) in enumerate(targets,1):
    mid=f"gepae4c_{kind[:2]}_{name[:24].replace(' ','_')}"
    registry.register_metric(mid,name,body)
    seed=MetricArtifact(metric_id=mid,kind=kind,body=body,name=name,description=name)
    try: seed_card=compute_scorecard(seed,texts,roles,cfg,np.random.default_rng(9)); seed_fid=seed_card["fidelity_scalar"]
    except Exception as e: import traceback; traceback.print_exc(); seed_fid=-1
    t0=time.time()
    try:
        summary=improve(seed,texts,roles,cfg,registry,caps=None,rounds=2,data_ids=[str(j) for j in range(len(texts))],run_id=f"e4c_{mid}",log=lambda *a,**k:None)
        accepted=bool(summary.get("accepted")) if summary else False
        acc=(summary or {}).get("best_fidelity_acceptance"); acc_s=f"{acc:.3f}" if isinstance(acc,(int,float)) else "na"
        best=body
        try:
            vers=registry.versions(mid,kind)
            bv=max(vers,key=lambda v:v.get("fidelity_scalar",v.get("scorecard",{}).get("fidelity_scalar",-1))) if vers else None
            if bv: best=registry.artifact_from_version(bv).body or body
        except: pass
    except Exception as e:
        import traceback; traceback.print_exc(); accepted=False; acc_s=f"ERR:{str(e)[:25]}"; best=body
    print(f"  ({i}/{len(targets)}) [{kind}] {name[:30]:32} seed={seed_fid:.3f} acc={acc_s} ok={'Y' if accepted else 'N'} ({time.time()-t0:.0f}s)",flush=True)
    out.append({"name":name,"kind":kind,"seed":body,"best":best,"seed_fid":round(seed_fid,3),"acc_fid":acc_s,"accepted":accepted})
with open(f"{DS}/news_gepa_e4c.jsonl","w") as f:
    for r in out: f.write(json.dumps(r)+"\n")
print(f"\n[gepa-e4c] DONE {len(out)} targets. Wrote news_gepa_e4c.jsonl",flush=True)
print("GEPA_E4C_DONE",flush=True)
