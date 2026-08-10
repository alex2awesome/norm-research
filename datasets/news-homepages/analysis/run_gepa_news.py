#!/usr/bin/env python3
"""Phase A: GEPA prompt-optimization (reconstruction-R objective) for the 10 NEW news-homepages metrics.
judge=Gemma-4-32B (served port 8006); reviser/reconstructor=GLM-5.2 (z.ai zai_anthropic) via make_roles_mixed.
Objective=fidelity_scalar (reconstruction accuracy + ...). Saves optimized prompts + fidelity gains."""
import os,sys,json,time,warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,"methods")
sys.path.insert(0,"datasets/news-homepages/analysis")
import numpy as np,pandas as pd
from metric_implementer.config import ImplementerConfig
from metric_implementer.backends import make_roles_mixed, LLMBackend, BACKENDS
from metric_implementer.optimizer import improve
from metric_implementer.artifact import MetricArtifact
from metric_implementer.measures import compute_scorecard
from metric_implementer.registry import Registry
from news_newmetrics import NEW_METRICS
DS="/lfs/skampere3/0/alexspan/norm-research/datasets/news-homepages"
open("/tmp/gemma_dummy_key.txt","w").write("dummy")
BACKENDS["local"]={"url":"http://127.0.0.1:8006/v1/chat/completions","key":"/tmp/gemma_dummy_key.txt","format":"openai"}
cfg=ImplementerConfig()
cfg.task="news-homepages"; cfg.backend="local"; cfg.judge_model="gemma"; cfg.reviser_model="glm-5.2"
cfg.domain_plural="news homepage articles (headline + summary + sibling-headline context)"
cfg.item_label="ARTICLE"; cfg.item_noun="article"
cfg.cf_keep_clause="keep the story topic, named actors, and stated facts"
cfg.cf_hold_default="length, section, and promotional tone"
cfg.recon_item_phrase="headline excerpts"; cfg.text_column="text"; cfg.id_column="snapshot_id"
cfg.random_seed=7; cfg.llm_concurrency=8; cfg.n_oracle_items=0; cfg.max_text_chars=2000; cfg.request_timeout_s=120
print("[gepa-news] judge=Gemma-4-32B(served:8006), strong=GLM-5.2(z.ai)",flush=True)
judge=LLMBackend("gemma","judge",cfg,cfg.judge_temperature)
roles=make_roles_mixed(judge,strong_model="glm-5.2",strong_backend="zai_anthropic",base_cfg=cfg)
d=pd.read_csv(f"{DS}/homepage_newsworthiness_clean_v8.csv.gz",compression="gzip"); d["text"]=d.text.fillna("")
texts=d.text.sample(n=min(40,len(d)),random_state=7).tolist()
registry=Registry(cfg.registry_dir()); rng=np.random.default_rng(7); out=[]
print(f"[gepa-news] {len(NEW_METRICS)} new metrics, {len(texts)} probe texts",flush=True)
for i,(name,guidance) in enumerate(NEW_METRICS,1):
    mid=f"gepanews_{name[:30].replace(' ','_')}"
    registry.register_metric(mid,name,guidance)
    seed=MetricArtifact(metric_id=mid,kind="prompt",body=guidance,name=name,description=guidance)
    seed_card=compute_scorecard(seed,texts,roles,cfg,np.random.default_rng(7)); seed_fid=seed_card["fidelity_scalar"]
    t0=time.time()
    try:
        summary=improve(seed,texts,roles,cfg,registry,caps=None,rounds=2,
                        data_ids=[str(j) for j in range(len(texts))],run_id=f"gepanews_{mid}",log=lambda *a,**k:None)
        accepted=bool(summary.get("accepted")) if summary else False
        acc=(summary or {}).get("best_fidelity_acceptance"); acc_s=f"{acc:.3f}" if isinstance(acc,(int,float)) else "na"
        # extract best prompt version from registry (highest-fidelity prompt version)
        best_prompt=guidance
        try:
            vers=registry.versions(mid,"prompt")
            best_v=max(vers,key=lambda v:v.get("fidelity_scalar",v.get("scorecard",{}).get("fidelity_scalar",-1))) if vers else None
            if best_v: best_prompt=registry.artifact_from_version(best_v).body or guidance
        except Exception as e: pass
    except Exception as e:
        import traceback; traceback.print_exc(); accepted=False; acc_s=f"ERR:{str(e)[:30]}"; best_prompt=guidance
    print(f"  ({i}/{len(NEW_METRICS)}) {name[:38]:40} seed_fid={seed_fid:.3f} acc_fid={acc_s} accepted={'Y' if accepted else 'N'} ({time.time()-t0:.0f}s)",flush=True)
    out.append({"name":name,"seed_prompt":guidance,"best_prompt":best_prompt,"seed_fid":round(seed_fid,3),"acc_fid":acc_s,"accepted":accepted})
with open(f"{DS}/news_gepa_optimized.jsonl","w") as f:
    for r in out: f.write(json.dumps(r)+"\n")
imp=sum(1 for r in out if r["accepted"])
print(f"\n[gepa-news] DONE {len(out)} metrics, {imp} accepted (improved). Wrote news_gepa_optimized.jsonl",flush=True)
print("GEPA_NEWS_DONE",flush=True)
