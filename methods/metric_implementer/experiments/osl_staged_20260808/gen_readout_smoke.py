"""3b generative-readout SMOKE (gates any think-mode ladder): 3 humor definition rubrics x
20 probes. Reports, per mode (no_think / think): nan rate, hard agreement vs the logprob
readout, think-trace lengths + truncation, think-vs-nothink verdict flips. Saves raw
samples for eyeballing. Usage: gen_readout_smoke.py <executor> <out.json>"""
import json
import sys

import numpy as np

sys.path.insert(0, "/lfs/skampere3/0/alexspan/norm-research")
from methods.metric_implementer.experiments import alpha_probe as ap
from methods.metric_implementer.experiments.osl_sweep import EXECUTORS
from methods.metric_implementer.experiments.run_real_test import _load_texts
from methods.metric_implementer import config as cfgmod
from methods.metric_implementer.vllm_backend import make_judge_backend

B = "/lfs/skampere3/0/alexspan"
OM = f"{B}/outputs/osl_multi"
EX = sys.argv[1]
OUT = sys.argv[2]
N_PROBES = 20
N_BASES = 3

slate = [m for m in json.load(open(f"{OM}/zxa_slate_v1.json"))
         if m["task"] == "humor" and m["class"] == "TACIT-CANDIDATE"]
v2rub = {m["name"]: m["rubric"] for m in json.load(open(f"{OM}/freeze_humor_v2.json"))["metrics"]}
bases = [m["name"] for m in slate if m["name"] in v2rub][:N_BASES]

cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), "humor")
texts, _ = _load_texts("humor", 360, cfg)
probes = texts[60:60 + N_PROBES]
MX = cfg.max_text_chars

prompts = [ap._YESNO_TEXTFIRST.format(text=t[:MX], rubric=v2rub[b])
           for b in bases for t in probes]
ex = make_judge_backend(EXECUTORS[EX][0], cfgmod.ImplementerConfig(), temperature=None)

lp = np.asarray(ex.score_binary(prompts), float)
g0, r0 = ex.score_binary_gen(prompts, thinking=False, return_texts=True)
g1, r1 = ex.score_binary_gen(prompts, thinking=True, return_texts=True)
g0, g1 = np.asarray(g0, float), np.asarray(g1, float)

def agree(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.mean((a[ok] > .5) == (b[ok] > .5))) if ok.sum() else float("nan")

think_lens = [len(t.split("</think>")[0]) if "</think>" in t else len(t) for t in r1]
res = {
    "executor": EX, "n_prompts": len(prompts), "bases": bases,
    "nan_rate": {"logprob": float(np.mean(~np.isfinite(lp))),
                 "gen_nothink": float(np.mean(~np.isfinite(g0))),
                 "gen_think": float(np.mean(~np.isfinite(g1)))},
    "agree_gen0_vs_logprob": agree(g0, lp),
    "agree_gen1_vs_gen0": agree(g1, g0),
    "think_trace_chars": {"mean": float(np.mean(think_lens)), "max": int(np.max(think_lens))},
    "think_truncated_frac": float(np.mean(["</think>" not in t for t in r1])),
    "nothink_has_think_tag_frac": float(np.mean(["</think>" in t for t in r0])),
    "samples": {"nothink": r0[:3], "think": [t[-600:] for t in r1[:3]]},
}
json.dump(res, open(OUT, "w"), indent=1)
for k in ("nan_rate", "agree_gen0_vs_logprob", "agree_gen1_vs_gen0",
          "think_trace_chars", "think_truncated_frac", "nothink_has_think_tag_frac"):
    print(f"SMOKE[{EX}] {k} = {res[k]}", flush=True)
print("DONE ->", OUT)
