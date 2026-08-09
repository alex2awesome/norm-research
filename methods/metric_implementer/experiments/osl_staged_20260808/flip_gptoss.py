"""Flip-selected sets scored at a FRONTIER receiver (criterion-3 caveat removal,
user-approved 2026-08-09): the functional rubrics from flip_functional_v2
{llama70b,qwen25-72b} selections (frontier+encoder objectives), humor/CW/math only
(news excluded — probe-universe landmine), scored by gpt-oss-120b via the GENERATIVE
Harmony readout (logprob readout is invalid for Harmony models: the analysis channel
precedes the verdict token). Definition arm scored in the same run, same readout, so
the fun-vs-def comparison at frontier is same-instrument. Writes
mbar_flipgptoss_{EX}.npz + reuses flipladder_mask_v1.json for masking at harvest.
Usage: flip_gptoss.py [executor=gpt-oss-120b]
"""
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
EX = sys.argv[1] if len(sys.argv) > 1 else "gpt-oss-120b"
EX_TRUNC = 500
MAX_GEN = 1536
TASKS = ["humor", "creative_writing", "math"]

sel = {s: json.load(open(f"{OM}/flip_functional_v2_{s}.json"))["results"]
       for s in ("llama70b", "qwen25-72b")}
v2rub_all = {}

executor = make_judge_backend(EXECUTORS[EX][0], cfgmod.ImplementerConfig(), temperature=None)
names, m_rows = [], []
for TASK in TASKS:
    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), TASK.replace("_", "-"))
    texts, _ = _load_texts(TASK.replace("_", "-"), 360, cfg)
    probes = texts[60:360]
    MX = cfg.max_text_chars

    def trunc(t):
        t = t.strip()
        return t if len(t) <= EX_TRUNC else t[:EX_TRUNC].rsplit(" ", 1)[0] + " ..."

    rubs, def_bases = [], {}
    for s in sel:
        for base, rec in sel[s].get(TASK, {}).items():
            for obj in ("frontier", "encoder"):
                po = rec.get("objectives", {}).get(obj)
                if not po or not po.get("selected"):
                    continue
                items = [(int(i), int(l)) for i, l in po["selected"]]
                pos = [i for i, l in items if l == 1]
                neg = [i for i, l in items if l == 0]
                r = base
                if pos:
                    r += "\nExamples that satisfy this criterion:\n" + \
                         "\n".join("- " + trunc(probes[i]) for i in pos)
                if neg:
                    r += "\nExamples that do NOT satisfy it:\n" + \
                         "\n".join("- " + trunc(probes[i]) for i in neg)
                rubs.append((f"{TASK}|{base}||functional_{s}_{obj}", r))
                def_bases[base] = True
    # definition arm, same readout, once per base
    fz = f"{OM}/freeze_{TASK}_v2.json"
    import os
    v2rub = ({m["name"]: m["rubric"] for m in json.load(open(fz))["metrics"]}
             if os.path.exists(fz) else {})
    for base in def_bases:
        rubs.append((f"{TASK}|{base}||definition_gen", v2rub.get(base, base)))
    if not rubs:
        continue
    prompts = [ap._YESNO_TEXTFIRST.format(text=t[:MX], rubric=r) for _, r in rubs
               for t in probes]
    flat = np.asarray(executor.score_binary_gen(prompts, thinking=True,
                                                max_gen_tokens=MAX_GEN), float)
    M = flat.reshape(len(rubs), len(probes))
    nan_rate = float(np.mean(~np.isfinite(M)))
    for (key, _), row in zip(rubs, M):
        names.append(key)
        m_rows.append(row)
    print(f"[{EX}|{TASK}] {len(rubs)} rubrics scored, nan_rate={nan_rate:.4f}", flush=True)

np.savez(f"{OM}/mbar_flipgptoss_{EX}.npz", m_bar=np.stack(m_rows),
         names=np.array(names, object))
print("DONE", EX, len(names), flush=True)
