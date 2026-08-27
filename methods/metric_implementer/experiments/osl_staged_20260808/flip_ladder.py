"""Flip-selected (functional) rubrics scored across the full local executor ladder
(logprob readout): is the receiver-relative flip channel a CURVE in z or a point?
Rubrics reconstructed from flip_functional_v2_{llama70b,qwen25-72b}.json selected sets
(frontier + encoder objectives). Definition baselines already exist in the mbar panels.
Sidecar mask json stores exemplar indices per rubric for harvest-time masking.
Usage: flip_ladder.py <executor>"""
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
EX_TRUNC = 500
TASKS = ["humor", "creative_writing", "math", "news_homepages"]

sel = {s: json.load(open(f"{OM}/flip_functional_v2_{s}.json"))["results"]
       for s in ("llama70b", "qwen25-72b")}

executor = make_judge_backend(EXECUTORS[EX][0], cfgmod.ImplementerConfig(), temperature=None)
names, m_rows, mask_map = [], [], {}
for TASK in TASKS:
    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), TASK.replace("_", "-"))
    texts, _ = _load_texts(TASK.replace("_", "-"), 360, cfg)
    probes = texts[60:360]
    MX = cfg.max_text_chars

    def trunc(t):
        t = t.strip()
        return t if len(t) <= EX_TRUNC else t[:EX_TRUNC].rsplit(" ", 1)[0] + " ..."

    rubs = []
    for s in sel:
        for base, rec in sel[s].get(TASK, {}).items():
            for obj in ("frontier", "encoder"):
                po = rec.get("objectives", {}).get(obj) if "objectives" in rec else rec.get(obj)
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
                key = f"{TASK}|{base}||functional_{s}_{obj}"
                rubs.append((key, r))
                mask_map[key] = sorted(i for i, _ in items)
    if not rubs:
        continue
    prompts = [ap._YESNO_TEXTFIRST.format(text=t[:MX], rubric=r) for _, r in rubs
               for t in probes]
    flat = np.asarray(executor.score_binary(prompts), float)
    M = flat.reshape(len(rubs), len(probes))
    for (key, _), row in zip(rubs, M):
        names.append(key)
        m_rows.append(row)
    print(f"[{EX}|{TASK}] {len(rubs)} functional rubrics scored", flush=True)

np.savez(f"{OM}/mbar_flipladder_{EX}.npz", m_bar=np.stack(m_rows),
         names=np.array(names, object))
json.dump(mask_map, open(f"{OM}/flipladder_mask_v1.json", "w"))
print("DONE", EX, len(names), flush=True)
