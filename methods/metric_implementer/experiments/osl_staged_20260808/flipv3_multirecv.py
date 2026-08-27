"""Examples-help ledger at scale (user 2026-08-09: "more metrics, more models"):
score the flip-v3 selected sets (humor full bank, 222 bases) + definition + mm control
(donor-rotation) at a given receiver. Readout: generative Harmony for gpt-oss,
logprob score_binary otherwise. Writes mbar_flipv3sets_{EX}.npz + flipv3sets_mask.json
(own+donor exemplar indices per key — mask the UNION at harvest).
Usage: flipv3_multirecv.py <executor>
"""
import json
import os
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
MAX_GEN = 1536
TASK = "humor"

cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), TASK)
texts, _ = _load_texts(TASK, 360, cfg)
probes = texts[60:360]
MX = cfg.max_text_chars

v3 = json.load(open(f"{OM}/flip_functional_v3_llama70b.json"))["results"][TASK]
fz = f"{OM}/freeze_{TASK}_v2.json"
v2rub = ({m["name"]: m["rubric"] for m in json.load(open(fz))["metrics"]}
         if os.path.exists(fz) else {})


def trunc(t):
    t = t.strip()
    return t if len(t) <= EX_TRUNC else t[:EX_TRUNC].rsplit(" ", 1)[0] + " ..."


def ex_block(items):
    pos = [i for i, l in items if l == 1]
    neg = [i for i, l in items if l == 0]
    r = ""
    if pos:
        r += "\nExamples that satisfy this criterion:\n" + \
             "\n".join("- " + trunc(probes[i]) for i in pos)
    if neg:
        r += "\nExamples that do NOT satisfy it:\n" + \
             "\n".join("- " + trunc(probes[i]) for i in neg)
    return r


entries = []
for base, rec in v3.items():
    po = rec.get("objectives", {}).get("frontier2v")
    if po and po.get("selected"):
        entries.append((base, [(int(i), int(l)) for i, l in po["selected"]]))
print(f"[{EX}] {len(entries)} bases with frontier2v sets", flush=True)

rubs, mask = [], {}
for k, (base, items) in enumerate(entries):
    donor_base, donor_items = entries[(k + 1) % len(entries)]
    d = v2rub.get(base, base)
    for arm, block, midx in (("definition", "", []),
                             ("functional", ex_block(items), [i for i, _ in items]),
                             ("functionalmm", ex_block(donor_items),
                              [i for i, _ in donor_items])):
        key = f"{TASK}|{base}||{arm}"
        rubs.append((key, (d if arm == "definition" else base + block)))
        mask[key] = sorted(midx)

prompts = [ap._YESNO_TEXTFIRST.format(text=t[:MX], rubric=r) for _, r in rubs
           for t in probes]
print(f"[{EX}] {len(rubs)} rubrics x {len(probes)} probes = {len(prompts)} prompts",
      flush=True)
executor = make_judge_backend(EXECUTORS[EX][0], cfgmod.ImplementerConfig(), temperature=None)
if "gpt-oss" in EX:
    flat = np.asarray(executor.score_binary_gen(prompts, thinking=True,
                                                max_gen_tokens=MAX_GEN), float)
else:
    flat = np.asarray(executor.score_binary(prompts), float)
M = flat.reshape(len(rubs), len(probes))
np.savez(f"{OM}/mbar_flipv3sets_{EX}.npz", m_bar=M,
         names=np.array([k for k, _ in rubs], object))
json.dump(mask, open(f"{OM}/flipv3sets_mask.json", "w"))
print(f"DONE-V3SETS {EX} {len(rubs)} nan={float(np.mean(~np.isfinite(M))):.4f}",
      flush=True)
