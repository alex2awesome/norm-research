"""Mismatched-exemplar control for flip_gptoss (2026-08-09): same base rubrics, but the
exemplar block comes from a DIFFERENT base's flip-selected set (deterministic rotation
within task+selector+objective). If fun-def at gpt-oss-120b is content-bearing, true
sets must beat these mm sets; if mm ~= true, the frontier win is decoder
un-collapsing/anchoring (entropy-mediated), not construct content.
Writes mbar_flipgptossmm_{EX}.npz. Same gen Harmony readout, same masking key
convention (mask indices of the DONOR base's exemplars apply — recorded in sidecar).
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

executor = make_judge_backend(EXECUTORS[EX][0], cfgmod.ImplementerConfig(), temperature=None)
names, m_rows, mm_mask = [], [], {}
for TASK in TASKS:
    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), TASK.replace("_", "-"))
    texts, _ = _load_texts(TASK.replace("_", "-"), 360, cfg)
    probes = texts[60:360]
    MX = cfg.max_text_chars

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

    rubs = []
    for s in sel:
        for obj in ("frontier", "encoder"):
            entries = []
            for base, rec in sel[s].get(TASK, {}).items():
                po = rec.get("objectives", {}).get(obj)
                if po and po.get("selected"):
                    entries.append((base, [(int(i), int(l)) for i, l in po["selected"]]))
            if len(entries) < 2:
                continue
            for k, (base, _own) in enumerate(entries):
                donor_base, donor_items = entries[(k + 1) % len(entries)]
                key = f"{TASK}|{base}||functionalmm_{s}_{obj}"
                rubs.append((key, base + ex_block(donor_items)))
                mm_mask[key] = sorted(i for i, _ in donor_items)
    if not rubs:
        continue
    prompts = [ap._YESNO_TEXTFIRST.format(text=t[:MX], rubric=r) for _, r in rubs
               for t in probes]
    flat = np.asarray(executor.score_binary_gen(prompts, thinking=True,
                                                max_gen_tokens=MAX_GEN), float)
    M = flat.reshape(len(rubs), len(probes))
    for (key, _), row in zip(rubs, M):
        names.append(key)
        m_rows.append(row)
    print(f"[{EX}|{TASK}] {len(rubs)} mm rubrics, nan={float(np.mean(~np.isfinite(M))):.4f}",
          flush=True)

np.savez(f"{OM}/mbar_flipgptossmm_{EX}.npz", m_bar=np.stack(m_rows),
         names=np.array(names, object))
json.dump(mm_mask, open(f"{OM}/flipgptossmm_mask_v1.json", "w"))
print("DONE-MM", EX, len(names), flush=True)
