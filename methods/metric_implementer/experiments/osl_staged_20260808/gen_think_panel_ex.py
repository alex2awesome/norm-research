"""3b-EX think x examples crossing (user 2026-08-06): exemplar arms under both modes (smoke-gated, passed 2026-08-06): definition-arm rubrics
(all humor slate bases incl. planted) x 300 probes, generative readout in both modes.
Writes mbar_zxagen_{mode}_humor_{exec}.npz compatible with the zxa harvest tooling.
Usage: gen_think_panel.py <executor>"""
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
TASK = sys.argv[2] if len(sys.argv) > 2 else "humor"
MAX_GEN = 1536

EXARMS = {"exemplars", "exemplars_mm", "def_exemplars", "exemplars_authored",
          "exemplars_authored_mm"}
f1 = json.load(open(f"{OM}/freeze_zxa_ex_" + TASK + "_v1.json"))
rubs = [(e["zxa"]["base"] + "||" + e["zxa"]["arm"], e["rubric"]) for e in f1["metrics"]
        if e["zxa"]["arm"] in EXARMS or e["zxa"]["arm"].startswith("exemplars")
        or e["zxa"]["arm"] == "def_exemplars"]
assert rubs, "no definition-arm rubrics found"

cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), TASK.replace("_", "-"))
texts, _ = _load_texts(TASK.replace("_", "-"), 360, cfg)
probes = texts[60:360]
MX = cfg.max_text_chars
ex = make_judge_backend(EXECUTORS[EX][0], cfgmod.ImplementerConfig(), temperature=None)

prompts = [ap._YESNO_TEXTFIRST.format(text=t[:MX], rubric=r) for _, r in rubs for t in probes]
names = np.array([b for b, _ in rubs], object)
print(f"[{EX}] {len(rubs)} rubrics x {len(probes)} probes = {len(prompts)} prompts/mode",
      flush=True)
for mode, think in (("nothink", False), ("think", True)):
    flat = np.asarray(ex.score_binary_gen(prompts, thinking=think, max_gen_tokens=MAX_GEN),
                      float)
    M = flat.reshape(len(rubs), len(probes))
    np.savez(f"{OM}/mbar_zxagenex_{mode}_{TASK}_{EX}.npz", m_bar=M, names=names)
    print(f"[{EX}|{mode}] saved, nan_rate={float(np.mean(~np.isfinite(M))):.4f}", flush=True)
print("DONE", EX, flush=True)
