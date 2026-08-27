"""1c k-scaling arms (expansion 1): exemplars_k2 / k16 / k32 for the 14 crowd-decisive humor
bases (k8 = the existing `exemplars` arm). k = total exemplars, half satisfy / half not, chosen
by crowd-consensus rank exactly as the k8 build; deeper lists relax purity — per-base achieved
purity recorded in zxa.purity (min consensus among chosen YES, max among chosen NO). Items are
supersets: k2 ⊂ k8 ⊂ k16 ⊂ k32. Mask zxa.exemplar_idx at fit time.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/lfs/skampere3/0/alexspan/norm-research")
from methods.metric_implementer.experiments.run_real_test import _load_texts
from methods.metric_implementer import config as cfgmod

B = "/lfs/skampere3/0/alexspan"
OM = f"{B}/outputs/osl_multi"
O = f"{B}/outputs/osl"
LOCAL_MID = ["llama1b", "llama3b", "llama8b", "llama70b", "qwen25-3b", "qwen25-7b",
             "qwen25-14b", "qwen25-72b", "mistral7b", "phi4", "gemma2-27b"]
KS = [2, 16, 32]
LEN_CAP = 400

v1 = json.load(open(f"{OM}/freeze_zxa_ex_humor_v1.json"))
bases = sorted({e["zxa"]["base"] for e in v1["metrics"] if e["zxa"]["arm"] == "exemplars"})
cls_of = {e["zxa"]["base"]: e["zxa"]["class"] for e in v1["metrics"]}

panels, names_ref = {}, None
for ex in LOCAL_MID:
    z = np.load(f"{O}/mbar285_{ex}.npz", allow_pickle=True)
    nm = [str(n) for n in z["names"]]
    names_ref = names_ref or nm
    assert nm == names_ref
    panels[ex] = z["m_bar"]
row = {n: i for i, n in enumerate(names_ref)}

cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), "humor")
texts, _ = _load_texts("humor", 360, cfg)
probes = texts[60:360]

entries = []
for b in bases:
    cons = np.stack([(panels[ex][row[b]] > 0.5).astype(float) for ex in LOCAL_MID]).mean(0)
    ok = [j for j, t in enumerate(probes) if len(t) <= LEN_CAP and t.strip()]
    pos_all = sorted(ok, key=lambda j: -cons[j])
    neg_all = sorted(ok, key=lambda j: cons[j])
    for k in KS:
        h = k // 2
        pos, neg = pos_all[:h], neg_all[:h]
        blk_p = "\n".join("- " + probes[j].strip() for j in pos)
        blk_n = "\n".join("- " + probes[j].strip() for j in neg)
        rub = ("%s\nExamples that satisfy this criterion:\n%s\n"
               "Examples that do NOT satisfy it:\n%s" % (b, blk_p, blk_n))
        entries.append({
            "name": f"{b}||exemplars_k{k}", "kind": f"{cls_of[b]}|exemplars_k{k}",
            "rubric": rub, "criteria": [],
            "zxa": {"base": b, "arm": f"exemplars_k{k}", "class": cls_of[b],
                    "exemplar_idx": sorted(pos + neg), "mismatch_src": None,
                    "purity": [round(float(cons[pos[-1]]), 3), round(float(cons[neg[-1]]), 3)]}})

out = {"meta": dict(v1["meta"], kscale_freeze="ex3", ks=KS,
                    note="k8 = the ex_v1 `exemplars` arm; supersets by consensus rank; "
                         "mask exemplar_idx at fit time; purity=[min-yes,max-no] consensus"),
       "metrics": entries}
json.dump(out, open(f"{OM}/freeze_zxa_ex3_humor_v1.json", "w"), indent=1)
pur = [e["zxa"]["purity"] for e in entries if e["zxa"]["arm"] == "exemplars_k32"]
print(f"entries {len(entries)} ({len(bases)} bases x {len(KS)} ks); "
      f"k32 purity floor: min-yes {min(p[0] for p in pur)}, max-no {max(p[1] for p in pur)}")
