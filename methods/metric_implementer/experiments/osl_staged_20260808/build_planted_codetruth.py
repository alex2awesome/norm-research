"""Planted exemplars re-selected by CODE TRUTH (user 2026-08-06): kills the crowd-label
confound on the stated-vs-shown planted claim. Arms per planted metric (humor):
  exemplars_ct    — 4+4 items chosen by the RULE's computed truth (deterministic pick)
  exemplars_ct_mm — another planted rule's code-true examples (content placebo)
Same template/masking rules as ex_v1.
"""
import hashlib
import json
import sys

sys.path.insert(0, "/lfs/skampere3/0/alexspan/norm-research")
from methods.metric_implementer.experiments.run_real_test import _load_texts
from methods.metric_implementer.experiments.osl_sweep import planted_metrics, _wc
from methods.metric_implementer import config as cfgmod

import numpy as np

B = "/lfs/skampere3/0/alexspan"
OM = f"{B}/outputs/osl_multi"
LEN_CAP = 400

cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), "humor")
texts, _ = _load_texts("humor", 360, cfg)
probes = texts[60:360]
k_med = int(np.median([_wc(t) for t in probes]))
planted = planted_metrics(probes, k_med)

def pick(truth, want, base, n=4):
    idx = [j for j, t in enumerate(probes)
           if truth[j] == want and len(probes[j]) <= LEN_CAP and probes[j].strip()]
    idx.sort(key=lambda j: hashlib.md5((base + str(j)).encode()).hexdigest())
    return idx[:n]

sel = {}
for pm in planted:
    name, truth = pm["name"], pm["truth"]
    pos, neg = pick(truth, True, name), pick(truth, False, name)
    sel[name] = (pos, neg)

names = sorted(sel)
partner = {a: b for a, b in zip(names, names[1:] + names[:1])}

def block(pos, neg):
    p = "\n".join("- " + probes[j].strip() for j in pos)
    n = "\n".join("- " + probes[j].strip() for j in neg)
    return ("\nExamples that satisfy this criterion:\n%s\n"
            "Examples that do NOT satisfy it:\n%s" % (p, n))

entries = []
for name in names:
    pos, neg = sel[name]
    mp, mn = sel[partner[name]]
    for arm, rub, idx in (("exemplars_ct", name + block(pos, neg), sorted(pos + neg)),
                          ("exemplars_ct_mm", name + block(mp, mn), sorted(mp + mn))):
        entries.append({"name": f"{name}||{arm}", "kind": f"PLANTED|{arm}", "rubric": rub,
                        "criteria": [],
                        "zxa": {"base": name, "arm": arm, "class": "PLANTED",
                                "exemplar_idx": idx,
                                "mismatch_src": partner[name] if "mm" in arm else None}})

v1meta = json.load(open(f"{OM}/freeze_zxa_ex_humor_v1.json"))["meta"]
out = {"meta": dict(v1meta, codetruth_freeze="v1",
                    note="planted exemplars selected by computed rule truth (no crowd); "
                         "mask exemplar_idx at fit time"),
       "metrics": entries}
json.dump(out, open(f"{OM}/freeze_zxa_exct_humor_v1.json", "w"), indent=1)
print(f"entries: {len(entries)} ({len(names)} planted rules)")

