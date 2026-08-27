"""Harvest code-truth planted exemplar arms: does stated>>shown survive provably-correct
examples? Scores vs (a) frontier-dossier reference (comparable to crowd-exemplar numbers)
and (b) the computed rule truth itself."""
import json
import sys

import numpy as np

sys.path.insert(0, "/lfs/skampere3/0/alexspan/norm-research")
from methods.metric_implementer.experiments.osl_sweep import planted_metrics, _wc
from methods.metric_implementer.experiments.run_real_test import _load_texts
from methods.metric_implementer import config as cfgmod

B = "/lfs/skampere3/0/alexspan"
OM = f"{B}/outputs/osl_multi"
LOCALS = ["llama1b", "llama3b", "llama8b", "qwen25-3b", "qwen25-7b",
          "qwen25-14b", "qwen25-32b", "llama70b", "qwen25-72b"]
BIG = {"qwen25-14b", "qwen25-32b", "llama70b", "qwen25-72b"}


def load(p):
    z = np.load(p, allow_pickle=True)
    return {str(n): z["m_bar"][i] for i, n in enumerate([str(x) for x in z["names"]])}

ct = {ex: load(f"{OM}/mbar_zxaexct_humor_{ex}.npz") for ex in LOCALS}
ex1 = {ex: load(f"{OM}/mbar_zxaex_humor_{ex}.npz") for ex in LOCALS}
v1p = {ex: load(f"{OM}/mbar_zxa_humor_{ex}.npz") for ex in LOCALS}
glm = {ex: load(f"{OM}/mbar_zxaglm_humor_{ex}.npz") for ex in ["glm-47", "glm-52"]}
fct = json.load(open(f"{OM}/freeze_zxa_exct_humor_v1.json"))
f1 = json.load(open(f"{OM}/freeze_zxa_ex_humor_v1.json"))

cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), "humor")
texts, _ = _load_texts("humor", 360, cfg)
probes = texts[60:360]
k_med = int(np.median([_wc(t) for t in probes]))
truth = {pm["name"]: np.asarray(pm["truth"], int) for pm in planted_metrics(probes, k_med)}

idx_ct = {e["zxa"]["base"]: set(e["zxa"]["exemplar_idx"]) for e in fct["metrics"]
          if e["zxa"]["arm"] == "exemplars_ct"}
idx_cr = {e["zxa"]["base"]: set(e["zxa"]["exemplar_idx"]) for e in f1["metrics"]
          if e["zxa"]["arm"] == "exemplars"}

def frontier_ref(base):
    votes = []
    for ex in ["llama70b", "qwen25-72b"]:
        r = v1p[ex].get(f"{base}||dossier")
        if r is not None:
            votes.append((np.asarray(r, float) > .5).astype(float))
    for ex in ["glm-47", "glm-52"]:
        r = glm[ex].get(f"{base}||dossier")
        if r is not None:
            votes.append((np.asarray(r, float) > .5).astype(float))
    mean = np.stack(votes).mean(0)
    return np.where(mean > .5, 1, np.where(mean < .5, 0, -1))

def bal(pred, lab, mask):
    ok = (lab >= 0) & mask & np.isfinite(pred)
    if ok.sum() < 20:
        return None
    p = (pred[ok] > .5).astype(int)
    l = lab[ok]
    accs = [float(np.mean(p[l == c] == c)) for c in (0, 1) if (l == c).sum() >= 3]
    return float(np.mean(accs)) if len(accs) == 2 else None

import collections
agg = collections.defaultdict(list)
for base, tr in truth.items():
    fref = frontier_ref(base)
    mask = np.ones(300, bool)
    for j in idx_ct.get(base, set()) | idx_cr.get(base, set()):
        mask[j] = False
    for exe in LOCALS:
        tier = "big" if exe in BIG else "small"
        for arm, src, key in (("exemplars_ct", ct, f"{base}||exemplars_ct"),
                              ("exemplars_ct_mm", ct, f"{base}||exemplars_ct_mm"),
                              ("exemplars_crowd", ex1, f"{base}||exemplars"),
                              ("definition", v1p, f"{base}||definition"),
                              ("name", v1p, f"{base}||name")):
            r = src[exe].get(key)
            if r is None:
                continue
            pred = np.asarray(r, float)
            for refname, lab in (("frontier", fref), ("codetruth", tr)):
                y = bal(pred, np.asarray(lab), mask)
                if y is not None:
                    agg[(tier, refname, arm)].append(y)
for tier in ("big", "small"):
    for refname in ("codetruth", "frontier"):
        line = []
        for arm in ("name", "definition", "exemplars_crowd", "exemplars_ct", "exemplars_ct_mm"):
            v = agg.get((tier, refname, arm))
            line.append(f"{arm}={np.mean(v):.3f}(n={len(v)})" if v else f"{arm}=--")
        print(f"[{tier}|vs-{refname}] " + "  ".join(line))
