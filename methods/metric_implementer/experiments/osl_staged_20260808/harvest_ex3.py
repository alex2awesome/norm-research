"""k-scaling harvest: exemplar dose-response k=2/8/16/32 (k8 = ex_v1 `exemplars` arm),
same frontier-dossier reference and masking rules as the mode-grid harvests."""
import json
import os

import numpy as np

B = "/lfs/skampere3/0/alexspan"
OM = f"{B}/outputs/osl_multi"
LOCALS = ["llama1b", "llama3b", "llama8b", "qwen25-3b", "qwen25-7b",
          "qwen25-14b", "qwen25-32b", "llama70b", "qwen25-72b"]
BIG = {"qwen25-14b", "qwen25-32b", "llama70b", "qwen25-72b"}
SMALL = set(LOCALS) - BIG


def load(p):
    z = np.load(p, allow_pickle=True)
    return {str(n): z["m_bar"][i] for i, n in enumerate([str(x) for x in z["names"]])}

v1p = {ex: load(f"{OM}/mbar_zxa_humor_{ex}.npz") for ex in LOCALS}
glm = {ex: load(f"{OM}/mbar_zxaglm_humor_{ex}.npz") for ex in ["glm-47", "glm-52"]}
ex1 = {ex: load(f"{OM}/mbar_zxaex_humor_{ex}.npz") for ex in LOCALS}
ex3 = {ex: load(f"{OM}/mbar_zxaex3_humor_{ex}.npz") for ex in LOCALS}
f1 = json.load(open(f"{OM}/freeze_zxa_ex_humor_v1.json"))
f3 = json.load(open(f"{OM}/freeze_zxa_ex3_humor_v1.json"))
bases = sorted({e["zxa"]["base"] for e in f3["metrics"]})
cls_of = {e["zxa"]["base"]: e["zxa"]["class"] for e in f3["metrics"]}
# mask = union of exemplar_idx across ALL k-arms of a base (k32 superset covers all) + k8 idx
idx = {}
for e in f3["metrics"]:
    if e["zxa"]["arm"] == "exemplars_k32":
        idx[e["zxa"]["base"]] = set(e["zxa"]["exemplar_idx"])
for e in f1["metrics"]:
    if e["zxa"]["arm"] == "exemplars":
        idx.setdefault(e["zxa"]["base"], set()).update(e["zxa"]["exemplar_idx"])

ref = {}
for b in bases:
    votes = []
    for ex in ["llama70b", "qwen25-72b"]:
        r = v1p[ex].get(f"{b}||dossier")
        if r is not None:
            votes.append((np.asarray(r, float) > .5).astype(float))
    for ex in ["glm-47", "glm-52"]:
        r = glm[ex].get(f"{b}||dossier")
        if r is not None:
            votes.append((np.asarray(r, float) > .5).astype(float))
    if len(votes) >= 3:
        mean = np.stack(votes).mean(0)
        ref[b] = np.where(mean > .5, 1, np.where(mean < .5, 0, -1))


def bal(pred, lab, mask):
    ok = (lab >= 0) & mask & np.isfinite(pred)
    if ok.sum() < 20:
        return None
    p = (pred[ok] > .5).astype(int)
    l = lab[ok]
    accs = [float(np.mean(p[l == c] == c)) for c in (0, 1) if (l == c).sum() >= 3]
    return float(np.mean(accs)) if len(accs) == 2 else None

ARMS = [("name", v1p), ("definition", v1p), ("exemplars_k2", ex3), ("exemplars", ex1),
        ("exemplars_k16", ex3), ("exemplars_k32", ex3), ("exemplars_mm", ex1)]
rows = []
for b in bases:
    if b not in ref:
        continue
    m = np.ones(300, bool)
    for j in idx.get(b, ()):
        m[j] = False
    for exe in LOCALS:
        for arm, src in ARMS:
            r = src[exe].get(f"{b}||{arm}")
            if r is not None:
                y = bal(np.asarray(r, float), ref[b], m)
                if y is not None:
                    rows.append((b, cls_of[b], arm, exe, y))

import collections
for tier, tset in (("big", BIG), ("small", SMALL)):
    print(f"== {tier} tier ==")
    agg = collections.defaultdict(list)
    for b, c, a, e, y in rows:
        if e in tset:
            agg[(c, a)].append(y)
    for cls in sorted({c for _, c, *_ in rows}):
        line = []
        for arm, _ in ARMS:
            v = agg.get((cls, arm))
            line.append(f"{arm.replace('exemplars', 'ex')}={np.mean(v):.3f}" if v else f"{arm}=--")
        print(f"  {cls:16s} " + "  ".join(line))
json.dump([dict(base=b, cls=c, arm=a, exec=e, y=round(y, 4)) for b, c, a, e, y in rows],
          open(f"{OM}/zxaex3_kcurve.json", "w"))
print("rows:", len(rows))
