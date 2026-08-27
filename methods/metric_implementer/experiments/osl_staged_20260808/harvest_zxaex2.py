"""1c first read: score exemplar arms against the v1 frontier-dossier reference.

Reference per base = per-item majority of 4 frontier executors' dossier-arm verdicts
(llama70b/qwen25-72b soft panels binarized at .5; glm-47/glm-52 hard panels). Ties dropped.
y(base, arm, exec) = balanced agreement (mean of per-reference-class accuracies) with that
reference, masking each base's zxa.exemplar_idx items for its own arms (verbatim leakage) and
undetermined items. All local-panel readouts are logprob-based (deterministic, n_forms=1), so
comparing arms across runs carries no sampling noise; that is why v1 arms (name/definition/
dossier) and the new exemplar arms may share a table.
"""
import json
from collections import defaultdict

import numpy as np

B = "/lfs/skampere3/0/alexspan"
OM = f"{B}/outputs/osl_multi"
LOCALS = ["llama1b", "llama3b", "llama8b", "qwen25-3b", "qwen25-7b",
          "qwen25-14b", "qwen25-32b", "llama70b", "qwen25-72b"]
FRONTIER_SOFT = ["llama70b", "qwen25-72b"]
FRONTIER_HARD = ["glm-47", "glm-52"]
V1_ARMS = ["name", "definition", "explanation", "dossier", "dossier_mismatched",
           "definition_padded"]
EX_ARMS = ["exemplars", "def_exemplars", "exemplars_mm", "exemplars_authored",
           "exemplars_authored_mm"]
EX2_ARMS = ["exemplars_fmt", "exemplars_shuf"]


def load(path):
    z = np.load(path, allow_pickle=True)
    names = [str(n) for n in z["names"]]
    return {n: z["m_bar"][i] for i, n in enumerate(names)}

v1p = {ex: load(f"{OM}/mbar_zxa_humor_{ex}.npz") for ex in LOCALS}
glm = {ex: load(f"{OM}/mbar_zxaglm_humor_{ex}.npz") for ex in FRONTIER_HARD}
exp = {ex: load(f"{OM}/mbar_zxaex_humor_{ex}.npz") for ex in LOCALS}
import os
exp2 = {ex: load(f"{OM}/mbar_zxaex2_humor_{ex}.npz") for ex in LOCALS
        if os.path.exists(f"{OM}/mbar_zxaex2_humor_{ex}.npz")}
freeze = json.load(open(f"{OM}/freeze_zxa_ex_humor_v1.json"))
zmeta = {e["name"]: e["zxa"] for e in freeze["metrics"]}
bases = sorted({z_["base"] for z_ in zmeta.values()})
cls_of = {z_["base"]: z_["class"] for z_ in zmeta.values()}
exidx = {}
for e in freeze["metrics"]:
    if e["zxa"]["arm"] == "exemplars":
        exidx[e["zxa"]["base"]] = set(e["zxa"]["exemplar_idx"])

# also need classes for bases lacking new arms rows? bases all covered (authored arms).
ref, refstats = {}, {}
for b in bases:
    votes = []
    for ex in FRONTIER_SOFT:
        r = v1p[ex].get(f"{b}||dossier")
        if r is not None:
            votes.append((np.asarray(r, float) > 0.5).astype(float))
    for ex in FRONTIER_HARD:
        r = glm[ex].get(f"{b}||dossier")
        if r is not None:
            v = np.asarray(r, float)
            votes.append((v > 0.5).astype(float))
    if len(votes) < 3:
        continue
    V = np.stack(votes)
    mean = np.nanmean(V, 0)
    lab = np.where(mean > 0.5, 1, np.where(mean < 0.5, 0, -1))  # -1 = tie/undetermined
    ref[b] = lab
    refstats[b] = dict(n_voters=len(votes),
                       frontier_unanimity=float(np.mean((mean == 0) | (mean == 1))),
                       pos_rate=float(np.mean(lab == 1)), tie_rate=float(np.mean(lab == -1)))

def balanced(pred, lab, mask):
    ok = (lab >= 0) & mask & np.isfinite(pred)
    if ok.sum() < 20:
        return None
    p = (pred[ok] > 0.5).astype(int)
    l = lab[ok]
    accs = []
    for c in (0, 1):
        sel = l == c
        if sel.sum() >= 3:
            accs.append(float(np.mean(p[sel] == c)))
    return round(float(np.mean(accs)), 4) if len(accs) == 2 else None

rows = []
for b in bases:
    if b not in ref:
        continue
    lab = ref[b]
    mask_ex = np.ones(300, bool)
    for j in exidx.get(b, ()):  # mask corpus-exemplar items for ALL this base's arms (uniform)
        mask_ex[j] = False
    for exe in LOCALS:
        for arm in V1_ARMS:
            r = v1p[exe].get(f"{b}||{arm}")
            if r is not None:
                y = balanced(np.asarray(r, float), lab, mask_ex)
                if y is not None:
                    rows.append((b, cls_of[b], arm, exe, y))
        for arm in EX_ARMS:
            r = exp[exe].get(f"{b}||{arm}")
            if r is not None:
                y = balanced(np.asarray(r, float), lab, mask_ex)
                if y is not None:
                    rows.append((b, cls_of[b], arm, exe, y))
        for arm in EX2_ARMS:
            r = exp2.get(exe, {}).get(f"{b}||{arm}")
            if r is not None:
                y = balanced(np.asarray(r, float), lab, mask_ex)
                if y is not None:
                    rows.append((b, cls_of[b], arm, exe, y))

SMALL = {"llama1b", "llama3b", "llama8b", "qwen25-3b", "qwen25-7b"}
agg = defaultdict(list)
for b, c, arm, exe, y in rows:
    tier = "small" if exe in SMALL else "big"
    agg[(c, arm, tier)].append(y)
summary = {f"{c}|{arm}|{t}": [round(float(np.mean(v)), 4), len(v)]
           for (c, arm, t), v in sorted(agg.items())}
out = dict(refstats={b: refstats[b] for b in sorted(refstats)},
           rows=[dict(base=b, cls=c, arm=a, exec=e, y=y) for b, c, a, e, y in rows],
           summary=summary)
json.dump(out, open(f"{OM}/zxaex_read2.json", "w"), indent=1)
print("bases with reference:", len(ref), "/", len(bases))
print("rows:", len(rows))
for cls in ("REACHES-ANCHOR", "DIALECT-SUSPECT", "TACIT-CANDIDATE", "PLANTED"):
    print(f"== {cls} (big tier) ==")
    for arm in ["name", "definition", "dossier", "exemplars", "def_exemplars",
                "exemplars_mm", "exemplars_fmt", "exemplars_shuf",
                "exemplars_authored", "exemplars_authored_mm"]:
        v = summary.get(f"{cls}|{arm}|big")
        if v:
            print("  %-24s %.3f  (n=%d)" % (arm, v[0], v[1]))
