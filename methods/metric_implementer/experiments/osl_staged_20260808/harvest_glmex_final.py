"""Final harvest: glm-5.2 exemplar-arm content-transmission table (job 2026-08-07: "do examples
become useful beyond definitions in the limit of model capacity", frontier receiver = glm-5.2).

Same frontier-dossier-reference convention as outputs/osl_multi/harvest_zxaex_all.py: reference
per base = majority vote of {llama70b, qwen25-72b}-dossier (soft, binarized>.5) +
{glm-47, glm-52}-dossier (hard) from the EXISTING mbar_zxa_/mbar_zxaglm_ npz files; ties (-1)
dropped. Masking = the freeze's "exemplars" arm exemplar_idx per base (the only arm with
verbatim probe-corpus leakage; exemplars_authored/_mm use dossier-extracted, non-corpus examples
so their own exemplar_idx is always [] and needs no separate mask), applied uniformly to every
arm scored for that base.

glm-5.2's EX_ARMS (exemplars_authored/_mm, exemplars/_mm, def_exemplars) come from the NEW
mbar_zxaglmex_<task>_glm-52.npz produced by zxa_glm_ex.py (this job); its V1_ARMS (definition
etc.) come from the pre-existing mbar_zxaglm_<task>_glm-52.npz. qwen25-32b's numbers come from
the pre-existing mbar_zxa_<task>_qwen25-32b.npz (V1_ARMS) + mbar_zxaex_<task>_qwen25-32b.npz
(EX_ARMS) -- no new scoring for qwen, side-by-side comparison only.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

B = "/lfs/skampere3/0/alexspan"
OM = f"{B}/outputs/osl_multi"
TASK = sys.argv[1] if len(sys.argv) > 1 else "humor"

FRONTIER_SOFT = ["llama70b", "qwen25-72b"]
FRONTIER_HARD = ["glm-47", "glm-52"]
V1_ARMS = ["name", "definition", "explanation", "dossier", "dossier_mismatched",
          "definition_padded"]
EX_ARMS = ["exemplars", "def_exemplars", "exemplars_mm", "exemplars_authored",
          "exemplars_authored_mm"]


def load(path):
    if not os.path.exists(path):
        print(f"  [warn] missing {path}", file=sys.stderr)
        return {}
    z = np.load(path, allow_pickle=True)
    names = [str(n) for n in z["names"]]
    return {n: z["m_bar"][i] for i, n in enumerate(names)}


v1p_q32 = load(f"{OM}/mbar_zxa_{TASK}_qwen25-32b.npz")
exp_q32 = load(f"{OM}/mbar_zxaex_{TASK}_qwen25-32b.npz")
v1p_70 = load(f"{OM}/mbar_zxa_{TASK}_llama70b.npz")
v1p_72 = load(f"{OM}/mbar_zxa_{TASK}_qwen25-72b.npz")
glm47 = load(f"{OM}/mbar_zxaglm_{TASK}_glm-47.npz")
glm52 = load(f"{OM}/mbar_zxaglm_{TASK}_glm-52.npz")
glm52_ex = load(f"{OM}/mbar_zxaglmex_{TASK}_glm-52.npz")

freeze = json.load(open(f"{OM}/freeze_zxa_ex_{TASK}_v1.json"))
zmeta = {e["name"]: e["zxa"] for e in freeze["metrics"]}
bases = sorted({z_["base"] for z_ in zmeta.values()})
cls_of = {z_["base"]: z_["class"] for z_ in zmeta.values()}
exidx = {}
for e in freeze["metrics"]:
    if e["zxa"]["arm"] == "exemplars":
        exidx[e["zxa"]["base"]] = set(e["zxa"]["exemplar_idx"])

N_PROBES = 300
ref = {}
for b in bases:
    votes = []
    for src in (v1p_70, v1p_72):
        r = src.get(f"{b}||dossier")
        if r is not None:
            votes.append((np.asarray(r, float) > 0.5).astype(float))
    for src in (glm47, glm52):
        r = src.get(f"{b}||dossier")
        if r is not None:
            votes.append((np.asarray(r, float) > 0.5).astype(float))
    if len(votes) < 3:
        continue
    V = np.stack(votes)
    mean = np.nanmean(V, 0)
    lab = np.where(mean > 0.5, 1, np.where(mean < 0.5, 0, -1))
    ref[b] = lab


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


def score_all(sources_by_arm):
    rows = []
    for b in bases:
        if b not in ref:
            continue
        lab = ref[b]
        mask_ex = np.ones(N_PROBES, bool)
        for j in exidx.get(b, ()):
            mask_ex[j] = False
        for arm, src in sources_by_arm.items():
            r = src.get(f"{b}||{arm}")
            if r is not None:
                y = balanced(np.asarray(r, float), lab, mask_ex)
                if y is not None:
                    rows.append((b, cls_of[b], arm, y))
    return rows


glm_sources = {a: glm52 for a in V1_ARMS}
glm_sources.update({a: glm52_ex for a in EX_ARMS})
glm_rows = score_all(glm_sources)

q32_sources = {a: v1p_q32 for a in V1_ARMS}
q32_sources.update({a: exp_q32 for a in EX_ARMS})
q32_rows = score_all(q32_sources)


def agg_by(rows):
    agg = defaultdict(list)
    for b, c, a, y in rows:
        agg[(c, a)].append(y)
    return {k: (float(np.mean(v)), len(v)) for k, v in agg.items()}


glm_agg = agg_by(glm_rows)
q32_agg = agg_by(q32_rows)

CLASSES = sorted({c for _, c in glm_agg} | {c for _, c in q32_agg})
print(f"=== TASK={TASK}: glm-5.2 vs qwen25-32b, per-class balanced agreement w/ "
     f"frontier-dossier reference (bases with reference: {len(ref)}/{len(bases)}) ===")


def fmt(x):
    return f"{x[0]:.3f}(n={x[1]})" if x else "--"


for cls in CLASSES:
    print(f"\n-- class={cls} --")
    print(f"  {'exec':11s} {'definition':13s} {'auth_true':13s} {'auth_mm':13s} "
         f"{'auth_Δ(t-mm)':13s} {'bare_true':13s} {'bare_mm':13s} {'bare_Δ(t-mm)':13s}")
    for label, agg in (("glm-5.2", glm_agg), ("qwen25-32b", q32_agg)):
        d = agg.get((cls, "definition"))
        at = agg.get((cls, "exemplars_authored"))
        am = agg.get((cls, "exemplars_authored_mm"))
        bt = agg.get((cls, "exemplars"))
        bm = agg.get((cls, "exemplars_mm"))
        da = (at[0] - am[0]) if (at and am) else None
        db = (bt[0] - bm[0]) if (bt and bm) else None
        print(f"  {label:11s} {fmt(d):13s} {fmt(at):13s} {fmt(am):13s} "
             f"{(f'{da:+.3f}' if da is not None else '--'):13s} "
             f"{fmt(bt):13s} {fmt(bm):13s} "
             f"{(f'{db:+.3f}' if db is not None else '--'):13s}")

print(f"\n=== classes where ANY exemplar arm >= definition (glm-5.2, TASK={TASK}) ===")
for cls in CLASSES:
    d = glm_agg.get((cls, "definition"))
    if not d:
        continue
    hits = []
    for arm in EX_ARMS:
        v = glm_agg.get((cls, arm))
        if v and v[0] >= d[0]:
            hits.append((arm, v[0]))
    if hits:
        hits.sort(key=lambda x: -x[1])
        print(f"  {cls}: " + ", ".join(f"{a}={v:.3f}" for a, v in hits)
             + f"  [definition={d[0]:.3f}]")
    else:
        print(f"  {cls}: none (definition={d[0]:.3f})")

json.dump({"glm_agg": {f"{c}|{a}": v for (c, a), v in glm_agg.items()},
          "q32_agg": {f"{c}|{a}": v for (c, a), v in q32_agg.items()},
          "n_bases_with_ref": len(ref), "n_bases_total": len(bases)},
         open(f"{OM}/zxaglmex_harvest_{TASK}.json", "w"), indent=1)
print(f"\n-> {OM}/zxaglmex_harvest_{TASK}.json")
