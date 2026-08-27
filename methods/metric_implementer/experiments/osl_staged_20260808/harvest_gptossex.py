"""gpt-oss-120b frontier exemplar-arm harvest (criterion 3, second frontier receiver).
Same conventions as harvest_glmex_final.py: reference per base = majority of
{llama70b,qwen25-72b}-dossier (soft>.5) + {glm-47,glm-52}-dossier votes, ties dropped,
>=3 votes required; exemplar-idx masking from freeze_zxa_ex_humor_v1.json "exemplars"
arm applied uniformly; balanced accuracy >=20 usable items, >=3 per class.
Sources: definition arm from mbar_zxagen_think_humor_gpt-oss-120b.npz (gen readout),
EX arms from mbar_zxagenex_think_humor_gpt-oss-120b.npz. Paired 20k bootstrap on
(exemplars - definition) and content gates (true - mm) per class. CPU-only.
"""
import json
import os

import numpy as np

B = "/lfs/skampere3/0/alexspan"
OM = f"{B}/outputs/osl_multi"
TASK = "humor"
EX_ARMS = ["exemplars", "def_exemplars", "exemplars_mm", "exemplars_authored",
           "exemplars_authored_mm"]


def load(path):
    if not os.path.exists(path):
        print(f"  [warn] missing {path}")
        return {}
    z = np.load(path, allow_pickle=True)
    names = [str(n) for n in z["names"]]
    return {n: z["m_bar"][i] for i, n in enumerate(names)}


v1p_70 = load(f"{OM}/mbar_zxa_{TASK}_llama70b.npz")
v1p_72 = load(f"{OM}/mbar_zxa_{TASK}_qwen25-72b.npz")
glm47 = load(f"{OM}/mbar_zxaglm_{TASK}_glm-47.npz")
glm52 = load(f"{OM}/mbar_zxaglm_{TASK}_glm-52.npz")
gen_def = load(f"{OM}/mbar_zxagen_think_{TASK}_gpt-oss-120b.npz")
gen_ex = load(f"{OM}/mbar_zxagenex_think_{TASK}_gpt-oss-120b.npz")

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
    mean = np.nanmean(np.stack(votes), 0)
    ref[b] = np.where(mean > 0.5, 1, np.where(mean < 0.5, 0, -1))


def balanced(pred, lab, mask):
    ok = (lab >= 0) & mask & np.isfinite(pred)
    if ok.sum() < 20:
        return None
    p = (pred[ok] > 0.5).astype(int)
    l = lab[ok]
    accs = [float(np.mean(p[l == c] == c)) for c in (0, 1) if (l == c).sum() >= 3]
    return round(float(np.mean(accs)), 4) if len(accs) == 2 else None


glm52_ex = load(f"{OM}/mbar_zxaglmex_{TASK}_glm-52.npz")
RECEIVERS = {
    "gpt-oss-120b": {"definition": gen_def, **{a: gen_ex for a in EX_ARMS}},
    "glm-52": {"definition": glm52, **{a: glm52_ex for a in EX_ARMS}},
}

rng = np.random.default_rng(0)
out = {"receivers": {}}
for rcv, sources in RECEIVERS.items():
    per_base = {}
    for b in bases:
        if b not in ref:
            continue
        mask = np.ones(N_PROBES, bool)
        for j in exidx.get(b, ()):
            mask[j] = False
        row = {}
        for arm, src in sources.items():
            r = src.get(f"{b}||{arm}")
            if r is not None:
                y = balanced(np.asarray(r, float), ref[b], mask)
                if y is not None:
                    row[arm] = y
        if row:
            per_base[b] = {"class": cls_of[b], "arms": row}

    def paired(arm_a, arm_b, cls=None):
        d = [per_base[b]["arms"][arm_a] - per_base[b]["arms"][arm_b] for b in per_base
             if (cls is None or per_base[b]["class"] == cls)
             and arm_a in per_base[b]["arms"] and arm_b in per_base[b]["arms"]]
        if len(d) < 4:
            return None
        d = np.array(d)
        idx = rng.integers(0, len(d), size=(20000, len(d)))
        bm = d[idx].mean(1)
        return {"n": len(d), "mean": round(float(d.mean()), 4),
                "ci": [round(float(np.percentile(bm, 2.5)), 4),
                       round(float(np.percentile(bm, 97.5)), 4)],
                "fracpos": round(float((d > 0).mean()), 3)}

    classes = sorted(set(v["class"] for v in per_base.values()))
    rout = {"per_base": per_base, "contrasts": {}}
    print(f"===== receiver {rcv}: n bases {len(per_base)} =====")
    for cls in classes + [None]:
        tag = cls or "ALL"
        subs = [b for b in per_base if cls is None or per_base[b]["class"] == cls]
        means = {}
        for arm in ["definition"] + EX_ARMS:
            vals = [per_base[b]["arms"][arm] for b in subs if arm in per_base[b]["arms"]]
            if vals:
                means[arm] = round(float(np.mean(vals)), 4)
        print(f"--- {tag} (n={len(subs)}) arm means: " +
              " ".join(f"{a}={v}" for a, v in means.items()))
        for ca, cb, lbl in (("exemplars", "definition", "ex-def"),
                            ("exemplars", "exemplars_mm", "gate_corpus"),
                            ("exemplars_authored", "exemplars_authored_mm", "gate_authored"),
                            ("def_exemplars", "definition", "additive")):
            r = paired(ca, cb, cls)
            if r:
                star = "*" if r["ci"][0] > 0 or r["ci"][1] < 0 else " "
                print(f"    {lbl:14s} n={r['n']:2d} {r['mean']:+.4f}{star} "
                      f"CI[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] pos={r['fracpos']}")
                rout["contrasts"][f"{tag}|{lbl}"] = r
    out["receivers"][rcv] = rout

json.dump(out, open(f"{OM}/frontier_limit_{TASK}_v2.json", "w"), indent=1)
print("DONE ->", f"{OM}/frontier_limit_{TASK}_v2.json")
