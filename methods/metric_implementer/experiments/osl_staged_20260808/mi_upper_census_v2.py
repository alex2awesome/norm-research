"""MI upper-bound census v2 (task #29): extend v1 beyond z×a V1 arms to
(1) exemplar arms (mbar_zxaex_{task}_{ex} + gpt-oss/glm ex panels),
(2) gen-readout panels (mbar_zxagen_{mode}_{task}_{ex}, mbar_zxagenex_*),
(3) flip-selected functional rows (mbar_flipladder_{ex}, exemplar-idx masked).
Same conventions as v1: upper bound = H(pred); MI vs 4-voter frontier ref;
efficiency = MI/min(H_pred,H_ref); degenerate = frac_yes <=.02 or >=.98.
Aligned tasks only for panel<->fresh joins (news EXCLUDED here entirely — the
gen/flip rows are fresh-probe scored and the news ref universe differs).
Exemplar-arm rows mask that base's freeze exemplar_idx; flip rows mask
flipladder_mask_v1 indices. CPU-only. Output mi_upper_census_v2.json.
"""
import glob
import json
import math
import os
import re

import numpy as np

B = "/lfs/skampere3/0/alexspan"
OM = f"{B}/outputs/osl_multi"
TASKS = ["humor", "creative_writing", "math", "peer_review"]
DEG_LO, DEG_HI = 0.02, 0.98
N_PROBES = 300


def load(p):
    if not os.path.exists(p):
        return {}
    z = np.load(p, allow_pickle=True)
    return {str(n): z["m_bar"][i] for i, n in enumerate([str(x) for x in z["names"]])}


def h_bin(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def mi_bits(x, y):
    mi = 0.0
    px1 = float(np.mean(x))
    py1 = float(np.mean(y))
    for a in (0, 1):
        for b_ in (0, 1):
            pab = float(np.mean((x == a) & (y == b_)))
            pa = px1 if a else 1 - px1
            pb = py1 if b_ else 1 - py1
            if pab > 0 and pa > 0 and pb > 0:
                mi += pab * math.log2(pab / (pa * pb))
    return max(mi, 0.0), h_bin(px1), h_bin(py1)


# refs + masks per task
refs, exmask, flipmask = {}, {}, json.load(open(f"{OM}/flipladder_mask_v1.json"))
for TASK in TASKS:
    v70 = load(f"{OM}/mbar_zxa_{TASK}_llama70b.npz")
    v72 = load(f"{OM}/mbar_zxa_{TASK}_qwen25-72b.npz")
    g47 = load(f"{OM}/mbar_zxaglm_{TASK}_glm-47.npz")
    g52 = load(f"{OM}/mbar_zxaglm_{TASK}_glm-52.npz")
    bases = sorted({k.split("||")[0] for d in (v70, v72, g47, g52) for k in d})
    refs[TASK] = {}
    for b in bases:
        votes = []
        for d in (v70, v72, g47, g52):
            r = d.get(f"{b}||dossier")
            if r is not None:
                votes.append((np.asarray(r, float) > .5).astype(float))
        if len(votes) < 2:
            continue
        mean = np.stack(votes).mean(0)
        refs[TASK][b] = np.where(mean > .5, 1, np.where(mean < .5, 0, -1))
    fz = f"{OM}/freeze_zxa_ex_{TASK}_v1.json"
    exmask[TASK] = {}
    if os.path.exists(fz):
        for e in json.load(open(fz))["metrics"]:
            if e["zxa"]["arm"] == "exemplars":
                exmask[TASK][e["zxa"]["base"]] = set(e["zxa"]["exemplar_idx"])


def row_stats(r, task, base, mask_idx=()):
    r = np.asarray(r, float)
    keep = np.ones(min(len(r), N_PROBES), bool)
    for j in mask_idx:
        if j < len(keep):
            keep[j] = False
    r = r[:len(keep)]
    fin = np.isfinite(r) & keep
    if fin.sum() < 30:
        return None
    pred = (r[fin] > .5).astype(int)
    fy = float(pred.mean())
    rec = {"frac_yes": round(fy, 4), "H_pred": round(h_bin(fy), 4),
           "degenerate": bool(fy <= DEG_LO or fy >= DEG_HI)}
    lab = refs[task].get(base)
    if lab is not None:
        lab = np.asarray(lab)[:len(keep)]
        ok = fin & (lab >= 0)
        if ok.sum() >= 30:
            mi, hx, hy = mi_bits((r[ok] > .5).astype(int), lab[ok].astype(int))
            cap = min(hx, hy)
            rec.update({"MI": round(mi, 4),
                        "eff": round(mi / cap, 4) if cap > 1e-9 else None})
    return rec


rows_out = []

# (1) exemplar-arm panels: local zxaex + glm ex + gpt-oss genex
for TASK in TASKS:
    srcs = {}
    for p in glob.glob(f"{OM}/mbar_zxaex_{TASK}_*.npz"):
        srcs["zxaex|" + p.split("_")[-1][:-4]] = load(p)
    g = load(f"{OM}/mbar_zxaglmex_{TASK}_glm-52.npz")
    if g:
        srcs["glmex|glm-52"] = g
    g = load(f"{OM}/mbar_zxagenex_think_{TASK}_gpt-oss-120b.npz")
    if g:
        srcs["genex|gpt-oss-120b"] = g
    for tag, d in srcs.items():
        fam, ex = tag.split("|")
        for key, row in d.items():
            b, arm = key.rsplit("||", 1)
            rec = row_stats(row, TASK, b, exmask[TASK].get(b, ()))
            if rec:
                rec.update({"task": TASK, "ex": ex, "arm": arm, "base": b,
                            "family": fam})
                rows_out.append(rec)

# (2) gen-readout definition panels (both modes)
for p in glob.glob(f"{OM}/mbar_zxagen_*.npz"):
    m = re.match(r".*/mbar_zxagen_(nothink|think)_(.+)_([^_]+)\.npz$", p)
    if not m:
        continue
    mode, TASK, ex = m.groups()
    if TASK not in TASKS:
        continue
    d = load(p)
    for key, row in d.items():
        b, arm = key.rsplit("||", 1)
        rec = row_stats(row, TASK, b)
        if rec:
            rec.update({"task": TASK, "ex": ex, "arm": f"{arm}@{mode}", "base": b,
                        "family": "gen"})
            rows_out.append(rec)

# (3) flip-selected functional rows (aligned tasks only)
for p in glob.glob(f"{OM}/mbar_flipladder_*.npz"):
    ex = p.split("_")[-1][:-4]
    d = load(p)
    for key, row in d.items():
        TASK, rest = key.split("|", 1)
        if TASK not in TASKS:
            continue
        b, tag = rest.split("||")
        rec = row_stats(row, TASK, b, flipmask.get(key, ()))
        if rec:
            rec.update({"task": TASK, "ex": ex, "arm": "functional", "base": b,
                        "family": "flip", "sel_obj": tag})
            rows_out.append(rec)

json.dump({"deg_lo": DEG_LO, "deg_hi": DEG_HI, "rows": rows_out},
          open(f"{OM}/mi_upper_census_v2.json", "w"))
print(f"rows={len(rows_out)}")


def summarize(label, sub):
    if not sub:
        return
    mis = [r["MI"] for r in sub if "MI" in r]
    effs = [r["eff"] for r in sub if r.get("eff") is not None]
    print(f"{label:42s} n={len(sub):5d} deg={100*np.mean([r['degenerate'] for r in sub]):5.1f}% "
          f"medH={np.median([r['H_pred'] for r in sub]):.3f} "
          f"medMI={(np.median(mis) if mis else float('nan')):.3f} "
          f"maxMI={(np.max(mis) if mis else float('nan')):.3f} "
          f"medEff={(np.median(effs) if effs else float('nan')):.3f}")


print("\n=== exemplar channels (pooled tasks/receivers) ===")
for arm in ["exemplars", "def_exemplars", "exemplars_mm", "exemplars_authored",
            "exemplars_authored_mm"]:
    summarize(f"ex-arm {arm}", [r for r in rows_out if r["arm"] == arm])

print("\n=== flip functional rows per receiver ===")
fl = [r for r in rows_out if r["family"] == "flip"]
for ex in sorted({r["ex"] for r in fl}):
    summarize(f"flip {ex}", [r for r in fl if r["ex"] == ex])

print("\n=== gen panels: definition arm, think vs nothink per receiver ===")
gn = [r for r in rows_out if r["family"] == "gen"]
for ex in sorted({r["ex"] for r in gn}):
    for mode in ("nothink", "think"):
        summarize(f"gen {ex} @{mode}",
                  [r for r in gn if r["ex"] == ex and r["arm"].endswith("@" + mode)])

print("\nDONE ->", f"{OM}/mi_upper_census_v2.json")
