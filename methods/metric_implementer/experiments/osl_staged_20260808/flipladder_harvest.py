"""Flip-ladder harvest: is the functional (flip-selected exemplar) channel a CURVE
in receiver z or a point? Reads mbar_flipladder_{exec}.npz (functional rubrics, all
300 probes) + flipladder_mask_v1.json (exemplar indices) + mbar_zxa panels
(definition baselines + dossier reference votes). Protocol identical to
flip_functional_v2 holdout: stable-hash H split only, exemplar items masked,
balanced accuracy vs the objective's own reference, paired 20k bootstrap over bases.
CPU-only analysis; no new scoring.
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/lfs/skampere3/0/alexspan/norm-research")
from methods.metric_implementer.experiments.run_real_test import _load_texts
from methods.metric_implementer import config as cfgmod

B = "/lfs/skampere3/0/alexspan"
OM = f"{B}/outputs/osl_multi"
# news EXCLUDED (2026-08-08): the news z×a panel was scored on news_probes.jsonl
# (360 curated probes, ZERO text overlap with _load_texts) — any panel↔fresh join
# in news is item-wise unrelated noise. PLANTED alignment: humor .92, news .50.
TASKS = ["humor", "creative_writing", "math"]
LADDER = ["llama1b", "llama3b", "qwen25-3b", "mistral7b", "qwen25-7b", "llama8b",
          "phi4", "qwen25-14b", "gemma2-27b", "qwen25-32b", "llama70b", "qwen25-72b"]
NBOOT = 20000


def load_npz(p):
    if not os.path.exists(p):
        return {}
    z = np.load(p, allow_pickle=True)
    return {str(n): z["m_bar"][i] for i, n in enumerate([str(x) for x in z["names"]])}


def balanced(pred, lab, keep):
    ok = (lab >= 0) & keep & np.isfinite(pred)
    if ok.sum() < 12:
        return None
    p = (pred[ok] > .5).astype(int)
    l = lab[ok]
    accs = [float(np.mean(p[l == c] == c)) for c in (0, 1) if (l == c).sum() >= 3]
    return float(np.mean(accs)) if len(accs) == 2 else None


mask_map = json.load(open(f"{OM}/flipladder_mask_v1.json"))
lad = {ex: load_npz(f"{OM}/mbar_flipladder_{ex}.npz") for ex in LADDER}

# per-task context: holdout mask, refs, per-rung definition rows
ctx = {}
for TASK in TASKS:
    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), TASK.replace("_", "-"))
    texts, _ = _load_texts(TASK.replace("_", "-"), 360, cfg)
    probes = texts[60:360]
    n_items = 300

    def split_of(t):
        h = int(hashlib.md5(("split1cv3|" + t).encode()).hexdigest(), 16)
        return "H" if h % 2 == 1 else "AB"
    S_H = np.array([split_of(t) == "H" for t in probes[:n_items]])

    v1p = {ex: load_npz(f"{OM}/mbar_zxa_{TASK}_{ex}.npz") for ex in ["llama70b", "qwen25-72b"]}
    glm = {ex: load_npz(f"{OM}/mbar_zxaglm_{TASK}_{ex}.npz") for ex in ["glm-47", "glm-52"]}
    zxa = {ex: load_npz(f"{OM}/mbar_zxa_{TASK}_{ex}.npz") for ex in LADDER}

    def frontier_ref(base, v1p=v1p, glm=glm):
        # default-arg binding: without it every task's closure late-binds to the
        # LAST task's panels and non-news frontier rows silently vanish
        votes = []
        for ex in ["llama70b", "qwen25-72b"]:
            r = v1p[ex].get(f"{base}||dossier")
            if r is not None:
                votes.append((np.asarray(r, float) > .5).astype(float))
        for ex in ["glm-47", "glm-52"]:
            r = glm[ex].get(f"{base}||dossier")
            if r is not None:
                votes.append((np.asarray(r, float) > .5).astype(float))
        if len(votes) < 2:
            return None
        mean = np.stack(votes).mean(0)
        return np.where(mean > .5, 1, np.where(mean < .5, 0, -1))

    ctx[TASK] = {"S_H": S_H, "frontier_ref": frontier_ref,
                 "enc": v1p["qwen25-72b"], "zxa": zxa}

# collect paired rows: (task, base, sel, obj, ex) -> (y_def, y_fun)
rows = []
some_ex = next(ex for ex in LADDER if lad[ex])
for key in lad[some_ex]:
    TASK, rest = key.split("|", 1)
    if TASK not in ctx:          # excluded task (news) — ladder npz still carries its rows
        continue
    base, tag = rest.split("||")
    _, sel, obj = tag.rsplit("_", 2)          # functional_{sel}_{obj}
    sel = tag[len("functional_"):-len(obj) - 1]
    c = ctx[TASK]
    if obj == "frontier":
        ref = c["frontier_ref"](base)
    else:
        r = c["enc"].get(f"{base}||dossier")
        ref = (np.asarray(r, float) > .5).astype(int) if r is not None else None
    if ref is None:
        continue
    keep = c["S_H"].copy()
    for i in mask_map.get(key, []):
        keep[i] = False
    for ex in LADDER:
        fun = lad[ex].get(key)
        if fun is None:
            continue
        y_fun = balanced(np.asarray(fun, float), np.asarray(ref), keep)
        drow = c["zxa"][ex].get(f"{base}||definition")
        y_def = (balanced(np.asarray(drow, float), np.asarray(ref), keep)
                 if drow is not None else None)
        if y_fun is not None:
            rows.append({"task": TASK, "base": base, "sel": sel, "obj": obj,
                         "ex": ex, "y_fun": y_fun, "y_def": y_def})

# aggregate per (ex, sel, obj), pooled across tasks; paired bootstrap over bases
rng = np.random.default_rng(0)
out = {"rows_n": len(rows), "cells": []}
for sel in sorted(set(r["sel"] for r in rows)):
    for obj in ("frontier", "encoder"):
        for ex in LADDER:
            sub = [r for r in rows if r["sel"] == sel and r["obj"] == obj
                   and r["ex"] == ex and r["y_def"] is not None]
            lvl = [r for r in rows if r["sel"] == sel and r["obj"] == obj and r["ex"] == ex]
            if not lvl:
                continue
            cell = {"sel": sel, "obj": obj, "ex": ex,
                    "n_pairs": len(sub), "n_fun": len(lvl),
                    "fun_mean": round(float(np.mean([r["y_fun"] for r in lvl])), 4)}
            if len(sub) >= 6:
                d = np.array([r["y_fun"] - r["y_def"] for r in sub])
                cell["def_mean"] = round(float(np.mean([r["y_def"] for r in sub])), 4)
                cell["delta_mean"] = round(float(d.mean()), 4)
                idx = rng.integers(0, len(d), size=(NBOOT, len(d)))
                bm = d[idx].mean(1)
                cell["ci"] = [round(float(np.percentile(bm, 2.5)), 4),
                              round(float(np.percentile(bm, 97.5)), 4)]
                cell["frac_pos"] = round(float((d > 0).mean()), 3)
            out["cells"].append(cell)

# per-task pooled deltas at each rung (for the note's per-domain table)
out["per_task"] = []
for TASK in TASKS:
    for sel in sorted(set(r["sel"] for r in rows)):
        for obj in ("frontier", "encoder"):
            for ex in LADDER:
                sub = [r for r in rows if r["task"] == TASK and r["sel"] == sel
                       and r["obj"] == obj and r["ex"] == ex and r["y_def"] is not None]
                if len(sub) < 6:
                    continue
                d = np.array([r["y_fun"] - r["y_def"] for r in sub])
                idx = rng.integers(0, len(d), size=(NBOOT, len(d)))
                bm = d[idx].mean(1)
                out["per_task"].append({
                    "task": TASK, "sel": sel, "obj": obj, "ex": ex, "n": len(sub),
                    "delta_mean": round(float(d.mean()), 4),
                    "ci": [round(float(np.percentile(bm, 2.5)), 4),
                           round(float(np.percentile(bm, 97.5)), 4)]})

json.dump(out, open(f"{OM}/flipladder_curve_v2_nonews.json", "w"), indent=1)

print(f"rows={len(rows)}")
for c in out["cells"]:
    if "delta_mean" in c:
        star = "*" if c["ci"][0] > 0 or c["ci"][1] < 0 else " "
        print(f"{c['sel']:12s} {c['obj']:8s} {c['ex']:12s} n={c['n_pairs']:3d} "
              f"def={c['def_mean']:.3f} fun={c['fun_mean']:.3f} "
              f"d={c['delta_mean']:+.4f}{star} CI[{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}] "
              f"pos={c['frac_pos']:.2f}")
    else:
        print(f"{c['sel']:12s} {c['obj']:8s} {c['ex']:12s} n_fun={c['n_fun']:3d} "
              f"fun={c['fun_mean']:.3f} (no definition panel)")
print("DONE ->", f"{OM}/flipladder_curve_v2_nonews.json")
