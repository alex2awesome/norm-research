"""1c-v3b NULL-ON-ALL-BASES control (advisor-mandated 2026-08-06): the label-permuted
selection-null from flip_functional_v2 ran on every 3rd base only, making the null-vs-true
contrast cross-composition. This run executes ONLY the null objective (independent
md5-random labels per item) for EVERY base, same selector/protocol/theta/splits as v2, so
the null column pairs within-base against v2's frontier/encoder columns.
Holdout scored vs the TRUE frontier reference, as in v2. Same rngperm construction as v2
=> every-3rd bases replicate v2's null rows exactly (consistency check).
Usage: flip_functional_v2b.py <selection_executor> <out.json>
"""
import hashlib
import json
import os
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
O = f"{B}/outputs/osl"
SEL_EX = sys.argv[1]
OUT = sys.argv[2]
LOCAL_MID = ["llama1b", "llama3b", "llama8b", "llama70b", "qwen25-3b", "qwen25-7b",
             "qwen25-14b", "qwen25-72b", "mistral7b", "phi4", "gemma2-27b"]
TASKS = ["humor", "creative_writing", "peer_review", "math", "news_homepages"]
MAX_SET = 12
N_CAND = 24
ROUNDS, BATCH = 6, 8
ACCEPT_THETA = 0.01
EX_TRUNC = 500


def load_npz(p):
    if not os.path.exists(p):
        return {}
    z = np.load(p, allow_pickle=True)
    return {str(n): z["m_bar"][i] for i, n in enumerate([str(x) for x in z["names"]])}

executor = make_judge_backend(EXECUTORS[SEL_EX][0], cfgmod.ImplementerConfig(), temperature=None)
slate_all = json.load(open(f"{OM}/zxa_slate_v1.json")) + json.load(open(f"{OM}/news_slate_v1.json"))
results = {}

for TASK in TASKS:
    slate = [m for m in slate_all if m["task"] == TASK and m["class"] != "PLANTED"]
    if not slate:
        continue
    v1p = {ex: load_npz(f"{OM}/mbar_zxa_{TASK}_{ex}.npz") for ex in ["llama70b", "qwen25-72b"]}
    glm = {ex: load_npz(f"{OM}/mbar_zxaglm_{TASK}_{ex}.npz") for ex in ["glm-47", "glm-52"]}
    crowd = {}
    for ex in LOCAL_MID:
        d = {}
        if TASK == "humor":
            d.update(load_npz(f"{O}/mbar285_{ex}.npz"))
            d.update(load_npz(f"{OM}/mbar2_humor_sup_{ex}.npz"))
        else:
            d.update(load_npz(f"{OM}/mbar2_{TASK}_{ex}.npz"))
        crowd[ex] = d
    v2f = f"{OM}/freeze_{TASK}_v2.json"
    v2rub = ({m["name"]: m["rubric"] for m in json.load(open(v2f))["metrics"]}
             if os.path.exists(v2f) else {})
    for m in slate:
        v2rub.setdefault(m["name"], m.get("rubric") or m["name"])

    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), TASK.replace("_", "-"))
    texts, _ = _load_texts(TASK.replace("_", "-"), 360, cfg)
    probes = texts[60:360]
    MX = cfg.max_text_chars
    n_items = 300

    def split_of(t):
        h = int(hashlib.md5(("split1cv3|" + t).encode()).hexdigest(), 16)
        if h % 2 == 1:
            return "H"
        return "A" if int(hashlib.md5(("conf|" + t).encode()).hexdigest(), 16) % 2 == 0 else "B"
    S_A = np.array([split_of(t) == "A" for t in probes[:n_items]])
    S_B = np.array([split_of(t) == "B" for t in probes[:n_items]])
    S_H = np.array([split_of(t) == "H" for t in probes[:n_items]])
    IA, IB, IH = np.where(S_A)[0], np.where(S_B)[0], np.where(S_H)[0]

    def trunc(t):
        t = t.strip()
        return t if len(t) <= EX_TRUNC else t[:EX_TRUNC].rsplit(" ", 1)[0] + " ..."

    def rubric_for(base, items):
        pos = [i for i, l in items if l == 1]
        neg = [i for i, l in items if l == 0]
        r = base
        if pos:
            r += "\nExamples that satisfy this criterion:\n" + \
                 "\n".join("- " + trunc(probes[i]) for i in pos)
        if neg:
            r += "\nExamples that do NOT satisfy it:\n" + \
                 "\n".join("- " + trunc(probes[i]) for i in neg)
        return r

    def score_rubrics(rubrics, item_idx):
        prompts, spans = [], []
        for r in rubrics:
            s = len(prompts)
            for j in item_idx:
                prompts.append(ap._YESNO_TEXTFIRST.format(text=probes[j][:MX], rubric=r))
            spans.append((s, len(prompts)))
        flat = np.asarray(executor.score_binary(prompts), float)
        return np.stack([flat[a:b] for a, b in spans])

    def balanced(pred, lab, keep):
        ok = (lab >= 0) & keep & np.isfinite(pred)
        if ok.sum() < 12:
            return None
        p = (pred[ok] > .5).astype(int)
        l = lab[ok]
        accs = [float(np.mean(p[l == c] == c)) for c in (0, 1) if (l == c).sum() >= 3]
        return float(np.mean(accs)) if len(accs) == 2 else None

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
        if len(votes) < 2:
            return None
        mean = np.stack(votes).mean(0)
        return np.where(mean > .5, 1, np.where(mean < .5, 0, -1))

    task_res = {}
    for bi, m in enumerate(slate):
        base = m["name"]
        fr = frontier_ref(base)
        cons_rows = [crowd[ex].get(base) for ex in LOCAL_MID]
        cons_rows = [r for r in cons_rows if r is not None]
        if fr is None or len(cons_rows) < 7:
            continue
        fr = np.asarray(fr)
        cons = np.stack([(np.asarray(r, float) > .5).astype(float) for r in cons_rows]).mean(0)
        rngperm = np.array([int(hashlib.md5((base + "|null|" + str(j)).encode()
                                            ).hexdigest(), 16) % 2 for j in range(n_items)])
        cand = [j for j in IA if .3 < cons[j] < .7 and probes[j].strip()][:N_CAND]
        decis = [j for j in IA if probes[j].strip()]
        if not decis or not cand:
            continue
        seed = [(max(decis, key=lambda j: cons[j]), 1), (min(decis, key=lambda j: cons[j]), 0)]
        per_obj = {}
        for obj, lab in {"null": rngperm}.items():
            lab = np.asarray(lab)
            true_lab = fr
            S = list(seed)
            def ev(items, idx, l):
                keep = np.zeros(n_items, bool); keep[idx] = True
                for i, _ in items:
                    keep[i] = False
                y = balanced(score_rubrics([rubric_for(base, items)], idx)[0], l[idx], keep[idx])
                return -1.0 if y is None else y
            yA = ev(S, IA, lab)
            yB = ev(S, IB, lab)
            pool = [(j, l) for j in cand for l in (1, 0)]
            pool.sort(key=lambda t: hashlib.md5((base + obj + str(t)).encode()).hexdigest())
            for r0 in range(0, min(len(pool), ROUNDS * BATCH), BATCH):
                if len(S) >= MAX_SET:
                    break
                trials = [S + [c] for c in pool[r0:r0 + BATCH]
                          if not any(c[0] == i for i, _ in S)]
                if not trials:
                    continue
                mats = score_rubrics([rubric_for(base, t) for t in trials], IA)
                ys = []
                for t, mat in zip(trials, mats):
                    keep = np.zeros(n_items, bool); keep[IA] = True
                    for i, _ in t:
                        keep[i] = False
                    yv = balanced(mat, lab[IA], keep[IA])
                    ys.append(-1.0 if yv is None else yv)
                k = int(np.argmax(ys))
                if ys[k] > yA + ACCEPT_THETA:
                    yB_new = ev(trials[k], IB, lab)
                    if yB_new >= yB:
                        S, yA, yB = trials[k], ys[k], yB_new
            arms = {"name": base, "definition": v2rub[base],
                    "functional": rubric_for(base, S)}
            keep = np.zeros(n_items, bool); keep[IH] = True
            for i, _ in S:
                keep[i] = False
            mats = score_rubrics(list(arms.values()), IH)
            yh = {a: balanced(mt, true_lab[IH], keep[IH]) for a, mt in zip(arms, mats)}
            per_obj[obj] = {
                "selected": [[int(i), int(l)] for i, l in S],
                "n_flips_from_crowd": sum(1 for i, l in S if (cons[i] > .5) != bool(l)),
                "yA": round(yA, 4), "yB": round(yB, 4),
                "holdout": {a: (round(v, 4) if v is not None else None)
                            for a, v in yh.items()}}
            print(f"[{SEL_EX}|{TASK}] {base[:36]:36s} {obj:8s} "
                  + " ".join(f"{a}={v}" for a, v in per_obj[obj]["holdout"].items()), flush=True)
        if per_obj:
            task_res[base] = {"class": m["class"], "objectives": per_obj}
    results[TASK] = task_res
    json.dump({"selection_executor": SEL_EX, "protocol": "NULL-ALL-BASES (v2b)",
               "theta": ACCEPT_THETA, "max_set": MAX_SET, "results": results},
              open(OUT, "w"), indent=1)

print("DONE ->", OUT)
