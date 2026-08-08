"""1c-v3: FUNCTIONAL exemplar selection via flip-influence (user design 2026-08-06).

For each contested (TACIT-CANDIDATE) humor base: search over (item, label) exemplar
assignments — crowd-ambivalent items, BOTH polarities — accepting what improves TRAIN-half
transmission; report on the HOLDOUT half only. Two objectives per base:
  ref=frontier  per-item majority of 4 frontier executors' dossier verdicts (ties dropped)
  ref=encoder   qwen25-72b's own dossier-arm verdicts (the literal m_omega design)
Usage: flip_functional.py <selection_executor> <out.json>
Cost model: every round scores all live rubrics x train items in ONE score_binary batch.
"""
import hashlib
import json
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
LEN_CAP = 400
MAX_SET = 8
ACCEPT_THETA = 0.005


def load_npz(p):
    z = np.load(p, allow_pickle=True)
    return {str(n): z["m_bar"][i] for i, n in enumerate([str(x) for x in z["names"]])}

v1p = {ex: load_npz(f"{OM}/mbar_zxa_humor_{ex}.npz") for ex in ["llama70b", "qwen25-72b"]}
glm = {ex: load_npz(f"{OM}/mbar_zxaglm_humor_{ex}.npz") for ex in ["glm-47", "glm-52"]}
panels = {ex: load_npz(f"{O}/mbar285_{ex}.npz") for ex in LOCAL_MID}
v2 = json.load(open(f"{OM}/freeze_humor_v2.json"))
v2rub = {m["name"]: m["rubric"] for m in v2["metrics"]}
exv1 = json.load(open(f"{OM}/freeze_zxa_ex_humor_v1.json"))
slate = json.load(open(f"{OM}/zxa_slate_v1.json"))
bases = [m["name"] for m in slate if m["task"] == "humor" and m["class"] == "TACIT-CANDIDATE"]

cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), "humor")
texts, _ = _load_texts("humor", 360, cfg)
probes = texts[60:360]

# stable hash split (never seeded-shuffle): md5(text) parity
is_train = np.array([int(hashlib.md5(("split1cv3|" + t).encode()).hexdigest(), 16) % 2 == 0
                     for t in probes])
TR = np.where(is_train)[0]
HO = np.where(~is_train)[0]

def refs_for(base):
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
    frontier = np.where(mean > .5, 1, np.where(mean < .5, 0, -1))
    enc = v1p["qwen25-72b"].get(f"{base}||dossier")
    encoder = (np.asarray(enc, float) > .5).astype(int) if enc is not None else None
    return {"frontier": frontier, "encoder": encoder}

def rubric_for(base, items):  # items: list of (idx, label) label in {1,0}
    pos = [i for i, l in items if l == 1]
    neg = [i for i, l in items if l == 0]
    r = base
    if pos:
        r += "\nExamples that satisfy this criterion:\n" + \
             "\n".join("- " + probes[i].strip() for i in pos)
    if neg:
        r += "\nExamples that do NOT satisfy it:\n" + \
             "\n".join("- " + probes[i].strip() for i in neg)
    return r

executor = make_judge_backend(EXECUTORS[SEL_EX][0], cfgmod.ImplementerConfig(), temperature=None)
MX = cfg.max_text_chars

def score_rubrics(rubrics, item_idx):
    """rubrics: list of strings; returns matrix [n_rubrics, n_items] of P(YES)."""
    prompts, spans = [], []
    for r in rubrics:
        s = len(prompts)
        for j in item_idx:
            prompts.append(ap._YESNO_TEXTFIRST.format(text=probes[j][:MX], rubric=r))
        spans.append((s, len(prompts)))
    flat = np.asarray(executor.score_binary(prompts), float)
    return np.stack([flat[a:b] for a, b in spans])

def balanced(pred, lab, mask_ok):
    ok = (lab >= 0) & mask_ok & np.isfinite(pred)
    if ok.sum() < 15:
        return None
    p = (pred[ok] > .5).astype(int)
    l = lab[ok]
    accs = [float(np.mean(p[l == c] == c)) for c in (0, 1) if (l == c).sum() >= 3]
    return float(np.mean(accs)) if len(accs) == 2 else None

results = {}
for base in bases:
    refs = refs_for(base)
    if base not in panels[LOCAL_MID[0]]:
        continue
    cons = np.stack([(panels[ex][base] > .5).astype(float) for ex in LOCAL_MID]).mean(0)
    cand = [j for j in TR if .3 < cons[j] < .7 and len(probes[j]) <= LEN_CAP][:30]
    # seed: 1+1 most-decisive train items (anchors the format)
    dec = [j for j in TR if len(probes[j]) <= LEN_CAP]
    seed = [(max(dec, key=lambda j: cons[j]), 1), (min(dec, key=lambda j: cons[j]), 0)]
    per_obj = {}
    for obj, lab in refs.items():
        if lab is None:
            continue
        lab = np.asarray(lab)
        def ev_train(items):
            mask = np.zeros(300, bool); mask[TR] = True
            for i, _ in items: mask[i] = False
            y = balanced(score_rubrics([rubric_for(base, items)], TR)[0],
                         lab[TR], mask[TR])
            return -1.0 if y is None else y
        S = list(seed)
        y_cur = ev_train(S)
        pool = [(j, l) for j in cand for l in (1, 0) if (j, 1) not in S and (j, 0) not in S]
        rng_order = sorted(pool, key=lambda t: hashlib.md5((base + str(t)).encode()).hexdigest())
        for r0 in range(0, min(len(rng_order), 24), 8):   # 3 batches of 8 proposals
            batch = rng_order[r0:r0 + 8]
            trials = [S + [c] for c in batch
                      if not any(c[0] == i for i, _ in S)]
            if not trials or len(S) >= MAX_SET:
                break
            mats = score_rubrics([rubric_for(base, t) for t in trials], TR)
            ys = []
            for t, m in zip(trials, mats):
                mask = np.zeros(300, bool); mask[TR] = True
                for i, _ in t: mask[i] = False
                yv = balanced(m, lab[TR], mask[TR])
                ys.append(-1.0 if yv is None else yv)
            k = int(np.argmax(ys))
            if ys[k] > y_cur + ACCEPT_THETA:
                S = trials[k]; y_cur = ys[k]
        # holdout readout: name / definition / crowd-ex (from exv1 if exists) / functional
        arms = {"name": base, "definition": v2rub.get(base, base),
                "functional": rubric_for(base, S)}
        crowd_e = next((e for e in exv1["metrics"]
                        if e["zxa"]["base"] == base and e["zxa"]["arm"] == "exemplars"), None)
        if crowd_e:
            arms["crowd_ex"] = crowd_e["rubric"]
        mask = np.zeros(300, bool); mask[HO] = True
        for i, _ in S: mask[i] = False
        mats = score_rubrics(list(arms.values()), HO)
        yh = {a: balanced(m, lab[HO], mask[HO]) for a, m in zip(arms, mats)}
        per_obj[obj] = {"selected": [[int(i), int(l)] for i, l in S],
                        "n_flips_from_crowd": sum(1 for i, l in S
                                                  if (cons[i] > .5) != bool(l)),
                        "y_train_final": round(y_cur, 4),
                        "holdout": {a: (round(v, 4) if v is not None else None)
                                    for a, v in yh.items()}}
        print(f"[{SEL_EX}] {base[:40]:40s} {obj:8s} "
              f"hold: " + " ".join(f"{a}={v}" for a, v in per_obj[obj]["holdout"].items()),
              flush=True)
    results[base] = per_obj

json.dump({"selection_executor": SEL_EX, "theta": ACCEPT_THETA, "max_set": MAX_SET,
           "split": "md5 parity split1cv3", "results": results},
          open(OUT, "w"), indent=1)
print("DONE", len(results), "bases ->", OUT)
