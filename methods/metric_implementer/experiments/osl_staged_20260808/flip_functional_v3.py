"""1c-v3 FULL-BANK SCALE-UP (user 2026-08-07): functional exemplar selection scaled from the
~31/41-base curated humor slate (flip_functional_v2.py) to the FULL humor metric bank (284
bank-kind bases; the 5 PLANTED-* length/format/digit/quote sanity-control metrics are excluded,
same exclusion v2 applied via `class != "PLANTED"` on the curated slate -- here applied by name
prefix since full-bank bases are enumerated straight from the crowd-panel npz files, not from a
class-labeled slate). llama70b-family selector fleet.

PREREG NOTE (reference change, 2026-08-07): full-bank bases have no dossier-based frontier
reference -- v2's 4-voter "frontier" objective was built from mbar_zxa_*/mbar_zxaglm_* dossier
scoring runs that only ever covered the curated 41-base humor slate, never the full ~285-metric
bank. So the transmission reference here is NOT v2's frontier: it is a 2-voter majority built
from the SAME full-bank crowd-panel npz files used for the candidate-ambivalence signal
(mbar285_<ex>.npz + mbar2_humor_sup_<ex>.npz), using ONLY the llama70b and qwen25-72b executors'
per-item verdicts (both are members of LOCAL_MID, so no extra npz families are loaded). Items
where the two voters disagree get label -1 (dropped from scoring, not imputed toward either
class). Objective key: ref="frontier2v". The full 11-executor LOCAL_MID mean crowd-consensus
(`cons`) is retained UNCHANGED as the candidate-ambivalence signal that picks the seed pair and
the .3<cons<.7 ambivalent candidate pool -- exactly as in v2 -- it is never used as ground truth.
There is no "encoder" objective in v3 (that arm was single-executor qwen25-72b dossier scoring,
which does not exist for full-bank bases); every base gets "frontier2v" ALWAYS + "null" ALWAYS
(v2 ran its label-permuted null on every 3rd base only, as a cross-composition leak-detector
subsample; v2b already established running null on every base as the right default, so v3 folds
that in rather than sub-sampling -- the null objective is still scored against the TRUE
frontier2v holdout labels, same as v2/v2b).

Leakage protocol (three-way stable-hash split, no seeded shuffles) -- UNCHANGED from v2:
  train-A  -- selection: greedy accepts need delta_A >= theta
  train-B  -- confirmation: accepted candidate must ALSO not degrade B (delta_B >= 0)
  holdout  -- touched exactly once, at the end, for the reported numbers
Exemplar items masked from all scoring; long-text tasks truncate exemplars to 500 chars.
Greedy (item,label) selection with theta=0.01 acceptance + B-confirmation, MAX_SET 12,
balanced-agreement scoring via ap._YESNO_TEXTFIRST + score_binary -- all UNCHANGED from v2.

Resumability: dumps partial results every 20 completed bases (v2 only dumped once per task,
i.e. once total here since v3 is humor-only) to the --out path; on startup, if --out already
exists, it is loaded and bases already present in results["humor"] are skipped.

Dry run: set env FLIP_V3_DRY=1 to print base count + resume state + split sizes + a per-base
probe for the first 3 bases, then sys.exit(0) BEFORE any judge backend / engine is created (no
GPU touched -- safe to run directly on a CPU-only shell).

Usage: flip_functional_v3.py <selection_executor> <out.json>
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
TASK = "humor"
MAX_SET = 12
N_CAND = 24            # items (x2 labels = 48 proposals)
ROUNDS, BATCH = 6, 8
ACCEPT_THETA = 0.01
EX_TRUNC = 500
DUMP_EVERY = 20         # bases between partial-results dumps


def load_npz(p):
    if not os.path.exists(p):
        return {}
    z = np.load(p, allow_pickle=True)
    return {str(n): z["m_bar"][i] for i, n in enumerate([str(x) for x in z["names"]])}


# full-bank crowd panels, merged per LOCAL_MID executor (mbar285 + mbar2_humor_sup) -- same
# merge v2 used for the humor task's ambivalence signal; here it is ALSO the frontier2v source
# (llama70b + qwen25-72b rows of this same dict).
crowd = {}
for ex in LOCAL_MID:
    d = {}
    d.update(load_npz(f"{O}/mbar285_{ex}.npz"))
    d.update(load_npz(f"{OM}/mbar2_humor_sup_{ex}.npz"))
    crowd[ex] = d

# base list = every metric name present for >= 9/11 LOCAL_MID executors in the full-bank crowd
# panels, PLANTED-* sanity controls excluded (parity with v2's `class != "PLANTED"` filter).
_name_counts = {}
for ex in LOCAL_MID:
    for name in crowd[ex]:
        _name_counts[name] = _name_counts.get(name, 0) + 1
base_list = sorted(n for n, c in _name_counts.items() if c >= 9 and not n.startswith("PLANTED-"))

v2f = f"{OM}/freeze_{TASK}_v2.json"
v2rub = ({m["name"]: m["rubric"] for m in json.load(open(v2f))["metrics"]}
         if os.path.exists(v2f) else {})
for base in base_list:
    v2rub.setdefault(base, base)

# provenance-only "class" tag: reuse the curated-slate class where this base was already in
# v2's 41-base humor slate, else BANK-FULL for the newly-included full-bank names. Metadata
# only -- never read by the selection/scoring logic below.
_slate_class = {}
_slate_path = f"{OM}/zxa_slate_v1.json"
if os.path.exists(_slate_path):
    for m in json.load(open(_slate_path)):
        if m["task"] == TASK:
            _slate_class[m["name"]] = m["class"]

cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), TASK.replace("_", "-"))
texts, _ = _load_texts(TASK.replace("_", "-"), 360, cfg)
probes = texts[60:360]
MX = cfg.max_text_chars
n_items = 300


def split_of(t):
    h = int(hashlib.md5(("split1cv3|" + t).encode()).hexdigest(), 16)
    if h % 2 == 1:
        return "H"                      # holdout (same as pilot's parity split)
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


def frontier2v_ref(base):
    """2-voter majority (llama70b, qwen25-72b) from the full-bank crowd panels. -1 = disagree."""
    v0 = crowd["llama70b"].get(base)
    v1 = crowd["qwen25-72b"].get(base)
    if v0 is None or v1 is None:
        return None
    v0 = (np.asarray(v0, float) > .5).astype(int)
    v1 = (np.asarray(v1, float) > .5).astype(int)
    return np.where(v0 == v1, v0, -1)


def dump(task_res):
    json.dump({"selection_executor": SEL_EX, "protocol": "A-select/B-confirm/H-report (v3 full-bank)",
               "ref": "frontier2v", "theta": ACCEPT_THETA, "max_set": MAX_SET,
               "results": {TASK: task_res}}, open(OUT, "w"), indent=1)


task_res = {}
if os.path.exists(OUT):
    try:
        prior = json.load(open(OUT))
        task_res = prior.get("results", {}).get(TASK, {})
    except (json.JSONDecodeError, OSError):
        task_res = {}
done = set(task_res)
todo = [b for b in base_list if b not in done]

if os.environ.get("FLIP_V3_DRY"):
    print(f"[FLIP_V3_DRY] base_count={len(base_list)} already_done={len(done)} todo={len(todo)}")
    print(f"[FLIP_V3_DRY] split_sizes A={len(IA)} B={len(IB)} H={len(IH)} n_items={n_items}")
    for base in base_list[:3]:
        fr = frontier2v_ref(base)
        n_fr_valid = int((fr >= 0).sum()) if fr is not None else None
        cons_rows = [crowd[ex].get(base) for ex in LOCAL_MID]
        cons_rows = [r for r in cons_rows if r is not None]
        print(f"[FLIP_V3_DRY] base={base!r} n_crowd_rows={len(cons_rows)} "
              f"frontier2v_valid_items={n_fr_valid}")
    sys.exit(0)

executor = make_judge_backend(EXECUTORS[SEL_EX][0], cfgmod.ImplementerConfig(), temperature=None)


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


since_dump = 0
for base in todo:
    fr = frontier2v_ref(base)
    cons_rows = [crowd[ex].get(base) for ex in LOCAL_MID]
    cons_rows = [r for r in cons_rows if r is not None]
    if fr is None or len(cons_rows) < 7:
        continue
    cons = np.stack([(np.asarray(r, float) > .5).astype(float) for r in cons_rows]).mean(0)
    objectives = {"frontier2v": np.asarray(fr)}
    rngperm = np.array([int(hashlib.md5((base + "|null|" + str(j)).encode()
                                        ).hexdigest(), 16) % 2 for j in range(n_items)])
    objectives["null"] = rngperm
    cand = [j for j in IA if .3 < cons[j] < .7 and probes[j].strip()][:N_CAND]
    decis = [j for j in IA if probes[j].strip()]
    if not decis or not cand:
        continue
    seed = [(max(decis, key=lambda j: cons[j]), 1), (min(decis, key=lambda j: cons[j]), 0)]
    per_obj = {}
    for obj, lab in objectives.items():
        lab = np.asarray(lab)
        true_lab = objectives["frontier2v"] if obj == "null" else lab
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
                yB_new = ev(trials[k], IB, lab)          # confirmation gate
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
        task_res[base] = {"class": _slate_class.get(base, "BANK-FULL"), "objectives": per_obj}
        since_dump += 1
        if since_dump >= DUMP_EVERY:
            dump(task_res)
            since_dump = 0

dump(task_res)
print("DONE ->", OUT)
