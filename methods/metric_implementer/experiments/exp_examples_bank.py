"""EXP-EXAMPLES-BANK-1 — examples-vs-definition at bank scale, task x category
(prereg notes/2026-08-17__exp-examples-bank-prereg.md, sha prefix c42a8f54db2e6f54).

flip_functional_v2 lineage with the prereg's three declared changes:
  1. reference = llama70b + qwen25-72b consensus under the metric's own bank rubric
     (crowd panels mbar2_<task>_<exec>.npz; humor = mbar285 + humor_sup merge);
     disagreement/NaN -> -1; metrics with <60 decided items skipped (counted).
  2. selection executor = llama8b only (never an evaluator).
  3. holdout evaluated at llama70b AND qwen25-72b (BF16 TP=2), reported separately.
Everything else identical to flip_functional_v2: three-way stable-hash split
("split1cv3|"/"conf|"), A-select at theta=.01 / B-confirm / holdout-once, exemplar masking,
EX_TRUNC=500, MAX_SET=12, N_CAND=24, ROUNDS=6, BATCH=8, selection-null on every 3rd metric.
gemma2 excluded everywhere (prereg exclusion) -> crowd ambivalence pool has 10 mid executors.

Sample: seed 0, up to 6 metrics per (task x 6-category M6) cell over the 1,270 fitted metrics
(taxonomy osl_metric_types_20260728.json labels_12type; O-id alignment = sorted curves_*.json
per-domain fitted order, as in notebook 4.2g-b). Sample FIXED in the manifest before any
selection call.

Modes:
  build      (CPU, laptop): manifest w/ sample, reference labels, cons vectors, projection
  select     (sk3, llama8b, 1 GPU): greedy selection -> <out>/ebank_select_state.jsonl
  evaluate   (sk3, --judge llama70b|qwen25-72b, TP=2): holdout arms -> <out>/ebank_eval_<judge>.jsonl
  score      (CPU, laptop): outputs/exp_examples_bank/bank_flips_v1.json + report tables
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

SEED = 0
TASKS = ["creative_writing", "humor", "math", "news_homepages", "notice_and_comment",
         "patents", "peer_review", "press_releases"]
MID = ["llama1b", "llama3b", "llama8b", "llama70b", "qwen25-3b", "qwen25-7b",
       "qwen25-14b", "qwen25-72b", "mistral7b", "phi4"]      # gemma2-27b EXCLUDED (prereg)
VOTERS = ["llama70b", "qwen25-72b"]
M6 = {"COMPACT_DEVICE": "local patterns", "EXTENDED_STRUCTURE": "whole-text structure",
      "SURFACE_CHECK": "verifiable", "ETHICS_HARM": "verifiable",
      "VERIFICATION_SOURCING": "verifiable", "LOGICAL_RIGOR": "verifiable",
      "ORIGINALITY_NOVELTY": "originality/audience", "AUDIENCE_FIT": "originality/audience",
      "COMMUNITY_STANCE": "identity/social", "IDENTITY_PERSONA": "identity/social",
      "ECOSYSTEM_ARTIFACTS": "beyond-text", "RECEPTION_OUTCOME": "beyond-text"}
PER_CELL = 6
MIN_DECIDED = 60
MAX_SET, N_CAND, ROUNDS, BATCH = 12, 24, 6, 8
ACCEPT_THETA, EX_TRUNC = 0.01, 500
N_ITEMS = 300
CALL_BUDGET = 1_500_000


def _rng(*parts):
    import random
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def split_of(t):                                    # identical to flip_functional_v2
    h = int(hashlib.md5(("split1cv3|" + t).encode()).hexdigest(), 16)
    if h % 2 == 1:
        return "H"
    return "A" if int(hashlib.md5(("conf|" + t).encode()).hexdigest(), 16) % 2 == 0 else "B"


def load_npz(p):
    if not os.path.exists(p):
        return {}
    z = np.load(p, allow_pickle=True)
    return {str(n): z["m_bar"][i] for i, n in enumerate([str(x) for x in z["names"]])}


def panels_for(task, ex, panel_dir):
    d = {}
    if task == "humor":
        d.update(load_npz(f"{panel_dir}/mbar285_{ex}.npz"))
        d.update(load_npz(f"{panel_dir}/mbar2_humor_sup_{ex}.npz"))
    else:
        d.update(load_npz(f"{panel_dir}/mbar2_{task}_{ex}.npz"))
    return d


# ---------------------------------------------------------------- build ----------------------
def build(a):
    tax = json.load(open(os.path.join(a.root, "outputs/analyses/osl_metric_types_20260728.json")))
    l12 = tax["labels_12type"]
    keep = []
    for f in sorted(glob.glob(os.path.join(
            a.root, "notebooks/data/2026-07-07-osl-multi/curves_*.json"))):
        dom = os.path.basename(f).split("curves_")[1].replace(".json", "")
        for name, v in json.load(open(f)).items():
            if v.get("verdict") in ("RISING", "REACHES", "BOUNDED"):
                keep.append((name, dom))
    ids = sorted(tax["key"], key=lambda s: int(s[1:]))
    assert len(ids) == len(keep) == 1270, (len(ids), len(keep))
    cat_of = {}
    for oid, (name, dom) in zip(ids, keep):
        lab = l12[oid]["label"] if isinstance(l12[oid], dict) else l12[oid]
        cat_of[(dom, name)] = (M6[lab], oid)

    panel_dir = os.path.join(a.root, "outputs/articulation_story_20260810/crowd_panels")
    freeze_dir = a.freeze_dir
    rubrics = {}
    for t in TASKS:
        for fz in ([f"freeze_{t}_v2.json"] + (["freeze_humor_sup_v2.json"] if t == "humor"
                                              else [])):
            p = os.path.join(freeze_dir, fz)
            if os.path.exists(p):
                for m in json.load(open(p))["metrics"]:
                    rubrics[(t, m["name"])] = m["rubric"]

    skip = {"no_category": 0, "no_rubric": 0, "missing_voter_panel": 0,
            "under_60_decided": 0, "under_7_crowd": 0}
    eligible, l2_agree = {}, {}
    for task in TASKS:
        vot = {ex: panels_for(task, ex, panel_dir) for ex in VOTERS}
        crowd = {ex: panels_for(task, ex, panel_dir) for ex in MID}
        for (dom, name), (cat, oid) in cat_of.items():
            if dom != task:
                continue
            if (task, name) not in rubrics:
                skip["no_rubric"] += 1
                continue
            rows = [vot[ex].get(name) for ex in VOTERS]
            if any(r is None or len(r) != N_ITEMS for r in rows):
                skip["missing_voter_panel"] += 1
                continue
            v = [np.asarray(r, float) for r in rows]
            fin = np.isfinite(v[0]) & np.isfinite(v[1])
            b = [(x > .5).astype(int) for x in v]
            ref = np.full(N_ITEMS, -1, int)
            agree = fin & (b[0] == b[1])
            ref[agree] = b[0][agree]
            n_dec = int((ref >= 0).sum())
            if n_dec < MIN_DECIDED:
                skip["under_60_decided"] += 1
                continue
            crows = [crowd[ex].get(name) for ex in MID]
            crows = [np.asarray(r, float) for r in crows if r is not None and len(r) == N_ITEMS]
            if len(crows) < 7:
                skip["under_7_crowd"] += 1
                continue
            cons = np.stack([(r > .5).astype(float) for r in crows]).mean(0)
            l2_agree.setdefault(task, []).append(float((b[0][fin] == b[1][fin]).mean()))
            eligible.setdefault((task, cat), []).append({
                "task": task, "name": name, "category": cat, "o_id": oid,
                "n_decided": n_dec,
                "voter_agreement": float((b[0][fin] == b[1][fin]).mean()),
                "ref": ref.tolist(), "cons": [round(float(x), 4) for x in cons],
                "rubric": rubrics[(task, name)], "news_flag": task == "news_homepages"})

    sample = []
    for task in TASKS:
        for cat in sorted(set(c for _, c in eligible if _ == task)):
            pool = sorted(eligible[(task, cat)], key=lambda m: m["name"])
            picks = (_rng("ebank-sample", SEED, task, cat)
                     .sample(pool, min(PER_CELL, len(pool))))
            sample += sorted(picks, key=lambda m: m["name"])
    for i, m in enumerate(sample):
        m["null_control"] = (i % 3 == 0)                # every 3rd metric, fixed order

    n_sel = len(sample) + sum(m["null_control"] for m in sample)
    proj = n_sel * (ROUNDS * BATCH * 75 + 2 * 75 + 6 * 75)   # ~4275 per metric-objective
    man = {"experiment": "EXP-EXAMPLES-BANK-1",
           "prereg": "notes/2026-08-17__exp-examples-bank-prereg.md",
           "prereg_sha": "c42a8f54db2e6f54", "seed": SEED,
           "selection_executor": "llama8b", "evaluators": VOTERS,
           "reference": "llama70b+qwen25-72b bank-rubric consensus, ties/NaN -> -1",
           "crowd_mid_executors": MID, "gemma2_excluded": True,
           "protocol": {"theta": ACCEPT_THETA, "max_set": MAX_SET, "n_cand": N_CAND,
                        "rounds": ROUNDS, "batch": BATCH, "ex_trunc": EX_TRUNC,
                        "split": "split1cv3/conf stable-hash (flip_functional_v2)"},
           "skip_counts": skip,
           "cells": {f"{t}|{c}": len(v) for (t, c), v in eligible.items()},
           "n_sample": len(sample),
           "n_null": sum(m["null_control"] for m in sample),
           "l2_task_median_agreement": {t: float(np.median(v)) for t, v in l2_agree.items()},
           "projected_llama8b_calls": proj, "call_budget": CALL_BUDGET,
           "sample": sample}
    os.makedirs(a.out, exist_ok=True)
    json.dump(man, open(os.path.join(a.out, "ebank_manifest.json"), "w"), indent=1)
    print(f"[build] sample={len(sample)} metrics ({man['n_null']} with null control); "
          f"skips={skip}")
    print(f"[build] per-cell sizes: {sorted(man['cells'].items())}")
    print(f"[build] L2 median agreement: {man['l2_task_median_agreement']}")
    print(f"[build] projected llama8b calls: {proj:,} (budget {CALL_BUDGET:,}) "
          f"{'OK' if proj <= CALL_BUDGET else 'OVER — STOP'}")


# ---------------------------------------------------------------- shared runtime -------------
def _task_setup(task, root_repo):
    from ..experiments.run_real_test import _load_texts
    from .. import config as cfgmod
    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), task.replace("_", "-"))
    texts, _ = _load_texts(task.replace("_", "-"), 360, cfg)
    probes = texts[60:360]
    sp = [split_of(t) for t in probes[:N_ITEMS]]
    IA = np.array([i for i, s in enumerate(sp) if s == "A"])
    IB = np.array([i for i, s in enumerate(sp) if s == "B"])
    IH = np.array([i for i, s in enumerate(sp) if s == "H"])
    return probes, cfg.max_text_chars, IA, IB, IH


def _trunc(t):
    t = t.strip()
    return t if len(t) <= EX_TRUNC else t[:EX_TRUNC].rsplit(" ", 1)[0] + " ..."


def _rubric_for(base, items, probes):
    pos = [i for i, l in items if l == 1]
    neg = [i for i, l in items if l == 0]
    r = base
    if pos:
        r += "\nExamples that satisfy this criterion:\n" + \
             "\n".join("- " + _trunc(probes[i]) for i in pos)
    if neg:
        r += "\nExamples that do NOT satisfy it:\n" + \
             "\n".join("- " + _trunc(probes[i]) for i in neg)
    return r


def _balanced(pred, lab, keep):
    ok = (lab >= 0) & keep & np.isfinite(pred)
    if ok.sum() < 12:
        return None
    p = (pred[ok] > .5).astype(int)
    l = lab[ok]
    accs = [float(np.mean(p[l == c] == c)) for c in (0, 1) if (l == c).sum() >= 3]
    return float(np.mean(accs)) if len(accs) == 2 else None


class _Scorer:
    def __init__(self, executor, probes, mx):
        from ..experiments import alpha_probe as ap
        self.ap, self.ex, self.probes, self.mx = ap, executor, probes, mx
        self.n_calls = 0

    def score_rubrics(self, rubrics, item_idx):
        prompts, spans = [], []
        for r in rubrics:
            s = len(prompts)
            for j in item_idx:
                prompts.append(self.ap._YESNO_TEXTFIRST.format(
                    text=self.probes[j][:self.mx], rubric=r))
            spans.append((s, len(prompts)))
        self.n_calls += len(prompts)
        flat = np.asarray(self.ex.score_binary(prompts), float)
        return np.stack([flat[a:b] for a, b in spans])


# ---------------------------------------------------------------- select (sk3, llama8b) ------
def select(a):
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    from .. import config as cfgmod
    from ..vllm_backend import make_judge_backend
    from .osl_sweep import EXECUTORS
    man = json.load(open(os.path.join(a.out, "ebank_manifest.json")))
    sf = os.path.join(a.out, "ebank_select_state.jsonl")
    done = {(json.loads(l)["task"], json.loads(l)["name"])
            for l in open(sf)} if os.path.exists(sf) else set()
    executor = make_judge_backend(EXECUTORS["llama8b"][0], cfgmod.ImplementerConfig(),
                                  temperature=None)
    total_calls = 0
    by_task = {}
    for m in man["sample"]:
        by_task.setdefault(m["task"], []).append(m)
    fout = open(sf, "a")
    for task in TASKS:
        if task not in by_task:
            continue
        todo = [m for m in by_task[task] if (task, m["name"]) not in done]
        if not todo:
            continue
        probes, mx, IA, IB, IH = _task_setup(task, a.root)
        sc = _Scorer(executor, probes, mx)
        for m in todo:
            # AMENDMENT 2 (sha a9d410335e879840): NO crowd anywhere in selection — empty
            # starting set, candidate pool = 24 seeded-random train-A items (stable hash keyed
            # on metric name, seed 0), no consensus filter, either polarity proposable.
            base, ref = m["name"], np.asarray(m["ref"], int)
            nonblank = [j for j in IA if probes[j].strip()]
            cand = sorted(nonblank, key=lambda j: hashlib.sha256(
                f"ebank-a2-cand|{SEED}|{base}|{j}".encode()).hexdigest())[:N_CAND]
            rec = {"task": task, "name": base, "category": m["category"],
                   "null_control": m["null_control"], "objectives": {}}
            if cand:
                objectives = {"reference": ref}
                if m["null_control"]:
                    objectives["null"] = np.array(
                        [int(hashlib.md5((base + "|null|" + str(j)).encode()
                                         ).hexdigest(), 16) % 2 for j in range(N_ITEMS)])
                for obj, lab in objectives.items():
                    lab = np.asarray(lab)
                    S = []                               # empty start (no crowd seeds)

                    def ev(items, idx, l):
                        keep = np.zeros(N_ITEMS, bool)
                        keep[idx] = True
                        for i, _ in items:
                            keep[i] = False
                        y = _balanced(sc.score_rubrics(
                            [_rubric_for(m["rubric"], items, probes)], idx)[0],
                            l[idx], keep[idx])
                        return -1.0 if y is None else y
                    yA, yB = ev(S, IA, lab), ev(S, IB, lab)
                    pool = [(j, l) for j in cand for l in (1, 0)]
                    pool.sort(key=lambda t: hashlib.md5(
                        (base + obj + str(t)).encode()).hexdigest())
                    for r0 in range(0, min(len(pool), ROUNDS * BATCH), BATCH):
                        if len(S) >= MAX_SET:
                            break
                        trials = [S + [c] for c in pool[r0:r0 + BATCH]
                                  if not any(c[0] == i for i, _ in S)]
                        if not trials:
                            continue
                        mats = sc.score_rubrics(
                            [_rubric_for(m["rubric"], t, probes) for t in trials], IA)
                        ys = []
                        for t, mat in zip(trials, mats):
                            keep = np.zeros(N_ITEMS, bool)
                            keep[IA] = True
                            for i, _ in t:
                                keep[i] = False
                            yv = _balanced(mat, lab[IA], keep[IA])
                            ys.append(-1.0 if yv is None else yv)
                        k = int(np.argmax(ys))
                        if ys[k] > yA + ACCEPT_THETA:
                            yB_new = ev(trials[k], IB, lab)
                            if yB_new >= yB:
                                S, yA, yB = trials[k], ys[k], yB_new
                    rec["objectives"][obj] = {
                        "selected": [[int(i), int(l)] for i, l in S],
                        "yA": round(yA, 4), "yB": round(yB, 4)}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            total_calls = sc.n_calls
            print(f"[select|{task}] {base[:40]:40s} objs={list(rec['objectives'])} "
                  f"cum_calls={total_calls:,}", flush=True)
        print(f"[select] {task} done; task calls={sc.n_calls:,}", flush=True)
    fout.close()
    print("[select] ALL DONE", flush=True)


# AMENDMENT 1 (2026-08-17, sha 49dd908dacce36f9): primary key = LOFO family-balanced
# 11-executor panel consensus; 2-voter bank key demoted to sensitivity; per-item arm vectors
# SAVED; silver AUC cross-check on sound-silver tasks; selection-vs-evaluation key asymmetry
# disclosed (conservative for functional).
PANEL11 = ["llama1b", "llama3b", "llama8b", "llama70b", "qwen25-3b", "qwen25-7b",
           "qwen25-14b", "qwen25-72b", "mistral7b", "phi4", "gemma2-27b"]
FAMILY = {**{e: "llama" for e in PANEL11 if e.startswith("llama")},
          **{e: "qwen" for e in PANEL11 if e.startswith("qwen")},
          "mistral7b": "mistral", "phi4": "phi", "gemma2-27b": "gemma2"}
JUDGE_FAMILY = {"llama70b": "llama", "qwen25-72b": "qwen"}
SILVER_TASKS = ["humor", "creative_writing", "peer_review"]   # code_review not in the 8-task
                                                              # sample (disclosed skip)


def lofo_key(task, name, judge, panel_dir):
    """LOFO family-balanced consensus over PANEL11, excluding the judge's family.
    Per item: binarize each finite executor row at .5; mean within family; mean across
    available families; >.5 -> 1, <.5 -> 0, exact .5 or no data -> -1."""
    fams = {}
    for ex in PANEL11:
        if FAMILY[ex] == JUDGE_FAMILY[judge]:
            continue
        r = panels_for(task, ex, panel_dir).get(name)
        if r is None or len(r) != N_ITEMS:
            continue
        fams.setdefault(FAMILY[ex], []).append(np.asarray(r, float))
    if not fams:
        return None
    fam_means = []
    for rows in fams.values():
        R = np.stack(rows)
        fin = np.isfinite(R)
        b = np.where(R > .5, 1.0, 0.0)
        with np.errstate(invalid="ignore"):
            fm = np.where(fin.any(0), (b * fin).sum(0) / np.maximum(fin.sum(0), 1), np.nan)
        fm[~fin.any(0)] = np.nan
        fam_means.append(fm)
    F = np.stack(fam_means)
    ok = np.isfinite(F)
    with np.errstate(invalid="ignore"):
        mean = np.where(ok.any(0), np.nansum(np.where(ok, F, 0), 0)
                        / np.maximum(ok.sum(0), 1), np.nan)
    key = np.full(N_ITEMS, -1, int)
    key[np.isfinite(mean) & (mean > .5)] = 1
    key[np.isfinite(mean) & (mean < .5)] = 0
    return key


# ---------------------------------------------------------------- evaluate (sk3, TP=2) -------
def evaluate(a):
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    from .. import config as cfgmod
    from ..vllm_backend import make_judge_backend
    from .osl_sweep import EXECUTORS
    man = json.load(open(os.path.join(a.out, "ebank_manifest.json")))
    meta = {(m["task"], m["name"]): m for m in man["sample"]}
    sel = [json.loads(l) for l in open(os.path.join(a.out, "ebank_select_state.jsonl"))]
    ef = os.path.join(a.out, f"ebank_eval_{a.judge}.jsonl")
    done = {(json.loads(l)["task"], json.loads(l)["name"])
            for l in open(ef)} if os.path.exists(ef) else set()
    cfg = cfgmod.ImplementerConfig()
    cfg.vllm_tp_size = int(os.environ.get("VOICE_TP", "1"))
    executor = make_judge_backend(EXECUTORS[a.judge][0], cfg, temperature=None)
    fout = open(ef, "a")
    for task in TASKS:
        todo = [r for r in sel if r["task"] == task and (task, r["name"]) not in done
                and r["objectives"]]
        if not todo:
            continue
        probes, mx, IA, IB, IH = _task_setup(task, a.root)
        sc = _Scorer(executor, probes, mx)
        for r in todo:
            m = meta[(task, r["name"])]
            arms = {"name": r["name"], "definition": m["rubric"]}
            S = [(i, l) for i, l in r["objectives"]["reference"]["selected"]]
            arms["functional"] = _rubric_for(m["rubric"], S, probes)
            masked = {i for i, _ in S}
            if "null" in r["objectives"]:
                Sn = [(i, l) for i, l in r["objectives"]["null"]["selected"]]
                arms["functional_null"] = _rubric_for(m["rubric"], Sn, probes)
                masked |= {i for i, _ in Sn}
            mats = sc.score_rubrics(list(arms.values()), IH)
            # AMENDMENT 1 change 3: save the RAW per-item P(YES) vectors (keys applied at
            # score time; any future key can rescore without re-running judges)
            fout.write(json.dumps({
                "task": task, "name": r["name"], "category": r["category"],
                "judge": a.judge, "ih_items": [int(i) for i in IH],
                "masked_items": sorted(int(i) for i in masked),
                "p_yes": {arm: [None if v != v else round(float(v), 4) for v in mat]
                          for arm, mat in zip(arms, mats)}}) + "\n")
            fout.flush()
            print(f"[eval|{a.judge}|{task}] {r['name'][:44]:44s} vectors saved", flush=True)
    fout.close()

    # legacy rescore rows (coordinator addition): same judge load, vectors only
    lrf = os.path.join(a.out, "ebank_legacy_rows.jsonl")
    if os.path.exists(lrf):
        lrows = [json.loads(l) for l in open(lrf)]
        lef = os.path.join(a.out, f"ebank_legacy_eval_{a.judge}.jsonl")
        ldone = {(x["sel_exec"], x["task"], x["name"], x["objective"])
                 for x in (json.loads(l) for l in open(lef))} if os.path.exists(lef) else set()
        lout = open(lef, "a")
        for task in sorted({r["task"] for r in lrows}):
            todo = [r for r in lrows if r["task"] == task and
                    (r["sel_exec"], task, r["name"], r["objective"]) not in ldone]
            if not todo:
                continue
            probes, mx, IA, IB, IH = _task_setup(task, a.root)
            sc = _Scorer(executor, probes, mx)
            for r in todo:
                S = [(int(i), int(l)) for i, l in r["selected"]]
                arms = {"name": r["name"], "definition": r["def_rubric"],
                        "functional": _rubric_for(r["def_rubric"], S, probes)}
                mats = sc.score_rubrics(list(arms.values()), IH)
                lout.write(json.dumps({
                    "sel_exec": r["sel_exec"], "task": task, "name": r["name"],
                    "objective": r["objective"], "judge": a.judge,
                    "ih_items": [int(i) for i in IH],
                    "masked_items": sorted({i for i, _ in S}),
                    "p_yes": {arm: [None if v != v else round(float(v), 4) for v in mat]
                              for arm, mat in zip(arms, mats)}}) + "\n")
                lout.flush()
                print(f"[legacy|{a.judge}|{task}] {r['sel_exec']}:{r['name'][:36]:36s} "
                      f"{r['objective']}", flush=True)
        lout.close()
    print(f"[eval] {a.judge} ALL DONE", flush=True)


# ---------------------------------------------------------------- score ----------------------
def score(a):
    man = json.load(open(os.path.join(a.out, "ebank_manifest.json")))
    sel = {(r["task"], r["name"]): r for r in
           (json.loads(l) for l in open(os.path.join(a.out, "ebank_select_state.jsonl")))}
    rows = []
    for j in VOTERS:
        f = os.path.join(a.out, f"ebank_eval_{j}.jsonl")
        if os.path.exists(f):
            rows += [json.loads(l) for l in open(f)]
    meta = {(m["task"], m["name"]): m for m in man["sample"]}
    panel_dir = os.path.join(a.root, "outputs/articulation_story_20260810/crowd_panels")

    # silver labels (AMENDMENT change 4): same subsample as _load_texts, judgement column.
    # Prefer the sk3-extracted sidecar (mode: silver-extract) — the panels/IH indices were
    # built from the sk3 corpora, so the sidecar guarantees row alignment.
    silver = {}
    sidecar = os.path.join(a.out, "ebank_silver.json")
    if os.path.exists(sidecar):
        for t, v in json.load(open(sidecar)).items():
            silver[t] = np.asarray(v, int) if v is not None else None
    else:
        from ..manifest import full_manifest, load_corpus_labels
        for t in SILVER_TASKS:
            try:
                entry = next(e for e in full_manifest().datasets
                             if e.task == t.replace("_", "-"))
                texts, labs, _ = load_corpus_labels(entry, 360)
                if len(labs) >= 360:
                    silver[t] = np.asarray(labs[60:360], int)
            except Exception as e:
                silver[t] = None
                print(f"[score] silver join SKIPPED for {t}: {e}")

    def _auc(p, y):
        ok = np.isfinite(p)
        p, y = p[ok], y[ok]
        if len(set(y.tolist())) < 2 or len(y) < 12:
            return None
        pos, neg = p[y == 1], p[y == 0]
        gt = (pos[:, None] > neg[None, :]).mean()
        eq = (pos[:, None] == neg[None, :]).mean()
        return float(gt + .5 * eq)

    per_metric = []
    for r in rows:
        k = (r["task"], r["name"])
        m = meta[k]
        ih = np.asarray(r["ih_items"], int)
        masked = set(r["masked_items"])
        keepH = np.array([i not in masked for i in ih])
        pv = {arm: np.asarray([np.nan if v is None else v for v in vec], float)
              for arm, vec in r["p_yes"].items()}
        keys = {"two_voter": np.asarray(m["ref"], int)}
        lk = lofo_key(r["task"], r["name"], r["judge"], panel_dir)
        if lk is not None:
            keys["lofo"] = lk
        hold, dec_frac = {}, {}
        for kn, key in keys.items():
            dec_frac[kn] = float((key[ih] >= 0).mean())
            hold[kn] = {arm: _balanced(pv[arm], key[ih], keepH) for arm in pv}
        sauc = None
        if r["task"] in silver and silver[r["task"]] is not None:
            sl = silver[r["task"]][ih]
            sauc = {arm: _auc(pv[arm][keepH], sl[keepH]) for arm in pv}
        def _delta(h):
            return (h["functional"] - h["definition"]
                    if h.get("functional") is not None and h.get("definition") is not None
                    else None)
        per_metric.append({
            "task": r["task"], "category": r["category"], "name": r["name"],
            "judge": r["judge"], "news_flag": m["news_flag"],
            "n_decided_two_voter": m["n_decided"],
            "decided_fraction_holdout": {kn: round(v, 4) for kn, v in dec_frac.items()},
            "voter_agreement": m["voter_agreement"], "null_control": m["null_control"],
            "holdout": {kn: {arm: (round(v, 4) if v is not None else None)
                             for arm, v in h.items()} for kn, h in hold.items()},
            "silver_auc": ({arm: (round(v, 4) if v is not None else None)
                            for arm, v in sauc.items()} if sauc else None),
            "selected": sel[k]["objectives"].get("reference", {}).get("selected"),
            "delta_lofo": _delta(hold.get("lofo", {})),
            "delta_two_voter": _delta(hold["two_voter"])})

    def tab(group_key, delta_key):
        out = {}
        for r in per_metric:
            if r.get(delta_key) is None:
                continue
            out.setdefault(r[group_key], []).append(r[delta_key])
        return {k: {"mean_delta": round(float(np.mean(v)), 4), "n": len(v)}
                for k, v in sorted(out.items())}

    # per-category divergence between keys (AMENDMENT change 2)
    div = {}
    for r in per_metric:
        if r["delta_lofo"] is not None and r["delta_two_voter"] is not None:
            div.setdefault(r["category"], []).append(r["delta_lofo"] - r["delta_two_voter"])
    divergence = {c: {"mean": round(float(np.mean(v)), 4),
                      "mean_abs": round(float(np.mean(np.abs(v))), 4), "n": len(v)}
                  for c, v in sorted(div.items())}

    # L1 under both keys: pooled null-control holdout vs name arm
    def l1_of(kn):
        nulls = [r for r in per_metric if r["null_control"]
                 and r["holdout"].get(kn, {}).get("functional_null") is not None]
        if not nulls:
            return None
        return {"pooled_null_holdout":
                round(float(np.mean([r["holdout"][kn]["functional_null"]
                                     for r in nulls])), 4),
                "pooled_name_holdout_null_metrics":
                round(float(np.mean([r["holdout"][kn]["name"] for r in nulls
                                     if r["holdout"][kn].get("name") is not None])), 4),
                "n": len(nulls)}
    l1 = {kn: l1_of(kn) for kn in ("lofo", "two_voter")}
    # OLS: delta ~ task dummies + category dummies, metric-level bootstrap (per key)
    def ols_of(delta_key):
        dec = [r for r in per_metric if r.get(delta_key) is not None]
        if not dec:
            return None
        tasks_u = sorted({r["task"] for r in dec})
        cats_u = sorted({r["category"] for r in dec})

        def design(rs):
            X = [[1.0] + [1.0 if r["task"] == t else 0.0 for t in tasks_u[1:]]
                 + [1.0 if r["category"] == c else 0.0 for c in cats_u[1:]] for r in rs]
            return np.asarray(X), np.asarray([r[delta_key] for r in rs])
        X, y = design(dec)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        keys_ = list({(r["task"], r["name"]) for r in dec})
        bygrp = {}
        for r in dec:
            bygrp.setdefault((r["task"], r["name"]), []).append(r)
        rng = np.random.default_rng(SEED)
        boots = []
        for _ in range(1000):
            pick = [keys_[i] for i in rng.integers(len(keys_), size=len(keys_))]
            rs = [r for k in pick for r in bygrp[k]]
            Xb, yb = design(rs)
            try:
                boots.append(np.linalg.lstsq(Xb, yb, rcond=None)[0])
            except np.linalg.LinAlgError:
                pass
        B = np.asarray(boots)
        names = (["intercept"] + [f"task:{t}" for t in tasks_u[1:]]
                 + [f"cat:{c}" for c in cats_u[1:]])
        return {nm: {"beta": round(float(b), 4),
                     "ci95": [round(float(np.percentile(B[:, i], 2.5)), 4),
                              round(float(np.percentile(B[:, i], 97.5)), 4)]}
                for i, (nm, b) in enumerate(zip(names, beta))}

    art = {"experiment": man["experiment"], "prereg_sha": man["prereg_sha"],
           "amendment1_sha": "49dd908dacce36f9",
           "disclosure": "selection optimized the 2-voter bank key; PRIMARY evaluation uses "
                         "the LOFO family-balanced 11-executor panel key (asymmetry biases "
                         "against the functional arm; accepted as conservative)",
           "manifest_summary": {k: man[k] for k in
                                ("n_sample", "n_null", "skip_counts", "cells",
                                 "l2_task_median_agreement", "selection_executor",
                                 "evaluators", "crowd_mid_executors")},
           "panel11": PANEL11,
           "n_eval_rows": len(rows),
           "delta_by_task": {"lofo": tab("task", "delta_lofo"),
                             "two_voter": tab("task", "delta_two_voter")},
           "delta_by_category": {"lofo": tab("category", "delta_lofo"),
                                 "two_voter": tab("category", "delta_two_voter")},
           "key_divergence_by_category": divergence,
           "silver_tasks": {t: (silver.get(t) is not None) for t in SILVER_TASKS},
           "silver_note": "code_review has sound silver but is not among the 8 sampled tasks",
           "L1_null_gate": l1, "L2_task_median_agreement": man["l2_task_median_agreement"],
           "ols_delta_on_task_plus_category": {"lofo": ols_of("delta_lofo"),
                                               "two_voter": ols_of("delta_two_voter")},
           "per_metric": per_metric}
    dst = os.path.join(a.root, "outputs/exp_examples_bank/bank_flips_v1.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(art, open(dst, "w"), indent=1)
    print(f"[score] {len(per_metric)} metric-judge rows -> {dst}")

    # ---- legacy rescore artifact --------------------------------------------------------
    lrows = []
    for j in VOTERS:
        f = os.path.join(a.out, f"ebank_legacy_eval_{j}.jsonl")
        if os.path.exists(f):
            lrows += [json.loads(l) for l in open(f)]
    if not lrows:
        return
    if not a.legacy_dir:
        print("[score] legacy eval rows present but --legacy-dir missing; legacy skipped")
        return
    old = {}
    for l in open(os.path.join(a.out, "ebank_legacy_rows.jsonl")):
        r = json.loads(l)
        old[(r["sel_exec"], r["task"], r["name"], r["objective"])] = r.get("old_holdout")
    per_leg = []
    for r in lrows:
        ih = np.asarray(r["ih_items"], int)
        keepH = np.array([i not in set(r["masked_items"]) for i in ih])
        pv = {arm: np.asarray([np.nan if v is None else v for v in vec], float)
              for arm, vec in r["p_yes"].items()}
        keys = {}
        lk = lofo_key(r["task"], r["name"], r["judge"], panel_dir)
        if lk is not None:
            keys["lofo"] = lk
        dk = legacy_dossier_key(r["task"], r["name"], a.legacy_dir)
        if dk is not None:
            keys["dossier"] = np.asarray(dk, int)
        hold = {kn: {arm: _balanced(pv[arm], key[ih], keepH) for arm in pv}
                for kn, key in keys.items()}
        per_leg.append({
            "sel_exec": r["sel_exec"], "task": r["task"], "name": r["name"],
            "objective": r["objective"], "judge": r["judge"],
            "decided_fraction_holdout": {kn: round(float((key[ih] >= 0).mean()), 4)
                                         for kn, key in keys.items()},
            "holdout": {kn: {arm: (round(v, 4) if v is not None else None)
                             for arm, v in h.items()} for kn, h in hold.items()},
            "old_artifact_holdout": old.get((r["sel_exec"], r["task"], r["name"],
                                             r["objective"]))})

    def leg_tab(kn):
        out = {}
        for r in per_leg:
            h = r["holdout"].get(kn, {})
            if h.get("functional") is not None and h.get("definition") is not None:
                out.setdefault(r["task"], []).append(h["functional"] - h["definition"])
        return {t: {"mean_delta": round(float(np.mean(v)), 4), "n": len(v)}
                for t, v in sorted(out.items())}

    old_deltas = {}
    for r in per_leg:
        oh = r.get("old_artifact_holdout") or {}
        if oh.get("functional") is not None and oh.get("definition") is not None:
            old_deltas.setdefault(r["task"], []).append(oh["functional"] - oh["definition"])
    lart = {"experiment": "EXP-EXAMPLES-BANK-1 legacy rescore",
            "amendment1_sha": "49dd908dacce36f9",
            "note": "selected sets from flip_functional_v2 artifacts (no re-selection); "
                    "functional arm rebuilt as definition+selected per coordinator spec; "
                    "old artifact numbers carried for continuity",
            "n_rows": len(per_leg),
            "delta_by_task": {"lofo_primary": leg_tab("lofo"),
                              "dossier_recomputed": leg_tab("dossier"),
                              "old_artifact_stored": {
                                  t: {"mean_delta": round(float(np.mean(v)), 4), "n": len(v)}
                                  for t, v in sorted(old_deltas.items())}},
            "per_metric": per_leg}
    ldst = os.path.join(a.root, "outputs/exp_examples_bank/legacy38_rescored_v1.json")
    json.dump(lart, open(ldst, "w"), indent=1)
    print(f"[score] legacy: {len(per_leg)} rows -> {ldst}")


# ---------------------------------------------------------------- legacy 38/56 rescore -------
# Coordinator addition 2026-08-17: rescore the legacy flip_functional_v2 winners under the
# Amendment-1 PRIMARY (LOFO) key. Selected sets come from the stored artifacts (no re-selection).
# Arms per coordinator spec: name = base; definition = the v2 rubric the original used;
# functional = definition + stored selected set (rubric_for format, 500-char truncation).
LEGACY_SEL = ["llama70b", "qwen25-72b"]


def legacy_build(a):
    """Local: flatten the two legacy artifacts -> <out>/ebank_legacy_rows.jsonl."""
    d = a.legacy_dir
    slate = (json.load(open(os.path.join(d, "zxa_slate_v1.json")))
             + json.load(open(os.path.join(d, "news_slate_v1.json"))))
    srub = {(m["task"], m["name"]): (m.get("rubric") or m["name"]) for m in slate}
    rows = []
    for se in LEGACY_SEL:
        res = json.load(open(os.path.join(d, f"flip_functional_v2_{se}.json")))["results"]
        for task, dd in res.items():
            fz = os.path.join(d, f"freeze_{task}_v2.json")
            v2rub = ({m["name"]: m["rubric"] for m in json.load(open(fz))["metrics"]}
                     if os.path.exists(fz) else {})
            for base, rec in dd.items():
                rub = v2rub.get(base) or srub.get((task, base)) or base
                for obj, po in rec.get("objectives", {}).items():
                    if obj == "null":
                        continue
                    rows.append({"sel_exec": se, "task": task, "name": base,
                                 "objective": obj, "selected": po["selected"],
                                 "def_rubric": rub,
                                 "old_holdout": po.get("holdout")})
    with open(os.path.join(a.out, "ebank_legacy_rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[legacy-build] {len(rows)} rows -> {a.out}/ebank_legacy_rows.jsonl")


def legacy_dossier_key(task, name, legacy_dir):
    """Recompute the original dossier-anchored key (frontier_ref of flip_functional_v2)."""
    votes = []
    for f, ex in ([(f"mbar_zxa_{task}_{ex}.npz", ex) for ex in LEGACY_SEL]
                  + [(f"mbar_zxaglm_{task}_{g}.npz", g) for g in ("glm-47", "glm-52")]):
        z = load_npz(os.path.join(legacy_dir, f))
        r = z.get(f"{name}||dossier")
        if r is not None:
            votes.append((np.asarray(r, float) > .5).astype(float))
    if len(votes) < 2:
        return None
    mean = np.stack(votes).mean(0)
    return np.where(mean > .5, 1, np.where(mean < .5, 0, -1))


def silver_extract(a):
    """Run ON SK3: dump the 300-item silver label arrays (judgement column, same subsample as
    _load_texts) for the sound-silver tasks -> <out>/ebank_silver.json."""
    from ..manifest import full_manifest, load_corpus_labels
    out = {}
    for t in SILVER_TASKS:
        try:
            entry = next(e for e in full_manifest().datasets
                         if e.task == t.replace("_", "-"))
            texts, labs, _ = load_corpus_labels(entry, 360)
            out[t] = labs[60:360] if len(labs) >= 360 else None
        except Exception as e:
            out[t] = None
            print(f"[silver] {t} SKIPPED: {e}")
    json.dump(out, open(os.path.join(a.out, "ebank_silver.json"), "w"))
    print("[silver] ->", os.path.join(a.out, "ebank_silver.json"),
          {t: (len(v) if v else None) for t, v in out.items()})


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["build", "select", "evaluate", "score",
                                    "silver-extract", "legacy-build"])
    p.add_argument("--root", default=".")
    p.add_argument("--out", required=True)
    p.add_argument("--freeze-dir", default=None)
    p.add_argument("--legacy-dir", default=None)
    p.add_argument("--judge", choices=VOTERS)
    a = p.parse_args(argv)
    if a.mode == "build" and not a.freeze_dir:
        sys.exit("build needs --freeze-dir")
    if a.mode == "evaluate" and not a.judge:
        sys.exit("evaluate needs --judge")
    if a.mode in ("legacy-build",) and not a.legacy_dir:
        sys.exit("legacy-build needs --legacy-dir")
    {"build": build, "select": select, "evaluate": evaluate, "score": score,
     "silver-extract": silver_extract, "legacy-build": legacy_build}[a.mode](a)


if __name__ == "__main__":                                   # spawn-safety (sk3 vLLM rule)
    main()
