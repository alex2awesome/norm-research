"""Channel x scale exhibit v2 (criterion 2 fold-in, 2026-08-11). Local-only.

Coverage vs the criterion's channel list (name, definition, units, examples, thinking):
- definition/rubric: FULL BANK (humor 284) x 11 local receivers (crowd panels).
- examples (flip-selected functional): FULL BANK at llama70b (v3 holdout), qwen25-72b,
  gpt-oss-120b (v3sets); slate x 12 receivers (flip ladder).
- name: FULL BANK at llama70b (v3 holdout name arm); slate x ladder (zxa panels).
- explanation/dossier: slate x ladder only (disclosed).
- thinking: slate x qwen3 rungs only (disclosed; gen panels).
- "units": no construct-side scored unit channel exists; the 4.1 unit decomposition is
  definitional (codability), not a scored channel — mapping DISCLOSED.
Classes: slate classes (PLANTED/TACIT/REACHES/DIALECT) where defined; bank-scale rows
carry blind category (norm-boundary/mechanics/other) + saturation group. Headline
cells get 20k paired bootstraps. Output: outputs/analyses/channel_scale_v2/.
"""
import json
import os

import numpy as np

D = "outputs/articulation_story_20260810"
OUT = "outputs/analyses/channel_scale_v2"
os.makedirs(OUT, exist_ok=True)
QF = ["qwen25-3b", "qwen25-7b", "qwen25-14b", "qwen25-72b"]
LF = ["llama1b", "llama3b", "llama8b", "llama70b"]


def lp(p):
    if not os.path.exists(p):
        return {}
    z = np.load(p, allow_pickle=True)
    return {str(n): z["m_bar"][i] for i, n in enumerate([str(x) for x in z["names"]])}


def balanced(pred, lab):
    ok = (lab >= 0) & np.isfinite(pred)
    if ok.sum() < 30:
        return None
    p = (pred[ok] > .5).astype(int)
    l = lab[ok]
    accs = [float(np.mean(p[l == c] == c)) for c in (0, 1) if (l == c).sum() >= 5]
    return float(np.mean(accs)) if len(accs) == 2 else None


rng = np.random.default_rng(0)


def ci(d):
    d = np.asarray(d, float)
    idx = rng.integers(0, len(d), size=(20000, len(d)))
    bm = d[idx].mean(1)
    return round(float(d.mean()), 4), round(float(np.percentile(bm, 2.5)), 4), \
        round(float(np.percentile(bm, 97.5)), 4)


# ---- bank-scale: humor crowd panels (rubric channel), all local receivers ----
def crowd(ex):
    d = {}
    d.update(lp(f"{D}/crowd_panels/mbar285_{ex}.npz"))
    d.update(lp(f"{D}/crowd_panels/mbar2_humor_sup_{ex}.npz"))
    return d


EXES = ["llama1b", "llama3b", "llama8b", "llama70b", "qwen25-3b", "qwen25-7b",
        "qwen25-14b", "qwen25-72b", "mistral7b", "phi4", "gemma2-27b"]
CR = {ex: crowd(ex) for ex in EXES}


VOTERS = ("llama70b", "qwen25-72b")


def cref(base, scored_ex=None):
    # leave-one-out: a voter rung is never scored against a ref containing its own vote
    # (with 2 voters, kept items are exactly the agreements -> voter recovery = 1.0 by
    # construction otherwise). Voter rungs therefore measure cross-voter agreement
    # (dialect-biased DOWN); non-voters measure vs the 2-voter consensus. Disclosed.
    votes = []
    for ex in VOTERS:
        if ex == scored_ex:
            continue
        r = CR[ex].get(base)
        if r is not None:
            votes.append((np.asarray(r, float) > .5).astype(float))
    if not votes:
        return None
    m = np.stack(votes).mean(0)
    return np.where(m > .5, 1, np.where(m < .5, 0, -1))


cat = json.load(open(f"{D}/analyses/metric_categories_blind_v1.json"))
fj = json.load(open(f"{D}/analyses/family_verdict_join_v1.json"))["full_rows"]


def satgroup(r, t=.02):
    tm = [v for v in r["top_minus_mid"].values() if v is not None]
    if len(tm) < 3:
        return None
    if all(v > t for v in tm):
        return "rising"
    if all(v <= t for v in tm):
        return "plateaued"
    return "family-dependent"


SAT = {r["name"]: satgroup(r) for r in fj if r["task"] == "humor"}

bank_rows = []
bases = sorted({b for b in CR["llama70b"] if not b.startswith("PLANTED")})
for b in bases:
    row = {"base": b, "category": cat.get(b), "sat": SAT.get(b)}
    any_valid = False
    for ex in EXES:
        r = CR[ex].get(b)
        lab = cref(b, scored_ex=ex if ex in VOTERS else None)
        y = balanced(np.asarray(r, float), lab) if (r is not None and lab is not None) else None
        row[ex] = y
        any_valid = any_valid or (y is not None)
    if any_valid:
        bank_rows.append(row)
print(f"bank rubric-channel rows: {len(bank_rows)}")

# per-class curves (blind category), rubric channel over capability
exhibit = {"bank_rubric_curves": {}, "bank_examples": {}, "headline_cells": {},
           "coverage_disclosures": [
               "explanation/dossier channels: slate-only (41 bases)",
               "thinking channel: slate x qwen3 rungs only",
               "'units' channel: no construct-side scored instrument; 4.1 unit "
               "decomposition is definitional (codability), not a channel score",
               "name channel at bank: llama70b holdout only"]}
for grp_name, grp_of in (("category", lambda r: r["category"]),
                         ("saturation", lambda r: r["sat"])):
    curves = {}
    for g in sorted({grp_of(r) for r in bank_rows if grp_of(r)}):
        sub = [r for r in bank_rows if grp_of(r) == g]
        curve = {}
        for ex in EXES:
            vals = [r[ex] for r in sub if r[ex] is not None]
            curve[ex] = {"mean": round(float(np.mean(vals)), 4), "n": len(vals)} if vals else None
        curves[g] = curve
    exhibit["bank_rubric_curves"][grp_name] = curves

# headline paired-bootstrap cells: top-vs-bottom rung per class (qwen family; paired over bases)
for g in sorted({r["category"] for r in bank_rows if r["category"]}):
    sub = [r for r in bank_rows if r["category"] == g
           and r["qwen25-3b"] is not None and r["qwen25-72b"] is not None]
    if len(sub) >= 6:
        m, lo, hi = ci([r["qwen25-72b"] - r["qwen25-3b"] for r in sub])
        exhibit["headline_cells"][f"rubric|{g}|qwen72b-minus-3b"] = \
            {"n": len(sub), "delta": m, "ci": [lo, hi]}

# ---- bank-scale examples channel: name/def/examples at llama70b (v3 holdout) + 2 receivers ----
v3 = json.load(open(f"{D}/flips/flip_functional_v3_llama70b.json"))["results"]["humor"]
trip = []
for b, rec in v3.items():
    h = rec["objectives"].get("frontier2v", {}).get("holdout", {})
    if all(h.get(k) is not None for k in ("name", "definition", "functional")):
        trip.append({"base": b, "category": cat.get(b), **{k: h[k] for k in
                                                           ("name", "definition", "functional")}})
exhibit["bank_examples"]["llama70b_holdout_n"] = len(trip)
for arm_a, arm_b, lbl in (("definition", "name", "def-minus-name"),
                          ("functional", "name", "examples-minus-name"),
                          ("functional", "definition", "examples-minus-def")):
    m, lo, hi = ci([t[arm_a] - t[arm_b] for t in trip])
    exhibit["headline_cells"][f"bank|llama70b|{lbl}"] = {"n": len(trip), "delta": m, "ci": [lo, hi]}
led = json.load(open(f"{D}/analyses/v3sets_ledger_v1.json"))
for rcv in ("qwen25-72b", "gpt-oss-120b"):
    rows = led[rcv]
    m, lo, hi = ci([r["functional"] - r["definition"] for r in rows])
    exhibit["headline_cells"][f"bank|{rcv}|examples-minus-def"] = {"n": len(rows), "delta": m, "ci": [lo, hi]}
    m, lo, hi = ci([r["functional"] - r["functionalmm"] for r in rows])
    exhibit["headline_cells"][f"bank|{rcv}|examples-content"] = {"n": len(rows), "delta": m, "ci": [lo, hi]}

json.dump(exhibit, open(f"{OUT}/channel_scale_v2.json", "w"), indent=1)

print("\n=== bank rubric channel: per-category curve (qwen family means) ===")
for g, c in exhibit["bank_rubric_curves"]["category"].items():
    line = " ".join(f"{ex.split('-')[-1]}={c[ex]['mean']:.3f}" for ex in QF if c.get(ex))
    print(f"{g:14s} {line}")
print("\n=== headline cells (paired 20k CIs) ===")
for k, v in exhibit["headline_cells"].items():
    star = "*" if v["ci"][0] > 0 or v["ci"][1] < 0 else " "
    print(f"{k:44s} n={v['n']:3d} {v['delta']:+.4f}{star} CI[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]")
print("\nDONE ->", f"{OUT}/channel_scale_v2.json")
