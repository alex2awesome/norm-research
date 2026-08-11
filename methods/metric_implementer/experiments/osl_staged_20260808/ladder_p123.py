"""Criterion-5: freeze rung assignments from full-bank data, THEN score the frozen
abstractness prereg P1-P3 (notes/2026-08-05, frozen 2026-08-06).

RUNG RULE (fixed before any correlate is computed; first match wins):
  R1 name-articulable:        humor bank llama70b holdout name >= .75
  R2 statement-articulable:   definition/rubric strong — humor: holdout def >= .75;
                              all tasks fallback: best NON-VOTER local crowd recovery >= .75
  R3 demonstration-articulable: certified exemplar content (fun-mm >= +.05, either receiver)
  R4 listener-bound:          still-gaining tail OR family-dependent saturation
  R5 low-ceiling/uncertified: remainder
Coverage: R1/R3 humor-instrumented (other tasks can't enter those rungs — disclosed).

P1: Brysbaert Conc.M mean over content words of name+definition decreases down rungs
    (Spearman rung-index vs concreteness, predict rho < 0).
P2: beyond-text share (osl_metric_types coarse axis, kappa .73) increases down rungs.
P3: R2 vs R3 indistinguishable in concreteness (predict n.s.) but different in 9-type
    profile (predict gestalt/holistic enriched in R3; chi-square on type counts).
Output: outputs/analyses/ladder_p123_v1/
"""
import glob
import json
import os
import re
from collections import Counter

import numpy as np

D = "outputs/articulation_story_20260810"
OUT = "outputs/analyses/ladder_p123_v1"
os.makedirs(OUT, exist_ok=True)
SP = "/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research/5dab3f74-48cc-4698-a04d-3d07440a91bf/scratchpad"

# ---- reconstruct O-id -> (dom, name) mapping (osl_deep join logic, assert-verified) ----
types = json.load(open("outputs/analyses/osl_metric_types_20260728.json"))
labels, key = types["labels"], types["key"]
AX = {"MECHANICAL_CHECK": 0, "CRAFT_OPERATION": 0, "EVIDENCE_RIGOR": 0, "HOLISTIC_TASTE": 0,
      "AUDIENCE_FIT": 1, "COMMUNITY_TRANSFORM": 1,
      "IDENTITY_PERSONA": 2, "EXTERNAL_BUNDLE": 2, "RECEPTION_OUTCOME": 2}
oid = 0
TYPE = {}
for f in sorted(glob.glob("notebooks/data/2026-07-07-osl-multi/curves_*.json")):
    dom = os.path.basename(f)[7:-5]
    for name, v in json.load(open(f)).items():
        if v.get("verdict") not in ("RISING", "REACHES", "BOUNDED"):
            continue
        i = "O%04d" % oid
        assert key[i]["verdict"] == v["verdict"] and key[i]["dom"] == dom, (i, name)
        lab = labels[i]["label"] if isinstance(labels[i], dict) else labels[i]
        TYPE[(dom, name)] = {"type9": lab, "axis": AX[lab]}
        oid += 1
print(f"type labels joined: {len(TYPE)} (of 1270)")

# ---- rung assignment inputs ----
fj = json.load(open(f"{D}/analyses/family_verdict_join_v1.json"))["full_rows"]
led = json.load(open(f"{D}/analyses/v3sets_ledger_v1.json"))
tail = {(r["task"], r["name"]) for r in json.load(open(f"{D}/flips/decensor_tail_v1.json"))}
v3 = json.load(open(f"{D}/flips/flip_functional_v3_llama70b.json"))["results"]["humor"]
content = {}
for rcv in ("qwen25-72b", "gpt-oss-120b"):
    for r in led[rcv]:
        content.setdefault(r["b"], {})[rcv] = r["functional"] - r["functionalmm"]
hold = {b: rec["objectives"].get("frontier2v", {}).get("holdout", {}) for b, rec in v3.items()}


def lp(p):
    if not os.path.exists(p):
        return {}
    z = np.load(p, allow_pickle=True)
    return {str(n): z["m_bar"][i] for i, n in enumerate([str(x) for x in z["names"]])}


def crowd(task, ex):
    d = {}
    if task == "humor":
        d.update(lp(f"{D}/crowd_panels/mbar285_{ex}.npz"))
        d.update(lp(f"{D}/crowd_panels/mbar2_humor_sup_{ex}.npz"))
    else:
        d.update(lp(f"{D}/crowd_panels/mbar2_{task}_{ex}.npz"))
    return d


def balanced(pred, lab):
    ok = (lab >= 0) & np.isfinite(pred)
    if ok.sum() < 30:
        return None
    p = (pred[ok] > .5).astype(int)
    l = lab[ok]
    accs = [float(np.mean(p[l == c] == c)) for c in (0, 1) if (l == c).sum() >= 5]
    return float(np.mean(accs)) if len(accs) == 2 else None


NONVOTER_TOPS = ["qwen25-14b", "mistral7b", "phi4", "gemma2-27b"]
CR = {}


def top_recovery(task, name):
    if task not in CR:
        CR[task] = {ex: crowd(task, ex) for ex in ("llama70b", "qwen25-72b") + tuple(NONVOTER_TOPS)}
    va, vb = CR[task]["llama70b"].get(name), CR[task]["qwen25-72b"].get(name)
    if va is None or vb is None:
        return None
    a = (np.asarray(va, float) > .5).astype(int)
    b = (np.asarray(vb, float) > .5).astype(int)
    m = (a + b) / 2.0
    ref = np.where(m > .5, 1, np.where(m < .5, 0, -1))
    tops = [balanced(np.asarray(CR[task][ex][name], float), ref)
            for ex in NONVOTER_TOPS if CR[task][ex].get(name) is not None]
    tops = [t for t in tops if t is not None]
    return max(tops) if tops else None


def satgroup(r, t=.02):
    tm = [v for v in r["top_minus_mid"].values() if v is not None]
    if len(tm) < 3:
        return None
    if all(v > t for v in tm):
        return "rising"
    if all(v <= t for v in tm):
        return "plateaued"
    return "family-dependent"


RUNGS = ["R1-name", "R2-statement", "R3-demonstration", "R4-listener-bound", "R5-low-ceiling"]
rows = []
for r in fj:
    task, name = r["task"], r["name"]
    h = hold.get(name, {}) if task == "humor" else {}
    c = content.get(name, {}) if task == "humor" else {}
    sat = satgroup(r)
    if task == "humor" and h.get("name") is not None and h["name"] >= .75:
        rung = "R1-name"
    elif (task == "humor" and h.get("definition") is not None and h["definition"] >= .75) or \
         ((task != "humor" or h.get("definition") is None) and (top_recovery(task, name) or 0) >= .75):
        rung = "R2-statement"
    elif any(v >= .05 for v in c.values()):
        rung = "R3-demonstration"
    elif (task, name) in tail or sat == "family-dependent":
        rung = "R4-listener-bound"
    else:
        rung = "R5-low-ceiling"
    rows.append({"task": task, "name": name, "rung": rung, "rung_idx": RUNGS.index(rung)})
json.dump({"rule": "frozen per script header", "rows": rows},
          open(f"{OUT}/rung_assignments_frozen.json", "w"), indent=0)
print("rung counts:", dict(Counter(r["rung"] for r in rows)))

# ---- P1: concreteness ----
import csv
CONC = {}
with open(f"{SP}/brysbaert.csv") as fh:
    for rec in csv.DictReader(fh, delimiter="\t"):
        try:
            CONC[rec["Word"].lower()] = float(rec["Conc.M"])
        except (ValueError, KeyError):
            pass
print(f"concreteness norms: {len(CONC)} words")
DEFS = {}
for t in ("humor", "creative_writing", "math", "peer_review", "news_homepages"):
    for m in json.load(open(f"{D}/code_metrics/defs_{t}.json")):
        DEFS[(t, m["name"])] = m["definition"]
STOP = set("the a an and or of to in for with on by is are be as that this it its not no".split())


def conc_of(text):
    toks = [w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP]
    vals = [CONC[w] for w in toks if w in CONC]
    return float(np.mean(vals)) if len(vals) >= 5 else None


for r in rows:
    txt = r["name"] + " " + DEFS.get((r["task"], r["name"]), "")
    r["conc"] = conc_of(txt)
    tp = TYPE.get((r["task"], r["name"]))
    r["type9"] = tp["type9"] if tp else None
    r["axis"] = tp["axis"] if tp else None

ok = [r for r in rows if r["conc"] is not None]
from scipy import stats
rho, p = stats.spearmanr([r["rung_idx"] for r in ok], [r["conc"] for r in ok])
print(f"\nP1: Spearman(rung, concreteness) rho={rho:+.3f} p={p:.2e} n={len(ok)} "
      f"(prereg predicts rho<0)")
print("   mean concreteness by rung:",
      {g: round(float(np.mean([r['conc'] for r in ok if r['rung'] == g])), 3)
       for g in RUNGS if any(r['rung'] == g for r in ok)})

axr = [r for r in rows if r["axis"] is not None]
rho2, p2 = stats.spearmanr([r["rung_idx"] for r in axr],
                           [1.0 * (r["axis"] == 2) for r in axr])
print(f"\nP2: Spearman(rung, beyond-text) rho={rho2:+.3f} p={p2:.2e} n={len(axr)} "
      f"(prereg predicts rho>0)")
print("   beyond-text share by rung:",
      {g: round(float(np.mean([r['axis'] == 2 for r in axr if r['rung'] == g])), 3)
       for g in RUNGS if any(r['rung'] == g for r in axr)})

r2 = [r for r in ok if r["rung"] == "R2-statement"]
r3 = [r for r in ok if r["rung"] == "R3-demonstration"]
if len(r3) >= 5:
    u, pu = stats.mannwhitneyu([r["conc"] for r in r2], [r["conc"] for r in r3])
    print(f"\nP3a: concreteness R2 (n={len(r2)}, m={np.mean([r['conc'] for r in r2]):.3f}) vs "
          f"R3 (n={len(r3)}, m={np.mean([r['conc'] for r in r3]):.3f}): MW p={pu:.3f} "
          f"(prereg predicts n.s.)")
    c2 = Counter(r["type9"] for r in rows if r["rung"] == "R2-statement" and r["type9"])
    c3 = Counter(r["type9"] for r in rows if r["rung"] == "R3-demonstration" and r["type9"])
    cats = sorted(set(c2) | set(c3))
    tab = np.array([[c2.get(c, 0) for c in cats], [c3.get(c, 0) for c in cats]])
    keep = tab.sum(0) > 0
    chi, pc, dof, _ = stats.chi2_contingency(tab[:, keep])
    print(f"P3b: 9-type profile R2 vs R3: chi2={chi:.1f} p={pc:.4f} (prereg predicts differs)")
    print("   R3 profile:", dict(c3.most_common(5)))
    print("   R2 top:", dict(c2.most_common(5)))
json.dump({"rows": rows, "P1": {"rho": rho, "p": p}, "P2": {"rho": rho2, "p": p2}},
          open(f"{OUT}/ladder_p123_v1.json", "w"), indent=0)
print("\nDONE ->", OUT)
