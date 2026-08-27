"""Criterion-6 reliability filter (2026-08-11): before quoting the plateaued-everywhere
share (362/1032 = 35.1%), screen each plateaued construct's REFERENCE reliability —
a flat curve against a noisy reference is unreliability, not articulable-ceiling.
Filter per plateaued construct (crowd panels, humor uses mbar285+sup; others mbar2):
  - voter_agreement: llama70b-vs-qwen25-72b raw agreement on all probes (2-voter ref
    keeps only agreements, so low agreement = few kept items + noisy construct).
  - kept_n: items surviving the ref (agreement count).
  - top_recovery: best non-voter local receiver recovery (mistral7b/phi4/gemma2-27b/
    qwen25-14b max) vs the ref — a plateau at HIGH recovery = genuine ceiling; a
    plateau at chance = never-transmitted (different claim).
RELIABLE-PLATEAU = agreement >= .70 AND kept_n >= 150 AND top_recovery >= .65.
Frontier spot-check (humor only, no new scoring): plateaued ∩ v3sets — gpt-oss
definition-arm recovery should sit at/below local top (consistency), not above
(which would contradict 'ceiling reached').
Output: outputs/analyses/reliability_filter_v1/
"""
import json
import os

import numpy as np

D = "outputs/articulation_story_20260810"
OUT = "outputs/analyses/reliability_filter_v1"
os.makedirs(OUT, exist_ok=True)
TASKS = ["humor", "creative_writing", "math", "peer_review", "news_homepages"]
NONVOTER_TOPS = ["qwen25-14b", "mistral7b", "phi4", "gemma2-27b"]


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


plateaued = [(r["task"], r["name"]) for r in fj if satgroup(r) == "plateaued"]
print(f"plateaued-everywhere set: {len(plateaued)}")

CR = {}
rows = []
for task, name in plateaued:
    if task not in CR:
        CR[task] = {ex: crowd(task, ex) for ex in ("llama70b", "qwen25-72b") + tuple(NONVOTER_TOPS)}
    va = CR[task]["llama70b"].get(name)
    vb = CR[task]["qwen25-72b"].get(name)
    if va is None or vb is None:
        rows.append({"task": task, "name": name, "status": "no-voter-rows"})
        continue
    a = (np.asarray(va, float) > .5).astype(int)
    b = (np.asarray(vb, float) > .5).astype(int)
    fin = np.isfinite(np.asarray(va, float)) & np.isfinite(np.asarray(vb, float))
    agree = float((a[fin] == b[fin]).mean())
    kept = int((a[fin] == b[fin]).sum())
    m = (a + b) / 2.0
    ref = np.where(m > .5, 1, np.where(m < .5, 0, -1))
    tops = []
    for ex in NONVOTER_TOPS:
        r = CR[task][ex].get(name)
        if r is not None:
            y = balanced(np.asarray(r, float), ref)
            if y is not None:
                tops.append(y)
    top = max(tops) if tops else None
    reliable = agree >= .70 and kept >= 150 and top is not None and top >= .65
    rows.append({"task": task, "name": name, "agreement": round(agree, 3), "kept_n": kept,
                 "top_recovery": round(top, 3) if top is not None else None,
                 "reliable_plateau": bool(reliable), "status": "ok"})

ok_rows = [r for r in rows if r["status"] == "ok"]
rel = [r for r in ok_rows if r["reliable_plateau"]]
print(f"screened {len(ok_rows)}; RELIABLE plateaus: {len(rel)} "
      f"({100*len(rel)/max(1,len(ok_rows)):.1f}% of screened; "
      f"{100*len(rel)/1032:.1f}% of the 1,032 universe)")
from collections import Counter
print("reliable by task:", dict(Counter(r["task"] for r in rel)))
fail = Counter()
for r in ok_rows:
    if not r["reliable_plateau"]:
        if r["agreement"] < .70:
            fail["low-voter-agreement"] += 1
        elif r["kept_n"] < 150:
            fail["few-kept-items"] += 1
        elif r["top_recovery"] is None or r["top_recovery"] < .65:
            fail["plateau-at-low-recovery"] += 1
print("failure modes:", dict(fail))

# frontier spot-check: plateaued humor ∩ v3sets gptoss definition rows
led = json.load(open(f"{D}/analyses/v3sets_ledger_v1.json"))
gd = {r["b"]: r["definition"] for r in led["gpt-oss-120b"]}
spot = []
for r in rel:
    if r["task"] == "humor" and r["name"] in gd and r["top_recovery"] is not None:
        spot.append({"name": r["name"], "local_top": r["top_recovery"], "gptoss_def": gd[r["name"]]})
above = [s for s in spot if s["gptoss_def"] > s["local_top"] + .05]
print(f"frontier spot-check (humor reliable-plateaus with gptoss rows): {len(spot)}; "
      f"contradicted (frontier > local top + .05): {len(above)}")
json.dump({"thresholds": {"agreement": .70, "kept_n": 150, "top_recovery": .65},
           "rows": rows, "spot_check": spot},
          open(f"{OUT}/reliability_filter_v1.json", "w"), indent=0)
print("DONE ->", OUT)
