#!/usr/bin/env python3
"""Pairwise-accuracy matrix: each source's binary same/different agreement vs each arbiter, on the
common labeled spectrum pairs. Sources: Llama v6 (partition->pairwise), GLM-4.7 pairwise (= the
GEPA-toward-5.2 WINNER, since pairwise GEPA reverted to baseline), GLM-5.2 pairwise. Reports
accuracy / recall / precision (source as a same-detector vs arbiter).
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

spec = {json.loads(l)["pid"]: json.loads(l) for l in open("outputs/analyses/spectrum_pairs.jsonl")}
def load(p):
    return {json.loads(l)["pid"]: json.loads(l)["label"] for l in open(p)} if os.path.exists(p) else {}
opus = load("outputs/analyses/spectrum_opus_labels.jsonl")
glm52 = load("outputs/analyses/spectrum_glm52_labels.jsonl")
glm47 = load("outputs/analyses/spectrum_glm47_labels.jsonl")

ex = json.load(open("outputs/analyses/structural_metrics/clusters_peer-review.json"))
v6_pred = {pid: 1 if (p["key_a"] in ex and p["key_b"] in ex and ex[p["key_a"]] == ex[p["key_b"]]) else 0
           for pid, p in spec.items()}

sources = [
    ("Llama v6 (partition->pairwise)", v6_pred),
    ("GLM-4.7 baseline pairwise", {pid: 1 if l == 2 else 0 for pid, l in glm47.items()}),
    ("GLM-4.7 tuned->5.2 pairwise", {pid: 1 if l == 2 else 0 for pid, l in glm47.items()}),  # ==baseline (GEPA reverted)
    ("GLM-5.2 pairwise", {pid: 1 if l == 2 else 0 for pid, l in glm52.items()}),
]


def metrics(pred, arb, pids):
    tp = fn = fp = tn = 0
    for pid in pids:
        if pid not in pred or pid not in arb:
            continue
        ps = pred[pid]; ag = 1 if arb[pid] == 2 else 0
        if ps and ag: tp += 1
        elif (not ps) and ag: fn += 1
        elif ps and (not ag): fp += 1
        else: tn += 1
    n = tp + fn + fp + tn
    return ((tp + tn) / n if n else None, tp / (tp + fn) if tp + fn else None,
            tp / (tp + fp) if tp + fp else None, tp, fn, fp, tn)


common = set(opus) & set(glm52) & set(glm47)
print(f"common labeled pairs (Opus & GLM-5.2 & GLM-4.7): {len(common)}")
print(f"Opus same-rate: {sum(1 for p in common if opus[p]==2)/len(common):.3f} | "
      f"GLM-5.2 same-rate: {sum(1 for p in common if glm52[p]==2)/len(common):.3f} | "
      f"GLM-4.7 same-rate: {sum(1 for p in common if glm47[p]==2)/len(common):.3f}")
fmt = lambda x: f"{x:.3f}" if x is not None else "  n/a"
for arbname, arb in [("Opus", opus), ("GLM-5.2", glm52)]:
    print(f"\n=== source pairwise accuracy vs {arbname} (on {len(common)} pairs) ===")
    print(f"{'source':34s} {'acc':>6s} {'recall':>7s} {'prec':>6s}  (tp fn fp tn)")
    print("-" * 72)
    for sname, spred in sources:
        if sname.startswith("GLM-5.2") and arbname == "GLM-5.2":
            continue  # self
        acc, rec, prec, tp, fn, fp, tn = metrics(spred, arb, common)
        tag = "  (=baseline, GEPA reverted)" if "tuned" in sname else ""
        print(f"{sname:34s} {fmt(acc):>6s} {fmt(rec):>7s} {fmt(prec):>6s}  ({tp:3d} {fn:3d} {fp:3d} {tn:3d}){tag}")
