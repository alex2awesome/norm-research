"""Analyze GEPA lineage joined with specificity class (subtask_breadth):
  (1) which OPERATORS fire in which breadth class,
  (2) per-operator Δlength (child-parent instruction_tokens) and Δfidelity — does the change grow
      or shrink the prompt, and does it help,
  (3) per-breadth INIT length, best-fidelity rubric length, best fidelity — which classes converge
      to LONGER rubrics (need more articulation) vs tap out short.
"""
import json
import sys
import collections
import numpy as np

OUT = "/lfs/skampere3/0/alexspan/tmp_vinfo"
fname = sys.argv[1] if len(sys.argv) > 1 else f"{OUT}/gepa_lineage.json"
rows = json.load(open(fname))
V = {r["version_id"]: r for r in rows if r.get("version_id")}
CLASSES = ["very_narrow", "narrow", "moderate", "broad", "very_broad"]


def m(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


print(f"file={fname}  versions={len(rows)}  metrics={len({(r['task'],r['metric_id']) for r in rows})}\n")

print("=== (1) operator counts by breadth class (non-INIT mutations) ===")
ct = collections.defaultdict(collections.Counter)
for r in rows:
    if r["operator"] and r["operator"] != "INIT":
        ct[r["breadth"]][r["operator"]] += 1
for b in CLASSES:
    if ct[b]:
        print(f"  {b:12s} {dict(ct[b])}")

print("\n=== (2) per-operator: Δlength (child-parent instr_tokens) and Δfidelity ===")
dl = collections.defaultdict(list)
df = collections.defaultdict(list)
for r in rows:
    op = r["operator"]
    if not op or op == "INIT":
        continue
    par = V.get(r.get("parent"))
    if par:
        if r["instruction_tokens"] is not None and par["instruction_tokens"] is not None:
            dl[op].append(r["instruction_tokens"] - par["instruction_tokens"])
        if r["fidelity"] is not None and par["fidelity"] is not None:
            df[op].append(r["fidelity"] - par["fidelity"])
print(f"  {'operator':10s} {'Δlen':>8s} {'n':>4s}   {'Δfidelity':>10s} {'n':>4s}   grows?")
for op in sorted(set(dl) | set(df)):
    L = m(dl.get(op, [])); F = m(df.get(op, []))
    print(f"  {op:10s} {L:+8.1f} {len(dl.get(op,[])):>4d}   {F:+10.3f} {len(df.get(op,[])):>4d}   "
          f"{'GROWS' if L>0 else 'shrinks'}")

print("\n=== (3) per-breadth: INIT length -> best-fidelity rubric length (need longer rubrics?) ===")
byb = collections.defaultdict(list)
for r in rows:
    byb[r["breadth"]].append(r)
print(f"  {'breadth':12s} {'INIT_len':>9s} {'best_len':>9s} {'best_fid':>9s} {'Δlen(best-INIT)':>16s}")
for b in CLASSES:
    rs = byb.get(b, [])
    if not rs:
        continue
    inits = m([r["instruction_tokens"] for r in rs if r["operator"] == "INIT"])
    withf = [r for r in rs if r["fidelity"] is not None and r["instruction_tokens"] is not None]
    best = max(withf, key=lambda r: r["fidelity"]) if withf else None
    bl = best["instruction_tokens"] if best else float("nan")
    bf = best["fidelity"] if best else float("nan")
    print(f"  {b:12s} {inits:9.0f} {bl:9.0f} {bf:9.3f} {bl-inits:+16.0f}")
