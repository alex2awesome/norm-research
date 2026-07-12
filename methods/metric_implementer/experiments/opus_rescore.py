#!/usr/bin/env python3
"""(1) Opus vs GLM-5.2 pairwise-judgment agreement on the spectrum, and (2) rescore every partition
(existing v6 + GLM-4.7 variants) against BOTH arbiters. GLM-5.2 spectrum labels are optional — present
once spectrum_glm52_labels.jsonl exists.
"""
import json, os, sys
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from methods.metric_implementer.experiments.glm_cluster import reconcile

SPEC = "outputs/analyses/spectrum_pairs.jsonl"
OPUS = "outputs/analyses/spectrum_opus_labels.jsonl"
GLM52 = "outputs/analyses/spectrum_glm52_labels.jsonl"
EX = "outputs/analyses/structural_metrics/clusters_peer-review.json"
GLM_RUNS = [("GLM-4.7 tuned+tfidf", "outputs/analyses/glm_cluster_peer_tuned.json"),
            ("GLM-4.7 tuned+openai", "outputs/analyses/glm_cluster_peer_tuned_openai.json"),
            ("GLM-4.7 baseline+tfidf", "outputs/analyses/glm_cluster_peer_pilot.json")]

spec = {json.loads(l)["pid"]: json.loads(l) for l in open(SPEC)}
opus = {json.loads(l)["pid"]: json.loads(l)["label"] for l in open(OPUS)}
glm52 = {json.loads(l)["pid"]: json.loads(l)["label"] for l in open(GLM52)} if os.path.exists(GLM52) else {}
keys300 = set(json.load(open("outputs/analyses/glm_cluster_peer_tuned_openai.json"))["keys"])
spec300 = {pid: p for pid, p in spec.items() if p["key_a"] in keys300 and p["key_b"] in keys300}

print(f"spectrum pairs: {len(spec)} | among-300-forms: {len(spec300)}")
print(f"Opus dist:   {dict(Counter(opus.values()))}")
if glm52:
    print(f"GLM-5.2 dist: {dict(Counter(glm52.values()))}")

# ---- (1) Opus vs GLM-5.2 agreement ----
if glm52:
    common = [p for p in opus if p in glm52]
    agree = sum(1 for p in common if opus[p] == glm52[p])
    # collapse to binary same(2) vs not
    def binv(x):
        return 1 if x == 2 else 0
    bagree = sum(1 for p in common if binv(opus[p]) == binv(glm52[p]))
    tp = sum(1 for p in common if opus[p] == 2 and glm52[p] == 2)
    print(f"\n(1) Opus vs GLM-5.2 on {len(common)} common pairs:")
    print(f"    exact-label agreement: {agree/len(common):.3f}")
    print(f"    binary(same/not) agreement: {bagree/len(common):.3f}  (both-same: {tp})")
    # confusion 3x3
    conf = Counter((glm52[p], opus[p]) for p in common)
    print("    confusion (rows=GLM-5.2, cols=Opus):")
    for g in (0, 1, 2):
        print(f"      {g}: " + " ".join(f"{conf.get((g,o),0):4d}" for o in (0, 1, 2)))
else:
    print("\n(1) Opus vs GLM-5.2: GLM-5.2 spectrum labels not ready yet (rerun later).")


def score(part, pairset, labels):
    s2 = s0 = tog2 = sep0 = 0
    for pid, p in pairset.items():
        g = labels.get(pid)
        ka, kb = p["key_a"], p["key_b"]
        if g is None or ka not in part or kb not in part:
            continue
        same = part[ka] == part[kb]
        if g == 2:
            s2 += 1; tog2 += same
        elif g == 0:
            s0 += 1; sep0 += (not same)
    rec = tog2 / s2 if s2 else None
    pre = sep0 / s0 if s0 else None
    return rec, pre, s2, s0


# ---- (2) rescore partitions on Opus (and GLM-5.2 if present) ----
ex = json.load(open(EX))
_fmt = lambda x: f"{x:.3f}" if x is not None else "  n/a"
print(f"\n(2) partitions vs arbiters  [n_same, n_diff used]")
print(f"{'partition':30s} {'arbiter':8s} {'recall':>7s} {'precision':>9s}  (n2 n0)")
print("-" * 70)
for arbname, arb in [("Opus", opus), ("GLM-5.2", glm52)]:
    if not arb:
        continue
    r, p, s2, s0 = score(ex, spec, arb)
    print(f"{'existing v6/Llama':30s} {arbname:8s} {_fmt(r):>7s} {_fmt(p):>9s}  ({s2} {s0})  [all {len(spec)} spectrum pairs]")
    for name, path in GLM_RUNS:
        d = json.load(open(path))
        if "batch_glob" not in d:
            if "glm_cid" in d:
                r, p, s2, s0 = score(d["glm_cid"], spec300, arb)
                print(f"{name+' (stored mv)':30s} {arbname:8s} {_fmt(r):>7s} {_fmt(p):>9s}  ({s2} {s0})  [{len(spec300)} pairs]")
            continue
        for mv in (1, 2):
            cid = reconcile(d["batch_glob"], d["n_forms"], mv)
            part = {d["keys"][i]: cid[i] for i in range(d["n_forms"])}
            r, p, s2, s0 = score(part, spec300, arb)
            print(f"{name+f' mv={mv}':30s} {arbname:8s} {_fmt(r):>7s} {_fmt(p):>9s}  ({s2} {s0})  [{len(spec300)} pairs]")
    print()
