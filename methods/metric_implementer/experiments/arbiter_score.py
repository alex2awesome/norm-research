#!/usr/bin/env python3
"""Score every partition (existing v6/Llama + each GLM run, re-reconciled at min_votes in {1,2})
against the arbiter labels. recall = arbiter-same pairs kept together; precision = arbiter-different
pairs kept apart. This is the 'better than before?' verdict (arbiter = GLM-5.2; confirm with Opus).
"""
import json, os, sys
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from methods.metric_implementer.experiments.glm_cluster import reconcile

PAIRS = "outputs/analyses/arbiter_pairs.jsonl"
LABELS = "outputs/analyses/arbiter_labels.jsonl"
EX = "outputs/analyses/structural_metrics/clusters_peer-review.json"
GLM_RUNS = [("GLM-4.7 baseline+tfidf", "outputs/analyses/glm_cluster_peer_pilot.json"),
            ("GLM-4.7 tuned+tfidf", "outputs/analyses/glm_cluster_peer_tuned.json"),
            ("GLM-4.7 tuned+openai", "outputs/analyses/glm_cluster_peer_tuned_openai.json")]

pairs = {json.loads(l)["pid"]: json.loads(l) for l in open(PAIRS)}
labels = {json.loads(l)["pid"]: json.loads(l)["label"] for l in open(LABELS)}

parts = {}
parts["existing (v6/Llama)"] = json.load(open(EX))
for name, path in GLM_RUNS:
    d = json.load(open(path))
    if "batch_glob" in d and "keys" in d:
        for mv in (1, 2):
            cid = reconcile(d["batch_glob"], d["n_forms"], mv)
            parts[f"{name} mv={mv}"] = {d["keys"][i]: cid[i] for i in range(d["n_forms"])}
    elif "glm_cid" in d:
        parts[f"{name} (stored)"] = d["glm_cid"]


def score(part):
    s2 = s0 = tog2 = sep0 = 0
    for pid, p in pairs.items():
        lab = labels.get(pid)
        ka, kb = p["key_a"], p["key_b"]
        if lab is None or ka not in part or kb not in part:
            continue
        same = part[ka] == part[kb]
        if lab == 2:
            s2 += 1; tog2 += same
        elif lab == 0:
            s0 += 1; sep0 += (not same)
    rec = tog2 / s2 if s2 else None
    pre = sep0 / s0 if s0 else None
    f1 = (2 * rec * pre / (rec + pre)) if (rec and pre) else None
    return rec, pre, f1, s2, s0


c = Counter(labels.values())
print(f"arbiter (GLM-5.2) label dist: same(2)={c[2]}  diff(0)={c[0]}  borderline(1)={c[1]}")
print(f"{'partition':38s} {'recall':>7s} {'precision':>9s} {'F1':>6s}   (n_same n_diff)")
print("-" * 72)
for name, part in parts.items():
    rec, pre, f1, s2, s0 = score(part)
    print(f"{name:38s} {rec:7.3f} {pre:9.3f} {f1:6.3f}   ({s2:4d} {s0:4d})")
