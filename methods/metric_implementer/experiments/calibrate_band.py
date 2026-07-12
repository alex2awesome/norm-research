#!/usr/bin/env python3
"""Calibrate the ambiguous band from the arbiter labels: bin the 500 GLM-5.2-labeled pairs by
openai cosine sim, report same-rate (label=2) and diff-rate (label=0) per bin. The band where
same-rate is neither ~0 nor ~1 is where embeddings can't decide confidently -> that's what GLM-4.7
must judge. Then map that band onto the all-3257 pair distribution for the GLM call count.
"""
import json, collections

PAIRS = sys.argv[1] if len(sys.argv) > 1 else "outputs/analyses/arbiter_pairs.jsonl"
LABELS = sys.argv[2] if len(sys.argv) > 2 else "outputs/analyses/arbiter_labels.jsonl"
pairs = {json.loads(l)["pid"]: json.loads(l) for l in open(PAIRS)}
labels = {json.loads(l)["pid"]: json.loads(l)["label"] for l in open(LABELS)}
rows = [(pairs[pid]["sim"], labels[pid]) for pid in pairs if pid in labels]

bins = collections.defaultdict(list)
for sim, lab in rows:
    bins[round(sim * 20) / 20].append(lab)   # 0.05-width bins

print(f"{'sim':>5} {'n':>4} {'same%':>6} {'diff%':>6}  verdict")
print("-" * 48)
for b in sorted(bins):
    labs = bins[b]
    n = len(labs)
    same = sum(1 for l in labs if l == 2) / n
    diff = sum(1 for l in labs if l == 0) / n
    if same >= 0.75:
        v = "confidently SAME (embedding can auto-merge)"
    elif diff >= 0.75:
        v = "confidently DIFF (embedding can auto-split)"
    else:
        v = "** AMBIGUOUS -> needs GLM **"
    print(f"{b:5.2f} {n:4d} {same*100:5.0f}% {diff*100:5.0f}%  {v}")
