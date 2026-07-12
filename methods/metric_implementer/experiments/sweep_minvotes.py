#!/usr/bin/env python3
"""Offline min_votes sweep on a saved glm_cluster run (which now stores batch_glob + keys).
Re-reconciles with min_votes in {1,2,3} and re-scores recall/precision vs v6 — ZERO GLM calls.

Usage: python -m methods.metric_implementer.experiments.sweep_minvotes <run.json> [task]
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from methods.metric_implementer.experiments.glm_cluster import reconcile, compare, existing_clusters_for

path = sys.argv[1]
d = json.load(open(path))
task = sys.argv[2] if len(sys.argv) > 2 else d["task"]
keys = d["keys"]; n = len(keys); bg = d["batch_glob"]
ex = existing_clusters_for(keys, task)
print(f"{os.path.basename(path)}  rule={d.get('rule','?')[:40]!r}  n={n}  batches={len(bg)}")
for mv in (1, 2, 3):
    cid = reconcile(bg, n, mv)
    c = compare(cid, keys, ex, task)
    print(f"  min_votes={mv}: recall={c['v6_score2_kept_together']}  prec={c['v6_score0_kept_apart']}  "
          f"(n_s2={c['n_v2']} n_s0={c['n_v0']})  rand_vs_ex={c['rand_vs_existing']}  glm_same={c['glm_same_rate']}")
