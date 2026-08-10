"""Final per-task articulability distribution after the v7 L3 re-judgement.

v6 classified all 24K clusters. The audit found L3 systematically inflated by a
technical-vocabulary confound. v7 (sharpened anti-jargon 2-vs-3 prompt) re-judged
every v6-L3 cluster. This script combines: non-L3 clusters keep their v6 score
(audit found L1/L2/L4 reliable); L3 clusters take their v7 score.

Output: outputs/analyses/articulability_v7final_per_task.parquet + printed tables.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
OUT = ROOT / "outputs" / "analyses"
FULL = OUT / "cluster_2axis_full.jsonl"
L3V7 = OUT / "cluster_2axis_l3_v7.jsonl"


def load_dedup(path):
    seen, rows = set(), []
    for line in path.open():
        try:
            r = json.loads(line)
        except Exception:
            continue
        res = r.get("result", {})
        if "_error" in res or not res:
            continue
        key = (r.get("medoid_name", ""), r.get("medoid_description", ""))
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)
    return rows


def main():
    v6 = load_dedup(FULL)
    v7_map = {}
    if L3V7.exists():
        for r in load_dedup(L3V7):
            v7_map[(r["medoid_name"], r["medoid_description"])] = r["result"].get("articulability")
    print(f"v6: {len(v6)} clusters | v7 re-judged L3: {len(v7_map)}\n")

    by_task = {}
    n_changed = Counter()
    for r in v6:
        res = r["result"]
        a6 = res.get("articulability")
        key = (r.get("medoid_name", ""), r.get("medoid_description", ""))
        a_final = a6
        if a6 == 3 and key in v7_map and v7_map[key] is not None:
            a_final = v7_map[key]
            if a_final != 3:
                n_changed[(3, a_final)] += 1
        by_task.setdefault(r["task"], []).append(a_final)

    print(f"v7 moved v6-L3 clusters: {dict(n_changed)}")
    total_l3 = sum(1 for r in v6 if r['result'].get('articulability') == 3)
    moved = sum(n_changed.values())
    print(f"  {moved}/{total_l3} v6-L3 clusters reassigned ({moved/max(1,total_l3)*100:.0f}%)\n")

    print("=" * 80)
    print("FINAL (v7-corrected) articulability per task")
    print("=" * 80)
    print(f"{'task':<24} {'n':>5}  {'L1':>5} {'L2':>5} {'L3':>5} {'L4':>5}  {'mean':>5}")
    out = []
    for t in sorted(by_task):
        vals = by_task[t]
        n = len(vals)
        c = Counter(vals)
        fr = {k: c.get(k, 0) / n * 100 for k in (1, 2, 3, 4)}
        mean = sum(k * c.get(k, 0) for k in (1, 2, 3, 4)) / n
        print(f"{t:<24} {n:>5}  {fr[1]:>4.0f}% {fr[2]:>4.0f}% {fr[3]:>4.0f}% {fr[4]:>4.0f}%  {mean:>5.2f}")
        out.append({"task": t, "n": n, **{f"L{k}_pct": fr[k] for k in (1,2,3,4)}, "mean": mean})

    pd.DataFrame(out).to_parquet(OUT / "articulability_v7final_per_task.parquet")
    print(f"\nwrote {OUT/'articulability_v7final_per_task.parquet'}")


if __name__ == "__main__":
    main()
