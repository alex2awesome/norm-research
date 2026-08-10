"""Aggregate the full cluster 2-axis classification into per-task distributions.

Dedups multi-homed clusters by (medoid_name, medoid_description) — the ~1.7%
hierarchy-builder inflation — then reports per-task distributions over all four
axes: articulability (primary), reasoning_depth, indeterminacy, surface.

Output: outputs/analyses/cluster_2axis_per_task.parquet + printed tables.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
OUT = ROOT / "outputs" / "analyses"
FULL = OUT / "cluster_2axis_full.jsonl"

AXES = ["articulability", "reasoning_depth", "indeterminacy", "surface_vs_substance"]


def load_dedup() -> list[dict]:
    seen = set()
    rows = []
    for line in FULL.open():
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
    rows = load_dedup()
    print(f"{len(rows)} distinct clusters (after dedup)\n")

    per_task = {}
    for r in rows:
        t = r["task"]
        per_task.setdefault(t, []).append(r["result"])

    # primary axis: articulability, per task, as fractions
    out_rows = []
    print("=" * 78)
    print("ARTICULABILITY per task (% of clusters at each level)")
    print(f"  1=code  2=LLM-judge  3=expert  4=tacit")
    print("=" * 78)
    print(f"{'task':<26} {'n':>5}  {'L1':>5} {'L2':>5} {'L3':>5} {'L4':>5}  {'mean':>5}")
    for t in sorted(per_task):
        results = per_task[t]
        n = len(results)
        c = Counter(x.get("articulability") for x in results)
        fr = {k: c.get(k, 0) / n * 100 for k in (1, 2, 3, 4)}
        mean = sum(k * c.get(k, 0) for k in (1, 2, 3, 4)) / n
        print(f"{t:<26} {n:>5}  {fr[1]:>4.0f}% {fr[2]:>4.0f}% {fr[3]:>4.0f}% {fr[4]:>4.0f}%  {mean:>5.2f}")
        row = {"task": t, "n_clusters": n, "artic_mean": mean}
        for k in (1, 2, 3, 4):
            row[f"artic_L{k}_pct"] = fr[k]
        out_rows.append(row)

    # diagnostic axes: per-task mean
    print()
    print("=" * 78)
    print("DIAGNOSTIC AXES — per-task mean (1-4)")
    print("=" * 78)
    print(f"{'task':<26} {'reasoning_depth':>16} {'indeterminacy':>14} {'surface':>9}")
    for t in sorted(per_task):
        results = per_task[t]
        n = len(results)
        rd = sum(x.get("reasoning_depth", 0) for x in results) / n
        ind = sum(x.get("indeterminacy", 0) for x in results) / n
        sv = sum(x.get("surface_vs_substance", 0) for x in results) / n
        print(f"{t:<26} {rd:>16.2f} {ind:>14.2f} {sv:>9.2f}")
        for row in out_rows:
            if row["task"] == t:
                row["reasoning_depth_mean"] = rd
                row["indeterminacy_mean"] = ind
                row["surface_mean"] = sv

    df = pd.DataFrame(out_rows)
    df.to_parquet(OUT / "cluster_2axis_per_task.parquet")
    print(f"\nwrote {OUT/'cluster_2axis_per_task.parquet'}")


if __name__ == "__main__":
    main()
