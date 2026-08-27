"""Build audit samples from the full 2-axis classification, for manual /
subagent re-assessment of possible language confounds.

Three audit files (deduped by medoid name+desc):
  audit1_L3.jsonl     — patents + notice-and-comment articulability=3 clusters
                        (jargon-confound test: really L3, or L2 mis-bumped?)
  audit2_L2.jsonl     — 30 articulability=2 clusters per task
                        (should any be L1 / code-checkable?)
  audit3_random.jsonl — 10 random clusters per (articulability score, task)
"""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
OUT = ROOT / "outputs" / "analyses"
FULL = OUT / "cluster_2axis_full.jsonl"
AUD = OUT / "audit_samples"
AUD.mkdir(parents=True, exist_ok=True)
random.seed(15)


def load_dedup():
    seen, rows = set(), []
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


def rec(r):
    res = r["result"]
    return {
        "task": r["task"],
        "medoid_name": r.get("medoid_name", ""),
        "medoid_description": r.get("medoid_description", ""),
        "cluster_size": r.get("cluster_size"),
        "assigned": {
            "articulability": res.get("articulability"),
            "articulability_why": res.get("articulability_why", ""),
            "reasoning_depth": res.get("reasoning_depth"),
            "indeterminacy": res.get("indeterminacy"),
            "surface_vs_substance": res.get("surface_vs_substance"),
        },
    }


def main():
    rows = load_dedup()
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)

    # audit 1: patents + N&C L3
    a1 = []
    for task, cap in [("patents", 999), ("notice-and-comment", 60)]:
        l3 = [r for r in by_task[task] if r["result"].get("articulability") == 3]
        random.shuffle(l3)
        a1 += [rec(r) for r in l3[:cap]]
    (AUD / "audit1_L3.jsonl").write_text("\n".join(json.dumps(x) for x in a1))
    print(f"audit1_L3.jsonl: {len(a1)} clusters (patents + N&C articulability=3)")

    # audit 2: 30 L2 per task
    a2 = []
    for task in sorted(by_task):
        l2 = [r for r in by_task[task] if r["result"].get("articulability") == 2]
        random.shuffle(l2)
        a2 += [rec(r) for r in l2[:30]]
    (AUD / "audit2_L2.jsonl").write_text("\n".join(json.dumps(x) for x in a2))
    print(f"audit2_L2.jsonl: {len(a2)} clusters (<=30 articulability=2 per task)")

    # audit 3: 10 random per (score, task)
    a3 = []
    for task in sorted(by_task):
        for score in (1, 2, 3, 4):
            pool = [r for r in by_task[task] if r["result"].get("articulability") == score]
            random.shuffle(pool)
            a3 += [rec(r) for r in pool[:10]]
    (AUD / "audit3_random.jsonl").write_text("\n".join(json.dumps(x) for x in a3))
    print(f"audit3_random.jsonl: {len(a3)} clusters (<=10 per score per task)")


if __name__ == "__main__":
    main()
