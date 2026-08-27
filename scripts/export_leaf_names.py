"""Export leaf-rubric NAMES (not the LLM-elaborated cluster descriptions) for
re-embedding on sk3.

The earlier singleton audit embedded `medoid_name + medoid_description`. The
medoid DESCRIPTION is LLM-generated per cluster and diverges between two
clusters that mean the same thing — which masked genuine duplicates. The leaf
`name` is the short, verbatim rubric text and is the right unit for measuring
how often the same concept recurs.

Output: outputs/analyses/_sk3_leaf_input.jsonl
        {task, idx, key, name, cluster_id}   (idx = row order per task)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
HIER = ROOT / "outputs" / "hierarchy"
OUT = ROOT / "outputs" / "analyses"

TASKS = [
    "code-review", "creative-writing", "grant-funding", "humor",
    "legal-outcome-prediction", "math-stackexchange", "news-homepages",
    "notice-and-comment", "patents", "peer-review", "press-releases",
]


def main():
    n = 0
    with (OUT / "_sk3_leaf_input.jsonl").open("w") as f:
        for task in TASKS:
            d = json.loads((HIER / f"{task}_general_r1_refined.json").read_text())
            seen = set()
            idx = 0
            for par in d.get("parented_trees", []):
                for ch in par.get("children", []):
                    k = (ch.get("medoid_name", ""), ch.get("medoid_description", ""))
                    if k in seen:
                        continue
                    seen.add(k)
                    cid = ch.get("cluster_id", "")
                    for r in ch.get("rubrics", []):
                        nm = r.get("name", "") or ""
                        if not nm.strip():
                            continue
                        f.write(json.dumps({"task": task, "idx": idx,
                                            "key": r.get("key", ""), "name": nm,
                                            "cluster_id": cid}) + "\n")
                        idx += 1
                        n += 1
    print(f"wrote {n} leaf rows -> {OUT/'_sk3_leaf_input.jsonl'}")


if __name__ == "__main__":
    main()
