"""Export specific + hyper_specific bucket leaf names for canonicalization.

Mirrors export_leaf_names.py but for the two non-general buckets. The general
bucket was already canonicalized (canon_sk3_prod.jsonl).

Output: outputs/analyses/_sk3_leaf_input_spec.jsonl
        {task, bucket, idx, key, name, cluster_id}   (idx per task+bucket)
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
BUCKETS = ["specific", "hyper_specific"]


def main():
    n = 0
    per_bucket = {b: 0 for b in BUCKETS}
    with (OUT / "_sk3_leaf_input_spec.jsonl").open("w") as f:
        for task in TASKS:
            for bucket in BUCKETS:
                p = HIER / f"{task}_{bucket}_r1_refined.json"
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
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
                            f.write(json.dumps({
                                "task": task, "bucket": bucket, "idx": idx,
                                "key": r.get("key", ""), "name": nm,
                                "cluster_id": cid}) + "\n")
                            idx += 1
                            n += 1
                            per_bucket[bucket] += 1
    print(f"wrote {n} leaves  ({per_bucket})  -> {OUT/'_sk3_leaf_input_spec.jsonl'}")


if __name__ == "__main__":
    main()
