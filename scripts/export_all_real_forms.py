"""Combine general + specific/hyper_specific canonical forms into one file for
nemotron embedding. Keeps only real, on-topic leaves (ok, canonical != null,
not off_topic); dedups by key; re-indexes per (bucket, task).

Inputs:  canon_sk3_prod.jsonl (general), canon_sk3_spec.jsonl (specific buckets)
Output:  canon_all_real_forms.jsonl  {task, bucket, idx, key, canonical}
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

OUT = Path("/Users/spangher/Projects/stanford-research/norm-research/outputs/analyses")


def load_real(path, default_bucket, seen):
    rows = []
    for line in path.open():
        r = json.loads(line)
        if not r.get("ok") or r.get("canonical") is None or r.get("off_topic"):
            continue
        if r["key"] in seen:
            continue
        seen.add(r["key"])
        r["bucket"] = r.get("bucket") or default_bucket
        rows.append(r)
    return rows


def main():
    seen: set = set()
    rows = load_real(OUT / "canon_sk3_prod.jsonl", "general", seen)
    spec_path = OUT / "canon_sk3_spec.jsonl"
    if spec_path.exists():
        rows += load_real(spec_path, "specific", seen)
    else:
        print("WARNING: canon_sk3_spec.jsonl not found — general only")

    groups = defaultdict(list)
    for r in rows:
        groups[(r["bucket"], r["task"])].append(r)

    n = 0
    counts = defaultdict(int)
    with (OUT / "canon_all_real_forms.jsonl").open("w") as f:
        for (bucket, task), rs in sorted(groups.items()):
            for idx, r in enumerate(rs):
                f.write(json.dumps({"task": task, "bucket": bucket, "idx": idx,
                                    "key": r["key"],
                                    "canonical": r["canonical"]}) + "\n")
                n += 1
            counts[bucket] += len(rs)
    print(f"wrote {n} real on-topic canonical forms  ({dict(counts)})")
    print(f"  -> {OUT/'canon_all_real_forms.jsonl'}")


if __name__ == "__main__":
    main()
