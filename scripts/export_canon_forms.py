"""Export the real, on-topic canonical leaf forms for re-embedding + re-clustering.

From canon_sk3_prod.jsonl, keep leaves that are: ok, canonical != null (not a
bare heading), and not off_topic. Dedup by key (the input has multi-homed
duplicate keys). One row per unique leaf, grouped by task with a per-task idx.

Output: outputs/analyses/canon_real_forms.jsonl  {task, idx, key, name, canonical}
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
OUT = ROOT / "outputs" / "analyses"


def main():
    rows = [json.loads(l) for l in (OUT / "canon_sk3_prod.jsonl").open()]
    seen, per_task = set(), {}
    for r in rows:
        if not r.get("ok") or r.get("canonical") is None or r.get("off_topic"):
            continue
        if r["key"] in seen:
            continue
        seen.add(r["key"])
        per_task.setdefault(r["task"], []).append(r)

    n = 0
    with (OUT / "canon_real_forms.jsonl").open("w") as f:
        for task in sorted(per_task):
            for idx, r in enumerate(per_task[task]):
                f.write(json.dumps({"task": task, "idx": idx, "key": r["key"],
                                    "name": r["name"],
                                    "canonical": r["canonical"]}) + "\n")
                n += 1
            print(f"  {task:<26} {len(per_task[task])}")
    print(f"\nwrote {n} real on-topic canonical forms -> {OUT/'canon_real_forms.jsonl'}")


if __name__ == "__main__":
    main()
