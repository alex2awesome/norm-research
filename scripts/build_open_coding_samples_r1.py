"""
Sample N R1 families (Fork3-merged) per task for open-ended coding.

Writes one JSONL per task to outputs/attr_open_coding_r1/inputs/<task>.jsonl
with rows of {task, r1_id, name, description, n_clusters, n_source_fams}.

Source: outputs/analyses/structural_metrics/r1_v4a_lora_fork3_merge/r1_families_<task>.json

Run:
  python scripts/build_open_coding_samples_r1.py --n 100 --seed 11
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
SRC  = ROOT / "outputs/analyses/structural_metrics/r1_v4a_lora_fork3_merge"
OUT  = ROOT / "outputs/attr_open_coding_r1/inputs"
OUT.mkdir(parents=True, exist_ok=True)

# Tasks use kebab-case in the R1 files
TASKS = [
    "code-review",
    "creative-writing",
    "grant-funding",
    "humor",
    "legal-outcome-prediction",
    "math-stackexchange",
    "news-homepages",
    "notice-and-comment",
    "patents",
    "peer-review",
    "press-releases",
]


def load_families(task: str):
    p = SRC / f"r1_families_{task}.json"
    if not p.exists():
        return None
    obj = json.loads(p.read_text())
    fams = obj.get("families", [])
    out = []
    for i, f in enumerate(fams):
        out.append({
            "r1_id": f"r1_{i:04d}",
            "name": f.get("name", ""),
            "description": f.get("description", ""),
            "n_clusters": len(f.get("cluster_ids", [])),
            "n_source_fams": len(f.get("source_family_ids", [])),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    grand = 0
    for task in TASKS:
        rows = load_families(task)
        if rows is None:
            print(f"!! missing {task}")
            continue
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        picks = rows[: args.n]
        # Use snake_case for consistency with R2 pipeline
        task_slug = task.replace("-", "_")
        out_path = OUT / f"{task_slug}.jsonl"
        with open(out_path, "w") as f:
            for r in picks:
                f.write(json.dumps({"task": task_slug, **r},
                                   ensure_ascii=False) + "\n")
        print(f"  {task:30s} {len(rows):>5} families total -> "
              f"sampled {len(picks)} -> {out_path.name}")
        grand += len(picks)

    print(f"\nwrote samples for {len(TASKS)} tasks, total {grand} families.")


if __name__ == "__main__":
    main()
