"""
Build per-task labeling inputs for the full R2 attribute-tagging pass.

For each R2 aspect from structural_metrics/r2_v1_subagent/, include:
  - aspect name + description
  - 3 R1 family examples (name + description) from
    structural_metrics/r1_v4a_lora_fork3_merge/ via source_family_ids

Writes outputs/r2_attr_labeling/inputs/<task>.jsonl

Run:
  python scripts/build_r2_labeling_input.py
"""
import json
from pathlib import Path

ROOT = Path("/Users/spangher/Projects/stanford-research/norm-research")
R2_DIR = ROOT / "outputs/analyses/structural_metrics/r2_v1_subagent"
R1_DIR = ROOT / "outputs/analyses/structural_metrics/r1_v4a_lora_fork3_merge"
OUT = ROOT / "outputs/r2_attr_labeling/inputs"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = [
    "code-review", "creative-writing", "grant-funding", "humor",
    "legal-outcome-prediction", "math-stackexchange", "news-homepages",
    "notice-and-comment", "patents", "peer-review", "press-releases",
]

N_R1_EXAMPLES = 3


def main():
    totals = {}
    for task in TASKS:
        r2 = json.loads((R2_DIR / f"r2_aspects_{task}.json").read_text())
        r1 = json.loads((R1_DIR / f"r1_families_{task}.json").read_text())
        r1_fams = r1.get("families", [])

        aspects = r2.get("aspects", [])
        task_slug = task.replace("-", "_")
        out_path = OUT / f"{task_slug}.jsonl"
        with open(out_path, "w") as f:
            for i, a in enumerate(aspects):
                src_ids = a.get("family_ids", []) or a.get("source_family_ids", []) or []
                # take up to N R1 examples, in original order
                examples = []
                for sid in src_ids[:N_R1_EXAMPLES]:
                    if 0 <= sid < len(r1_fams):
                        fam = r1_fams[sid]
                        examples.append({
                            "name": fam.get("name", ""),
                            "description": fam.get("description", ""),
                        })
                row = {
                    "task": task_slug,
                    "aspect_id": a.get("aspect_id", f"r2_{i:04d}"),
                    "name": a.get("name", ""),
                    "description": a.get("description", ""),
                    "n_source_r1": len(src_ids),
                    "r1_examples": examples,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        totals[task] = (len(aspects), out_path.name)
        print(f"  {task:30s} {len(aspects):>4} aspects -> {out_path.name}")

    grand = sum(v[0] for v in totals.values())
    print(f"\nwrote labeling inputs for {len(totals)} tasks, total {grand} aspects.")


if __name__ == "__main__":
    main()
