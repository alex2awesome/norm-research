"""Extend smoke run with code-gen prompts for ALL (or top-K) R1 members
within each R2 aspect — for the aggregation experiment.

For each smoke R2 aspect, generate code-gen prompts for its top-K R1 sub-families.
Each prompt uses the R1 framing (specific rule + 3 member texts).

Output:
  runs/validity_pilot/<run>/codegen/prompts_r1_within_r2/<aspect_idx>__<fi>.txt
  runs/validity_pilot/<run>/codegen/r1_within_r2_manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from validity_pilot_codegen_prep import CODEGEN_SYSTEM


def r1_user(name, description, samples):
    samples_block = "\n".join(f"  - {s[:140]}" for s in samples)
    return (
        f"Rubric NAME: \"{name}\"\n"
        f"DESCRIPTION: {description}\n\n"
        f"Examples of equivalent rubric statements (same underlying rule, "
        f"different wording):\n{samples_block}\n\n"
        f"Write the score(text: str) function.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="smoke")
    ap.add_argument("--task", default="peer-review")
    ap.add_argument("--top-k", type=int, default=5,
                    help="Top-K R1 families per R2 aspect (by n_clusters desc)")
    args = ap.parse_args()

    base = Path(f"runs/validity_pilot/{args.run_name}")
    metrics = json.loads((base / "metrics.json").read_text())

    r2 = json.loads(Path("outputs/analyses/structural_metrics/r2_v1_subagent/"
                         f"r2_aspects_{args.task}.json").read_text())
    r1 = json.loads(Path("outputs/analyses/structural_metrics/"
                         f"r1_v4a_lora_fork3_merge/r1_families_{args.task}.json"
                         ).read_text())
    reps = json.loads(Path(f"/tmp/r1_subagent_lora_bs400/{args.task}/"
                           "clusters_repr.json").read_text())
    fam_meta = json.loads(Path(f"/tmp/r2_subagent/{args.task}/"
                               "family_meta.json").read_text())

    cg = base / "codegen"
    (cg / "prompts_r1_within_r2").mkdir(parents=True, exist_ok=True)
    (cg / "responses_r1_within_r2_llama").mkdir(parents=True, exist_ok=True)

    manifest = []
    n_written = 0
    for m in metrics:
        # find the aspect
        asp = next((a for a in r2["aspects"] if a["name"] == m["r2_aspect_name"]), None)
        if not asp:
            continue
        # rank R1 families by n_clusters desc
        ranked = []
        for fi in asp["family_ids"]:
            f = r1["families"][fi]
            n_cl = len(f.get("cluster_ids", []))
            ranked.append((fi, f, n_cl))
        ranked.sort(key=lambda x: -x[2])
        chosen = ranked[:args.top_k]

        for fi, f, n_cl in chosen:
            cids = [int(str(c).lstrip("C")) for c in f.get("cluster_ids", [])
                    if str(c).lstrip("C").isdigit()]
            samples = [reps[str(c)] for c in cids[:5] if str(c) in reps]
            user = r1_user(f.get("name", ""), f.get("description", ""), samples)
            key = f"{m['metric_id']}__F{fi}"
            prompt = CODEGEN_SYSTEM + "\n\n=== USER ===\n" + user
            (cg / "prompts_r1_within_r2" / f"{key}.txt").write_text(prompt)
            manifest.append({
                "key": key,
                "metric_id": m["metric_id"],          # = R2 aspect smoke id
                "r2_aspect_name": m["r2_aspect_name"],
                "r1_family_id": fi,
                "r1_family_name": f.get("name", ""),
                "n_clusters": n_cl,
            })
            n_written += 1

    (cg / "r1_within_r2_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {n_written} prompts -> {cg}/prompts_r1_within_r2/")
    print(f"manifest -> {cg}/r1_within_r2_manifest.json")
    # Per-aspect summary
    from collections import Counter
    by_aspect = Counter(m["metric_id"] for m in manifest)
    print("\nPer-aspect coverage:")
    for mid, n in sorted(by_aspect.items()):
        print(f"  {mid}: {n} R1 families queued")


if __name__ == "__main__":
    main()
