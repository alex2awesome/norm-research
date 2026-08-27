"""Approach A.2: Re-batch R1 family centroids and run R1 prompt at meta level.

Treats each R1 family as a "cluster" with:
  - centroid = mean of member L0 cluster centroids in LoRA-bge space (L2-normed)
  - representative text = family name + ": " + short description
  - support = 3 sampled member L0 cluster reps for grounding

Then cover-once batches the families with the LoRA-bge centroids and emits
chat messages with a META-level R1 prompt asking the LLM to identify which
families should be merged because they're really the same rule.

Output:
  /tmp/r1_meta_merge/<task>/batches/batch_<i>.txt  (subagent prompt files)
  /tmp/r1_meta_merge/<task>/batches.jsonl          (per-batch records w/ family idx list)
  /tmp/r1_meta_merge/<task>/family_meta.json       (fi -> name/desc/clusters/centroid_idx)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import sk3_build_r1
sk3_build_r1.EMB = Path("notebooks/_explore_cache/bge_lora")
from sk3_build_r1 import TASK_INFO, cluster_data, load_task, make_batches


META_SYSTEM = """You are reviewing the output of a rubric-clustering system applied to {task}.

Background:
{background}

A previous LLM pass grouped raw rubric clusters into "rule families" — each family is meant to represent a single underlying rule, just expressed in different wording across the rubric sources. But the system was conservative and likely split some families that should have been together.

Your task: review the list of families and identify which should be MERGED because they actually express the SAME underlying rule. Two families should be merged ONLY IF the rule statement they encode is essentially the same — not merely thematically related.

Examples of WHEN TO MERGE:
- "Methods reported in sufficient detail to replicate" and "Methods are described with enough detail for reproduction" → SAME rule
- "Figures should be clear and focused" and "Tables and figures should be clear and easy to understand" → SAME rule (about visual asset clarity)
- "Code is released for reproducibility" and "Source code and dependencies are downloadable" → SAME rule

Examples of WHEN NOT TO MERGE:
- "Methodology is appropriate for the research question" and "Methodology is rigorously justified" → DIFFERENT rules (fit vs justification)
- "Conclusions are not misleading" and "Limitations are properly disclosed" → DIFFERENT rules (truthfulness vs scope-marking)
- "Data is openly accessible" and "Code is openly accessible" → DIFFERENT rules (data vs code; could be enforced separately)

Output a JSON object listing every family in exactly one group. Singleton groups (1 family) mean "no merge — keep this family as-is". Multi-family groups mean "merge these into one R1 family".

Schema:
{{"groups": [{{"name": "<canonical name for the merged family>", "description": "<one-sentence description>", "members": ["F123", "F456", ...]}}, ...]}}

Every "F<n>" id that appears in the USER must appear in exactly one group."""


META_FEWSHOT_USER = """Here are 10 candidate families. Group families that should be merged into the same rule.

F1: Methods reported in detail sufficient to replicate
    desc: Methods, procedures, and techniques should be described with enough detail to allow independent reproduction.
    members: ["The work should provide sufficient methodological detail to allow for reproducibility.", "The methods used should be clearly and transparently described.", "The Materials & Methods section should describe a methodology that is replicable."]

F2: Code and reproducibility artifacts are released
    desc: Source code, scripts, and reproducibility packages should be made available.
    members: ["The work should provide supporting resources, such as code and data, to facilitate reproducibility.", "The work's code and checklist should be sufficient to reproduce the results."]

F3: Work is reproducible by independent researchers
    desc: The work, its results, and workflows should be reproducible by others.
    members: ["The work should be reproducible by others.", "The results should be reproducible by others."]

F4: Computational results are reproducible
    desc: Computational results should be reproducible and the software used properly reported.
    members: ["The computational results should be reproducible by others."]

F5: Methods are clearly described and transparent
    desc: Methods, assumptions, and design choices should be presented transparently.
    members: ["The methods should be reported transparently and be reproducible.", "The methods and assumptions should be clearly presented."]

F6: Compute requirements for reproduction are reported
    desc: The paper should report the compute resources required for reproducing the work.
    members: ["The document should include the compute requirements and discuss the practicality of reproducing the work."]

F7: Methodology is appropriate for research question
    desc: The research design and methodology should fit the research question.
    members: ["The methodology used should be appropriate for the research question and objectives."]

F8: Methodology choice is justified with rationale
    desc: The chosen methodology should come with a justification.
    members: ["The choice of methodology should be clearly justified with a compelling rationale."]

F9: Figures and tables are clear and legible
    desc: Visual assets should be clear, legible, and accurately represent the data.
    members: ["Figures should be clear and focused.", "Tables and figures should be clear, accurate, and necessary."]

F10: Graphs and plots are reproducible
    desc: Figures, graphs, and plots should be reproducible.
    members: ["The graphs and plots in the work should be reproducible by others."]"""


META_FEWSHOT_ASSISTANT = """{
 "groups": [
  {
   "name": "Methods reported in sufficient detail to replicate",
   "description": "Methods, procedures, techniques, and assumptions should be described transparently with enough detail to allow independent reproduction.",
   "members": ["F1", "F5"]
  },
  {
   "name": "Code and reproducibility artifacts are released",
   "description": "Source code, scripts, dependencies, and reproducibility packages should be made available.",
   "members": ["F2"]
  },
  {
   "name": "Work is reproducible by independent researchers",
   "description": "The work, its computational results, and its figures should be reproducible by others.",
   "members": ["F3", "F4", "F10"]
  },
  {
   "name": "Compute requirements for reproduction are reported",
   "description": "The paper should report the compute resources required for reproducing the work.",
   "members": ["F6"]
  },
  {
   "name": "Methodology is appropriate for research question",
   "description": "The research design and methodology should fit the research question.",
   "members": ["F7"]
  },
  {
   "name": "Methodology choice is justified with rationale",
   "description": "The chosen methodology should come with a justification.",
   "members": ["F8"]
  },
  {
   "name": "Figures and tables are clear and legible",
   "description": "Visual assets should be clear, legible, and accurately represent the data.",
   "members": ["F9"]
  }
 ]
}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="peer-review")
    ap.add_argument("--r1-dir", default="r1_v4a_subagent_lora_bs400")
    ap.add_argument("--batch-size", type=int, default=400)
    ap.add_argument("--output-dir", default="/tmp/r1_meta_merge")
    args = ap.parse_args()

    out = Path(args.output_dir) / args.task
    (out / "batches").mkdir(parents=True, exist_ok=True)
    (out / "responses").mkdir(parents=True, exist_ok=True)

    # Load L0 embeddings + clusters
    forms = [json.loads(l)
             for l in open("outputs/analyses/canon_all_real_forms.jsonl")
             if json.loads(l)["task"] == args.task]
    rows, emb = load_task(args.task, forms)
    cl_path = Path("outputs/analyses/structural_metrics") / f"clusters_{args.task}.json"
    cl = json.loads(cl_path.read_text())
    reps, centroids, members = cluster_data(rows, emb, cl)

    # Load R1 families
    r1 = json.loads((Path("outputs/analyses/structural_metrics") / args.r1_dir
                     / f"r1_families_{args.task}.json").read_text())
    fams = r1["families"]
    print(f"loaded {len(fams)} families from {args.r1_dir}")

    # Compute family centroids + support text per family
    fam_centroids = {}
    fam_meta = []
    for fi, f in enumerate(fams):
        cids = f.get("cluster_ids") or f.get("members") or []
        cids = [int(str(c).lstrip("C")) for c in cids
                if str(c).lstrip("C").isdigit()]
        cids = [c for c in cids if c in centroids]
        if not cids:
            continue
        v = np.stack([centroids[c] for c in cids]).mean(0)
        v /= (np.linalg.norm(v) + 1e-9)
        fam_centroids[fi] = v
        # Sample up to 3 member rep texts as support
        support = [reps[c] for c in cids[:3] if c in reps]
        fam_meta.append({
            "fi": fi,
            "name": f.get("name", ""),
            "description": f.get("description", ""),
            "n_clusters": len(cids),
            "cluster_ids": cids,
            "support": support,
        })
    (out / "family_meta.json").write_text(json.dumps(fam_meta, indent=1))

    # Cover-once batching on family centroids
    fis = list(fam_centroids.keys())
    batches, _ = make_batches(fis, fam_centroids, args.batch_size)
    print(f"{len(fis)} families -> {len(batches)} batches (bs={args.batch_size})")

    # Materialize prompt files
    info = TASK_INFO[args.task]
    system = META_SYSTEM.format(task=args.task, background=info["background"])
    meta_by_fi = {m["fi"]: m for m in fam_meta}
    with (out / "batches.jsonl").open("w") as bjf:
        for bi, batch in enumerate(batches):
            user_lines = ["Here are the families. Group them into rule families "
                          "(every F<n> in exactly one group; singletons OK):"]
            for fi in batch:
                m = meta_by_fi[fi]
                user_lines.append("")
                user_lines.append(f"F{fi}: {m['name']}")
                user_lines.append(f"    desc: {m['description']}")
                if m['support']:
                    sup = ", ".join(json.dumps(s[:120]) for s in m['support'])
                    user_lines.append(f"    members: [{sup}]")
            user_msg = "\n".join(user_lines)

            prompt_text = (
                system
                + "\n\n=== FEW-SHOT USER ===\n" + META_FEWSHOT_USER
                + "\n\n=== FEW-SHOT ASSISTANT ===\n" + META_FEWSHOT_ASSISTANT
                + "\n\n=== USER ===\n" + user_msg
            )
            (out / "batches" / f"batch_{bi}.txt").write_text(prompt_text)
            bjf.write(json.dumps({
                "batch_idx": bi,
                "family_ids": [int(x) for x in batch],
            }) + "\n")

    print(f"wrote {len(batches)} batch prompts -> {out}/batches/")


if __name__ == "__main__":
    main()
