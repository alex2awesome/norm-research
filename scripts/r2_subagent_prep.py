"""R2: group R1 rule-families into thematic super-families ("aspects").

Different from R1: R1 was about merging same-rule families; R2 is about
grouping different-but-related rules under a shared theme/aspect (e.g.,
"Methodology" theme containing several distinct rules about methods).

Approach:
  1. Compute family centroids in LoRA-bge space (one per Fork3-merged R1 family).
  2. Cover-once batch family centroids at bs=400.
  3. Subagent partitions a batch of 400 R1 families into R2 super-families.

Output:
  /tmp/r2_subagent/<task>/batches/batch_<i>.txt
  /tmp/r2_subagent/<task>/batches.jsonl
  /tmp/r2_subagent/<task>/family_meta.json
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


R2_SYSTEM = """You are organizing rules into thematic aspects for a {task} evaluation system.

Background:
{background}

A previous clustering pass produced "rule families" — each family encodes one specific rule (e.g., "Methods are reported in sufficient detail" or "Figures are clearly labeled"). The goal NOW is to group these specific rules into broader THEMATIC ASPECTS that share a unifying concern. Think of each aspect as a chapter in a guidelines manual: rules about reproducibility go in the "Reproducibility" chapter, rules about figure quality in the "Visual Presentation" chapter, rules about statistical methods in the "Statistical Methodology" chapter, etc.

What counts as a single THEMATIC ASPECT:
- The rules share a unifying CONCERN that a reviewer would naturally check together.
- A practical R2 aspect usually covers 3-20 R1 families. Singletons are fine for truly unique rules.
- The aspect name should be a noun phrase naming the area of concern (e.g., "Methodological Rigor", "Citation Practices", "Figure Quality") — NOT a rule statement.

What does NOT count as a single aspect:
- Don't lump everything about "writing" into one giant aspect — distinguish "Clarity of Prose" from "Grammar/Mechanics" from "Argumentation Structure".
- Don't merge surface concerns with substance concerns: "Capitalization rules" should NOT join "Argument validity".

Output a JSON object listing every R1 family in exactly one aspect. Singleton aspects (one family) are allowed.

Schema:
{{"aspects": [{{"name": "<aspect name as noun phrase>", "description": "<one sentence>", "members": ["F12", "F34", ...]}}, ...]}}

Every F<n> id in the USER input must appear in exactly one aspect."""


R2_FEWSHOT_USER = """Here are 12 R1 rule-families. Group them into thematic aspects:

F1: Methods reported in sufficient detail to replicate
    desc: Methods, procedures, and techniques should be described with enough detail to allow independent reproduction.

F2: Methodology is appropriate for the research question
    desc: The research design should fit the research question.

F3: Statistical methods are described clearly
    desc: Statistical methods used should be clearly described and appropriate.

F4: Code is released for reproducibility
    desc: Source code and dependencies should be made available.

F5: Data is shared and openly accessible
    desc: Underlying data should be made openly available.

F6: Pre-register research for transparency
    desc: Studies should be pre-registered to prevent post-hoc framing.

F7: Figures are clear and legible
    desc: Figures should be readable, with clear labels and units.

F8: Tables follow consistent formatting
    desc: Table layouts, units, and column headers should be consistent.

F9: Use clear and concise language
    desc: Sentences should be precise and avoid unnecessary jargon.

F10: Citations support all empirical claims
    desc: Every empirical claim in the manuscript should have a supporting citation.

F11: Cited references should be current and relevant
    desc: Citations should be to recent and topical work.

F12: No plagiarism or duplicate publication
    desc: Submitted work should not duplicate prior or other published work."""


R2_FEWSHOT_ASSISTANT = """{
 "aspects": [
  {
   "name": "Methodological Rigor",
   "description": "Methods should be appropriate, well-described, and reported in sufficient detail.",
   "members": ["F1", "F2", "F3"]
  },
  {
   "name": "Open Science Practices",
   "description": "Code, data, and protocols should be openly shared and pre-registered.",
   "members": ["F4", "F5", "F6"]
  },
  {
   "name": "Visual Presentation",
   "description": "Figures and tables should be clear, legible, and consistently formatted.",
   "members": ["F7", "F8"]
  },
  {
   "name": "Clarity of Prose",
   "description": "Writing should be clear, precise, and concise.",
   "members": ["F9"]
  },
  {
   "name": "Citation Practices",
   "description": "Citations should support claims and reference current relevant work.",
   "members": ["F10", "F11"]
  },
  {
   "name": "Publication Ethics",
   "description": "Work should be original, free of plagiarism or duplicate publication.",
   "members": ["F12"]
  }
 ]
}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="peer-review")
    ap.add_argument("--r1-dir", default="r1_v4a_lora_fork3_merge")
    ap.add_argument("--batch-size", type=int, default=400)
    ap.add_argument("--output-dir", default="/tmp/r2_subagent")
    args = ap.parse_args()

    out = Path(args.output_dir) / args.task
    (out / "batches").mkdir(parents=True, exist_ok=True)
    (out / "responses").mkdir(parents=True, exist_ok=True)

    forms = [json.loads(l)
             for l in open("outputs/analyses/canon_all_real_forms.jsonl")
             if json.loads(l)["task"] == args.task]
    rows, emb = load_task(args.task, forms)
    cl = json.loads((Path("outputs/analyses/structural_metrics")
                     / f"clusters_{args.task}.json").read_text())
    reps, centroids, members = cluster_data(rows, emb, cl)

    r1 = json.loads((Path("outputs/analyses/structural_metrics") / args.r1_dir
                     / f"r1_families_{args.task}.json").read_text())
    fams = r1["families"]

    fam_centroids = {}
    fam_meta = []
    for fi, f in enumerate(fams):
        cids = f.get("cluster_ids") or []
        cids = [int(str(c).lstrip("C")) for c in cids
                if str(c).lstrip("C").isdigit()]
        cids = [c for c in cids if c in centroids]
        if not cids:
            continue
        v = np.stack([centroids[c] for c in cids]).mean(0)
        v /= (np.linalg.norm(v) + 1e-9)
        fam_centroids[fi] = v
        fam_meta.append({
            "fi": fi,
            "name": f.get("name", ""),
            "description": f.get("description", ""),
            "n_clusters": len(cids),
        })
    (out / "family_meta.json").write_text(json.dumps(fam_meta, indent=1))

    fis = list(fam_centroids.keys())
    batches, _ = make_batches(fis, fam_centroids, args.batch_size)
    print(f"{args.task}: {len(fis)} R1 families -> {len(batches)} R2 batches "
          f"(bs={args.batch_size})")

    meta_by_fi = {m["fi"]: m for m in fam_meta}
    info = TASK_INFO[args.task]
    system = R2_SYSTEM.format(task=args.task, background=info["background"])

    with (out / "batches.jsonl").open("w") as bjf:
        for bi, batch in enumerate(batches):
            user_lines = ["Here are the R1 families. Group them into thematic "
                          "aspects (every F<n> in exactly one aspect; "
                          "singletons OK for unique rules):"]
            for fi in batch:
                m = meta_by_fi[fi]
                user_lines.append("")
                user_lines.append(f"F{fi}: {m['name']}")
                user_lines.append(f"    desc: {m['description']}")
            user_msg = "\n".join(user_lines)

            prompt_text = (
                system
                + "\n\n=== FEW-SHOT USER ===\n" + R2_FEWSHOT_USER
                + "\n\n=== FEW-SHOT ASSISTANT ===\n" + R2_FEWSHOT_ASSISTANT
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
