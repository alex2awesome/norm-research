"""R2.5: cross-batch merge pass for R2 aspects.

Mirror of Fork 3 (R1 pairwise merge) but at R2 level. Goal: consolidate
aspects that are conceptually the same theme but were produced in different
batches (e.g., "Whitespace and Spacing" appearing in both code-review batch 6
and batch 8 with disjoint members).

Steps:
  1. Compute R2 aspect centroid = mean of constituent R1-family LoRA-bge
     centroids, L2-normed.
  2. Compute pairwise cosine across aspects.
  3. Filter to cos >= threshold (default 0.65; aspects are coarser-grained
     than R1 families so the natural threshold is lower).
  4. Batch K=20 candidate aspect-pairs per subagent prompt with the
     "are these the SAME thematic aspect?" prompt (modified Fork 3 style).
  5. Apply union-find to consolidate aspects across batches.

Outputs subagent batches at /tmp/r2_5/<task>/batches/. Run `r25_aspect_merge_apply.py`
after subagents return verdicts.
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
from sk3_build_r1 import TASK_INFO, cluster_data, load_task


R25_SYSTEM = """You are deciding whether two thematic aspects from a {task} clustering should be MERGED because they encode the same overall theme/concern.

Background:
{background}

Each aspect groups together specific rule-families that share a unifying concern (e.g., "Methodological Rigor" covers rules about methods being appropriate, well-described, and reported in detail).

Two aspects should be MERGED if and only if they cover the SAME concern at the same level of abstraction — even if produced in different batches with slightly different wording. They should NOT be merged if one is a sub-aspect of the other, or if they cover related-but-distinct concerns.

MERGE examples:
- "Code Formatting and Whitespace" + "Whitespace and Spacing" → SAME aspect
- "Methodological Rigor" + "Methodological Rigor and Soundness" → SAME aspect (slight wording variation)
- "Citation Practices" + "Citation Style and Quality" → SAME aspect

DO NOT MERGE:
- "Statistical Methodology" + "Methodological Rigor" → DIFFERENT (more specific vs more general)
- "Code Documentation" + "Code Comments Quality" → DIFFERENT (different sub-aspects)
- "Visual Presentation" + "Figure Quality" → DIFFERENT (more specific vs more general)

For each pair, output YES if they should be merged, NO otherwise.

Schema:
{{"verdicts": [{{"pair_idx": 1, "merge": "YES"|"NO"}}, ...]}}

Every pair_idx in the USER input must have a verdict."""


R25_FEWSHOT_USER = """Pair 1
A: "Code Formatting and Whitespace"
   desc: Standards for whitespace, indentation, line length, and other formatting.
   sample members: ["Use spaces inside parentheses", "Cap line length and use blank lines", "Choose tabs vs spaces consistently"]
B: "Whitespace and Spacing"
   desc: Rules about how whitespace should be placed in code.
   sample members: ["Whitespace after commas/colons", "Format semicolons: no space before, space after", "Use spaces before trailing comments"]

Pair 2
A: "Methodological Rigor"
   desc: Methods should be appropriate, sound, and well-justified.
   sample members: ["Methodology fits research question", "Methods are sound", "Methodology choices are defensible"]
B: "Statistical Methodology and Reporting"
   desc: Statistics should be described clearly and applied correctly.
   sample members: ["Statistical methods clearly described", "Statistical claims well-supported", "Define error bars"]

Pair 3
A: "Citation Practices"
   desc: Standards for citing sources.
   sample members: ["Citations support all empirical claims", "Cite recent and relevant work"]
B: "Citation Style and Format"
   desc: How citations should be formatted.
   sample members: ["Use APA citation style consistently", "Place citations at end of relevant sentence"]

Pair 4
A: "Reproducibility Artifacts"
   desc: Code, scripts, and packages enabling reproduction.
   sample members: ["Submission includes reproducibility artifacts/packages", "Code is sufficient to reproduce results"]
B: "Code and Data Availability"
   desc: Data, code, and materials should be made available.
   sample members: ["Code made publicly available", "Data sharing plan for de-identified data"]"""


R25_FEWSHOT_ASSISTANT = """{"verdicts": [
 {"pair_idx": 1, "merge": "YES"},
 {"pair_idx": 2, "merge": "NO"},
 {"pair_idx": 3, "merge": "NO"},
 {"pair_idx": 4, "merge": "NO"}
]}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="peer-review")
    ap.add_argument("--r2-dir", default="r2_v1_subagent")
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--pairs-per-batch", type=int, default=20)
    ap.add_argument("--max-batches", type=int, default=10,
                    help="Cap total subagent batches (top-K by cos descending)")
    ap.add_argument("--output-dir", default="/tmp/r2_5")
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
    reps, cl_centroids, _members = cluster_data(rows, emb, cl)

    # Load R2 aspects + need R1 family info
    r2 = json.loads((Path("outputs/analyses/structural_metrics") / args.r2_dir
                     / f"r2_aspects_{args.task}.json").read_text())
    r1_dir = "r1_v4a_lora_fork3_merge"
    r1 = json.loads((Path("outputs/analyses/structural_metrics") / r1_dir
                     / f"r1_families_{args.task}.json").read_text())
    r1_fams = r1["families"]

    # Each R2 aspect's centroid = mean of its R1 families' centroids, which
    # are themselves mean of cluster centroids.
    aspect_centroids = []
    aspect_meta = []
    for ai, asp in enumerate(r2["aspects"]):
        all_cids = []
        for fi in asp["family_ids"]:
            if fi >= len(r1_fams):
                continue
            cids = r1_fams[fi].get("cluster_ids", [])
            cids = [int(str(c).lstrip("C")) for c in cids
                    if str(c).lstrip("C").isdigit()]
            all_cids.extend(c for c in cids if c in cl_centroids)
        if not all_cids:
            continue
        v = np.stack([cl_centroids[c] for c in all_cids]).mean(0)
        v /= (np.linalg.norm(v) + 1e-9)
        aspect_centroids.append(v)
        aspect_meta.append({
            "ai": ai,
            "aspect_id": asp["aspect_id"],
            "name": asp["name"],
            "description": asp["description"],
            "n_families": asp["n_families"],
            "source_batch": asp["source_batch"],
            "sample_member_names": [
                r1_fams[fi].get("name", "")[:80]
                for fi in asp["family_ids"][:3] if fi < len(r1_fams)
            ],
        })

    if not aspect_centroids:
        print(f"no aspects to process for {args.task}")
        return
    (out / "aspect_meta.json").write_text(json.dumps(aspect_meta, indent=1))

    centroids = np.stack(aspect_centroids)
    print(f"{args.task}: {len(centroids)} R2 aspects")

    sims = centroids @ centroids.T
    iu = np.triu_indices(len(centroids), k=1)
    pair_sims = sims[iu]
    print(f"  cos quantiles: q90={np.quantile(pair_sims, 0.90):.3f} "
          f"q99={np.quantile(pair_sims, 0.99):.3f}")
    keep = pair_sims >= args.threshold
    cand = [(float(pair_sims[k]), int(iu[0][k]), int(iu[1][k]))
            for k in np.where(keep)[0]]
    cand.sort(key=lambda x: -x[0])
    print(f"  {len(cand)} candidate pairs at cos>={args.threshold}")

    if not cand:
        return

    meta_by_ai = {m["ai"]: m for m in aspect_meta}
    ai_idx_to_meta = {i: aspect_meta[i] for i in range(len(aspect_meta))}
    info = TASK_INFO[args.task]
    system = R25_SYSTEM.format(task=args.task, background=info["background"])

    n_batches = min(args.max_batches,
                    (len(cand) + args.pairs_per_batch - 1) // args.pairs_per_batch)
    cand = cand[:n_batches * args.pairs_per_batch]

    with (out / "batches.jsonl").open("w") as bjf:
        for bi in range(0, len(cand), args.pairs_per_batch):
            chunk = cand[bi:bi + args.pairs_per_batch]
            bidx = bi // args.pairs_per_batch
            lines = []
            for k, (cos, ia, ib) in enumerate(chunk, 1):
                ma, mb = ai_idx_to_meta[ia], ai_idx_to_meta[ib]
                lines.append(f"\nPair {k}")
                lines.append(f"A: {ma['name']!r}  (batch {ma['source_batch']})")
                lines.append(f"   desc: {ma['description'][:160]}")
                lines.append(f"   sample members: [{', '.join(json.dumps(s) for s in ma['sample_member_names'])}]")
                lines.append(f"B: {mb['name']!r}  (batch {mb['source_batch']})")
                lines.append(f"   desc: {mb['description'][:160]}")
                lines.append(f"   sample members: [{', '.join(json.dumps(s) for s in mb['sample_member_names'])}]")
            user_msg = ("Here are the candidate R2 aspect pairs. For each, "
                        "decide YES/NO whether they encode the SAME thematic "
                        "aspect and should be merged.\n" + "\n".join(lines))
            prompt_text = (
                system
                + "\n\n=== FEW-SHOT USER ===\n" + R25_FEWSHOT_USER
                + "\n\n=== FEW-SHOT ASSISTANT ===\n" + R25_FEWSHOT_ASSISTANT
                + "\n\n=== USER ===\n" + user_msg
            )
            (out / "batches" / f"batch_{bidx}.txt").write_text(prompt_text)
            bjf.write(json.dumps({
                "batch_idx": bidx,
                "pairs": [
                    {"pair_idx": k+1, "cos": c, "ai_a": ia, "ai_b": ib,
                     "aspect_id_a": ai_idx_to_meta[ia]["aspect_id"],
                     "aspect_id_b": ai_idx_to_meta[ib]["aspect_id"]}
                    for k, (c, ia, ib) in enumerate(chunk)
                ],
            }) + "\n")
    print(f"  wrote {n_batches} batch prompts -> {out}/batches/")


if __name__ == "__main__":
    main()
