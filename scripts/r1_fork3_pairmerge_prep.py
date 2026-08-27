"""Fork 3: pairwise post-hoc merge on Claude+LoRA bs=400 base.

For each candidate family-pair (cos >= threshold in LoRA-bge centroid space),
batch K=20 pairs per subagent prompt. Subagent answers YES/NO per pair
indicating whether the two families encode the SAME rule.

Then `r1_fork3_pairmerge_apply.py` applies YES verdicts via union-find ordered
by cos descending (so the most-confident merges happen first).

Output:
  /tmp/r1_fork3/<task>/batches/batch_<i>.txt   (subagent prompts)
  /tmp/r1_fork3/<task>/batches.jsonl           (per-batch: pair indices, fa/fb, cos)
  /tmp/r1_fork3/<task>/family_meta.json        (fi -> name/desc/clusters/support)
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


FORK3_SYSTEM = """You are reviewing candidate merges for a rubric-clustering system applied to {task}.

Background:
{background}

You will be shown N candidate pairs of "rule families". Each family is meant to represent a single underlying rule. Your job: for each pair, decide whether the two families encode the SAME rule (in which case they should be merged into one family) or DIFFERENT rules (in which case they should stay separate).

Two families should be merged ONLY IF the rule they encode is essentially the same — not merely thematically related, not merely about the same artifact.

MERGE examples:
- "Methods described in detail" + "Methods reported with enough detail to replicate" → SAME
- "Code is released" + "Source code should be made available" → SAME
- "Figures are clear and focused" + "Tables and figures are clear" → SAME (visual clarity of all visual assets)

DO NOT MERGE examples:
- "Code is available" + "Data is available" → DIFFERENT (different artifacts)
- "Conclusions are not misleading" + "Methods are clearly described" → DIFFERENT
- "Methods are appropriate" + "Methods are clearly justified" → DIFFERENT (fit vs justification)

Output a JSON object with one verdict per pair. Use "YES" for SAME, "NO" for DIFFERENT.

Schema:
{{"verdicts": [{{"pair_idx": 1, "merge": "YES"}}, {{"pair_idx": 2, "merge": "NO"}}, ...]}}

Every pair_idx in the USER input must have a verdict."""


FORK3_FEWSHOT_USER = """Pair 1
A: "Methods are reported in sufficient detail to allow reproduction"
   desc: Methods should be described with enough detail for independent reproduction.
   members: ["The work should provide sufficient methodological detail.", "The methods should be reported with enough detail to replicate."]
B: "Methods are described transparently"
   desc: Methods should be presented transparently and reproducibly.
   members: ["The methods should be reported transparently and be reproducible.", "The methods and assumptions should be clearly presented."]

Pair 2
A: "Code is released for reproducibility"
   desc: Source code should be made available to support reproduction.
   members: ["The work should release its source code.", "Code is encouraged to support reproducibility."]
B: "Data is openly accessible"
   desc: Underlying data should be made openly available.
   members: ["Underlying data should be made openly available."]

Pair 3
A: "Methodology is appropriate for the research question"
   desc: Methodology should fit the research question.
   members: ["The methods should match the research question."]
B: "Methodology choice is justified with rationale"
   desc: Methodology choice should come with justification.
   members: ["The choice of methodology should be clearly justified."]

Pair 4
A: "Statistical methods are described"
   desc: Statistical methods used should be clearly described.
   members: ["The statistical methods should be reported."]
B: "Methods are clearly and transparently described"
   desc: Methods should be presented clearly.
   members: ["The methods used should be clearly described."]"""


FORK3_FEWSHOT_ASSISTANT = """{"verdicts": [
 {"pair_idx": 1, "merge": "YES"},
 {"pair_idx": 2, "merge": "NO"},
 {"pair_idx": 3, "merge": "NO"},
 {"pair_idx": 4, "merge": "YES"}
]}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="peer-review")
    ap.add_argument("--r1-dir", default="r1_v4a_subagent_lora_bs400")
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--max-threshold", type=float, default=1.01,
                    help="Upper bound on cos (exclusive) -- to judge only new pairs")
    ap.add_argument("--pairs-per-batch", type=int, default=20)
    ap.add_argument("--output-dir", default="/tmp/r1_fork3")
    args = ap.parse_args()

    out = Path(args.output_dir) / args.task
    (out / "batches").mkdir(parents=True, exist_ok=True)
    (out / "responses").mkdir(parents=True, exist_ok=True)

    # Load L0 + centroids
    forms = [json.loads(l)
             for l in open("outputs/analyses/canon_all_real_forms.jsonl")
             if json.loads(l)["task"] == args.task]
    rows, emb = load_task(args.task, forms)
    cl = json.loads((Path("outputs/analyses/structural_metrics")
                     / f"clusters_{args.task}.json").read_text())
    reps, centroids, members = cluster_data(rows, emb, cl)

    # Load R1 families + build family centroids
    r1 = json.loads((Path("outputs/analyses/structural_metrics") / args.r1_dir
                     / f"r1_families_{args.task}.json").read_text())
    fams = r1["families"]
    fam_meta = []
    fam_emb_list = []
    for fi, f in enumerate(fams):
        cids = [int(str(c).lstrip("C")) for c in
                (f.get("cluster_ids") or f.get("members") or [])
                if str(c).lstrip("C").isdigit()]
        cids = [c for c in cids if c in centroids]
        if not cids:
            continue
        v = np.stack([centroids[c] for c in cids]).mean(0)
        v /= (np.linalg.norm(v) + 1e-9)
        fam_emb_list.append((fi, v))
        fam_meta.append({
            "fi": fi,
            "name": f.get("name", ""),
            "description": f.get("description", ""),
            "n_clusters": len(cids),
            "cluster_ids": cids,
            "support": [reps[c] for c in cids[:3] if c in reps],
        })
    (out / "family_meta.json").write_text(json.dumps(fam_meta, indent=1))
    fis = [x[0] for x in fam_emb_list]
    fcs = np.stack([x[1] for x in fam_emb_list]).astype(np.float32)
    fi_to_idx = {fi: i for i, fi in enumerate(fis)}

    # Find candidate pairs
    sims = fcs @ fcs.T
    iu = np.triu_indices(len(fcs), k=1)
    pair_sims = sims[iu]
    keep = (pair_sims >= args.threshold) & (pair_sims < args.max_threshold)
    cand = [(float(pair_sims[k]), int(iu[0][k]), int(iu[1][k]))
            for k in np.where(keep)[0]]
    # Sort desc by cos so most-confident merges judged first
    cand.sort(key=lambda x: -x[0])
    print(f"{len(cand)} candidate pairs at cos>={args.threshold}")

    meta_by_fi = {m["fi"]: m for m in fam_meta}
    info = TASK_INFO[args.task]
    system = FORK3_SYSTEM.format(task=args.task, background=info["background"])

    # Batch and materialize
    bjf = (out / "batches.jsonl").open("w")
    for bi in range(0, len(cand), args.pairs_per_batch):
        chunk = cand[bi:bi + args.pairs_per_batch]
        bidx = bi // args.pairs_per_batch
        lines = []
        for k, (cos, ia, ib) in enumerate(chunk, 1):
            fa, fb = fis[ia], fis[ib]
            ma, mb = meta_by_fi[fa], meta_by_fi[fb]
            lines.append(f"\nPair {k}")
            lines.append(f"A: {ma['name']!r}")
            lines.append(f"   desc: {ma['description']}")
            if ma['support']:
                lines.append(f"   members: [{', '.join(json.dumps(s[:140]) for s in ma['support'])}]")
            lines.append(f"B: {mb['name']!r}")
            lines.append(f"   desc: {mb['description']}")
            if mb['support']:
                lines.append(f"   members: [{', '.join(json.dumps(s[:140]) for s in mb['support'])}]")
        user_msg = "Here are the candidate pairs. For each, decide YES/NO whether the families should be merged (same rule).\n" + "\n".join(lines)
        prompt_text = (
            system
            + "\n\n=== FEW-SHOT USER ===\n" + FORK3_FEWSHOT_USER
            + "\n\n=== FEW-SHOT ASSISTANT ===\n" + FORK3_FEWSHOT_ASSISTANT
            + "\n\n=== USER ===\n" + user_msg
        )
        (out / "batches" / f"batch_{bidx}.txt").write_text(prompt_text)
        bjf.write(json.dumps({
            "batch_idx": bidx,
            "pairs": [{"pair_idx": k+1, "cos": c, "fa": fis[ia], "fb": fis[ib]}
                      for k, (c, ia, ib) in enumerate(chunk)],
        }) + "\n")
    bjf.close()

    n_batches = (len(cand) + args.pairs_per_batch - 1) // args.pairs_per_batch
    print(f"wrote {n_batches} batch prompts -> {out}/batches/")


if __name__ == "__main__":
    main()
