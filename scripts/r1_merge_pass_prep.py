"""Step 1 of LoRA-bge post-hoc merge pass.

Computes per-R1-family centroids in LoRA-bge space and tabulates candidate
family-pair counts at various cosine thresholds, so we can pick the threshold
that yields a manageable number of LLM-judge calls.

Output:
  /tmp/r1_merge_pass/peer-review/family_centroids.npy
  /tmp/r1_merge_pass/peer-review/family_meta.json     (family_idx -> name/desc/cluster_ids)
  /tmp/r1_merge_pass/peer-review/candidate_pairs.jsonl  (all pairs above lowest threshold)
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
from sk3_build_r1 import cluster_data, load_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="peer-review")
    ap.add_argument("--r1-dir", default="r1_v4a_subagent_lora_bs400")
    ap.add_argument("--output-dir", default="/tmp/r1_merge_pass")
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.60, 0.70, 0.75, 0.80, 0.85, 0.90])
    args = ap.parse_args()

    out = Path(args.output_dir) / args.task
    out.mkdir(exist_ok=True, parents=True)

    # Load embeddings + clusters
    forms = [json.loads(l)
             for l in open("outputs/analyses/canon_all_real_forms.jsonl")
             if json.loads(l)["task"] == args.task]
    rows, emb = load_task(args.task, forms)
    cl = json.loads(Path("outputs/analyses/structural_metrics") /
                    f"clusters_{args.task}.json").read_text())
    reps, centroids, members = cluster_data(rows, emb, cl)

    # Load R1 families
    r1 = json.loads(Path("outputs/analyses/structural_metrics") /
                    args.r1_dir / f"r1_families_{args.task}.json")
                    .read_text())
    fams = r1["families"]

    # Compute family centroids = mean of cluster centroids, L2-normed
    fam_emb = np.zeros((len(fams), 1024), dtype=np.float32)
    fam_meta = []
    for fi, f in enumerate(fams):
        cids = f.get("cluster_ids") or f.get("members") or []
        cids = [int(str(c).lstrip("C")) for c in cids if str(c).lstrip("C").isdigit()]
        vecs = np.stack([centroids[c] for c in cids if c in centroids])
        v = vecs.mean(0)
        v /= (np.linalg.norm(v) + 1e-9)
        fam_emb[fi] = v
        fam_meta.append({
            "fi": fi,
            "name": f.get("name", ""),
            "description": f.get("description", ""),
            "n_clusters": len(cids),
            "cluster_ids": cids,
            "rep_texts": [reps[c] for c in cids[:6] if c in reps],
        })
    np.save(out / "family_centroids.npy", fam_emb)
    (out / "family_meta.json").write_text(json.dumps(fam_meta, indent=1))
    print(f"saved {len(fams)} family centroids -> {out}/family_centroids.npy")

    # Pair similarity distribution
    sims = fam_emb @ fam_emb.T
    iu = np.triu_indices(len(fams), k=1)
    pair_sims = sims[iu]
    print(f"\nAll {len(pair_sims):,} family-pair cosines distribution:")
    for q in [0.5, 0.75, 0.9, 0.95, 0.99, 0.999]:
        print(f"  q={q}: {np.quantile(pair_sims, q):.3f}")

    print(f"\nCandidate counts per threshold:")
    for t in args.thresholds:
        n = (pair_sims >= t).sum()
        print(f"  cos >= {t:.2f}: {n:,} pairs  "
              f"({n/len(pair_sims)*100:.2f}% of all pairs)")

    # Save the candidate list at the LOWEST threshold (so we can filter later)
    lowest = min(args.thresholds)
    keep = pair_sims >= lowest
    pairs_idx_a = iu[0][keep]
    pairs_idx_b = iu[1][keep]
    sims_keep = pair_sims[keep]
    order = np.argsort(-sims_keep)
    with (out / "candidate_pairs.jsonl").open("w") as f:
        for k in order:
            ai, bi = int(pairs_idx_a[k]), int(pairs_idx_b[k])
            f.write(json.dumps({
                "fa": ai, "fb": bi, "cos": float(sims_keep[k]),
                "name_a": fam_meta[ai]["name"],
                "name_b": fam_meta[bi]["name"],
            }) + "\n")
    print(f"\nsaved {keep.sum()} candidate pairs (cos>={lowest}) "
          f"-> {out}/candidate_pairs.jsonl")


if __name__ == "__main__":
    main()
