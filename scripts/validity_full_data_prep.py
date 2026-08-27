"""Full validity pipeline data prep for peer-review.

Selects:
  - all 218 R2 aspects
  - top-K R1 sub-families per aspect (default K=3)
  - N stratified peer-review datapoints (accept/reject 50/50)

Writes:
  runs/validity_full/<run>/r2_aspects.json       (218 aspects with their selected R1 children)
  runs/validity_full/<run>/r1_metrics.json       (flattened list of {metric_id, parent_r2, name, desc, samples})
  runs/validity_full/<run>/datapoints.json       (N stratified)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="peer-review")
    ap.add_argument("--top-k-r1", type=int, default=3)
    ap.add_argument("--n-datapoints", type=int, default=500)
    ap.add_argument("--max-text-chars", type=int, default=1500)
    ap.add_argument("--run-name", default="full_v1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(f"runs/validity_full/{args.run_name}")
    out.mkdir(parents=True, exist_ok=True)

    r2 = json.loads(Path("outputs/analyses/structural_metrics/r2_v1_subagent/"
                         f"r2_aspects_{args.task}.json").read_text())
    r1 = json.loads(Path("outputs/analyses/structural_metrics/"
                         f"r1_v4a_lora_fork3_merge/r1_families_{args.task}.json"
                         ).read_text())
    reps = json.loads(Path(f"/tmp/r1_subagent_lora_bs400/{args.task}/"
                           "clusters_repr.json").read_text())

    aspects_out = []
    r1_metrics_out = []
    for ai, asp in enumerate(r2["aspects"]):
        # Rank R1 sub-families by n_clusters desc
        ranked = []
        for fi in asp["family_ids"]:
            if fi >= len(r1["families"]):
                continue
            f = r1["families"][fi]
            n_cl = len(f.get("cluster_ids", []))
            ranked.append((fi, f, n_cl))
        ranked.sort(key=lambda x: -x[2])
        chosen = ranked[:args.top_k_r1]
        aspect_record = {
            "aspect_id": f"a{ai}",
            "name": asp["name"],
            "description": asp["description"],
            "n_r1_total": asp["n_families"],
            "n_r1_used": len(chosen),
            "r1_metric_ids": [],  # filled below
        }
        for fi, f, n_cl in chosen:
            cids = [int(str(c).lstrip("C")) for c in f.get("cluster_ids", [])
                    if str(c).lstrip("C").isdigit()]
            samples = [reps[str(c)] for c in cids[:5] if str(c) in reps]
            metric_id = f"a{ai}_f{fi}"
            r1_metrics_out.append({
                "metric_id": metric_id,
                "parent_aspect_id": f"a{ai}",
                "r1_family_id": fi,
                "name": f.get("name", ""),
                "description": f.get("description", ""),
                "samples": samples,
                "n_clusters": n_cl,
            })
            aspect_record["r1_metric_ids"].append(metric_id)
        aspects_out.append(aspect_record)

    (out / "r2_aspects.json").write_text(json.dumps(aspects_out, indent=1))
    (out / "r1_metrics.json").write_text(json.dumps(r1_metrics_out, indent=1))
    print(f"R2 aspects: {len(aspects_out)}")
    print(f"R1 metrics (top-{args.top_k_r1} per R2): {len(r1_metrics_out)}")

    # Datapoints — stratified by judgement
    import pandas as pd
    df = pd.read_csv("datasets/peer-review/peer_review_modeling_dataset.csv.gz")
    n_per_class = args.n_datapoints // 2
    sample = pd.concat([
        df[df.judgement == 0].sample(n=n_per_class, random_state=args.seed),
        df[df.judgement == 1].sample(n=n_per_class, random_state=args.seed),
    ]).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    datapoints = []
    for i, r in sample.iterrows():
        text = str(r["text"])[:args.max_text_chars]
        datapoints.append({
            "datapoint_id": f"d{i:04d}",
            "paper_id": r["paper_id"],
            "judgement": int(r["judgement"]),
            "venue": str(r.get("venue", "")),
            "text": text,
        })
    (out / "datapoints.json").write_text(json.dumps(datapoints, indent=1))
    print(f"Datapoints: {len(datapoints)} (accept={sum(d['judgement'] for d in datapoints)})")


if __name__ == "__main__":
    main()
