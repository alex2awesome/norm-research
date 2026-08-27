"""Validity pilot data prep — sample K (R1, R2) metric pairs + N peer-review datapoints.

For each R2 aspect with ≥2 R1 families, pick the largest R1 family inside it.
This gives us K paired metrics: same concept at two levels of abstraction.
The level comparison (R1 vs R2) reuses the SAME aspect content with two
different framings:
  - R1 framing: just the focal R1 family (specific rule + 3 member texts)
  - R2 framing: the parent R2 aspect (broader theme + names of all sub-R1 families)

Outputs:
  runs/validity_pilot/<run_name>/metrics.json    paired (R1, R2) metrics
  runs/validity_pilot/<run_name>/datapoints.json  N datapoints with text (truncated)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="peer-review")
    ap.add_argument("--n-metrics", type=int, default=5)
    ap.add_argument("--n-datapoints", type=int, default=10)
    ap.add_argument("--run-name", default="smoke")
    ap.add_argument("--max-text-chars", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(f"runs/validity_pilot/{args.run_name}")
    out.mkdir(exist_ok=True, parents=True)

    # Load R2 aspects + R1 (Fork3-merged) families + L0 cluster reps
    r2 = json.loads(Path("outputs/analyses/structural_metrics/r2_v1_subagent/"
                         f"r2_aspects_{args.task}.json").read_text())
    r1 = json.loads(Path("outputs/analyses/structural_metrics/"
                         f"r1_v4a_lora_fork3_merge/r1_families_{args.task}.json"
                         ).read_text())
    reps = json.loads(Path(f"/tmp/r1_subagent_lora_bs400/{args.task}/"
                           "clusters_repr.json").read_text())

    # Eligible R2 aspects: at least 2 R1 families, total >=4 L0 clusters
    eligible = []
    for aspect in r2["aspects"]:
        if aspect["n_families"] < 2:
            continue
        # Sort R1 families inside by n_clusters desc
        family_objs = []
        for fi in aspect["family_ids"]:
            if fi >= len(r1["families"]):
                continue
            f = r1["families"][fi]
            cids = f.get("cluster_ids") or []
            family_objs.append((fi, f, len(cids)))
        family_objs.sort(key=lambda x: -x[2])
        if not family_objs:
            continue
        eligible.append((aspect, family_objs))

    rng.shuffle(eligible)
    chosen = eligible[:args.n_metrics]
    print(f"Eligible R2 aspects: {len(eligible)}, choosing {len(chosen)}")

    metrics = []
    for aspect, family_objs in chosen:
        focal_fi, focal_f, n_cl = family_objs[0]
        # R1 framing: focal family details
        focal_cids = focal_f.get("cluster_ids") or []
        focal_cids = [int(str(c).lstrip("C")) for c in focal_cids
                      if str(c).lstrip("C").isdigit()]
        focal_samples = [reps[str(c)] for c in focal_cids[:5] if str(c) in reps]
        # R2 framing: aspect + all R1 family names inside (top 6 by size)
        r1_sibling_names = [obj[1].get("name", "") for obj in family_objs[:6]]
        metrics.append({
            "metric_id": f"m{len(metrics)}",
            "r2_aspect_name": aspect["name"],
            "r2_aspect_description": aspect["description"],
            "r2_n_r1_families": aspect["n_families"],
            "r2_r1_member_names": r1_sibling_names,
            "r1_focal_family_id": focal_fi,
            "r1_focal_name": focal_f.get("name", ""),
            "r1_focal_description": focal_f.get("description", ""),
            "r1_focal_n_clusters": n_cl,
            "r1_focal_samples": focal_samples,
        })

    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"wrote {len(metrics)} metrics -> {out}/metrics.json")

    # Datapoints — stratified by judgement
    import pandas as pd
    df = pd.read_csv("datasets/peer-review/peer_review_modeling_dataset.csv.gz")
    # stratified 50/50
    n_per_class = max(1, args.n_datapoints // 2)
    sample = pd.concat([
        df[df.judgement == 0].sample(n=n_per_class, random_state=args.seed),
        df[df.judgement == 1].sample(n=n_per_class, random_state=args.seed),
    ]).reset_index(drop=True)
    datapoints = []
    for i, r in sample.iterrows():
        text = str(r["text"])[:args.max_text_chars]
        datapoints.append({
            "datapoint_id": f"d{i}",
            "paper_id": r["paper_id"],
            "judgement": int(r["judgement"]),
            "venue": r.get("venue", ""),
            "text": text,
        })
    (out / "datapoints.json").write_text(json.dumps(datapoints, indent=1))
    print(f"wrote {len(datapoints)} datapoints -> {out}/datapoints.json")


if __name__ == "__main__":
    main()
