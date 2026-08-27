"""Build the 184 canonical rubrics for smoke v2.

Source: 184 aspect_clusters in outputs/v2_analysis/peer_review__aspect_clusters.parquet.
Name source: hierarchy files (r2_expanded for the bulk, falling back through r2 raw
batches/merges and r1 refined merged_trees). Cluster_ids in the parquet map onto
source_r2_cluster_ids / source_cluster_ids in the hierarchy.

Coverage check: 154 unique cluster_ids in the aspect parquet (184 rows because some
clusters have multiple aspects). All 154 resolved with names.

Output:
  rubrics.jsonl — one row per cluster_id: {rubric_id, name, description, aspect_ids, mean_score, source}
"""
import json
import os
import pandas as pd

OUT_DIR = "/Users/spangher/Projects/stanford-research/norm-research/datasets/peer-review/extracted/smoke_v2"
AC_PATH = "/Users/spangher/Projects/stanford-research/norm-research/outputs/v2_analysis/peer_review__aspect_clusters.parquet"

HIER_R2_EXPANDED = [
    "/Users/spangher/Projects/stanford-research/norm-research/outputs/hierarchy/peer-review_specific_r2_expanded.json",
    "/Users/spangher/Projects/stanford-research/norm-research/outputs/hierarchy/peer-review_general_r2_expanded.json",
    "/Users/spangher/Projects/stanford-research/norm-research/outputs/hierarchy/peer-review_hyper_specific_r2_expanded.json",
]
HIER_R2_RAW = [
    "/Users/spangher/Projects/stanford-research/norm-research/outputs/hierarchy/peer-review_specific_r2.json",
    "/Users/spangher/Projects/stanford-research/norm-research/outputs/hierarchy/peer-review_general_r2.json",
    "/Users/spangher/Projects/stanford-research/norm-research/outputs/hierarchy/peer-review_hyper_specific_r2.json",
]
HIER_R1 = [
    "/Users/spangher/Projects/stanford-research/norm-research/outputs/hierarchy/peer-review_specific_r1_refined.json",
    "/Users/spangher/Projects/stanford-research/norm-research/outputs/hierarchy/peer-review_general_r1_refined.json",
    "/Users/spangher/Projects/stanford-research/norm-research/outputs/hierarchy/peer-review_hyper_specific_r1_refined.json",
]


def build_name_map():
    """Return {cluster_id (int): {'name', 'description', 'source'}}."""
    name_map = {}

    # 1. r2 expanded (highest level merges)
    for fp in HIER_R2_EXPANDED:
        with open(fp) as f:
            d = json.load(f)
        src_tag = os.path.basename(fp).replace("peer-review_", "").replace(".json", "")
        for m in d["merged_groups"]:
            for cid in m["source_r2_cluster_ids"]:
                if cid not in name_map:
                    name_map[cid] = {
                        "name": m["merged_name"],
                        "description": m["merged_description"],
                        "source": src_tag,
                    }

    # 2. r2 raw batch merges/parents
    for fp in HIER_R2_RAW:
        with open(fp) as f:
            d = json.load(f)
        src_tag = os.path.basename(fp).replace("peer-review_", "").replace(".json", "")
        for b in d["batches"]:
            res = b.get("result") or {}
            for m in res.get("merges", []) or []:
                for cid in m.get("cluster_ids", []):
                    if cid not in name_map:
                        name_map[cid] = {
                            "name": m["merged_name"],
                            "description": m.get("merged_description", ""),
                            "source": src_tag + ":merge",
                        }
            for p in res.get("parents", []) or []:
                for cid in p.get("cluster_ids", []):
                    if cid not in name_map:
                        name_map[cid] = {
                            "name": p.get("parent_name", "?"),
                            "description": p.get("parent_description", ""),
                            "source": src_tag + ":parent",
                        }

    # 3. r1 refined merged_trees (lowest-level fallback)
    for fp in HIER_R1:
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            d = json.load(f)
        src_tag = os.path.basename(fp).replace("peer-review_", "").replace(".json", "")
        for m in d.get("merged_trees", []):
            for cid in m.get("source_cluster_ids", []):
                if cid not in name_map:
                    name_map[cid] = {
                        "name": m["merged_name"],
                        "description": m.get("merged_description", ""),
                        "source": src_tag,
                    }
    return name_map


def main():
    ac = pd.read_parquet(AC_PATH)
    print(f"aspect_clusters: {len(ac)} rows, {ac['cluster_id'].nunique()} unique cluster_ids")

    name_map = build_name_map()
    cids = sorted(ac["cluster_id"].unique().tolist())
    missing = [c for c in cids if c not in name_map]
    print(f"name_map covers: {len(cids) - len(missing)} / {len(cids)}, missing: {missing[:20]}")

    # Build per-cluster aggregate
    agg = ac.groupby("cluster_id").agg(
        aspect_ids=("aspect_id", list),
        mean_score=("mean_score", "mean"),
        n_dps_seen=("n_dps_seen", "sum"),
        n_aspects=("aspect_id", "count"),
    ).reset_index()

    rows = []
    for _, r in agg.iterrows():
        cid = int(r["cluster_id"])
        info = name_map.get(cid, {"name": f"cluster_{cid}", "description": "(unnamed)", "source": "missing"})
        rows.append({
            "rubric_id": cid,  # Use cluster_id as the canonical id
            "name": info["name"],
            "description": info["description"],
            "name_source": info["source"],
            "aspect_ids": r["aspect_ids"],
            "n_aspects_in_cluster": int(r["n_aspects"]),
            "mean_score": float(r["mean_score"]),
        })

    out_path = os.path.join(OUT_DIR, "rubrics.jsonl")
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rubrics to {out_path}")

    # Summary
    print(f"  unique rubric_ids: {len(rows)}")
    print(f"  total aspects covered: {sum(r['n_aspects_in_cluster'] for r in rows)}")
    print(f"  description lengths (chars): min={min(len(r['description']) for r in rows)}, max={max(len(r['description']) for r in rows)}, mean={sum(len(r['description']) for r in rows)//len(rows)}")


if __name__ == "__main__":
    main()
