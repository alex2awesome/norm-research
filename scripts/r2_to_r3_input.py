"""Convert R2 expanded output to a fake R1-expanded format so meta_merge.py
can run it as Round 3.

R2 output has 'merged_groups' (canonical parent concepts that collapsed
redundant R1 parents) and 'grandparents'. For R3 we feed R2's merged_groups
back into the algorithm as new "parents" — the algorithm will find
duplicates that R2 missed (because they were in different R2 batches)."""
import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="R2 _expanded.json")
    ap.add_argument("--output", required=True, help="fake-r1 _expanded.json for R3 meta-merge")
    ap.add_argument("--include-unmerged-r1", action="store_true",
                    help="Also include R1 parents that weren't touched by R2 merges (recovers more coverage).")
    args = ap.parse_args()

    r2 = json.loads(Path(args.input).read_text())
    merged = r2.get("merged_groups", [])

    parented_trees = []
    for m in merged:
        parented_trees.append({
            "parent_name": m["merged_name"],
            "parent_description": m["merged_description"],
            "n_children": 1,
            "total_leaf_rubrics": m["total_leaf_rubrics"],
            "children": [{
                "cluster_id": -1,
                "medoid_name": m["merged_name"],
                "medoid_description": m["merged_description"],
                "size": m["total_leaf_rubrics"],
                "rubrics": m["all_leaves"],
            }],
        })

    fake = {
        "task": r2.get("task"),
        "bucket": r2.get("bucket"),
        "round": 3,
        "n_clusters_in": len(parented_trees),
        "n_merge_trees": 0,
        "n_parent_trees": len(parented_trees),
        "merged_trees": [],
        "parented_trees": parented_trees,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(fake, indent=2, ensure_ascii=False))
    print(f"converted {len(parented_trees)} R2 merged_groups → R3 input")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
