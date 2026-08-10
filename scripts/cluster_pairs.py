"""
Build clusters from the LLM-labeled pair eval (creative_writing_2k.jsonl).

Algorithm: representative-anchored complete-linkage with bridge tracking.

For each specificity bucket:
  1. Build two graphs:
     G_merge: edges where verdict ∈ {duplicate, paraphrase}
     G_nonmerge: edges where verdict ∈ {related, different}
  2. Sort candidate rubrics by merge-degree (most-connected first).
  3. For each rubric R in order:
     a. Find which existing clusters R has merge-edges to.
     b. If 0: start new cluster, R is centroid.
     c. If 1: complete-linkage check — R must not have any verified
        non-merge edge to existing members. If clean, join. If conflict,
        start new cluster (singleton) and mark R as boundary.
     d. If ≥2: representative-pair verification (we have the relevant
        cross-cluster pair labels in our data already):
        - If any cross-cluster centroid pair is dup/paraphrase, merge
          those clusters + add R.
        - Else: R is a bridge. Assign R to highest-merge-edge cluster,
          and record bridge_to for the other candidates.

Outputs:
  - Per-bucket clusters with members, centroid, bridge metadata
  - Summary stats (cluster size distribution, # bridges)
  - Spot-check of largest clusters
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")

MERGE_VERDICTS = {"duplicate", "paraphrase"}
NONMERGE_VERDICTS = {"related", "different"}


def load_pairs(jsonl_path: Path) -> list[dict]:
    return [json.loads(l) for l in jsonl_path.open() if l.strip()]


def build_graphs(pairs: list[dict]) -> tuple[dict, dict, dict]:
    """Returns (G_merge, G_nonmerge, rubric_meta).
    G_merge[r] = set of rubrics merge-connected to r.
    G_nonmerge[r] = set of rubrics verified-non-merge to r.
    rubric_meta[r] = {'name', 'description', 'specificity'}"""
    G_merge: dict[str, set[str]] = defaultdict(set)
    G_nonmerge: dict[str, set[str]] = defaultdict(set)
    meta: dict[str, dict] = {}
    for p in pairs:
        a, b = p["a_key"], p["b_key"]
        v = p["verdict"]
        if v in MERGE_VERDICTS:
            G_merge[a].add(b); G_merge[b].add(a)
        elif v in NONMERGE_VERDICTS:
            G_nonmerge[a].add(b); G_nonmerge[b].add(a)
        meta.setdefault(a, {"name": p["a_name"], "description": p["a_description"], "specificity": p["specificity"]})
        meta.setdefault(b, {"name": p["b_name"], "description": p["b_description"], "specificity": p["specificity"]})
    return dict(G_merge), dict(G_nonmerge), meta


def cluster_bucket(rubrics: list[str], G_merge: dict, G_nonmerge: dict,
                   pair_lookup: dict) -> list[dict]:
    """Run representative-anchored complete-linkage clustering on rubrics
    in a specificity bucket. Returns list of clusters."""
    # Sort by merge-degree (descending)
    sorted_rubrics = sorted(rubrics, key=lambda r: -len(G_merge.get(r, set())))

    clusters: list[dict] = []
    assigned: dict[str, int] = {}

    for r in sorted_rubrics:
        if r in assigned: continue

        # Find candidate clusters R could join
        merge_neighbors = G_merge.get(r, set())
        candidate_cluster_idxs: set[int] = set()
        for nb in merge_neighbors:
            if nb in assigned:
                candidate_cluster_idxs.add(assigned[nb])

        if not candidate_cluster_idxs:
            # New cluster, R is centroid
            clusters.append({
                "centroid": r,
                "members": [r],
                "bridge_to": [],   # cluster_ids R also paraphrased
                "boundary": [],    # rubrics with non-merge conflict in this cluster
            })
            assigned[r] = len(clusters) - 1
            continue

        if len(candidate_cluster_idxs) == 1:
            idx = next(iter(candidate_cluster_idxs))
            # Complete-linkage check: R must not have any non-merge edge to
            # existing cluster members
            cm = clusters[idx]["members"]
            conflicts = [m for m in cm if m in G_nonmerge.get(r, set())]
            if conflicts:
                # Conflict — start new cluster, mark as boundary
                clusters.append({
                    "centroid": r,
                    "members": [r],
                    "bridge_to": [],
                    "boundary": conflicts,
                })
                assigned[r] = len(clusters) - 1
            else:
                clusters[idx]["members"].append(r)
                assigned[r] = idx
            continue

        # ≥2 candidate clusters — boundary case
        # Step 1: verify cluster centroids relate to each other
        centroids = [clusters[i]["centroid"] for i in candidate_cluster_idxs]
        # Check cross-centroid verdicts where we have pair labels
        merge_pairs: list[tuple[int, int]] = []  # (idx_i, idx_j) where centroids merge
        idxs_list = sorted(candidate_cluster_idxs)
        for i, ci in enumerate(idxs_list):
            for cj in idxs_list[i+1:]:
                rep_i = clusters[ci]["centroid"]; rep_j = clusters[cj]["centroid"]
                key = tuple(sorted([rep_i, rep_j]))
                if key in pair_lookup:
                    v = pair_lookup[key]
                    if v in MERGE_VERDICTS:
                        merge_pairs.append((ci, cj))

        if merge_pairs:
            # Some candidate clusters should be merged. Union them + add R.
            # Union-find quick & dirty: merge ci into the smallest-idx cluster
            unite_set: set[int] = set()
            for ci, cj in merge_pairs:
                unite_set.add(ci); unite_set.add(cj)
            keep = min(unite_set)
            for other in unite_set - {keep}:
                clusters[keep]["members"].extend(clusters[other]["members"])
                clusters[keep]["bridge_to"].extend(clusters[other]["bridge_to"])
                clusters[keep]["boundary"].extend(clusters[other]["boundary"])
                clusters[other]["members"] = []  # mark dead
                for m in [m for m, a in assigned.items() if a == other]:
                    assigned[m] = keep
            clusters[keep]["members"].append(r)
            assigned[r] = keep
            # Other candidate clusters NOT involved in merge: mark as bridges
            others = candidate_cluster_idxs - unite_set
            clusters[keep]["bridge_to"].extend(sorted(others))
        else:
            # No cross-cluster merge — R is a bridge
            # Assign R to highest-edge-count candidate cluster
            edge_counts: dict[int, int] = {i: 0 for i in candidate_cluster_idxs}
            for nb in merge_neighbors:
                if nb in assigned and assigned[nb] in edge_counts:
                    edge_counts[assigned[nb]] += 1
            best = max(edge_counts, key=lambda i: edge_counts[i])
            clusters[best]["members"].append(r)
            clusters[best]["bridge_to"].extend(sorted(candidate_cluster_idxs - {best}))
            assigned[r] = best

    # Filter out dead clusters (merged into others)
    return [c for c in clusters if c["members"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "outputs/dedup_eval/creative_writing_2k.jsonl"))
    ap.add_argument("--output", default=str(ROOT / "outputs/dedup_eval/clusters_v1.json"))
    args = ap.parse_args()

    pairs = load_pairs(Path(args.input))
    print(f"loaded {len(pairs):,} pairs")
    G_merge, G_nonmerge, meta = build_graphs(pairs)

    # Pair-verdict lookup for cross-centroid checks
    pair_lookup: dict[tuple[str, str], str] = {}
    for p in pairs:
        k = tuple(sorted([p["a_key"], p["b_key"]]))
        pair_lookup[k] = p["verdict"]

    # Group rubrics by specificity
    by_spec: dict[str, set[str]] = defaultdict(set)
    for k, m in meta.items():
        by_spec[m["specificity"]].add(k)

    all_clusters: dict[str, list[dict]] = {}
    for spec in ["vague", "general", "specific", "hyper_specific"]:
        if spec not in by_spec: continue
        rubrics = sorted(by_spec[spec])
        clusters = cluster_bucket(rubrics, G_merge, G_nonmerge, pair_lookup)
        all_clusters[spec] = clusters

        # Stats
        n_total = len(rubrics)
        n_singleton = sum(1 for c in clusters if len(c["members"]) == 1)
        n_multimember = sum(1 for c in clusters if len(c["members"]) > 1)
        n_bridges = sum(1 for c in clusters if c["bridge_to"])
        size_dist = Counter(len(c["members"]) for c in clusters)
        max_size = max(size_dist.keys()) if size_dist else 0

        print(f"\n=== {spec} ===")
        print(f"  rubrics: {n_total}")
        print(f"  clusters: {len(clusters)} ({n_singleton} singletons, {n_multimember} multi-member)")
        print(f"  bridges (clusters touching others): {n_bridges}")
        print(f"  cluster size distribution: {sorted(size_dist.items())}")
        print(f"  max cluster size: {max_size}")

        # Show top 5 largest clusters
        sorted_clusters = sorted(clusters, key=lambda c: -len(c["members"]))
        for c in sorted_clusters[:5]:
            if len(c["members"]) < 2: break
            centroid_name = meta[c["centroid"]]["name"]
            print(f"\n  CLUSTER (size={len(c['members'])}, centroid='{centroid_name[:60]}', bridges={len(c['bridge_to'])})")
            for m in c["members"][:6]:
                tag = " *BRIDGE*" if m != c["centroid"] and m in c.get("boundary", []) else ""
                print(f"     - {meta[m]['name'][:80]}{tag}")
            if len(c["members"]) > 6:
                print(f"     ... and {len(c['members'])-6} more")

    # Save
    out = {spec: [
        {"centroid": c["centroid"], "centroid_name": meta[c["centroid"]]["name"],
         "members": [{"key": m, "name": meta[m]["name"]} for m in c["members"]],
         "bridge_to": c["bridge_to"], "boundary": c.get("boundary", [])}
        for c in cs
    ] for spec, cs in all_clusters.items()}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {args.output}")


if __name__ == "__main__":
    main()
