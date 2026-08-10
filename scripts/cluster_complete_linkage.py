"""
Cluster all keep+work rubrics using embedding similarity with the hybrid
single-linkage-then-complete-linkage-refinement approach.

Why hybrid: pure complete-linkage on 12K rubrics needs O(N²)~O(N³) ops on
a full distance matrix. Instead:
  1. Sparse edge set: pairs with cos >= threshold (cross-doc, no name-dup).
  2. Single-linkage connected components — fast.
  3. For each component of size >= 5: run scipy hierarchical complete-linkage
     on its members only (size ~5-100, fast). Cut at 1 - threshold.
     This splits drifted single-linkage chains into clique-like clusters.
  4. Smaller components kept as-is.

Output:
  - outputs/clusters/{task}_complete_linkage.json
  - same structure as the embed-only version
  - reports # of cluster-pair candidates for post-hoc purification
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")
CHUNKS = ROOT / "outputs/classifier_chunks_FULL"


def load_keep_work(task: str) -> list[dict]:
    rows = []
    for cf in sorted(CHUNKS.glob("chunk_*.jsonl")):
        with cf.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if (r.get("task") == task and r.get("cls_ok")
                        and r.get("cls_keep") == "keep"
                        and r.get("cls_target") == "work"):
                        rows.append(r)
                except Exception:
                    pass
    return rows


def union_find_components(n: int, edges: list[tuple[int, int]]) -> list[int]:
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry: parent[rx] = ry
    for a, b in edges:
        union(a, b)
    return [find(i) for i in range(n)]


def compute_medoid(member_idx: list[int], embs: np.ndarray) -> int:
    if len(member_idx) == 1:
        return member_idx[0]
    sub = embs[member_idx]
    sims = sub @ sub.T
    np.fill_diagonal(sims, 0.0)
    return member_idx[int(np.argmax(sims.mean(axis=1)))]


def refine_component_complete_linkage(member_idx: list[int], embs: np.ndarray,
                                       merge_threshold: float) -> list[list[int]]:
    """Run complete-linkage on a single connected component, cut at threshold.
    Returns list of sub-clusters (each is a list of original indices)."""
    if len(member_idx) <= 2:
        return [member_idx]
    sub = embs[member_idx]
    sims = sub @ sub.T          # (m, m) in [-1, 1]
    np.clip(sims, -1.0, 1.0, out=sims)
    dist = 1.0 - sims           # cosine distance
    np.fill_diagonal(dist, 0.0)
    # Convert to condensed
    condensed = squareform(dist, checks=False)
    if condensed.size == 0:
        return [member_idx]
    Z = linkage(condensed, method='complete')
    labels = fcluster(Z, t=(1.0 - merge_threshold), criterion='distance')
    sub_clusters: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        sub_clusters[int(lab)].append(member_idx[i])
    return list(sub_clusters.values())


def cluster_bucket_complete(rubrics: list[dict], embs: np.ndarray,
                             merge_threshold: float, ambig_low: float,
                             refine_min_size: int = 5) -> dict:
    name_lc = [str(r.get('rubric_name') or '').lower().strip() for r in rubrics]
    pid = [r['page_id'] for r in rubrics]
    n = len(rubrics)

    # Step 1: sparse merge edges
    edges: list[tuple[int, int]] = []
    batch = 512
    for s in range(0, n, batch):
        e = min(s + batch, n)
        block = embs[s:e] @ embs.T
        for i, ri in enumerate(range(s, e)):
            sims = block[i]
            above = np.where(sims >= merge_threshold)[0]
            for j in above:
                j = int(j)
                if j <= ri: continue
                if pid[ri] == pid[j]: continue
                if name_lc[ri] == name_lc[j]: continue
                edges.append((ri, j))

    print(f"  merge edges (cos>={merge_threshold}): {len(edges):,}")

    # Step 2: single-linkage connected components
    comp = union_find_components(n, edges)
    sl_clusters: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(comp):
        sl_clusters[c].append(i)

    print(f"  single-linkage components: {len(sl_clusters):,}")
    sl_sizes = [len(v) for v in sl_clusters.values()]
    print(f"    components >= {refine_min_size}: {sum(1 for s in sl_sizes if s >= refine_min_size)}")

    # Step 3: refine large components with complete-linkage
    final_clusters: list[list[int]] = []
    n_split = 0; n_split_from = 0
    for members in sl_clusters.values():
        if len(members) < refine_min_size:
            final_clusters.append(members)
            continue
        subs = refine_component_complete_linkage(members, embs, merge_threshold)
        if len(subs) > 1:
            n_split += 1
            n_split_from += len(members)
        final_clusters.extend(subs)

    print(f"    refined {n_split} large components ({n_split_from} rubrics) into sub-clusters")

    # Step 4: medoids + stats
    cluster_records = []
    for members in final_clusters:
        med = compute_medoid(members, embs)
        cluster_records.append({"medoid_idx": med, "members": members})

    n_clusters = len(cluster_records)
    n_singletons = sum(1 for c in cluster_records if len(c["members"]) == 1)
    n_multi = n_clusters - n_singletons
    sizes = Counter(len(c["members"]) for c in cluster_records)

    # Step 5: post-hoc purification candidates (centroid_cos in ambig zone)
    medoid_idx = np.array([c["medoid_idx"] for c in cluster_records])
    medoid_embs = embs[medoid_idx]
    if len(medoid_embs) > 1:
        centroid_sims = medoid_embs @ medoid_embs.T
        np.fill_diagonal(centroid_sims, 0.0)
        # Count pairs in (ambig_low, merge_threshold)
        # Vectorized
        upper_tri = np.triu(centroid_sims, k=1)
        in_zone = (upper_tri >= ambig_low) & (upper_tri < merge_threshold)
        n_purif = int(in_zone.sum())
    else:
        n_purif = 0

    return {
        "clusters": cluster_records,
        "n_clusters": n_clusters,
        "n_singletons": n_singletons,
        "n_multimember": n_multi,
        "max_cluster_size": max((len(c["members"]) for c in cluster_records), default=0),
        "size_distribution": dict(sizes),
        "n_pairs_for_purification": n_purif,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="creative-writing")
    ap.add_argument("--merge-threshold", type=float, default=0.85)
    ap.add_argument("--ambig-low", type=float, default=0.65)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    rubrics_all = load_keep_work(args.task)
    print(f"loaded {len(rubrics_all):,} {args.task} keep+work rubrics")

    cache = ROOT / f"outputs/embeddings/{args.task}_work_embeddings.npz"
    d = np.load(cache, allow_pickle=True)
    expected = [f"{r['page_id']}::{r['rubric_idx']}" for r in rubrics_all]
    if list(d['keys']) != expected:
        key_to_emb = dict(zip(list(d['keys']), d['embs']))
        embs_all = np.array([key_to_emb[k] for k in expected], dtype=np.float32)
    else:
        embs_all = d['embs'].astype(np.float32)
    print(f"embeddings: {embs_all.shape}")

    SPECS = ["vague", "general", "specific", "hyper_specific"]
    out_path = Path(args.output or (ROOT / f"outputs/clusters/{args.task}_complete_linkage.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bucket_results: dict[str, dict] = {}
    total_purif = 0
    t_start = time.perf_counter()

    for spec in SPECS:
        idx = [i for i, r in enumerate(rubrics_all) if r.get('cls_specificity') == spec]
        if not idx:
            print(f"\n=== {spec}: empty"); continue
        rubrics_b = [rubrics_all[i] for i in idx]
        embs_b    = embs_all[idx]
        print(f"\n=== {spec}  N={len(rubrics_b):,} ===")

        t0 = time.perf_counter()
        res = cluster_bucket_complete(rubrics_b, embs_b,
                                       args.merge_threshold, args.ambig_low)
        print(f"  clusters: {res['n_clusters']:,} ({res['n_singletons']:,} singletons, {res['n_multimember']:,} multi)")
        print(f"  max cluster size: {res['max_cluster_size']}")
        size_str = sorted(res['size_distribution'].items())
        print(f"  size distribution: {size_str[:12]}{'...' if len(size_str) > 12 else ''}")
        print(f"  purification candidates (cluster cos in [{args.ambig_low}, {args.merge_threshold})): "
              f"{res['n_pairs_for_purification']:,}")
        print(f"  [bucket time: {time.perf_counter()-t0:.1f}s]")

        out_clusters = []
        for c in res["clusters"]:
            med = rubrics_b[c["medoid_idx"]]
            out_clusters.append({
                "medoid_key": f"{med['page_id']}::{med['rubric_idx']}",
                "medoid_name": med['rubric_name'],
                "medoid_description": med.get('rubric_description', ''),
                "size": len(c["members"]),
                "members": [{
                    "key": f"{rubrics_b[m]['page_id']}::{rubrics_b[m]['rubric_idx']}",
                    "name": rubrics_b[m]['rubric_name'],
                } for m in c["members"]],
            })
        bucket_results[spec] = {
            "n_rubrics": len(rubrics_b),
            "n_clusters": res["n_clusters"],
            "n_singletons": res["n_singletons"],
            "n_multimember": res["n_multimember"],
            "max_cluster_size": res["max_cluster_size"],
            "n_pairs_for_purification": res["n_pairs_for_purification"],
            "clusters": out_clusters,
        }
        total_purif += res["n_pairs_for_purification"]

    print(f"\n=== SUMMARY  (elapsed {time.perf_counter()-t_start:.1f}s) ===")
    print(f"total clusters: {sum(b['n_clusters'] for b in bucket_results.values()):,}")
    print(f"total multi-member clusters: {sum(b['n_multimember'] for b in bucket_results.values()):,}")
    print(f"total purification candidates: {total_purif:,}")
    print(f"projected purification LLM cost @ $0.001/pair: ${total_purif * 0.001:.2f}")

    out_path.write_text(json.dumps(bucket_results, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
