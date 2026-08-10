"""
Cluster ALL keep+work rubrics in a task using embedding similarity only —
no LLM calls. Per specificity bucket, take connected components where
cosine > merge_threshold, compute centroid (medoid) per cluster.

Then report how many cluster-pair centroids fall in the ambiguous zone
(centroid cos in [0.65, merge_threshold]) — those are candidates for the
post-hoc LLM purification pass.

Output: outputs/clusters/{task}_embed_only.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

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
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry: parent[rx] = ry
    for a, b in edges:
        union(a, b)
    return [find(i) for i in range(n)]


def compute_medoid(member_idx: list[int], embs: np.ndarray) -> int:
    """Return the index in member_idx that has highest avg cosine to others
    (the medoid). All embs assumed L2-normalized."""
    if len(member_idx) == 1:
        return member_idx[0]
    sub = embs[member_idx]
    sims = sub @ sub.T  # (m, m)
    np.fill_diagonal(sims, 0.0)
    avg = sims.mean(axis=1)
    return member_idx[int(np.argmax(avg))]


def cluster_one_bucket(rubrics: list[dict], embs: np.ndarray,
                       merge_threshold: float, ambig_low: float) -> dict:
    """Cluster a single bucket using connected components above threshold.
    Returns dict with clusters + purification candidates."""
    name_lc = [str(r.get('rubric_name') or '').lower().strip() for r in rubrics]
    pid = [r['page_id'] for r in rubrics]
    n = len(rubrics)

    # Build merge edges: cos >= merge_threshold, cross-doc, non-identical-name
    edges: list[tuple[int, int]] = []
    batch = 512
    for s in range(0, n, batch):
        e = min(s + batch, n)
        block = embs[s:e] @ embs.T  # (b, n)
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

    # Connected components
    comp = union_find_components(n, edges)
    clusters: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(comp):
        clusters[c].append(i)

    # Compute medoid centroids
    cluster_list = []
    for cid, members in clusters.items():
        med = compute_medoid(members, embs)
        cluster_list.append({"medoid_idx": med, "members": members})

    # Find cluster-pair candidates for post-hoc purification:
    # centroid_cos in [ambig_low, merge_threshold)
    n_clusters = len(cluster_list)
    medoids = np.array([c["medoid_idx"] for c in cluster_list])
    medoid_embs = embs[medoids]  # (K, d)
    medoid_specs = [rubrics[m]['cls_specificity'] for m in medoids]

    # Pairwise centroid sims (only K x K, much smaller)
    centroid_sims = medoid_embs @ medoid_embs.T
    np.fill_diagonal(centroid_sims, 0.0)

    purif_pairs = []
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            c = float(centroid_sims[i, j])
            if c < ambig_low: continue
            if c >= merge_threshold: continue  # would have been merged already
            purif_pairs.append((i, j, c))

    return {
        "clusters": cluster_list,
        "n_pairs_for_purification": len(purif_pairs),
        "purif_pairs_sample": purif_pairs[:10],
        "size_distribution": dict(Counter(len(c["members"]) for c in cluster_list)),
        "max_cluster_size": max((len(c["members"]) for c in cluster_list), default=0),
        "n_singletons": sum(1 for c in cluster_list if len(c["members"]) == 1),
        "n_multimember": sum(1 for c in cluster_list if len(c["members"]) > 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="creative-writing")
    ap.add_argument("--merge-threshold", type=float, default=0.85)
    ap.add_argument("--ambig-low", type=float, default=0.65,
                    help="Centroid cos lower bound for post-hoc purification candidates.")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    rubrics_all = load_keep_work(args.task)
    print(f"loaded {len(rubrics_all):,} {args.task} keep+work rubrics")

    cache = ROOT / f"outputs/embeddings/{args.task}_work_embeddings.npz"
    d = np.load(cache, allow_pickle=True)
    expected = [f"{r['page_id']}::{r['rubric_idx']}" for r in rubrics_all]
    if list(d['keys']) != expected:
        # Re-align
        key_to_emb = dict(zip(list(d['keys']), d['embs']))
        embs_all = np.array([key_to_emb[k] for k in expected], dtype=np.float32)
    else:
        embs_all = d['embs'].astype(np.float32)
    print(f"embeddings: {embs_all.shape}")

    SPECS = ["vague", "general", "specific", "hyper_specific"]
    out_path = Path(args.output or (ROOT / f"outputs/clusters/{args.task}_embed_only.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_pairs_for_purif = 0
    bucket_results: dict[str, dict] = {}

    for spec in SPECS:
        idx = [i for i, r in enumerate(rubrics_all) if r.get('cls_specificity') == spec]
        if not idx:
            print(f"\n=== {spec}: empty"); continue
        rubrics_b = [rubrics_all[i] for i in idx]
        embs_b    = embs_all[idx]
        print(f"\n=== {spec}  N={len(rubrics_b):,} ===")

        res = cluster_one_bucket(rubrics_b, embs_b, args.merge_threshold, args.ambig_low)
        n_clusters = len(res["clusters"])
        print(f"  clusters: {n_clusters:,} ({res['n_singletons']:,} singletons, {res['n_multimember']:,} multi-member)")
        print(f"  max cluster size: {res['max_cluster_size']}")
        print(f"  compression: {res['n_multimember']:,} multi-member clusters bundle "
              f"{len(rubrics_b) - res['n_singletons'] - res['n_multimember']:,} extra rubrics")
        print(f"  size distribution (size, count): {sorted(res['size_distribution'].items())[:10]}{'...' if len(res['size_distribution']) > 10 else ''}")
        print(f"  cluster-pair centroids in ambiguous zone [{args.ambig_low}, {args.merge_threshold}): {res['n_pairs_for_purification']:,}")

        # Resolve members to original rubric records for output
        out_clusters = []
        for c in res["clusters"]:
            med = rubrics_b[c["medoid_idx"]]
            out_clusters.append({
                "medoid_key": f"{med['page_id']}::{med['rubric_idx']}",
                "medoid_name": med['rubric_name'],
                "size": len(c["members"]),
                "members": [{
                    "key": f"{rubrics_b[m]['page_id']}::{rubrics_b[m]['rubric_idx']}",
                    "name": rubrics_b[m]['rubric_name'],
                } for m in c["members"]],
            })

        bucket_results[spec] = {
            "n_rubrics": len(rubrics_b),
            "n_clusters": n_clusters,
            "n_singletons": res["n_singletons"],
            "n_multimember": res["n_multimember"],
            "max_cluster_size": res["max_cluster_size"],
            "n_pairs_for_purification": res["n_pairs_for_purification"],
            "clusters": out_clusters,
        }
        total_pairs_for_purif += res["n_pairs_for_purification"]

    print(f"\n=== SUMMARY ===")
    print(f"total clusters across buckets: {sum(b['n_clusters'] for b in bucket_results.values()):,}")
    print(f"total ambiguous-zone cluster-pairs for purification: {total_pairs_for_purif:,}")
    print(f"projected purification LLM cost @ $0.001/pair: ${total_pairs_for_purif * 0.001:.2f}")

    out_path.write_text(json.dumps(bucket_results, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
