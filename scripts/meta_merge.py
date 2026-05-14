"""
Meta-merge (Round 2+) of LLM-proposed parents from a previous hierarchy run.

Takes an expanded hierarchy JSON (from build_hierarchy.py), treats each parent
as a new "cluster" input, and runs the SAME hierarchy algorithm on parents:
  - Embed parent (name + description) with text-embedding-3-small
  - Pick the parent with most leaf rubrics as anchor
  - Order by cosine similarity to anchor
  - Slice K=200 at a time → ask gpt-5 to MERGE redundant parents OR propose
    GRANDPARENTS (parents of parents)
  - Save expanded Round 2 trees with all underlying leaves

Output schemas:
  - {task}_{bucket}_r2.json: raw round-2 decisions
  - {task}_{bucket}_r2_expanded.json: each merged group / grandparent with
    full child→leaf rubric mapping
"""
from __future__ import annotations
import argparse, asyncio, json, os, sys, time
from pathlib import Path
import numpy as np
from openai import AsyncOpenAI, OpenAI

# Reuse the hierarchy machinery
sys.path.insert(0, str(Path(__file__).parent))
from build_hierarchy import (
    JSON_SCHEMA, build_system_prompt, call_llm, build_batch_user_msg,
    TASK_DESCRIPTIONS,
)

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")
KEY_PATH = Path("/lfs/skampere3/0/alexspan/.openai-salt-lab-key.txt")
EMBED_MODEL = "text-embedding-3-small"


def round1_parents_as_clusters(r1_expanded: dict) -> list[dict]:
    """Convert Round 1 parented_trees into 'cluster' dicts compatible with
    build_hierarchy's input format."""
    out = []
    for i, p in enumerate(r1_expanded.get("parented_trees", [])):
        all_leaves = []
        for child in p.get("children", []):
            all_leaves.extend(child.get("rubrics", []))
        out.append({
            "medoid_key": f"r1_parent_{i}",
            "medoid_name": p.get("parent_name", ""),
            "medoid_description": p.get("parent_description", ""),
            "size": len(all_leaves),  # total leaves under this parent
            "members": all_leaves,
            # Round-1 children kept for downstream expansion
            "_r1_children": p.get("children", []),
        })
    return out


def round1_merges_as_clusters(r1_expanded: dict, start_idx: int) -> list[dict]:
    """Also include Round 1 merged groups as 'clusters' for Round 2.
    A merged group is a single canonical concept; it can also be subsumed
    by a grandparent."""
    out = []
    for i, m in enumerate(r1_expanded.get("merged_trees", [])):
        idx = start_idx + i
        out.append({
            "medoid_key": f"r1_merge_{i}",
            "medoid_name": m.get("merged_name", ""),
            "medoid_description": m.get("merged_description", ""),
            "size": m.get("total_rubric_count", 0),
            "members": m.get("all_rubrics", []),
            "_r1_merge_source_ids": m.get("source_cluster_ids", []),
        })
    return out


def embed_texts(client: OpenAI, texts: list[str], batch_size: int = 256) -> np.ndarray:
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        embs.extend([d.embedding for d in resp.data])
        print(f"  embedded {i+len(batch)}/{len(texts)}")
    arr = np.array(embs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0, 1.0, norms)


async def main_async(args):
    r1 = json.loads(Path(args.input).read_text())
    print(f"loaded Round 1 expanded: {r1.get('n_parent_trees', 0)} parents, "
          f"{r1.get('n_merge_trees', 0)} merges")

    # Convert R1 parents (+ optionally merges) into R2 cluster inputs
    r2_clusters = round1_parents_as_clusters(r1)
    if args.include_merges:
        r2_clusters.extend(round1_merges_as_clusters(r1, len(r2_clusters)))
    print(f"R2 input clusters: {len(r2_clusters)}")

    if len(r2_clusters) < 5:
        print("Too few R1 parents — meta-merge not useful. Exiting.")
        return

    # Embed (name + description)
    api_key = KEY_PATH.read_text().strip() if KEY_PATH.exists() else os.environ.get("OPENAI_API_KEY", "")
    sync_client = OpenAI(api_key=api_key)
    print("\nembedding R2 cluster medoids...")
    texts = [f"{c['medoid_name']}\n{c['medoid_description']}"[:2000] for c in r2_clusters]
    embs = embed_texts(sync_client, texts)
    print(f"embeddings: {embs.shape}")

    # Anchor: largest (most leaves)
    sizes = np.array([c["size"] for c in r2_clusters])
    anchor_idx = int(np.argmax(sizes))
    print(f"anchor: r2_cluster {anchor_idx} '{r2_clusters[anchor_idx]['medoid_name']}' "
          f"(size={sizes[anchor_idx]})")

    anchor_emb = embs[anchor_idx]
    sims = embs @ anchor_emb
    sims[anchor_idx] = 2.0
    order = np.argsort(-sims).tolist()

    K = args.batch_size
    batches = [order[i:i+K] for i in range(0, len(order), K)]
    if args.max_batches > 0:
        batches = batches[: args.max_batches]
    print(f"batches: {len(batches)} (K={K})")

    async_client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(args.concurrency)

    system_prompt = build_system_prompt(args.task)

    async def process_batch(batch_idx: int, cluster_ids: list[int]) -> dict:
        async with sem:
            id_to_local = {cid: i+1 for i, cid in enumerate(cluster_ids)}
            user_msg = build_batch_user_msg(cluster_ids, r2_clusters, id_to_local)
            t0 = time.perf_counter()
            res = await call_llm(async_client, args.model, system_prompt, user_msg,
                                  service_tier=args.service_tier)
            return {
                "batch_idx": batch_idx,
                "cluster_ids": cluster_ids,
                "id_to_local": id_to_local,
                "result": res,
                "elapsed_sec": time.perf_counter() - t0,
            }

    print(f"\nlaunching {len(batches)} async batches (concurrency={args.concurrency})...")
    t_start = time.perf_counter()
    coros = [process_batch(i, b) for i, b in enumerate(batches)]
    results = await asyncio.gather(*coros)
    print(f"done in {time.perf_counter()-t_start:.1f}s")

    # Aggregate decisions
    n_merges = n_grandparents = 0
    n_r2_clusters_merged = n_r2_clusters_grandparented = 0
    for d in results:
        res = d.get("result", {})
        for m in res.get("merges", []):
            n_merges += 1
            n_r2_clusters_merged += len(m.get("cluster_ids", []))
        for p in res.get("parents", []):
            n_grandparents += 1
            n_r2_clusters_grandparented += len(p.get("child_cluster_ids", []))

    print(f"\nROUND 2 DECISIONS")
    print(f"  total merges (collapsing redundant R1 parents): {n_merges} "
          f"(touching {n_r2_clusters_merged} R1 parents)")
    print(f"  total grandparents: {n_grandparents} (covering {n_r2_clusters_grandparented} R1 parents)")

    # Save raw
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "task": args.task, "bucket": args.bucket, "round": 2, "model": args.model,
        "round1_input": str(args.input),
        "n_r2_clusters_in": len(r2_clusters),
        "anchor_idx": anchor_idx,
        "anchor_name": r2_clusters[anchor_idx]["medoid_name"],
        "batch_size": K,
        "batches": results,
    }, indent=2, ensure_ascii=False))
    print(f"\nsaved raw decisions -> {out_path}")

    # Expand: each merge/grandparent gets full underlying rubrics
    def r2_cluster_full_leaves(c: dict) -> list[dict]:
        return c.get("members", [])

    merged_groups = []
    grandparents = []
    for d in results:
        res = d.get("result", {})
        local_to_id = {v: int(k) for k, v in d["id_to_local"].items()}
        for m in res.get("merges", []):
            r2_ids = [local_to_id.get(lid) for lid in m.get("cluster_ids", [])]
            r2_ids = [g for g in r2_ids if g is not None]
            all_leaves = []
            for gid in r2_ids:
                all_leaves.extend(r2_cluster_full_leaves(r2_clusters[gid]))
            merged_groups.append({
                "merged_name": m.get("merged_name", ""),
                "merged_description": m.get("merged_description", ""),
                "source_r2_cluster_ids": r2_ids,
                "source_r2_cluster_names": [r2_clusters[g]["medoid_name"] for g in r2_ids],
                "total_leaf_rubrics": len(all_leaves),
                "all_leaves": all_leaves,
            })
        for gp in res.get("parents", []):
            r2_ids = [local_to_id.get(lid) for lid in gp.get("child_cluster_ids", [])]
            r2_ids = [g for g in r2_ids if g is not None]
            child_summaries = []
            total_leaves = 0
            for gid in r2_ids:
                rc = r2_clusters[gid]
                leaves = r2_cluster_full_leaves(rc)
                total_leaves += len(leaves)
                child_summaries.append({
                    "r2_cluster_id": gid,
                    "name": rc["medoid_name"],
                    "description": rc["medoid_description"],
                    "n_leaves": len(leaves),
                })
            grandparents.append({
                "grandparent_name": gp.get("parent_name", ""),
                "grandparent_description": gp.get("parent_description", ""),
                "n_children": len(child_summaries),
                "total_leaf_rubrics": total_leaves,
                "children": child_summaries,
            })

    expanded_path = out_path.with_name(out_path.stem + "_expanded.json")
    expanded_path.write_text(json.dumps({
        "task": args.task, "bucket": args.bucket, "round": 2, "model": args.model,
        "n_r2_clusters_in": len(r2_clusters),
        "n_merged_groups": len(merged_groups),
        "n_grandparents": len(grandparents),
        "merged_groups": merged_groups,
        "grandparents": grandparents,
    }, indent=2, ensure_ascii=False))
    print(f"saved expanded R2 -> {expanded_path}")
    print(f"  merged groups (collapsing redundant R1 parents): {len(merged_groups)}")
    print(f"  grandparent concepts: {len(grandparents)} "
          f"(covering {sum(g['total_leaf_rubrics'] for g in grandparents):,} leaves)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="creative-writing")
    ap.add_argument("--bucket", default="general")
    ap.add_argument("--input", required=True,
                    help="Path to Round 1 _expanded.json from build_hierarchy.py")
    ap.add_argument("--output", default=None)
    ap.add_argument("--model", default="gpt-5")
    ap.add_argument("--service-tier", default=None, choices=[None, "auto", "default", "flex"],
                    help="OpenAI service tier. 'flex' = ~50%% off, slower/lower priority.")
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=15)
    ap.add_argument("--include-merges", action="store_true",
                    help="Also include Round-1 merge concepts as Round-2 cluster inputs.")
    args = ap.parse_args()
    if args.output is None:
        args.output = str(ROOT / f"outputs/hierarchy/{args.task}_{args.bucket}_r2.json")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
