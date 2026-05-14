"""
Iterative LLM-driven hierarchy builder over complete-linkage clusters.

Algorithm:
  1. Start with the complete-linkage clusters.
  2. Pick the biggest cluster as the anchor; sort all other clusters by
     centroid cosine similarity to the anchor (descending).
  3. Slice K=100 clusters at a time, in similarity order, and pass them to
     gpt-5-mini. The LLM decides:
       a) which clusters to MERGE (same construct, concise enough to apply)
       b) which clusters are related but distinct, and propose a PARENT
          concept that subsumes them
  4. Persist decisions. After all batches, parents can themselves be merged
     in a follow-up pass (deferred).

Output:
  - outputs/hierarchy/{task}_{bucket}_hierarchy.json
"""
from __future__ import annotations
import argparse, asyncio, json, os, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
from openai import AsyncOpenAI

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")
CHUNKS = ROOT / "outputs/classifier_chunks_FULL"
KEY_PATH = Path("/lfs/skampere3/0/alexspan/.openai-salt-lab-key.txt")
DEFAULT_MODEL = "gpt-5-mini"


JSON_SCHEMA = {
    "name": "hierarchy_decisions",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning": {"type": "string"},
            "merges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "cluster_ids": {"type": "array", "items": {"type": "integer"}, "minItems": 2},
                        "merged_name": {"type": "string"},
                        "merged_description": {"type": "string"},
                    },
                    "required": ["cluster_ids", "merged_name", "merged_description"],
                },
            },
            "parents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "child_cluster_ids": {"type": "array", "items": {"type": "integer"}, "minItems": 2},
                        "parent_name": {"type": "string"},
                        "parent_description": {"type": "string"},
                    },
                    "required": ["child_cluster_ids", "parent_name", "parent_description"],
                },
            },
        },
        "required": ["reasoning", "merges", "parents"],
    },
}


TASK_DESCRIPTIONS = {
    "code-review": "evaluating proposed code changes (pull requests, patches) against quality, style, security, and maintainability criteria",
    "creative-writing": "evaluating the quality of creative writing — fiction, poetry, essays — against craft, structural, and aesthetic criteria",
    "grant-funding": "evaluating research grant proposals against funding-decision criteria (significance, innovation, approach, investigator quality)",
    "humor": "evaluating humor and comedy (writing, performance, joke construction) against criteria for effectiveness, taste, and form",
    "legal-outcome-prediction": "evaluating legal arguments and litigation documents against statutory, procedural, and doctrinal standards",
    "math-stackexchange": "evaluating mathematical writing and proofs against criteria for rigor, clarity, elegance, and precision",
    "news-homepages": "evaluating news article placement and editorial curation against newsworthiness, ethics, and audience criteria",
    "notice-and-comment": "evaluating regulatory documents, economic impact assessments, and benefit-cost analyses against analytic and procedural standards",
    "patents": "evaluating patent applications against patentability requirements (novelty, non-obviousness, utility, written description)",
    "peer-review": "evaluating academic manuscripts (papers, articles) against publication-quality criteria (novelty, methodology, clarity, contribution)",
    "press-releases": "evaluating press release effectiveness for corporate, governmental, and organizational communications",
}


SYSTEM_PROMPT_TEMPLATE = """You are building a hierarchical taxonomy of evaluation rubrics for {task_description}.

Each rubric measures some property of work in this task. You will see a batch of up to 100 rubric clusters (each represented by a name and description). Your job:

1. MERGE clusters that measure the SAME underlying construct. A merged rubric must remain CONCISE enough to be meaningfully applied to a real example. Avoid over-generalizing (e.g., "good writing" or "compelling characters" without qualification are too broad). Two clusters should be merged only if a human evaluator would score them identically on the same piece of work.

2. PARENT: For clusters that are related but measure distinct constructs (would score differently), propose a parent concept that captures their shared axis. The parent should be:
   - Specific enough to be a meaningful evaluation dimension
   - General enough to subsume each child as a sub-criterion
   - Avoid blandly broad parents that subsume too much

Default to keeping clusters separate if uncertain. It is OK to leave clusters un-merged and un-parented in a batch — only act when the relationship is clear.

Output:
- "reasoning": short summary of the analysis steps taken
- "merges": list of cluster ID groups to merge, with a single canonical name and description
- "parents": list of child cluster ID groups + proposed parent concept

Cluster IDs in your output MUST come from the cluster IDs shown in the user message."""


def build_system_prompt(task: str) -> str:
    desc = TASK_DESCRIPTIONS.get(task, "evaluating work in this task domain")
    return SYSTEM_PROMPT_TEMPLATE.format(task_description=desc)


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


def build_batch_user_msg(cluster_ids: list[int], clusters: list[dict],
                         id_to_local: dict[int, int]) -> str:
    """clusters: list of cluster dicts (from input JSON, with medoid_name, medoid_description, size).
    cluster_ids: list of cluster indices to include in this batch (in similarity order).
    id_to_local: mapping cluster_id -> local 1-based id shown to LLM (stable within batch)."""
    lines = ["CLUSTERS TO ANALYZE:\n"]
    for cid in cluster_ids:
        local = id_to_local[cid]
        c = clusters[cid]
        nm = c.get('medoid_name') or ''
        dsc = (c.get('medoid_description') or '')[:300]
        sz = c.get('size', 1)
        lines.append(f"[{local}] ({sz}x) NAME: {nm}\n     DESC: {dsc}\n")
    return "".join(lines)


async def call_llm(client: AsyncOpenAI, model: str, system: str, user: str,
                    timeout_sec: float = 300.0, service_tier: str | None = None) -> dict:
    """Call OpenAI with explicit per-call timeout + 3 retries.
    service_tier='flex' enables OpenAI's discounted flex tier (~50% off) at
    the cost of possibly slower / lower-priority responses."""
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "response_format": {"type": "json_schema", "json_schema": JSON_SCHEMA},
    }
    if service_tier:
        kwargs["service_tier"] = service_tier

    for attempt in range(3):
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(**kwargs),
                timeout=timeout_sec,
            )
            raw = resp.choices[0].message.content or ""
            return json.loads(raw)
        except asyncio.TimeoutError:
            if attempt == 2:
                return {"_error": f"timeout after {timeout_sec}s on all attempts"}
            await asyncio.sleep(2 ** attempt)
        except Exception as e:
            if attempt == 2:
                return {"_error": str(e)[:300]}
            await asyncio.sleep(2 ** attempt)
    return {}


async def main_async(args):
    rubrics_all = load_keep_work(args.task)
    print(f"loaded {len(rubrics_all):,} {args.task} keep+work rubrics")

    # Load complete-linkage clusters
    cluster_input = json.loads(Path(args.input).read_text())
    bucket = cluster_input[args.bucket]
    clusters = bucket["clusters"]
    print(f"bucket={args.bucket}: {len(clusters):,} clusters")

    # Load embeddings + lookup medoid embeddings
    cache = ROOT / f"outputs/embeddings/{args.task}_work_embeddings.npz"
    d = np.load(cache, allow_pickle=True)
    cache_keys = list(d['keys'])
    embs_all = d['embs'].astype(np.float32)
    key_to_emb = dict(zip(cache_keys, embs_all))

    medoid_embs = np.array([key_to_emb[c["medoid_key"]] for c in clusters], dtype=np.float32)
    print(f"medoid embeddings: {medoid_embs.shape}")

    # Pick the biggest cluster as anchor
    sizes = np.array([c["size"] for c in clusters])
    anchor_idx = int(np.argmax(sizes))
    print(f"anchor: cluster {anchor_idx} size={sizes[anchor_idx]} medoid='{clusters[anchor_idx]['medoid_name']}'")

    # Order all clusters by similarity to anchor (descending)
    anchor_emb = medoid_embs[anchor_idx]
    sims = medoid_embs @ anchor_emb
    sims[anchor_idx] = 2.0  # ensure anchor comes first
    order = np.argsort(-sims).tolist()

    # Build batches of size K
    K = args.batch_size
    if args.window_stride is not None and args.window_stride < K:
        # Sliding window: adjacent batches overlap by (K - stride) clusters.
        stride = args.window_stride
        batches = []
        i = 0
        while i < len(order):
            batches.append(order[i:i+K])
            if i + K >= len(order):
                break
            i += stride
        print(f"batches (sliding): {len(batches)}  K={K}  stride={stride}  "
              f"overlap={K - stride} per adjacent pair")
    else:
        batches = [order[i:i+K] for i in range(0, len(order), K)]
        print(f"batches (disjoint): {len(batches)} K={K}")
    if args.max_batches > 0:
        batches = batches[: args.max_batches]

    api_key = KEY_PATH.read_text().strip() if KEY_PATH.exists() else os.environ.get("OPENAI_API_KEY", "")
    client = AsyncOpenAI(api_key=api_key)

    all_decisions = []
    sem = asyncio.Semaphore(args.concurrency)

    system_prompt = build_system_prompt(args.task)

    async def process_batch(batch_idx: int, cluster_ids: list[int]) -> dict:
        async with sem:
            id_to_local = {cid: i+1 for i, cid in enumerate(cluster_ids)}
            local_to_id = {i+1: cid for i, cid in enumerate(cluster_ids)}
            user_msg = build_batch_user_msg(cluster_ids, clusters, id_to_local)
            t0 = time.perf_counter()
            res = await call_llm(client, args.model, system_prompt, user_msg,
                                  service_tier=args.service_tier)
            dt = time.perf_counter() - t0
            return {
                "batch_idx": batch_idx,
                "cluster_ids": cluster_ids,
                "id_to_local": id_to_local,
                "result": res,
                "elapsed_sec": dt,
            }

    print(f"\nlaunching {len(batches)} async batches (concurrency={args.concurrency})...")
    t_start = time.perf_counter()
    coros = [process_batch(i, b) for i, b in enumerate(batches)]
    results = await asyncio.gather(*coros)
    print(f"done in {time.perf_counter()-t_start:.1f}s")

    # Tally
    n_merges = 0
    n_parents = 0
    n_clusters_merged = 0
    n_clusters_parented = 0
    for d in results:
        res = d.get("result", {})
        for m in res.get("merges", []):
            n_merges += 1
            n_clusters_merged += len(m.get("cluster_ids", []))
        for p in res.get("parents", []):
            n_parents += 1
            n_clusters_parented += len(p.get("child_cluster_ids", []))

    print(f"\nDECISIONS")
    print(f"  total merges proposed: {n_merges} (touching {n_clusters_merged} clusters)")
    print(f"  total parents proposed: {n_parents} (touching {n_clusters_parented} clusters)")

    # Save raw decisions
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "task": args.task, "bucket": args.bucket, "model": args.model,
        "n_clusters_in": len(clusters),
        "anchor_idx": anchor_idx,
        "anchor_name": clusters[anchor_idx]["medoid_name"],
        "batch_size": K,
        "batches": results,
    }, indent=2, ensure_ascii=False))
    print(f"\nsaved raw decisions -> {out_path}")

    # ----- Expanded trees: every merge + every parent with full underlying rubrics -----
    def cluster_full_members(c: dict) -> list[dict]:
        return [{"key": m["key"], "name": m["name"]} for m in c.get("members", [])]

    merged_trees = []
    parented_trees = []
    for d in results:
        res = d.get("result", {})
        local_to_global = {v: int(k) for k, v in d["id_to_local"].items()}
        for m in res.get("merges", []):
            global_ids = [local_to_global.get(lid) for lid in m.get("cluster_ids", [])]
            global_ids = [g for g in global_ids if g is not None]
            all_leaf_rubrics = []
            for gid in global_ids:
                all_leaf_rubrics.extend(cluster_full_members(clusters[gid]))
            merged_trees.append({
                "merged_name": m.get("merged_name", ""),
                "merged_description": m.get("merged_description", ""),
                "source_cluster_ids": global_ids,
                "source_cluster_medoids": [clusters[g]["medoid_name"] for g in global_ids],
                "total_rubric_count": len(all_leaf_rubrics),
                "all_rubrics": all_leaf_rubrics,
            })
        for p in res.get("parents", []):
            global_ids = [local_to_global.get(lid) for lid in p.get("child_cluster_ids", [])]
            global_ids = [g for g in global_ids if g is not None]
            children = []
            total_leaves = 0
            for gid in global_ids:
                leaves = cluster_full_members(clusters[gid])
                total_leaves += len(leaves)
                children.append({
                    "cluster_id": gid,
                    "medoid_name": clusters[gid]["medoid_name"],
                    "medoid_description": clusters[gid].get("medoid_description", ""),
                    "size": clusters[gid].get("size", len(leaves)),
                    "rubrics": leaves,
                })
            parented_trees.append({
                "parent_name": p.get("parent_name", ""),
                "parent_description": p.get("parent_description", ""),
                "n_children": len(children),
                "total_leaf_rubrics": total_leaves,
                "children": children,
            })

    expanded_path = out_path.with_name(out_path.stem + "_expanded.json")
    expanded_path.write_text(json.dumps({
        "task": args.task, "bucket": args.bucket, "model": args.model,
        "n_clusters_in": len(clusters),
        "n_merge_trees": len(merged_trees),
        "n_parent_trees": len(parented_trees),
        "merged_trees": merged_trees,
        "parented_trees": parented_trees,
    }, indent=2, ensure_ascii=False))
    print(f"saved expanded trees -> {expanded_path}")
    print(f"  merge trees: {len(merged_trees)}  (collapsing {sum(t['total_rubric_count'] for t in merged_trees)} rubrics)")
    print(f"  parent trees: {len(parented_trees)}  (covering {sum(t['total_leaf_rubrics'] for t in parented_trees)} rubrics)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="creative-writing")
    ap.add_argument("--bucket", default="general", choices=["vague", "general", "specific", "hyper_specific"])
    ap.add_argument("--input", default=str(ROOT / "outputs/clusters/creative-writing_complete_linkage.json"))
    ap.add_argument("--output", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="OpenAI chat model (gpt-5-mini default; gpt-5 for higher-quality.")
    ap.add_argument("--service-tier", default=None, choices=[None, "auto", "default", "flex"],
                    help="OpenAI service tier. 'flex' = discounted ~50%% off, slower/lower priority.")
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--window-stride", type=int, default=None,
                    help="If set < batch-size, use sliding-window batching with this stride. "
                         "Adjacent batches overlap by (K - stride). E.g. K=200 stride=100 → 50%% overlap.")
    ap.add_argument("--max-batches", type=int, default=0,
                    help="0 = process all. Else, limit to first N batches (for testing).")
    ap.add_argument("--concurrency", type=int, default=20)
    args = ap.parse_args()
    if args.output is None:
        args.output = str(ROOT / f"outputs/hierarchy/{args.task}_{args.bucket}.json")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
