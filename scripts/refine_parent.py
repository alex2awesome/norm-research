"""
Within-parent refinement pass.

For each parent concept in an R1 expanded hierarchy:
  - Build a prompt showing the parent + all its children (their names + descriptions)
  - Ask gpt-5: which children DON'T belong under this parent? Which children are
    duplicates of each other and should be merged?
  - Drop misfits to a stranded pool (for later re-anchoring)
  - Merge duplicate children into one canonical child

This is a complement to meta_merge.py (cross-parent merging). meta_merge fixes
"3 different parents are all the same concept" by collapsing them. This script
fixes "the children under a parent include some that don't actually belong."

Output:
  - {task}_{bucket}_r1_refined.json   refined hierarchy (cleaner trees)
  - {task}_{bucket}_r1_stranded.json  children that got dropped to the stranded pool
"""
from __future__ import annotations
import argparse, asyncio, json, os, sys, time
from pathlib import Path
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).parent))
from build_hierarchy import TASK_DESCRIPTIONS

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")
KEY_PATH = Path("/lfs/skampere3/0/alexspan/.openai-salt-lab-key.txt")


REFINE_SCHEMA = {
    "name": "parent_refinement",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning": {"type": "string"},
            "misfits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "child_id": {"type": "integer"},
                        "reason":   {"type": "string"},
                    },
                    "required": ["child_id", "reason"],
                },
            },
            "duplicate_groups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "child_ids":          {"type": "array", "items": {"type": "integer"}, "minItems": 2},
                        "merged_name":        {"type": "string"},
                        "merged_description": {"type": "string"},
                    },
                    "required": ["child_ids", "merged_name", "merged_description"],
                },
            },
        },
        "required": ["reasoning", "misfits", "duplicate_groups"],
    },
}


SYSTEM_PROMPT_TEMPLATE = """You are refining a single parent concept in an evaluation-rubric taxonomy for {task_description}.

The parent concept and its current children are shown below. Each child is a more concrete rubric that the previous step grouped under this parent. Your job is to clean up the children list by identifying two specific issues:

1. MISFITS: children that DON'T BELONG under this parent. They share some surface similarity but actually measure a different construct that doesn't fit the parent's evaluation dimension. Be specific in the reason — what construct does the misfit child measure that the parent doesn't cover?

2. DUPLICATE GROUPS: groups of children that measure essentially the same construct as each other and should be merged into a single canonical child. For each group, propose the merged name + description.

Default to KEEP — only flag clear misfits or clear duplicates. If a child fits the parent's dimension and is distinct from its siblings, leave it alone.

Output JSON with the misfits list and duplicate_groups list. Either can be empty if the parent is already clean."""


def build_user_msg(parent_name: str, parent_description: str, children: list[dict]) -> str:
    lines = [f"PARENT CONCEPT:\n  name: {parent_name}\n  description: {parent_description}\n"]
    lines.append(f"\nCHILDREN ({len(children)} total):\n")
    for i, c in enumerate(children, 1):
        name = c.get("medoid_name") or c.get("name") or ""
        desc = (c.get("medoid_description") or c.get("description") or "")[:300]
        n_leaves = c.get("size") or c.get("n_leaves") or len(c.get("rubrics", []))
        lines.append(f"[{i}] NAME: {name}\n     DESC: {desc}\n     N_LEAVES: {n_leaves}\n")
    return "".join(lines)


async def call_llm(client: AsyncOpenAI, model: str, system: str, user: str,
                    service_tier: str | None = None, timeout_sec: float = 300.0) -> dict:
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "response_format": {"type": "json_schema", "json_schema": REFINE_SCHEMA},
    }
    if service_tier:
        kwargs["service_tier"] = service_tier
    for attempt in range(3):
        try:
            resp = await asyncio.wait_for(client.chat.completions.create(**kwargs), timeout=timeout_sec)
            return json.loads(resp.choices[0].message.content or "")
        except asyncio.TimeoutError:
            if attempt == 2:
                return {"_error": "timeout"}
            await asyncio.sleep(2 ** attempt)
        except Exception as e:
            if attempt == 2:
                return {"_error": str(e)[:200]}
            await asyncio.sleep(2 ** attempt)
    return {}


async def main_async(args):
    r1 = json.loads(Path(args.input).read_text())
    parents = r1.get("parented_trees", [])
    print(f"loaded R1 with {len(parents)} parents")

    api_key = KEY_PATH.read_text().strip() if KEY_PATH.exists() else os.environ.get("OPENAI_API_KEY", "")
    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(args.concurrency)

    desc = TASK_DESCRIPTIONS.get(args.task, "evaluating work in this task domain")
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(task_description=desc)

    async def refine_one(idx: int, p: dict) -> dict:
        children = p.get("children", [])
        if len(children) < 2:
            return {"parent_idx": idx, "result": {"misfits": [], "duplicate_groups": []}, "skipped": True}
        async with sem:
            user_msg = build_user_msg(p.get("parent_name", ""), p.get("parent_description", ""), children)
            t0 = time.perf_counter()
            res = await call_llm(client, args.model, system_prompt, user_msg, service_tier=args.service_tier)
            return {"parent_idx": idx, "result": res, "elapsed_sec": time.perf_counter() - t0}

    print(f"\nlaunching {len(parents)} refine calls (concurrency={args.concurrency})...")
    t_start = time.perf_counter()
    coros = [refine_one(i, p) for i, p in enumerate(parents)]
    results = await asyncio.gather(*coros)
    print(f"done in {time.perf_counter()-t_start:.1f}s")

    # Apply decisions
    refined_parents = []
    stranded_children: list[dict] = []  # misfits dropped from any parent
    n_total_misfits = 0
    n_total_merges = 0
    n_children_in_merges = 0

    for r in results:
        idx = r["parent_idx"]
        p = parents[idx]
        children = p.get("children", [])
        res = r.get("result", {})

        misfit_ids = {m["child_id"] for m in res.get("misfits", []) if isinstance(m.get("child_id"), int)}
        merge_groups = res.get("duplicate_groups", [])

        # Process misfits → stranded
        for mid in misfit_ids:
            if 1 <= mid <= len(children):
                child = children[mid - 1]
                stranded_children.append({
                    "from_parent_name": p.get("parent_name", ""),
                    "child": child,
                    "reason": next((m["reason"] for m in res.get("misfits", []) if m.get("child_id") == mid), ""),
                })
        n_total_misfits += len(misfit_ids)

        # Apply duplicate-group merges
        in_merge = set()
        merged_children_to_add: list[dict] = []
        for g in merge_groups:
            cids = [c for c in g.get("child_ids", []) if isinstance(c, int) and 1 <= c <= len(children)]
            if len(cids) < 2: continue
            n_total_merges += 1
            n_children_in_merges += len(cids)
            # Combine member rubrics from all merged children
            all_rubrics = []
            for cid in cids:
                child = children[cid - 1]
                all_rubrics.extend(child.get("rubrics", []))
                in_merge.add(cid)
            merged_children_to_add.append({
                "cluster_id": -1,
                "medoid_name": g.get("merged_name", ""),
                "medoid_description": g.get("merged_description", ""),
                "size": len(all_rubrics),
                "rubrics": all_rubrics,
            })

        # New children list: keep non-misfit, non-merged children + merged children
        new_children = []
        for ci, c in enumerate(children, 1):
            if ci in misfit_ids: continue
            if ci in in_merge: continue
            new_children.append(c)
        new_children.extend(merged_children_to_add)

        refined_parents.append({
            "parent_name": p.get("parent_name", ""),
            "parent_description": p.get("parent_description", ""),
            "n_children": len(new_children),
            "total_leaf_rubrics": sum(c.get("size", len(c.get("rubrics", []))) for c in new_children),
            "children": new_children,
            "_refinement_log": {
                "n_misfits_dropped": len(misfit_ids),
                "n_duplicate_groups": len(merge_groups),
            },
        })

    print(f"\nREFINEMENT SUMMARY")
    print(f"  parents refined: {len(refined_parents)}")
    print(f"  misfits dropped → stranded: {n_total_misfits}")
    print(f"  duplicate-groups within parents: {n_total_merges} (collapsing {n_children_in_merges} children)")

    out_main = Path(args.output)
    out_main.parent.mkdir(parents=True, exist_ok=True)
    out_main.write_text(json.dumps({
        "task": args.task, "bucket": args.bucket, "model": args.model,
        "round": "refined", "source_input": str(args.input),
        "n_clusters_in": r1.get("n_clusters_in", 0),
        "n_parent_trees": len(refined_parents),
        "n_merge_trees": r1.get("n_merge_trees", 0),
        "merged_trees": r1.get("merged_trees", []),
        "parented_trees": refined_parents,
    }, indent=2, ensure_ascii=False))
    print(f"  saved refined hierarchy → {out_main}")

    out_str = out_main.with_name(out_main.stem + "_stranded.json")
    out_str.write_text(json.dumps({
        "task": args.task, "bucket": args.bucket,
        "n_stranded": len(stranded_children),
        "stranded_children": stranded_children,
    }, indent=2, ensure_ascii=False))
    print(f"  saved stranded children → {out_str}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="creative-writing")
    ap.add_argument("--bucket", default="general")
    ap.add_argument("--input", required=True, help="R1 expanded.json")
    ap.add_argument("--output", default=None)
    ap.add_argument("--model", default="gpt-5")
    ap.add_argument("--service-tier", default=None, choices=[None, "auto", "default", "flex"])
    ap.add_argument("--concurrency", type=int, default=20)
    args = ap.parse_args()
    if args.output is None:
        args.output = str(ROOT / f"outputs/hierarchy/{args.task}_{args.bucket}_r1_refined.json")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
