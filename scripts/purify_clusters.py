"""
Post-hoc purification of complete-linkage clusters.

Two-pass design:
  Pass 1 (FREE): use existing LLM-labeled pairs to resolve cluster-pair
  decisions where both medoids happen to have a labeled pair. Merge any
  cluster pair where the labeled medoid-pair is dup/paraphrase.

  Pass 2 (PAID): for the remaining candidate cluster-pairs (centroid cos
  in [ambig_low, merge_threshold)), call gpt-5-mini on the medoid pair.
  Merge if dup/paraphrase.

Output:
  - outputs/clusters/{task}_purified.json — same structure as input but
    with merged clusters.
  - outputs/clusters/{task}_purify_log.jsonl — per-decision log.

Cost model: gpt-5-mini at ~$0.001 per pairwise call.
"""
from __future__ import annotations
import argparse, asyncio, json, os, sys, time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from openai import AsyncOpenAI

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")
CHUNKS = ROOT / "outputs/classifier_chunks_FULL"
KEY_PATH = Path("/lfs/skampere3/0/alexspan/.openai-salt-lab-key.txt")
FEW_SHOTS_DIR = ROOT / "outputs/few_shots_auto"
MODEL = "gpt-5-mini"

MERGE_VERDICTS = {"duplicate", "paraphrase"}

JSON_SCHEMA = {
    "name": "rubric_relationship",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning": {"type": "string"},
            "verdict":   {"type": "string", "enum": ["duplicate", "paraphrase", "related", "different"]},
        },
        "required": ["reasoning", "verdict"],
    },
}

SYSTEM_BASE = """You are evaluating two rubric items from an evaluation system to determine their relationship for the purpose of de-duplicating the rubric set.

The dedup target is CONSTRUCT IDENTITY — whether the two rubrics are attempting to measure the same underlying property of the work. This is distinct from SCORE AGREEMENT.

Verdicts:
- "duplicate": Same construct with substantively the same operational definition.
- "paraphrase": Same underlying construct with meaningfully different wording.
- "related": Share a topic/dimension but measure different constructs.
- "different": Unrelated constructs.

Reason before deciding. Walk through: (1) name the construct of each rubric, (2) is there a single coherent description that captures both, (3) what distinct information each provides if any, (4) is correlation merely empirical or constitutive.
"""


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


def load_existing_labels(jsonl_path: Path) -> dict[tuple[str, str], str]:
    """Returns {(a_key, b_key)_sorted: verdict} from existing eval."""
    out = {}
    if not jsonl_path.exists():
        return out
    with jsonl_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                v = r.get("verdict")
                if v not in ("duplicate", "paraphrase", "related", "different"):
                    continue
                k = tuple(sorted([r["a_key"], r["b_key"]]))
                out[k] = v
            except Exception:
                pass
    return out


def load_few_shots(task: str, specificity: str) -> list[dict]:
    paths = [
        FEW_SHOTS_DIR / task / f"{specificity}.json",
        FEW_SHOTS_DIR / task / "general.json",
    ]
    for p in paths:
        if p.exists():
            return json.loads(p.read_text())
    return []


def format_few_shots(few_shots: list[dict]) -> str:
    if not few_shots: return ""
    chunks = ["\n"]
    for i, ex in enumerate(few_shots, 1):
        chunks.append(f"--- EXAMPLE {i} ({ex['verdict']}) ---\n")
        chunks.append("Rubric A:\n")
        chunks.append(f"  name: {ex['a']['name']}\n")
        chunks.append(f"  description: {ex['a']['description']}\n")
        chunks.append("Rubric B:\n")
        chunks.append(f"  name: {ex['b']['name']}\n")
        chunks.append(f"  description: {ex['b']['description']}\n")
        chunks.append("Output:\n")
        chunks.append("{\n")
        chunks.append(f"  \"reasoning\": {json.dumps(ex['reasoning'])},\n")
        chunks.append(f"  \"verdict\": \"{ex['verdict']}\"\n")
        chunks.append("}\n\n")
    return "".join(chunks)


def build_user_msg(a: dict, b: dict) -> str:
    def fmt(r):
        return (
            f"  name: {r.get('rubric_name','')}\n"
            f"  description: {r.get('rubric_description','') or ''}\n"
            f"  source page: {r['page_id']}"
        )
    return f"Rubric A:\n{fmt(a)}\n\nRubric B:\n{fmt(b)}"


async def call_model(client: AsyncOpenAI, sem: asyncio.Semaphore,
                      system: str, user: str) -> dict:
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
                )
                raw = resp.choices[0].message.content or ""
                return json.loads(raw)
            except Exception as e:
                if attempt == 2:
                    return {"_error": str(e)[:200]}
                await asyncio.sleep(2 ** attempt)
        return {"_error": "unknown"}


def find_candidate_cluster_pairs(clusters: list[dict], embs: np.ndarray,
                                  medoid_idx_list: list[int],
                                  ambig_low: float, merge_threshold: float,
                                  multi_only: bool = False,
                                  multi_x_multi_only: bool = False) -> list[tuple[int, int, float]]:
    """Return candidate cluster index pairs whose medoid cosines are in
    [ambig_low, merge_threshold). Optionally filter to multi-member."""
    medoid_embs = embs[np.array(medoid_idx_list)]
    sims = medoid_embs @ medoid_embs.T
    np.fill_diagonal(sims, 0.0)
    n = len(clusters)
    pairs = []
    is_multi = [len(c["members"]) > 1 for c in clusters]
    for i in range(n):
        for j in range(i+1, n):
            c = float(sims[i, j])
            if c < ambig_low or c >= merge_threshold: continue
            if multi_x_multi_only and (not is_multi[i] or not is_multi[j]): continue
            if multi_only and not (is_multi[i] or is_multi[j]): continue
            pairs.append((i, j, c))
    pairs.sort(key=lambda t: -t[2])
    return pairs


def union_clusters_by_label(clusters: list[dict], merges: list[tuple[int, int]]) -> list[dict]:
    """Apply union-find on the merges, then rebuild clusters list."""
    parent = list(range(len(clusters)))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in merges:
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    grouped: dict[int, list[int]] = defaultdict(list)
    for i in range(len(clusters)):
        grouped[find(i)].append(i)
    new_clusters = []
    for root, idxs in grouped.items():
        # Pick the largest existing cluster's medoid as new medoid (heuristic)
        idxs.sort(key=lambda i: -len(clusters[i]["members"]))
        new_medoid = clusters[idxs[0]]["medoid_idx"]
        new_members = []
        for i in idxs:
            new_members.extend(clusters[i]["members"])
        new_clusters.append({"medoid_idx": new_medoid, "members": new_members})
    return new_clusters


async def purify_bucket(task: str, spec: str, clusters: list[dict],
                         rubrics: list[dict], embs: np.ndarray,
                         existing_labels: dict, args, client, sem) -> tuple[list[dict], dict]:
    """Run two-pass purification for one bucket. Returns (new_clusters, log)."""
    log = {"spec": spec, "n_clusters_pre": len(clusters)}
    medoid_idx_list = [c["medoid_idx"] for c in clusters]

    candidate_pairs = find_candidate_cluster_pairs(
        clusters, embs, medoid_idx_list,
        args.ambig_low, args.merge_threshold,
        multi_only=args.multi_only,
        multi_x_multi_only=args.multi_x_multi_only,
    )
    log["n_candidate_pairs"] = len(candidate_pairs)
    print(f"  [{spec}] candidate cluster-pairs: {len(candidate_pairs):,}")

    # Pass 1: use existing labels
    merges: list[tuple[int, int]] = []
    resolved_existing = 0
    keep_separate_existing = 0
    paid_pairs = []
    for ci, cj, cos in candidate_pairs:
        a_idx = clusters[ci]["medoid_idx"]
        b_idx = clusters[cj]["medoid_idx"]
        a_key = f"{rubrics[a_idx]['page_id']}::{rubrics[a_idx]['rubric_idx']}"
        b_key = f"{rubrics[b_idx]['page_id']}::{rubrics[b_idx]['rubric_idx']}"
        key = tuple(sorted([a_key, b_key]))
        if key in existing_labels:
            v = existing_labels[key]
            if v in MERGE_VERDICTS:
                merges.append((ci, cj))
            resolved_existing += 1
            if v not in MERGE_VERDICTS:
                keep_separate_existing += 1
        else:
            paid_pairs.append((ci, cj, cos))
    log["n_resolved_by_existing"] = resolved_existing
    log["n_existing_merges"] = sum(1 for c1, c2 in merges)
    log["n_paid_pending"] = len(paid_pairs)
    print(f"    pass1 (existing labels): resolved {resolved_existing:,}; "
          f"{len(merges):,} merges from existing labels; "
          f"{len(paid_pairs):,} pairs still pending LLM judgement")

    # Pass 2: LLM-judge remaining pairs (subject to --max-paid-pairs)
    paid_to_run = paid_pairs[: args.max_paid_pairs]
    log["n_paid_run"] = len(paid_to_run)
    if paid_to_run:
        print(f"    pass2 (LLM): judging {len(paid_to_run):,} cluster-pair centroids...")
        system_prompt = SYSTEM_BASE + format_few_shots(load_few_shots(task, spec))
        coros = []
        for ci, cj, _ in paid_to_run:
            a = rubrics[clusters[ci]["medoid_idx"]]
            b = rubrics[clusters[cj]["medoid_idx"]]
            coros.append(call_model(client, sem, system_prompt, build_user_msg(a, b)))
        results = await asyncio.gather(*coros)
        n_merge_pass2 = 0
        for (ci, cj, _), res in zip(paid_to_run, results):
            v = res.get("verdict")
            if v in MERGE_VERDICTS:
                merges.append((ci, cj))
                n_merge_pass2 += 1
        log["n_pass2_merges"] = n_merge_pass2

    # Apply merges
    new_clusters = union_clusters_by_label(clusters, merges)
    log["n_clusters_post"] = len(new_clusters)
    log["n_merges_applied"] = log["n_clusters_pre"] - log["n_clusters_post"]
    print(f"    AFTER PURIFICATION: {log['n_clusters_pre']:,} → {log['n_clusters_post']:,} clusters "
          f"({log['n_merges_applied']:,} merges applied)")
    return new_clusters, log


async def main_async(args):
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

    clusters_in = json.loads(Path(args.input).read_text())
    existing_labels = load_existing_labels(Path(args.existing_labels))
    print(f"loaded {len(existing_labels):,} existing labeled pairs")

    api_key = KEY_PATH.read_text().strip() if KEY_PATH.exists() else os.environ.get("OPENAI_API_KEY", "")
    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(args.concurrency)

    SPECS = ["vague", "general", "specific", "hyper_specific"]
    final = {}
    logs = []
    for spec in SPECS:
        if spec not in clusters_in: continue
        bucket = clusters_in[spec]
        # Build internal cluster repr keyed by medoid INDEX in embs_all
        # We need to remap medoid_key -> rubrics_all index
        key_to_idx = {f"{r['page_id']}::{r['rubric_idx']}": i for i, r in enumerate(rubrics_all)}
        clusters = []
        for c in bucket["clusters"]:
            med_key = c["medoid_key"]
            if med_key not in key_to_idx:
                continue
            members_idx = []
            for m in c["members"]:
                mk = m["key"]
                if mk in key_to_idx:
                    members_idx.append(key_to_idx[mk])
            if not members_idx:
                continue
            clusters.append({"medoid_idx": key_to_idx[med_key], "members": members_idx})

        print(f"\n=== {spec}  {len(clusters):,} clusters ===")
        new_clusters, log = await purify_bucket(
            args.task, spec, clusters, rubrics_all, embs_all,
            existing_labels, args, client, sem,
        )

        # Recompute medoids after merges
        for c in new_clusters:
            if len(c["members"]) > 1:
                from cluster_complete_linkage import compute_medoid
                c["medoid_idx"] = compute_medoid(c["members"], embs_all)

        # Rebuild output structure
        out_clusters = []
        for c in new_clusters:
            med = rubrics_all[c["medoid_idx"]]
            out_clusters.append({
                "medoid_key": f"{med['page_id']}::{med['rubric_idx']}",
                "medoid_name": med['rubric_name'],
                "medoid_description": med.get('rubric_description', ''),
                "size": len(c["members"]),
                "members": [{
                    "key": f"{rubrics_all[m]['page_id']}::{rubrics_all[m]['rubric_idx']}",
                    "name": rubrics_all[m]['rubric_name'],
                } for m in c["members"]],
            })
        final[spec] = {
            "n_rubrics": bucket["n_rubrics"],
            "n_clusters": len(out_clusters),
            "n_singletons": sum(1 for c in out_clusters if c["size"] == 1),
            "n_multimember": sum(1 for c in out_clusters if c["size"] > 1),
            "max_cluster_size": max((c["size"] for c in out_clusters), default=0),
            "clusters": out_clusters,
        }
        logs.append(log)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final, indent=2, ensure_ascii=False))
    log_path = out_path.with_suffix(".log.json")
    log_path.write_text(json.dumps(logs, indent=2))

    print("\n=== FINAL SUMMARY ===")
    total_pre = sum(l["n_clusters_pre"] for l in logs)
    total_post = sum(l["n_clusters_post"] for l in logs)
    print(f"total clusters: {total_pre:,} → {total_post:,}  ({total_pre - total_post:,} merges)")
    print(f"saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="creative-writing")
    ap.add_argument("--input", default=str(ROOT / "outputs/clusters/creative-writing_complete_linkage.json"))
    ap.add_argument("--existing-labels", default=str(ROOT / "outputs/dedup_eval/creative_writing_2k.jsonl"))
    ap.add_argument("--output", default=str(ROOT / "outputs/clusters/creative-writing_purified.json"))
    ap.add_argument("--ambig-low", type=float, default=0.65)
    ap.add_argument("--merge-threshold", type=float, default=0.85)
    ap.add_argument("--multi-only", action="store_true",
                    help="Only consider pairs where at least one cluster is multi-member.")
    ap.add_argument("--multi-x-multi-only", action="store_true",
                    help="Only consider pairs where BOTH clusters are multi-member.")
    ap.add_argument("--max-paid-pairs", type=int, default=2000,
                    help="Cap on LLM calls; useful to bound cost.")
    ap.add_argument("--concurrency", type=int, default=50)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
