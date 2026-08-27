"""
Run the dedup classifier on a stratified 2,000-pair sample from
creative-writing keep+work rubrics. Save full per-pair output to JSONL.

Sampling: 4 specificity buckets × 5 cosine zones × ~100 pairs each
(adapted when a bucket has fewer candidates).

Output schema (one JSONL row per pair):
  {
    "task", "specificity", "cosine_zone", "cosine",
    "a_key", "a_name", "a_description", "a_rubric_idx",
    "b_key", "b_name", "b_description", "b_rubric_idx",
    "verdict", "reasoning", "elapsed_sec",
  }
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import asyncio
import numpy as np
from openai import AsyncOpenAI

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")
CHUNKS = ROOT / "outputs/classifier_chunks_FULL"
KEY_PATH = Path("/lfs/skampere3/0/alexspan/.openai-salt-lab-key.txt")
FEW_SHOTS_DIR = ROOT / "outputs/few_shots_auto"

MODEL = "gpt-5-mini"

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

SYSTEM_BASE = """You are evaluating two rubric items from a creative-writing evaluation system to determine their relationship for the purpose of de-duplicating the rubric set.

The dedup target is CONSTRUCT IDENTITY — whether the two rubrics are attempting to measure the same underlying property of the work. This is distinct from SCORE AGREEMENT (whether they'd produce similar scores on the same texts). Two rubrics can correlate strongly on real data and still measure DIFFERENT constructs — collapsing them on correlation alone destroys information.

Verdicts:
- "duplicate": The two rubrics measure the IDENTICAL construct with substantively the same operational definition. One could be removed and the other kept with no loss of evaluative information.
- "paraphrase": The two rubrics measure the SAME underlying construct but with meaningfully different wording or framing. They could safely be merged.
- "related": The two rubrics share a topic/dimension and may correlate empirically, but they measure DIFFERENT underlying constructs. Each captures information the other does not.
- "different": Unrelated constructs.

Reason before deciding. Walk through: (1) name the construct of each rubric, (2) is there a single coherent description that captures both, (3) what distinct information each provides if any, (4) is correlation merely empirical or constitutive.

Each example below is drawn from the actual rubric corpus at the same specificity level as the pair you are evaluating.
"""


def load_keep_work_rubrics(task: str) -> list[dict]:
    rows = []
    for cf in sorted(CHUNKS.glob("chunk_*.jsonl")):
        with cf.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if (r.get("task") == task
                        and r.get("cls_ok")
                        and r.get("cls_keep") == "keep"
                        and r.get("cls_target") == "work"):
                        rows.append(r)
                except Exception:
                    pass
    return rows


def load_few_shots(task: str, specificity: str) -> list[dict]:
    """Load auto-generated few-shots for a (task, specificity) bucket.
    Falls back to general bucket if the requested one has none."""
    paths = [
        FEW_SHOTS_DIR / task / f"{specificity}.json",
        FEW_SHOTS_DIR / task / "general.json",
    ]
    for p in paths:
        if p.exists():
            return json.loads(p.read_text())
    return []


def format_few_shots(few_shots: list[dict]) -> str:
    if not few_shots:
        return "\n(No bucket-specific examples available.)\n"
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


def build_system_prompt(task: str, specificity: str) -> str:
    return SYSTEM_BASE + format_few_shots(load_few_shots(task, specificity))


def build_user_msg(a: dict, b: dict) -> str:
    def fmt(r):
        return (
            f"  name: {r['rubric_name']}\n"
            f"  description: {r.get('rubric_description','') or ''}\n"
            f"  guidance: {(r.get('rubric_guidance','') or '')[:300]}\n"
            f"  source page: {r['page_id']}"
        )
    return f"Rubric A:\n{fmt(a)}\n\nRubric B:\n{fmt(b)}"


# ---- Sampling ----

def find_candidate_pairs(rubrics, embs, top_k=10, n_queries=3000):
    name_lc = [str(r.get('rubric_name') or '').lower().strip() for r in rubrics]
    pid = [r['page_id'] for r in rubrics]
    rng = np.random.RandomState(42)
    q_idx = rng.choice(len(rubrics), size=min(n_queries, len(rubrics)), replace=False)
    seen = set()
    zones = {
        'duplicate':  [],  # cos >= 0.92
        'paraphrase': [],  # 0.80 - 0.92
        'related':    [],  # 0.65 - 0.80
        'mid':        [],  # 0.45 - 0.65
        'different':  [],  # < 0.30 (random sampling)
    }
    for qi in q_idx:
        sims = embs @ embs[qi]
        sims[qi] = -1
        top = np.argpartition(-sims, min(top_k, len(sims)-1))[:top_k]
        top = top[np.argsort(-sims[top])]
        for ti in top:
            if ti == qi: continue
            if pid[qi] == pid[ti]: continue
            if name_lc[qi] == name_lc[ti]: continue
            k = (min(int(qi), int(ti)), max(int(qi), int(ti)))
            if k in seen: continue
            seen.add(k)
            c = float(sims[ti])
            if c >= 0.92:        zones['duplicate'].append((int(qi), int(ti), c))
            elif c >= 0.80:      zones['paraphrase'].append((int(qi), int(ti), c))
            elif c >= 0.65:      zones['related'].append((int(qi), int(ti), c))
            elif c >= 0.45:      zones['mid'].append((int(qi), int(ti), c))

    # Random low-cos sampling for "different" zone
    rng2 = np.random.RandomState(99)
    for _ in range(5000):
        if len(zones['different']) >= 500: break
        i, j = rng2.choice(len(rubrics), size=2, replace=False)
        if pid[i] == pid[j]: continue
        if name_lc[i] == name_lc[j]: continue
        k = (min(int(i), int(j)), max(int(i), int(j)))
        if k in seen: continue
        c = float(embs[i] @ embs[j])
        if c < 0.30:
            seen.add(k)
            zones['different'].append((int(i), int(j), c))

    for z in zones:
        zones[z].sort(key=lambda t: -t[2])
    return zones


def stratified_sample(zones: dict, target_per_zone: int, seed: int = 7) -> dict:
    """Take up to target_per_zone pairs per zone (random within zone)."""
    rng = np.random.RandomState(seed)
    out = {}
    for z, items in zones.items():
        if len(items) <= target_per_zone:
            out[z] = items
        else:
            idx = rng.choice(len(items), size=target_per_zone, replace=False)
            out[z] = [items[i] for i in idx]
    return out


# ---- Eval ----

async def call_model(client: AsyncOpenAI, system: str, user: str) -> dict:
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
        return {"_error": str(e)[:300]}


async def eval_one(client, sem, idx, total, pair_meta, a, b, system_prompt, out_handle):
    async with sem:
        t0 = time.perf_counter()
        user_msg = build_user_msg(a, b)
        result = await call_model(client, system_prompt, user_msg)
        elapsed = time.perf_counter() - t0
        rec = {
            **pair_meta,
            "verdict": result.get("verdict"),
            "reasoning": result.get("reasoning"),
            "elapsed_sec": elapsed,
            "_error": result.get("_error"),
        }
        out_handle.write(json.dumps(rec) + "\n")
        out_handle.flush()
        if idx % 50 == 0 or idx == total - 1:
            print(f"  [{idx+1}/{total}] verdict={rec['verdict']} ({elapsed:.1f}s)")


async def main_async(args):
    rubrics = load_keep_work_rubrics(args.task)
    print(f"loaded {len(rubrics):,} {args.task} keep+work rubrics")

    # Load task-level embeddings (work-only cache)
    cache = ROOT / f"outputs/embeddings/{args.task}_work_embeddings.npz"
    if not cache.exists():
        sys.exit(f"no embedding cache at {cache} — run generate_few_shots.py first")
    d = np.load(cache, allow_pickle=True)
    expected_keys = [f"{r['page_id']}::{r['rubric_idx']}" for r in rubrics]
    if list(d['keys']) != expected_keys:
        sys.exit("embedding cache key mismatch")
    embs_all = d['embs'].astype(np.float32)
    print(f"embeddings: shape={embs_all.shape}")

    SPECS = ['vague', 'general', 'specific', 'hyper_specific']
    target_per_zone = args.n_pairs // (len(SPECS) * 5)  # ~100 per zone
    print(f"target pairs per (spec, zone): {target_per_zone}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = KEY_PATH.read_text().strip() if KEY_PATH.exists() else os.environ.get("OPENAI_API_KEY", "")
    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(args.concurrency)

    all_tasks = []
    pair_idx = 0

    with out_path.open("w") as out:
        for spec in SPECS:
            spec_idx = [i for i, r in enumerate(rubrics) if r.get('cls_specificity') == spec]
            if len(spec_idx) < 20:
                print(f"  {spec}: too small ({len(spec_idx)}), skip")
                continue
            rubrics_b = [rubrics[i] for i in spec_idx]
            embs_b    = embs_all[spec_idx]

            zones = find_candidate_pairs(rubrics_b, embs_b)
            sampled = stratified_sample(zones, target_per_zone)
            total_for_spec = sum(len(v) for v in sampled.values())
            print(f"\nspec={spec}  total sampled: {total_for_spec}")
            for z, items in sampled.items():
                print(f"  {z}: {len(items)}")

            system_prompt = build_system_prompt(args.task, spec)

            for zone, items in sampled.items():
                for ai, bi, cos in items:
                    a = rubrics_b[ai]; b = rubrics_b[bi]
                    pair_meta = {
                        "task": args.task,
                        "specificity": spec,
                        "cosine_zone": zone,
                        "cosine": cos,
                        "a_key": f"{a['page_id']}::{a['rubric_idx']}",
                        "a_name": a['rubric_name'],
                        "a_description": a.get('rubric_description', ''),
                        "a_rubric_idx": int(a['rubric_idx']),
                        "b_key": f"{b['page_id']}::{b['rubric_idx']}",
                        "b_name": b['rubric_name'],
                        "b_description": b.get('rubric_description', ''),
                        "b_rubric_idx": int(b['rubric_idx']),
                    }
                    all_tasks.append(eval_one(client, sem, pair_idx, args.n_pairs, pair_meta, a, b, system_prompt, out))
                    pair_idx += 1

        print(f"\nlaunching {len(all_tasks)} async eval tasks (concurrency={args.concurrency})...")
        t_start = time.perf_counter()
        await asyncio.gather(*all_tasks)
        print(f"\nDONE: {len(all_tasks)} pairs in {time.perf_counter()-t_start:.1f}s -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="creative-writing")
    ap.add_argument("--n-pairs", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--output", default=str(ROOT / "outputs/dedup_eval/creative_writing_2k.jsonl"))
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
