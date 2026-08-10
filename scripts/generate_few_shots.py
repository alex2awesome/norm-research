"""
Auto-generate per-(task, specificity) few-shot examples for the dedup
classifier.

For each (task, specificity) bucket with enough rubrics:
  1. Embed all keep+work rubrics for the task (cached after first run)
  2. Find candidate pairs per cosine zone within the bucket
  3. Pick: top duplicate-zone, top paraphrase-zone, top related-zone, random low-cos different
  4. For each pick, generate construct-identity reasoning via gpt-5-mini
  5. Save to outputs/few_shots_auto/<task>/<specificity>.json

Skips buckets with <50 rubrics.
"""
from __future__ import annotations
import argparse, asyncio, json, os, sys, random
from pathlib import Path
import numpy as np
from openai import OpenAI, AsyncOpenAI

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")
CHUNKS = ROOT / "outputs/classifier_chunks_FULL"
KEY_PATH = Path("/lfs/skampere3/0/alexspan/.openai-salt-lab-key.txt")
EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL = "gpt-5-mini"

TASKS = ['code-review', 'creative-writing', 'grant-funding', 'humor',
         'legal-outcome-prediction', 'math-stackexchange', 'news-homepages',
         'notice-and-comment', 'patents', 'peer-review', 'press-releases']
SPECS = ['vague', 'general', 'specific', 'hyper_specific']

MIN_BUCKET_SIZE = 50  # skip buckets smaller than this


# ---- Load + embed ----

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


def embed_text(r: dict) -> str:
    name = r.get("rubric_name") or ""
    desc = r.get("rubric_description") or ""
    guid = r.get("rubric_guidance") or ""
    text = f"{name}\n{desc}"
    if guid:
        text += f"\n{guid}"
    return text[:2000]


def get_or_compute_embeddings(rubrics: list[dict], client: OpenAI, cache: Path,
                              batch_size: int = 512) -> np.ndarray:
    keys = [f"{r['page_id']}::{r['rubric_idx']}" for r in rubrics]
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        if list(d['keys']) == keys:
            return d['embs'].astype(np.float32)
    texts = [embed_text(r) for r in rubrics]
    embs_list = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        embs_list.extend([d.embedding for d in resp.data])
    embs = np.array(embs_list, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.where(norms == 0, 1.0, norms)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, embs=embs, keys=np.array(keys))
    return embs


# ---- Candidate pair sampling ----

def find_candidate_pairs(rubrics: list[dict], embs: np.ndarray,
                          top_k: int = 8, n_queries: int = 2000) -> dict:
    """Return {zone_name: [(a_idx, b_idx, cos), ...]} cross-doc, no
    identical-names."""
    name_lc = [str(r.get('rubric_name') or '').lower().strip() for r in rubrics]
    pid = [r['page_id'] for r in rubrics]
    rng = np.random.RandomState(2026)
    q_idx = rng.choice(len(rubrics), size=min(n_queries, len(rubrics)), replace=False)

    seen = set()
    zones = {
        'duplicate':  [],
        'paraphrase': [],
        'related':    [],
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
            if c >= 0.92:
                zones['duplicate'].append((int(qi), int(ti), c))
            elif c >= 0.80:
                zones['paraphrase'].append((int(qi), int(ti), c))
            elif c >= 0.65:
                zones['related'].append((int(qi), int(ti), c))

    for z in zones:
        zones[z].sort(key=lambda t: -t[2])
    return zones


def sample_low_cos_pair(rubrics, embs, max_cos: float = 0.35, seed: int = 7) -> tuple | None:
    rng = random.Random(seed)
    n = len(rubrics)
    for _ in range(500):
        i, j = rng.sample(range(n), 2)
        if rubrics[i]['page_id'] == rubrics[j]['page_id']: continue
        if (rubrics[i].get('rubric_name') or '').lower().strip() == (rubrics[j].get('rubric_name') or '').lower().strip():
            continue
        c = float(embs[i] @ embs[j])
        if c < max_cos:
            return (i, j, c)
    return None


# ---- Reasoning generation ----

REASONING_SYSTEM = """You are writing pedagogical few-shot examples for a rubric de-duplication classifier. Given two rubrics and a verdict label (duplicate / paraphrase / related / different), write 3-5 sentences of construct-identity reasoning that justifies the verdict.

The framework: dedup decisions are about CONSTRUCT IDENTITY — whether two rubrics measure the same underlying property of the work. This is distinct from score-agreement. Two rubrics can correlate strongly on real data and still measure different constructs.

For each verdict:
- duplicate / paraphrase: a single coherent description captures both rubrics; neither carries evaluative information the other lacks
- related: the rubrics share a topic/dimension and may correlate empirically, but they measure different constructs; each captures distinct information
- different: no shared evaluative dimension

Walk through: (1) name the construct each rubric purports to measure, (2) whether a single description captures both or they need separate descriptions, (3) what distinct information each provides (if any), (4) for 'related', explicitly note that any score-agreement would be empirical/correlational rather than constitutive.

Output ONLY the reasoning string (no JSON wrapper, no preamble)."""


async def generate_reasoning_async(client: AsyncOpenAI, sem: asyncio.Semaphore,
                                    a: dict, b: dict, verdict: str) -> str:
    user_msg = (
        f"Rubric A:\n"
        f"  name: {a.get('rubric_name','')}\n"
        f"  description: {a.get('rubric_description','')}\n"
        f"  guidance: {(a.get('rubric_guidance','') or '')[:300]}\n\n"
        f"Rubric B:\n"
        f"  name: {b.get('rubric_name','')}\n"
        f"  description: {b.get('rubric_description','')}\n"
        f"  guidance: {(b.get('rubric_guidance','') or '')[:300]}\n\n"
        f"VERDICT: {verdict}\n\n"
        f"Write 3-5 sentences of construct-identity reasoning justifying this verdict."
    )
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=GEN_MODEL,
                    messages=[
                        {"role": "system", "content": REASONING_SYSTEM},
                        {"role": "user",   "content": user_msg},
                    ],
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                if attempt == 2:
                    return f"[reasoning_generation_failed: {str(e)[:200]}]"
                await asyncio.sleep(2 ** attempt)
        return ""


# ---- Driver ----

async def main_async(args):
    out_dir = Path(args.out_dir)
    api_key = KEY_PATH.read_text().strip() if KEY_PATH.exists() else os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        sys.exit("no API key")
    sync_client = OpenAI(api_key=api_key)
    async_client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(args.concurrency)

    # Phase 1: embed + pick candidates per (task, spec) — sequential because
    # embedding is cheap and disk IO bound.
    plan: list[dict] = []  # list of {task, spec, picks, out_path}
    for task in args.tasks:
        print(f"\n=== TASK: {task} ===")
        rubrics_all = load_keep_work_rubrics(task)
        print(f"  loaded {len(rubrics_all):,} keep+work rubrics")
        if len(rubrics_all) < 20:
            print(f"  too small overall, skip")
            continue

        work_cache = ROOT / f"outputs/embeddings/{task}_work_embeddings.npz"
        embs = get_or_compute_embeddings(rubrics_all, sync_client, work_cache)
        print(f"  embeddings: {embs.shape}")

        for spec in SPECS:
            spec_idx = [i for i, r in enumerate(rubrics_all) if r.get('cls_specificity') == spec]
            if len(spec_idx) < MIN_BUCKET_SIZE:
                print(f"  {spec:<18s}  N={len(spec_idx):<6d}  skip (< {MIN_BUCKET_SIZE})")
                continue
            rubrics_b = [rubrics_all[i] for i in spec_idx]
            embs_b    = embs[spec_idx]

            out_path = out_dir / task / f"{spec}.json"
            if out_path.exists() and not args.overwrite:
                print(f"  {spec:<18s}  N={len(rubrics_b):<6d}  exists, skip")
                continue

            zones = find_candidate_pairs(rubrics_b, embs_b,
                                          top_k=8,
                                          n_queries=min(2000, len(rubrics_b)))
            picks: dict[str, tuple | None] = {}
            picks['duplicate']  = zones['duplicate'][0]  if zones['duplicate']  else None
            picks['paraphrase'] = zones['paraphrase'][0] if zones['paraphrase'] else None
            picks['related']    = zones['related'][0]    if zones['related']    else None
            picks['different']  = sample_low_cos_pair(rubrics_b, embs_b)

            plan.append({
                'task': task, 'spec': spec, 'rubrics': rubrics_b,
                'picks': picks, 'out_path': out_path,
            })
            print(f"  {spec:<18s}  N={len(rubrics_b):<6d}  queued (picks: "
                  f"{', '.join(v for v, p in picks.items() if p)})")

    # Phase 2: launch all reasoning calls in parallel with semaphore
    print(f"\n=== launching reasoning generation (concurrency={args.concurrency}) ===")
    total_calls = sum(sum(1 for p in item['picks'].values() if p) for item in plan)
    print(f"total reasoning calls: {total_calls}")

    async def gen_one(item: dict, verdict: str, pick: tuple) -> tuple:
        ai, bi, cos = pick
        a = item['rubrics'][ai]; b = item['rubrics'][bi]
        reasoning = await generate_reasoning_async(async_client, sem, a, b, verdict)
        return (item, verdict, {
            "verdict": verdict,
            "a": {"name": a.get("rubric_name") or "", "description": a.get("rubric_description") or ""},
            "b": {"name": b.get("rubric_name") or "", "description": b.get("rubric_description") or ""},
            "cos": cos,
            "reasoning": reasoning,
        })

    coros = []
    for item in plan:
        for verdict, pick in item['picks'].items():
            if pick is None: continue
            coros.append(gen_one(item, verdict, pick))

    import time as _time
    t0 = _time.perf_counter()
    results = await asyncio.gather(*coros)
    print(f"  generated {len(results)} reasonings in {_time.perf_counter()-t0:.1f}s")

    # Phase 3: group results by out_path + write
    by_path: dict[Path, list[dict]] = {}
    for item, verdict, entry in results:
        by_path.setdefault(item['out_path'], []).append(entry)
    for path, entries in by_path.items():
        # Preserve verdict order
        order = {"duplicate": 0, "paraphrase": 1, "related": 2, "different": 3}
        entries.sort(key=lambda e: order.get(e["verdict"], 99))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
        print(f"  wrote {len(entries)} examples -> {path.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/few_shots_auto"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--concurrency", type=int, default=100)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
