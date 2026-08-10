"""
Embedding-based candidate finder for rubric de-duplication.

Embeds all keep rubrics for a given task, finds top-K nearest neighbors per
random sample, filters to cross-document non-identical-name pairs, buckets by
cosine similarity. Output is hand-picked few-shot candidates spanning the
duplicate / paraphrase / related / different verdict zones.

TASK-SPECIFIC: the matcher operates within a single task's rubric set; no
cross-task pairs are ever produced. Pass --task to switch.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from openai import OpenAI

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")
CHUNKS = ROOT / "outputs/classifier_chunks_FULL"
KEY_PATH = Path("/lfs/skampere3/0/alexspan/.openai-salt-lab-key.txt")
EMB_MODEL = "text-embedding-3-small"


def load_keep_rubrics(task: str) -> list[dict]:
    rows = []
    for cf in sorted(CHUNKS.glob("chunk_*.jsonl")):
        with cf.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("task") == task and r.get("cls_ok") and r.get("cls_keep") == "keep":
                        rows.append(r)
                except Exception:
                    pass
    return rows


def embed_text(rubric: dict) -> str:
    name = rubric.get("rubric_name") or ""
    desc = rubric.get("rubric_description") or ""
    guidance = rubric.get("rubric_guidance") or ""
    text = f"{name}\n{desc}"
    if guidance:
        text += f"\n{guidance}"
    return text[:2000]


def get_or_compute_embeddings(rubrics: list[dict], client: OpenAI, cache_path: Path,
                              batch_size: int = 512) -> tuple[np.ndarray, list[str]]:
    keys = [f"{r['page_id']}::{r['rubric_idx']}" for r in rubrics]

    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        if list(d['keys']) == keys:
            print(f"using cached embeddings: {cache_path}")
            return d['embs'].astype(np.float32), list(d['keys'])
        else:
            print("cache key mismatch — recomputing")

    texts = [embed_text(r) for r in rubrics]
    embs_list = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=EMB_MODEL, input=batch)
        embs_list.extend([d.embedding for d in resp.data])
        print(f"  embedded {i+len(batch):,} / {len(texts):,}")
    embs = np.array(embs_list, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.where(norms == 0, 1.0, norms)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, embs=embs, keys=np.array(keys))
    print(f"cached: {cache_path}")
    return embs, keys


def find_candidates(rubrics: list[dict], embs: np.ndarray,
                    top_k: int = 8, n_queries: int = 2000, seed: int = 42) -> list[tuple[int,int,float]]:
    rng = np.random.RandomState(seed)
    q_idx = rng.choice(len(rubrics), size=min(n_queries, len(rubrics)), replace=False)
    candidates = []
    for qi in q_idx:
        sims = embs @ embs[qi]
        sims[qi] = -1
        top = np.argpartition(-sims, top_k)[:top_k]
        top = top[np.argsort(-sims[top])]
        for ti in top:
            candidates.append((int(qi), int(ti), float(sims[ti])))
    return candidates


def filter_and_bucket(candidates, rubrics):
    name_lc = [str(r.get('rubric_name') or '').lower().strip() for r in rubrics]
    page_id = [r['page_id'] for r in rubrics]

    seen: set[tuple[int,int]] = set()
    buckets: dict[str, list[tuple[int,int,float]]] = {
        'duplicate-zone (cos>=0.92)':  [],
        'paraphrase-zone (0.80-0.92)': [],
        'related-zone (0.65-0.80)':    [],
        'mid-zone (0.45-0.65)':        [],
        'different-zone (cos<0.30)':   [],
    }
    for a, b, cos in candidates:
        if a == b: continue
        k = (min(a,b), max(a,b))
        if k in seen: continue
        seen.add(k)
        if name_lc[a] == name_lc[b]: continue   # exclude identical-name
        if page_id[a] == page_id[b]: continue   # cross-doc only

        if cos >= 0.92:      buckets['duplicate-zone (cos>=0.92)'].append((a,b,cos))
        elif cos >= 0.80:    buckets['paraphrase-zone (0.80-0.92)'].append((a,b,cos))
        elif cos >= 0.65:    buckets['related-zone (0.65-0.80)'].append((a,b,cos))
        elif cos >= 0.45:    buckets['mid-zone (0.45-0.65)'].append((a,b,cos))
        elif cos < 0.30:     buckets['different-zone (cos<0.30)'].append((a,b,cos))
    # Sort each bucket by similarity desc (so highest-cos shown first within each)
    for k in buckets:
        buckets[k].sort(key=lambda t: -t[2])
    return buckets


def show_rubric(r: dict, label: str):
    print(f"  {label}  name: {r.get('rubric_name','')}")
    desc = (r.get('rubric_description') or '')[:220]
    print(f"     desc: {desc}")
    guid = (r.get('rubric_guidance') or '')[:120]
    if guid:
        print(f"     guid: {guid}")
    print(f"     page: {r['page_id']}")
    print(f"     cls : target={r.get('cls_target')} action={r.get('cls_action')} verif={r.get('cls_verifiability')} spec={r.get('cls_specificity')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="creative-writing")
    ap.add_argument("--n-per-bucket", type=int, default=4)
    ap.add_argument("--n-queries", type=int, default=4000)
    ap.add_argument("--top-k", type=int, default=8)
    # Final filter (after operability triage): only target=work is reliably
    # observable in a finished work. Drop production_process (~8% operable),
    # selection_criterion (~20% operable, mostly award/eligibility), and
    # evaluation_judgment (~32% operable, mostly evaluator-discipline).
    ap.add_argument("--include-targets", nargs='*', default=['work'],
                    help="Only rubrics with these targets enter dedup. Default: work-only.")
    ap.add_argument("--specificities", nargs='*',
                    default=['vague', 'general', 'specific', 'hyper_specific'],
                    help="Specificity buckets to iterate; each is a separate dedup partition.")
    args = ap.parse_args()

    rubrics_all = load_keep_rubrics(args.task)
    print(f"loaded {len(rubrics_all):,} {args.task} keep rubrics")

    api_key = KEY_PATH.read_text().strip() if KEY_PATH.exists() else os.environ.get("OPENAI_API_KEY","")
    if not api_key:
        sys.exit("no API key")
    client = OpenAI(api_key=api_key)

    cache = ROOT / f"outputs/embeddings/{args.task}_embeddings.npz"
    print(f"embedding (cache: {cache})...")
    embs_all, _ = get_or_compute_embeddings(rubrics_all, client, cache)
    print(f"embeddings: shape={embs_all.shape}, dtype={embs_all.dtype}")

    # Filter to in-scope targets (operability-based; default: work-only)
    include = set(args.include_targets)
    in_scope = [i for i, r in enumerate(rubrics_all) if r.get('cls_target') in include]
    print(f"after target filter (incl {sorted(include)}): {len(in_scope):,} / {len(rubrics_all):,}")

    # Iterate per specificity bucket — each is its own Phase-1 dedup partition.
    for spec in args.specificities:
        spec_idx = [i for i in in_scope if rubrics_all[i].get('cls_specificity') == spec]
        if not spec_idx:
            continue
        rubrics = [rubrics_all[i] for i in spec_idx]
        embs    = embs_all[spec_idx]

        print(f"\n{'#'*90}")
        print(f"#  SPECIFICITY BUCKET: {spec}  ({len(rubrics):,} rubrics in this bucket)")
        print(f"{'#'*90}")

        # kNN candidates within this bucket only
        n_q = min(args.n_queries, len(rubrics))
        candidates = find_candidates(rubrics, embs, top_k=args.top_k, n_queries=n_q)

        cos_buckets = filter_and_bucket(candidates, rubrics)
        for k, v in cos_buckets.items():
            print(f"    {k:<35s} {len(v):>5}")

        for bucket_name, pairs in cos_buckets.items():
            if not pairs: continue
            print(f"\n  -- {bucket_name}  (showing top {min(args.n_per_bucket, len(pairs))} of {len(pairs)}) --")
            for a, b, cos in pairs[:args.n_per_bucket]:
                print(f"\n  cos={cos:.3f}")
                show_rubric(rubrics[a], 'A')
                show_rubric(rubrics[b], 'B')


if __name__ == "__main__":
    main()
