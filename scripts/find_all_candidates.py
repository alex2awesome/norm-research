"""
Find ALL within-bucket candidate pairs for a task, using cached text-embedding-3-small
embeddings. Reports total pair count + LLM cost estimate per bucket.

For each specificity bucket (vague / general / specific / hyper_specific):
  - All pairwise cosines (cross-doc, non-identical-name)
  - Stratified by cosine zone (dup, paraphrase, related, mid, different)
  - Pairs with cos >= threshold are written to JSONL for downstream LLM judgement

Output JSONL schema:
  {task, specificity, a_key, a_name, a_description, a_rubric_idx,
   b_key, b_name, b_description, b_rubric_idx, cosine, zone}
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import Counter
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
                    if (r.get("task") == task
                        and r.get("cls_ok")
                        and r.get("cls_keep") == "keep"
                        and r.get("cls_target") == "work"):
                        rows.append(r)
                except Exception:
                    pass
    return rows


def cosine_zone(c: float) -> str:
    if c >= 0.92:  return "duplicate"
    if c >= 0.80:  return "paraphrase"
    if c >= 0.65:  return "related"
    if c >= 0.45:  return "mid"
    return "different"


def find_all_pairs_in_bucket(rubrics: list[dict], embs: np.ndarray,
                              cos_threshold: float, top_k_per_rubric: int = 50):
    """For each rubric, find its top-K nearest neighbors (cos > threshold).
    Returns (zones_dict, total_pairs)."""
    name_lc = [str(r.get('rubric_name') or '').lower().strip() for r in rubrics]
    pid = [r['page_id'] for r in rubrics]
    n = len(rubrics)

    seen: set[tuple[int, int]] = set()
    zones: dict[str, list] = {"duplicate": [], "paraphrase": [], "related": [],
                              "mid": [], "different": []}

    # For each rubric, compute its sim row, find top-K above threshold
    # 12K × 12K cosines = 144M ops, fast with numpy (~1 minute)
    batch_size = 256
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        block = embs[start:end] @ embs.T  # (b, n)
        # Mask diagonal entries
        for i, row_idx in enumerate(range(start, end)):
            block[i, row_idx] = -1.0
        # Threshold + top-K
        for i, row_idx in enumerate(range(start, end)):
            sims = block[i]
            # Get indices where sims > threshold
            above = np.where(sims > cos_threshold)[0]
            if len(above) == 0: continue
            # Top-K of those (descending by sim)
            top_k = above[np.argsort(-sims[above])[:top_k_per_rubric]]
            for j in top_k:
                if j == row_idx: continue
                if pid[row_idx] == pid[int(j)]: continue
                if name_lc[row_idx] == name_lc[int(j)]: continue
                key = (min(row_idx, int(j)), max(row_idx, int(j)))
                if key in seen: continue
                seen.add(key)
                c = float(sims[j])
                zones[cosine_zone(c)].append((row_idx, int(j), c))

    return zones, len(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="creative-writing")
    ap.add_argument("--cos-threshold", type=float, default=0.65)
    ap.add_argument("--top-k-per-rubric", type=int, default=50)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    out_path = Path(args.output or (ROOT / f"outputs/dedup_eval/candidate_pairs_{args.task}.jsonl"))

    print(f"loading {args.task} keep+work rubrics...")
    rubrics_all = load_keep_work(args.task)
    print(f"  loaded {len(rubrics_all):,}")

    # Cache: try work_cache first (the one used by generate_few_shots), fall back to plain cache
    cache_candidates = [
        ROOT / f"outputs/embeddings/{args.task}_work_embeddings.npz",
        ROOT / f"outputs/embeddings/{args.task}_embeddings.npz",
    ]
    cache = None
    for c in cache_candidates:
        if c.exists():
            cache = c; break
    if cache is None:
        raise SystemExit(f"no embedding cache for {args.task}; expected one of {cache_candidates}")
    print(f"using embedding cache: {cache.name}")
    d = np.load(cache, allow_pickle=True)
    expected_keys = [f"{r['page_id']}::{r['rubric_idx']}" for r in rubrics_all]
    cache_keys = list(d['keys'])

    if cache_keys != expected_keys:
        # Cache may have been built over a different subset; align by key
        key_to_emb = dict(zip(cache_keys, d['embs']))
        missing = [k for k in expected_keys if k not in key_to_emb]
        if missing:
            raise SystemExit(f"cache missing {len(missing)} keys (e.g. {missing[0]})")
        embs_all = np.array([key_to_emb[k] for k in expected_keys], dtype=np.float32)
    else:
        embs_all = d['embs'].astype(np.float32)
    print(f"embeddings: shape={embs_all.shape}")

    # Process per-bucket
    SPECS = ["vague", "general", "specific", "hyper_specific"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_writeable = 0
    summary = {}
    with out_path.open("w") as out_f:
        for spec in SPECS:
            idx = [i for i, r in enumerate(rubrics_all) if r.get('cls_specificity') == spec]
            if not idx:
                print(f"\n=== {spec}: empty, skip"); continue
            rubrics_b = [rubrics_all[i] for i in idx]
            embs_b    = embs_all[idx]
            print(f"\n=== {spec}  N={len(rubrics_b):,}  ===")

            zones, n_pairs = find_all_pairs_in_bucket(
                rubrics_b, embs_b,
                cos_threshold=args.cos_threshold,
                top_k_per_rubric=args.top_k_per_rubric,
            )
            print(f"  candidate pairs (cos>={args.cos_threshold}): {n_pairs:,}")
            for z, items in zones.items():
                if items:
                    print(f"    {z:<11s} {len(items):>7,}")

            summary[spec] = {z: len(items) for z, items in zones.items()}
            summary[spec]['total'] = n_pairs

            for z, items in zones.items():
                if z == "different": continue  # below threshold, skip
                for ai, bi, c in items:
                    a = rubrics_b[ai]; b = rubrics_b[bi]
                    rec = {
                        "task": args.task,
                        "specificity": spec,
                        "cosine_zone": z,
                        "cosine": c,
                        "a_key": f"{a['page_id']}::{a['rubric_idx']}",
                        "a_name": a['rubric_name'],
                        "a_description": a.get('rubric_description', ''),
                        "a_rubric_idx": int(a['rubric_idx']),
                        "b_key": f"{b['page_id']}::{b['rubric_idx']}",
                        "b_name": b['rubric_name'],
                        "b_description": b.get('rubric_description', ''),
                        "b_rubric_idx": int(b['rubric_idx']),
                    }
                    out_f.write(json.dumps(rec) + "\n")
                    total_writeable += 1

    print(f"\n=== SUMMARY ===")
    print(f"total candidate pairs above threshold (excluding 'different'): {total_writeable:,}")
    print(f"output -> {out_path}")
    print(f"\n=== Cost estimate at gpt-5-mini ($0.001/pair): ${total_writeable * 0.001:.2f}")
    for spec, s in summary.items():
        eligible = s.get('duplicate', 0) + s.get('paraphrase', 0) + s.get('related', 0) + s.get('mid', 0)
        print(f"  {spec:<18s} {eligible:>8,}  → ${eligible*0.001:.2f}")


if __name__ == "__main__":
    main()
