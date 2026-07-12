"""Persistent P(YES) behavioral matrix: n_metrics x n_items, scored ONCE and reused everywhere
(difficulty metric / MCQ hard-negative mining / reconstruction-target selection).

The difficulty metric is Cohen's-kappa between binarized verdict vectors; it needs DATA:
  * more ITEMS  -> tighter kappa (SE ~ 1/sqrt(K): 60 items ~0.12, 400 items ~0.05),
  * more METRICS -> a richer pool so genuine near-misses exist at every target kappa.

This builds that substrate. Score with the campaign continuous readout P(YES) (1-token logprob),
batched in thousands per vLLM call. ONE GPU, offline batch. Persists long parquet + dense .npy +
metric/item sidecars so reconstruction can reuse the exact same items.

Run (sk3, 1 GPU):
  CUDA_VISIBLE_DEVICES=5 VLLM_GPU_MEM_UTIL=0.4 HOME=/lfs/skampere3/0/alexspan \
    python -m methods.metric_implementer.build_score_matrix --task creative_writing \
    --n-metrics 250 --n-items 400
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from .batch_scoring import _YESNO_TEMPLATE
from .config import ImplementerConfig, apply_task_preset
from .manifest import full_manifest, load_corpus, load_metrics
from .vllm_backend import make_judge_backend


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--task", default="creative_writing")
    ap.add_argument("--n-metrics", type=int, default=250)
    ap.add_argument("--n-items", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8000, help="P(YES) prompts per vLLM call")
    ap.add_argument("--out-dir", default="/lfs/skampere3/0/alexspan/tmp_vinfo/score_matrix")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args(argv)

    # Need enough rubric FILES scanned to yield n_metrics distinct seeds (each file ~12 metrics).
    man = full_manifest(metrics_per_task=args.n_metrics,
                        metric_files_cap=max(800, args.n_metrics * 3))
    entry = (next((e for e in man.datasets if e.name == args.task), None)
             or next((e for e in man.datasets if args.task in e.name), None))
    if entry is None:
        print("SKIP unknown task", args.task)
        return 1
    cfg = ImplementerConfig()
    apply_task_preset(cfg, entry.task)
    max_chars = getattr(cfg, "max_text_chars", 4000)

    cfg0 = ImplementerConfig()
    if args.fake:
        cfg0.vllm_fake = True
    backend = make_judge_backend(args.model, cfg0, temperature=None)

    metrics = load_metrics(entry)[: args.n_metrics]
    texts, _ = load_corpus(entry, args.n_items, seed=7)
    M, K = len(metrics), len(texts)
    print(f"=== {args.task}: scoring {M} metrics x {K} items = {M * K} P(YES) calls ===")

    pairs = [(mi, ii) for mi in range(M) for ii in range(K)]
    mat = np.full((M, K), np.nan)
    t0 = time.time()
    for s in range(0, len(pairs), args.batch):
        chunk = pairs[s:s + args.batch]
        prompts = [_YESNO_TEMPLATE.format(rubric=metrics[mi].body, text=texts[ii][:max_chars])
                   for mi, ii in chunk]
        py = backend.score_binary(prompts, pos="YES", neg="NO")
        for (mi, ii), p in zip(chunk, py):
            mat[mi, ii] = p
        print(f"  {s + len(chunk)}/{len(pairs)} ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    base = f"{args.out_dir}/{args.task}"
    np.save(f"{base}_matrix.npy", mat)
    pd.DataFrame([{"metric_idx": mi, "metric_id": metrics[mi].metric_id, "item_idx": ii,
                   "p_yes": float(mat[mi, ii])} for mi, ii in pairs]).to_parquet(f"{base}_pyes.parquet")
    pd.DataFrame([{"metric_idx": mi, "metric_id": m.metric_id, "name": m.name,
                   "description": m.description, "body": m.body}
                  for mi, m in enumerate(metrics)]).to_parquet(f"{base}_metrics.parquet")
    pd.DataFrame([{"item_idx": ii, "text": texts[ii]} for ii in range(K)]).to_parquet(f"{base}_items.parquet")

    means, stds = np.nanmean(mat, axis=1), np.nanstd(mat, axis=1)
    disc = int(np.sum((stds >= 0.12) & (means >= 0.1) & (means <= 0.9)))
    print(f"\nwrote {base}_{{matrix.npy,pyes.parquet,metrics.parquet,items.parquet}}")
    print(f"discriminating metrics (std>=0.12, 0.1<=mean<=0.9): {disc}/{M}")
    print("backend:", backend.stats.as_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
