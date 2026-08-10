"""Orchestrate the articulation-STaR loop.

Per iter:
  1. generate K rationales per artifact using current (LoRA-adapted) generator
  2. filter by bottleneck judge (rationale-only label decoding)
  3. SFT current LoRA on the kept rationales (rationale-only loss)

After each iter the latest LoRA is fed back to step 1 of the next iter.

NOTE: this runs steps in-process. On a single shared GPU with vLLM + HF
training in the same Python process, you'll hit memory contention -- prefer
launching each step as a subprocess (see scripts/run_articulation_star.sh
which we'll write once this works end-to-end on a small slice).
"""
from __future__ import annotations

import argparse

from .config import LoopConfig, TASKS
from . import generate_rationales, judge_filter, train_sft


def run(cfg: LoopConfig) -> None:
    if cfg.task not in TASKS:
        raise ValueError(f"unknown task {cfg.task}; have {list(TASKS)}")
    cur_lora: str | None = None
    for it in range(cfg.n_iters):
        print(f"\n=== iter {it} (task={cfg.task} run={cfg.run_name}) ===")
        generate_rationales.run(cfg, iter_idx=it, lora_path=cur_lora)
        judge_filter.run(cfg, iter_idx=it)
        lora_dir = train_sft.run(cfg, iter_idx=it, prev_lora=cur_lora)
        cur_lora = str(lora_dir)


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="peer_review")
    ap.add_argument("--run_name", default="v0")
    ap.add_argument("--n_iters", type=int, default=3)
    ap.add_argument("--n_train_subsample", type=int, default=4000)
    ap.add_argument("--n_rationales_per_input", type=int, default=4)
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    cfg = LoopConfig(
        task=a.task,
        run_name=a.run_name,
        n_iters=a.n_iters,
        n_train_subsample=a.n_train_subsample,
        n_rationales_per_input=a.n_rationales_per_input,
    )
    run(cfg)
