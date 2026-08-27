"""Merge per-shard rationale jsonls into a single rationales.jsonl.

Called after the data-parallel generation step finishes.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

from .config import LoopConfig


def run(cfg: LoopConfig, iter_idx: int) -> Path:
    out_dir = cfg.iter_dir(iter_idx)
    shards = sorted(glob.glob(str(out_dir / "rationales.shard*.jsonl")))
    if not shards:
        raise RuntimeError(f"no shards found in {out_dir}")
    out_path = out_dir / "rationales.jsonl"
    n = 0
    with out_path.open("w") as f:
        for s in shards:
            with open(s) as g:
                for line in g:
                    f.write(line)
                    n += 1
    print(f"[merge] {len(shards)} shards -> {out_path} ({n:,} rows)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="peer_review")
    ap.add_argument("--run_name", default="v0")
    ap.add_argument("--iter", type=int, required=True)
    args = ap.parse_args()
    run(LoopConfig(task=args.task, run_name=args.run_name), args.iter)
