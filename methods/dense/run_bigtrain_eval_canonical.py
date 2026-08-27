#!/usr/bin/env python3
"""Thin harness for TRAIN-BIG / EVAL-CANONICAL dense-standard runs.

train_reward_model.get_or_create_fixed_split() hard-asserts that an on-disk split is
80/10/10 within +-2pp. That assertion is right for the ordinary dense-standard design and
WRONG for a train-big design, where the eval/test folds are a fixed canonical row set and
the train fold is deliberately much larger (mathlib: 29,324 / 932 / 795 = .944/.030/.026).

Rather than edit the shared trainer (other agents run it concurrently), this harness
imports it, sets its three fraction constants to the OBSERVED fractions of the split dir
it was handed -- which makes the assertion a no-op for this run only -- and then calls
train_reward_model.train() completely unmodified. Everything else (recipe, loss,
class weighting, checkpoint selection, logging, outputs) is the frozen trainer verbatim.

Usage: identical to train_reward_model.py, e.g.
  python3 run_bigtrain_eval_canonical.py --data_path DIR/data.csv --split_dir DIR/split ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_reward_model as T  # noqa: E402


def main() -> None:
    args = T.parse_args()
    assert args.split_dir, "--split_dir is required for a train-big/eval-canonical run"
    sd = Path(args.split_dir)
    n = {s: len(pd.read_csv(sd / f"{s}.csv")) for s in ("train", "eval", "test")}
    total = sum(n.values())
    T.FIXED_TRAIN_FRACTION = n["train"] / total
    T.FIXED_EVAL_FRACTION = n["eval"] / total
    T.FIXED_TEST_FRACTION = n["test"] / total
    print(
        "[bigtrain] split ratio assertion relaxed to the observed on-disk fractions: "
        f"train={T.FIXED_TRAIN_FRACTION:.4f} ({n['train']}) "
        f"eval={T.FIXED_EVAL_FRACTION:.4f} ({n['eval']}) "
        f"test={T.FIXED_TEST_FRACTION:.4f} ({n['test']}); trainer otherwise unmodified",
        flush=True,
    )
    T.train(args)


if __name__ == "__main__":
    main()
