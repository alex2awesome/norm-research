#!/usr/bin/env python3
"""Direction-2 moredata launcher: byte-identical to invoking
methods/dense/train_reward_model.py with the same CLI, EXCEPT the on-disk
split-ratio gate (80/10/10, atol 2e-2) is relaxed to the observed ratios --
required because Direction 2 grows ONLY the train split (eval/test rows AS-IS),
so the grown split dir intentionally fails the stock 80/10/10 check.

Nothing else in the recipe changes: same trainer module, same args.
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

TRM_PATH = Path(__file__).resolve().parents[2] / "dense" / "train_reward_model.py"
sys.path.insert(0, str(TRM_PATH.parent))  # train_reward_model does `from eval_utils import ...`
spec = importlib.util.spec_from_file_location("train_reward_model_grown", str(TRM_PATH))
trm = importlib.util.module_from_spec(spec)
sys.modules["train_reward_model_grown"] = trm
spec.loader.exec_module(trm)


def main():
    args = trm.parse_args()
    split_dir = Path(args.split_dir)
    sizes = {s: len(pd.read_csv(split_dir / f"{s}.csv")) for s in ("train", "eval", "test")}
    total = sum(sizes.values())
    trm.FIXED_TRAIN_FRACTION = sizes["train"] / total
    trm.FIXED_EVAL_FRACTION = sizes["eval"] / total
    trm.FIXED_TEST_FRACTION = sizes["test"] / total
    print(f"[train_grown_split] relaxed split gate to observed fractions: "
          f"train={trm.FIXED_TRAIN_FRACTION:.4f} eval={trm.FIXED_EVAL_FRACTION:.4f} "
          f"test={trm.FIXED_TEST_FRACTION:.4f} (n={sizes})", flush=True)
    if args.use_optuna:
        trm.run_optuna(args)
    else:
        trm.train(args)


if __name__ == "__main__":
    main()
