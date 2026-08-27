#!/usr/bin/env python3
"""Same-rows dense (T) rescore for the peer-VERDICT Layer-3 closure pilot.

Design freeze change #2 (notes/2026-08-05__taste-decomposition-design.md §6):
Delta_beyond needs T evaluated on the EXACT A/V-scored population, not the dense
model's own eval split.  This scores the frozen chain-of-6 best_model
(datasets/peer-review/vat_3y/dense_llama/verdict/rm_out/best_model, LoRA on
Llama-3.1-8B) over all 6,030 population rows and saves per-row probabilities.

Rows that fell in the dense TRAIN split are IN-SAMPLE -- the CSV keeps the
dense_split column so the readout can restrict to eval/test rows.

Run on sk3, GPU0 only:
  export HOME=/lfs/skampere3/0/alexspan
  CUDA_VISIBLE_DEVICES=0 $HOME/envs/ai_usage/bin/python rescore_dense_same_rows.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")
sys.path.insert(0, str(ROOT / "methods" / "dense"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_dir",
        default=str(ROOT / "datasets/peer-review/vat_3y/dense_llama/verdict/rm_out/best_model"),
    )
    ap.add_argument(
        "--pop_csv",
        default=str(ROOT / "methods/taste_decomposition/closure/peer_verdict_population.csv"),
    )
    ap.add_argument(
        "--out",
        default=str(ROOT / "methods/taste_decomposition/closure/peer_verdict_dense_preds.csv"),
    )
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=32)
    a = ap.parse_args()

    from train_reward_model import score_texts  # noqa: E402

    df = pd.read_csv(a.pop_csv)
    print(f"[dense-rescore] n={len(df)} model={a.model_dir}", flush=True)
    probs = np.array(
        score_texts(a.model_dir, df.text.tolist(), max_length=a.max_length, batch_size=a.batch_size),
        dtype=float,
    )
    df_out = df[["i", "ntitle", "judgement", "split", "dense_split"]].copy()
    df_out["dense_prob"] = probs
    df_out.to_csv(a.out, index=False)

    y = df.judgement.astype(int).values
    rep = {
        "n": int(len(df)),
        "auc_all_rows_CONTAMINATED": float(roc_auc_score(y, probs)),
    }
    for name, mask in [
        ("dense_heldout", df.dense_split.isin(["eval", "test"]).values),
        ("dense_eval", (df.dense_split == "eval").values),
        ("dense_test", (df.dense_split == "test").values),
        ("dense_train_INSAMPLE", (df.dense_split == "train").values),
        ("monitor_all", (df.split == "monitor").values),
        ("monitor_heldout", ((df.split == "monitor") & df.dense_split.isin(["eval", "test"])).values),
        ("fitmine_heldout_MININGSLICE", ((df.split == "fit_mine") & df.dense_split.isin(["eval", "test"])).values),
    ]:
        if mask.sum() > 20 and len(set(y[mask])) == 2:
            rep[f"auc_{name}"] = float(roc_auc_score(y[mask], probs[mask]))
            rep[f"n_{name}"] = int(mask.sum())
    print("DENSE_RESCORE_REPORT " + json.dumps(rep), flush=True)
    Path(a.out).with_suffix(".report.json").write_text(json.dumps(rep, indent=2))
    print("DENSE_RESCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
