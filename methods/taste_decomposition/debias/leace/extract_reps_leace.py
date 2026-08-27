#!/usr/bin/env python3
"""LEACE pilot -- the single GPU step: pooled-h representations of the
TOKEN-STRIPPED realtok corpus under the FROZEN R06 model (vanilla trained on
corpus_realtok.csv).  Needed for the V3a ablation readout: the refit heads
(raw vs standard-nuisance-projected) are scored on token-present AND
token-stripped reps, and the token's contribution must survive projection.

Stripped text of corpus_realtok.csv IS population.csv's text column (the
builder prepends the token to a copy of the population text), asserted below.

Run from the sk3 debias_pilot dir (train_grl.py lives there):
  export HOME=/lfs/skampere3/0/alexspan
  CUDA_VISIBLE_DEVICES=<gpu> $HOME/envs/ai_usage/bin/python extract_reps_leace.py
No training: one forward pass over 9,521 rows.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_grl import build_model, score_split

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research/datasets/notice-and-comment/debias_pilot")
RUN = ROOT / "runs/R06_vanilla_realtok"
TOKEN = "⟦RS4⟧"   # <<RS4>> realtok token


def main():
    t0 = time.time()
    rt = pd.read_csv(ROOT / "build/corpus_realtok.csv")
    pop = pd.read_csv(ROOT / "build/population.csv")
    assert (rt["doc_id"].astype(str).values == pop["doc_id"].astype(str).values).all()
    # stripped realtok text == population text, row by row
    stripped = [t[len(TOKEN) + 1:] if str(t).startswith(TOKEN + " ") else str(t)
                for t in rt["text"]]
    mismatch = sum(1 for a, b in zip(stripped, pop["text"].astype(str)) if a != b)
    assert mismatch == 0, f"{mismatch} rows: stripped realtok text != population text"

    model, tok = build_model()
    sd = torch.load(RUN / "best_state.pt", map_location="cuda:0")
    missing = model.load_state_dict(sd, strict=False)
    print(f"loaded best_state.pt: {len(sd)} tensors, "
          f"{len(missing.unexpected_keys)} unexpected", flush=True)
    model.eval()

    probs, reps, _ = score_split(model, tok, stripped, "cuda:0", want_rep=True, bn=None)
    out = RUN / "reps_stripped.npz"
    np.savez_compressed(out, doc_id=rt["doc_id"].astype(str).values, rep=reps,
                        prob=probs.astype(np.float32),
                        split=rt["split"].values, y=rt["judgement"].astype(int).values)
    from sklearn.metrics import roc_auc_score
    ev = rt["split"].values == "eval"
    y = rt["judgement"].astype(int).values
    rj = json.loads((RUN / "result.json").read_text())
    print(json.dumps({
        "n": len(stripped),
        "auc_eval_stripped_modelhead": float(roc_auc_score(y[ev], probs[ev])),
        "result_json_auc_eval_ablated": rj["ablation"]["auc_eval_ablated"],
        "runtime_sec": round(time.time() - t0, 1),
    }), flush=True)
    print("EXTRACT_LEACE DONE", flush=True)


if __name__ == "__main__":
    main()
