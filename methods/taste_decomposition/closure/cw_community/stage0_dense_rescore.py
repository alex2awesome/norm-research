#!/usr/bin/env python3
"""STAGE 0 dense (T) rescore for the ENLARGED CW-community honest population.

Same model + same call pattern as
methods/taste_decomposition/samerows_scratch/score_samerows_dense.py
(Llama-3.1-8B + the wp_clean_rm_out/best_model LoRA adapter, num_labels=1,
sigmoid(logits[:,0]), max_length 1024).  Every row here is in the dense model's
eval/test split (prompt_id-grouped), so every prediction is honest.

Run on sk3, ONE GPU:
  export HOME=/lfs/skampere3/0/alexspan
  CUDA_VISIBLE_DEVICES=6 $HOME/envs/ai_usage/bin/python stage0_dense_rescore.py
"""
import json
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

R = Path("/lfs/skampere3/0/alexspan/norm-research")
HERE = R / "methods/taste_decomposition/closure/cw_community"
MODEL_DIR = R / "datasets/creative-writing/wp_clean_rm_out/best_model"
BASE_MODEL = "meta-llama/Llama-3.1-8B"
BATCH_SIZE = 16
MAX_LENGTH = 1024


def main():
    device = "cuda:0"
    t0 = time.time()
    df = pd.read_csv(HERE / "cw_honest_population.csv")
    print(f"[cw-dense] n={len(df)} model={MODEL_DIR}", flush=True)
    assert set(df.dense_split.unique()) <= {"eval", "test"}, "non-heldout rows present"

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=1, torch_dtype=torch.bfloat16, device_map=device)
    base.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(base, str(MODEL_DIR)).eval()

    texts = df["text"].astype(str).tolist()
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            b = tok(texts[i:i + BATCH_SIZE], truncation=True, max_length=MAX_LENGTH,
                    padding=True, return_tensors="pt").to(device)
            probs.extend(torch.sigmoid(model(**b).logits.float()[:, 0]).cpu().tolist())
            if (i // BATCH_SIZE) % 40 == 0:
                print(f"[cw-dense] {i}/{len(texts)}", flush=True)

    out = df[["id", "prompt_id", "judgement", "dense_split", "is_new"]].copy()
    out["dense_prob"] = probs
    out.to_csv(HERE / "cw_honest_dense_preds.csv", index=False)

    y = out.judgement.astype(int).values
    rep = {"n": int(len(out)), "model_dir": str(MODEL_DIR),
           "auc_honest_all": float(roc_auc_score(y, out.dense_prob.values))}
    for k, m in (("eval", out.dense_split == "eval"), ("test", out.dense_split == "test"),
                 ("old408", ~out.is_new.astype(bool)), ("new", out.is_new.astype(bool))):
        m = m.values
        if m.sum() > 20 and len(set(y[m])) == 2:
            rep[f"auc_{k}"] = float(roc_auc_score(y[m], out.dense_prob.values[m]))
            rep[f"n_{k}"] = int(m.sum())
    rep["runtime_sec"] = round(time.time() - t0, 1)
    (HERE / "cw_honest_dense_preds.report.json").write_text(json.dumps(rep, indent=2))
    print("CW_DENSE_REPORT " + json.dumps(rep), flush=True)
    print("CW_DENSE_DONE", flush=True)


if __name__ == "__main__":
    main()
