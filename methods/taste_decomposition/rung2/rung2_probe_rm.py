#!/usr/bin/env python3
"""ADDENDUM E / P1 — RM-representation probe (paper-4-adjacent).

Extract the certified reward model's LAST-LAYER hidden state at the final
(non-pad) token — the exact vector its preference head reads — for every
pool text, save to npz. The linear probe (real-vs-generated, grouped CV)
runs on the mac afterwards.

Run on sk3:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
  $HOME/envs/ai_usage/bin/python rung2_probe_rm.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")
HERE = ROOT / "methods/taste_decomposition/rung2"
sys.path.insert(0, str(ROOT / "methods/dense"))
MODEL_DIR = ROOT / "datasets/creative-writing/wp_clean_rm_out/best_model"


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = json.load(open(MODEL_DIR / "adapter_config.json"))
    base = AutoModelForSequenceClassification.from_pretrained(
        cfg["base_model_name_or_path"], num_labels=1,
        torch_dtype=torch.bfloat16, device_map="auto")
    base.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(base, MODEL_DIR)
    model.eval()

    df = pd.read_csv(HERE / "rung2v2_pool_cw.csv").fillna({"prompt": "", "story": ""})
    texts = [f"PROMPT: {p}\n\nSTORY: {s}" for p, s in zip(df.prompt, df.story)]

    embs = []
    B = 16
    for i in range(0, len(texts), B):
        enc = tok(texts[i:i + B], truncation=True, max_length=1024,
                  padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[-1]          # (B, T, D); left padding -> final tok = -1
        embs.append(h[:, -1, :].float().cpu().numpy())
        if (i // B) % 20 == 0:
            print(f"{i}/{len(texts)}", flush=True)
    E = np.concatenate(embs)
    np.savez_compressed(HERE / "rung2_rm_penult_embs.npz",
                        emb=E.astype(np.float16),
                        cand_ids=df.cand_id.values.astype(object),
                        prompt_ids=df.prompt_id.values.astype(object),
                        family=df.family.values.astype(object))
    print(f"PROBE_EMB_DONE {E.shape}", flush=True)


if __name__ == "__main__":
    main()
