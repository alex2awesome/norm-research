#!/usr/bin/env python3
"""RUNG 2, stage D — score candidates with a split-half dense model
(design §2.4.2). Run once with half-A (SELECTOR) and once with half-B
(ARBITER). Renders candidates EXACTLY like the dense training rows
("PROMPT: {prompt}\n\nSTORY: {story}" -- matched-pipeline rule) and plants a
blinded battery of REAL held-out pos/neg items in the same pass: the model
must rank them near its registered test AUC or the pass is void.

Run on sk3 (one GPU, PCI_BUS_ID order):
  export HOME=/lfs/skampere3/0/alexspan
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
    $HOME/envs/ai_usage/bin/python rung2_dense_score_cands.py --half A
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("/lfs/skampere3/0/alexspan/norm-research")
HERE = ROOT / "methods/taste_decomposition/rung2"
CWD_ = ROOT / "methods/taste_decomposition/closure/cw_community"
sys.path.insert(0, str(ROOT / "methods/dense"))

SEED = 20260822


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", required=True,
                    help="A | B | full (certified wp_clean model, Addendum C)")
    ap.add_argument("--cands", default=str(HERE / "rung2_candidates_cw_community_full.csv"))
    ap.add_argument("--anchor-k", type=int, default=100)
    ap.add_argument("--out-tag", default="", help="suffix for output files (e.g. v2)")
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=32)
    a = ap.parse_args()

    if a.half == "full":
        model_dir = ROOT / "datasets/creative-writing/wp_clean_rm_out/best_model"
    else:
        model_dir = (ROOT / "datasets/creative-writing" /
                     f"wp_clean_half{a.half}_rm_out" / "best_model")
    assert model_dir.exists(), f"no model at {model_dir}"

    from train_reward_model import score_texts

    df = pd.read_csv(a.cands).fillna({"prompt": "", "story": ""})
    texts = [f"PROMPT: {p}\n\nSTORY: {s}" for p, s in zip(df.prompt, df.story)]

    # blinded real-item battery from the honest population (held-out frame)
    pop = pd.read_csv(CWD_ / "cw_honest_population.csv")
    rng = random.Random(SEED)
    anchors = pd.concat([
        pop[pop.judgement == 1].sample(a.anchor_k, random_state=SEED).assign(tag="anchor_pos"),
        pop[pop.judgement == 0].sample(a.anchor_k, random_state=SEED).assign(tag="anchor_neg"),
    ], ignore_index=True)
    atexts = anchors.text.tolist()  # population 'text' is the canonical rendering

    order = list(range(len(texts) + len(atexts)))
    rng.shuffle(order)                     # blind: anchors interleaved
    alltexts = texts + atexts
    shuffled = [alltexts[i] for i in order]
    probs_shuf = np.array(score_texts(str(model_dir), shuffled,
                                      max_length=a.max_length,
                                      batch_size=a.batch_size), dtype=float)
    probs = np.empty(len(alltexts))
    probs[np.array(order)] = probs_shuf

    cand_probs, anchor_probs = probs[:len(texts)], probs[len(texts):]
    ya = (anchors.tag == "anchor_pos").astype(int).values
    battery_auc = float(roc_auc_score(ya, anchor_probs))

    keep = [c for c in ("cand_id", "prompt_id", "k_index", "family", "condition")
            if c in df.columns]
    out = df[keep].copy()
    tag = ("full" if a.half == "full" else f"half{a.half}") +         (f"_{a.out_tag}" if a.out_tag else "")
    out[f"dense_{tag}_prob"] = cand_probs
    fp = HERE / f"rung2_dense_scores_cw_{tag}.csv"
    out.to_csv(fp, index=False)
    rep = {"half": a.half, "model_dir": str(model_dir), "n_candidates": int(len(df)),
           "anchor_k_per_class": a.anchor_k,
           "anchor_pos_vs_neg_auc": battery_auc,
           "render": "PROMPT: {prompt}\\n\\nSTORY: {story} (population-identical)",
           "design": "notes/2026-08-21__rung12_design_gap_consequences.md §2.4.2"}
    fp.with_suffix(".report.json").write_text(json.dumps(rep, indent=2))
    print("RUNG2_DENSE_REPORT " + json.dumps(rep), flush=True)
    print("RUNG2_DENSE_DONE", flush=True)


if __name__ == "__main__":
    main()
