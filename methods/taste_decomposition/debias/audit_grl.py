#!/usr/bin/env python3
"""V2-failure audit for the adversarial-debiasing (GRL) pilot.

Three independent checks, run BEFORE any retraining (audit-first discipline):

  1. MECHANICS  -- gradient probe on one real batch: is the adversarial gradient
     actually sign-reversed and does it reach the SHARED (LoRA) parameters at the
     expected magnitude?  Verified numerically, not by reading code.
  2. LORA BLIND SPOT -- the prime suspect: the dense standard trains LoRA
     adapters over a FROZEN base, so the pooled representation is
     frozen-base activations + low-rank deltas.  If the plant is already
     strongly readable from the FROZEN BASE representation (no adapters, no
     training), then GRL-on-LoRA must actively CANCEL frozen-substrate
     information with rank-16 deltas rather than merely "not encode" it --
     removal may be impossible by construction.  Test: extract base-model
     reps on the planted corpus, probe them with the standard probe.
  3. PLANT VISIBILITY -- does the planted token survive tokenization +
     truncation on >95% of planted rows?

Usage (sk3, one ledger-claimed GPU):
  export HOME=/lfs/skampere3/0/alexspan
  CUDA_VISIBLE_DEVICES=<gpu> $HOME/envs/ai_usage/bin/python audit_grl.py \
      --corpus build/corpus_planted.csv --out results_audit_grl.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from train_grl import (Adversary, GradReverse, BASE_MODEL, MAX_LENGTH,
                       build_model, forward_batch)

PLANT_TOKEN = "⟦QX7⟧"


def flat_grads(model):
    return {n: p.grad.detach().float().clone()
            for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}


def cos_and_ratio(ga, gb):
    """cosine + norm ratio over the concatenation of shared params in both dicts."""
    keys = sorted(set(ga) & set(gb))
    a = torch.cat([ga[k].flatten() for k in keys])
    b = torch.cat([gb[k].flatten() for k in keys])
    return (float(nn.functional.cosine_similarity(a[None], b[None]).item()),
            float(a.norm().item()), float(b.norm().item()), len(keys))


def part1_mechanics(df, tok, model, device, out):
    """One-batch gradient probe."""
    model.gradient_checkpointing_disable()
    tr = df[df["split"] == "train"]
    rows = pd.concat([tr[tr["plant"] == 1].head(4), tr[tr["plant"] == 0].head(4)])
    enc = tok(list(rows["text"].astype(str)), truncation=True, max_length=MAX_LENGTH,
              padding=True, return_tensors="pt")
    batch = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    y = torch.tensor(rows["judgement"].values, dtype=torch.float32, device=device)
    plant = torch.tensor(rows["plant"].values, dtype=torch.float32, device=device)
    nuis = ((plant - plant.mean()) / plant.std().clamp_min(1e-6))[:, None]

    model.train()
    logit, rep = forward_batch(model, batch, device)

    # score-head equivalence: the pooled rep must be the exact vector the scalar
    # head reads (guards the probe's "representation the head consumes" claim)
    score_mod = [m for n, m in model.named_modules() if n.endswith("score")][0]
    logit2 = score_mod(rep)[:, 0].float()
    out["score_head_reads_hidden_states_minus1"] = {
        "max_abs_diff": float((logit - logit2).abs().max().item()),
        "equivalent": bool((logit - logit2).abs().max().item() < 1e-2),
    }

    adv = Adversary(rep.shape[1], 1).to(device).float()
    task_loss = nn.functional.binary_cross_entropy_with_logits(logit, y)

    model.zero_grad(set_to_none=True)
    task_loss.backward(retain_graph=True)
    g_task = flat_grads(model)

    grads = {}
    for lam in (1.0, 0.5):
        model.zero_grad(set_to_none=True)
        adv.zero_grad(set_to_none=True)
        a_rev = nn.functional.mse_loss(adv(GradReverse.apply(rep.float(), lam)), nuis)
        a_rev.backward(retain_graph=True)
        grads[f"rev_{lam}"] = flat_grads(model)

    model.zero_grad(set_to_none=True)
    adv.zero_grad(set_to_none=True)
    a_fwd = nn.functional.mse_loss(adv(rep.float()), nuis)
    a_fwd.backward()
    g_fwd = flat_grads(model)
    model.zero_grad(set_to_none=True)

    cos1, n_rev1, n_fwd, k = cos_and_ratio(grads["rev_1.0"], g_fwd)
    cos5, n_rev5, _, _ = cos_and_ratio(grads["rev_0.5"], g_fwd)
    _, n_task, _, _ = cos_and_ratio(g_task, g_task)
    lora_keys = [kk for kk in grads["rev_1.0"] if "lora_" in kk]
    out["gradient_probe"] = {
        "n_shared_params_with_grad": k,
        "n_lora_params_with_adv_grad": len(lora_keys),
        "n_lora_adv_grad_nonzero": int(sum(grads["rev_1.0"][kk].abs().max() > 0 for kk in lora_keys)),
        "cosine_rev1_vs_fwd": cos1, "norm_ratio_rev1_over_fwd": n_rev1 / max(n_fwd, 1e-12),
        "cosine_rev0.5_vs_fwd": cos5, "norm_ratio_rev0.5_over_fwd": n_rev5 / max(n_fwd, 1e-12),
        "adv_grad_norm_over_task_grad_norm_lam1": n_rev1 / max(n_task, 1e-12),
        "expected": "cosine = -1.0 exactly; norm ratios = lambda; nonzero LoRA grads throughout",
    }
    print("[mechanics]", json.dumps(out["gradient_probe"], indent=1), flush=True)
    del grads, g_fwd, g_task
    torch.cuda.empty_cache()


@torch.no_grad()
def part2_base_reps(df, device, run_dir):
    """Frozen-base (NO adapters, NO training) pooled reps over the whole corpus,
    saved in the reps.npz schema so probe_reps.py runs on them unchanged."""
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModel.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16,
                                     device_map="cuda:0", low_cpu_mem_usage=True)
    base.eval()
    texts = df["text"].astype(str).tolist()
    reps = []
    B = 48
    for i in range(0, len(texts), B):
        enc = tok(texts[i:i + B], truncation=True, max_length=MAX_LENGTH,
                  padding=True, return_tensors="pt")
        am = enc["attention_mask"].to(device)
        assert bool((am[:, -1] == 1).all())
        h = base(input_ids=enc["input_ids"].to(device), attention_mask=am).last_hidden_state
        reps.append(h[:, -1, :].float().cpu().numpy().astype(np.float16))
        if (i // B) % 40 == 0:
            print(f"[base-reps] {i}/{len(texts)}", flush=True)
    rep = np.concatenate(reps)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / "reps.npz", doc_id=df["doc_id"].astype(str).values,
                        rep=rep, prob=np.zeros(len(df), dtype=np.float32),
                        prob_ablated=np.zeros(0, dtype=np.float32),
                        split=df["split"].values, y=df["judgement"].values)
    del base
    torch.cuda.empty_cache()
    print(f"[base-reps] wrote {run_dir/'reps.npz'} shape={rep.shape}", flush=True)


def part3_tokenization(df, out):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    plant_ids = tok.encode(PLANT_TOKEN, add_special_tokens=False)
    planted = df[df["plant"] == 1]
    ok = 0
    for t in planted["text"].astype(str):
        ids = tok(t, truncation=True, max_length=MAX_LENGTH)["input_ids"]
        head = ids[: len(plant_ids) + 3]
        ok += any(head[j:j + len(plant_ids)] == plant_ids
                  for j in range(len(head) - len(plant_ids) + 1))
    out["plant_tokenization"] = {
        "plant_token_ids": plant_ids, "n_planted_rows": int(len(planted)),
        "fraction_surviving_truncation": ok / max(len(planted), 1),
        "gate": ">= .95",
    }
    print("[tokenization]", json.dumps(out["plant_tokenization"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="build/corpus_planted.csv")
    ap.add_argument("--out", default="results_audit_grl.json")
    ap.add_argument("--base_rep_dir", default="runs/BASE_frozen_planted")
    ap.add_argument("--skip", default="", help="comma list: mechanics,base,tok")
    args = ap.parse_args()
    skip = set(args.skip.split(","))
    device = "cuda:0"
    out = {}
    df = pd.read_csv(args.corpus)

    if "tok" not in skip:
        part3_tokenization(df, out)
    if "mechanics" not in skip:
        model, tok = build_model()
        part1_mechanics(df, tok, model, device, out)
        del model
        torch.cuda.empty_cache()
    if "base" not in skip:
        part2_base_reps(df, device, Path(args.base_rep_dir))

    Path(args.out).write_text(json.dumps(out, indent=2))
    print("AUDIT " + json.dumps(out, default=float)[:1500], flush=True)


if __name__ == "__main__":
    main()
