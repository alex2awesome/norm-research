"""Channel B, step 2 — LoRA distillation of the target's name-invoked policy (soft labels).

Causal-LM LoRA; loss = soft cross-entropy over the {YES, NO} label-token logits at the answer
position vs the target's p_yes. Prompts are chat-formatted EXACTLY like scoring time
(apply_chat_template, add_generation_prompt=True, enable_thinking=False fallback — mirrors
vllm_backend._flush / score_declared_binary) so the trained distribution and the scored
distribution live on the same rendering.

Low-N is the regime of interest (--n-train subsamples train split deterministically): the
exchange rate "N examples vs K articulation tokens" is the estimand; saturation cloning is a
failure mode, not a result.

GPU script (1 GPU, feedback_gpu_usage). Deps: torch, transformers, peft (already in repo env,
methods/dense/requirements.txt). Main-guarded for spawn (feedback_vllm_main_guard_moe_spawn).

Usage (sk2, after build_policy_distill_dataset):
  python -m methods.tacit_channels.channels.distill.train_policy_lora \
      --dataset outputs/tacit_channels/distill/humor_<cell>.jsonl \
      --base-model Qwen/Qwen2.5-7B-Instruct --n-train 32 \
      --out-dir outputs/tacit_channels/adapters/humor_<cell>_n32
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

from methods.tacit_channels.channels.common import read_jsonl


def resolve_label_ids(tokenizer, pos: str, neg: str) -> tuple[int, int]:
    """Single-token label resolution — same fail-closed contract as the scorer."""
    from methods.metric_implementer.vllm_backend import _single_token_label_id
    return _single_token_label_id(tokenizer, pos), _single_token_label_id(tokenizer, neg)


def render_chat(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def train(args) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [r for r in read_jsonl(args.dataset) if r["split"] == "train"]
    if not rows:
        raise SystemExit("no train rows in dataset")
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.n_train and args.n_train < len(rows):
        rows = rows[:args.n_train]
    print(f"training on {len(rows)} examples (low-N regime)")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    yes_id, no_id = resolve_label_ids(tokenizer, args.pos, args.neg)
    print(f"label ids: {args.pos}={yes_id} {args.neg}={no_id}")

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map={"": 0})
    lora = LoraConfig(
        task_type="CAUSAL_LM", r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.train()

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr)
    total_steps = max(1, (len(rows) // args.batch_size) * args.epochs)
    warmup = max(10, total_steps // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warmup, max(0.0, 1 - s / total_steps)))
    device = next(model.parameters()).device

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    step = 0
    quarantined = 0
    for epoch in range(args.epochs):
        rng.shuffle(rows)
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            texts = [render_chat(tokenizer, r["prompt"]) for r in batch]
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=args.max_length).to(device)
            targets = torch.tensor([r["p_yes"] for r in batch],
                                   dtype=torch.float32, device=device)
            logits = model(**enc).logits  # (B, T, V)
            # answer position = last non-pad token (add_generation_prompt ends the prefix)
            last = enc["attention_mask"].sum(dim=1) - 1
            idx = torch.arange(len(batch), device=device)
            pair = torch.stack([logits[idx, last, yes_id],
                                logits[idx, last, no_id]], dim=1)  # (B, 2)
            logp = torch.log_softmax(pair.float(), dim=1)
            # soft CE against the target's p_yes — the distillation objective
            loss = -(targets * logp[:, 0] + (1 - targets) * logp[:, 1]).mean()
            # fail-soft with accounting (v1b/v1c post-mortem: a specific batch produces a
            # non-finite forward at the same deterministic step). Quarantine + skip; abort
            # only if the quarantine rate exceeds 0.5% — never train THROUGH divergence,
            # never save NaN adapters, never let one batch kill the run silently.
            if not torch.isfinite(loss):
                quarantined += 1
                with open(out / "quarantine.jsonl", "a") as qf:
                    qf.write(json.dumps({
                        "epoch": epoch, "step": step,
                        "cells": [r["cell_id"] for r in batch],
                        "items": [r["item_sha256"][:12] for r in batch],
                        "max_abs_logit": float(pair.detach().abs().max()),
                        "finite_logits": bool(torch.isfinite(pair).all()),
                    }) + "\n")
                optimizer.zero_grad()
                step += 1
                if quarantined > max(5, total_steps // 200):
                    raise RuntimeError(
                        f"{quarantined} non-finite batches quarantined — systemic divergence")
                continue
            loss.backward()
            # gradient-finiteness guard (v1c2 post-mortem: a batch with FINITE loss produced
            # inf/NaN grads; the optimizer step poisoned the weights and every later forward
            # was NaN). No step ever consumes a non-finite gradient.
            trainable = [p for p in model.parameters() if p.requires_grad]
            if not all(torch.isfinite(p.grad).all() for p in trainable
                       if p.grad is not None):
                quarantined += 1
                with open(out / "quarantine.jsonl", "a") as qf:
                    qf.write(json.dumps({
                        "kind": "grad", "epoch": epoch, "step": step,
                        "cells": [r["cell_id"] for r in batch],
                        "items": [r["item_sha256"][:12] for r in batch]}) + "\n")
                optimizer.zero_grad()
                step += 1
                if quarantined > max(5, total_steps // 200):
                    raise RuntimeError(
                        f"{quarantined} quarantined batches — systemic divergence")
                continue
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            sched.step()
            optimizer.zero_grad()
            step += 1
            if step % 200 == 0:  # weight-integrity tripwire — corrupted weights abort HERE
                if not all(torch.isfinite(p).all() for p in trainable):
                    raise RuntimeError(f"non-finite parameters at step {step} — weights "
                                       "corrupted despite grad guard; aborting")
            if step % 10 == 0 or step == 1:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}", flush=True)
            # checkpoint trajectory for compilation-signature probes (P-DYN-1, P-GEN-5)
            if args.save_every and step % args.save_every == 0:
                ck = out / f"checkpoint-step{step}"
                ck.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ck)
                print(f"checkpoint saved at step {step}", flush=True)

    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    manifest = {
        "schema": "tacit_channels_policy_adapter/v1",
        "base_model": args.base_model,
        "dataset": args.dataset,
        "dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        "n_train": len(rows), "seed": args.seed, "epochs": args.epochs,
        "lr": args.lr, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "label_ids": {args.pos: yes_id, args.neg: no_id},
        "loss": "soft_ce_yes_no_logits",
        "quarantined_batches": quarantined,
    }
    (out / "adapter_provenance.json").write_text(json.dumps(manifest, indent=2))
    print(f"saved adapter + provenance -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-train", type=int, default=0, help="0 = all train rows; low-N sweep "
                    "values: 8/32/128/512")
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--save-every", type=int, default=0,
                    help="save a checkpoint adapter every K steps (0 = final only)")
    ap.add_argument("--pos", default="YES")
    ap.add_argument("--neg", default="NO")
    ap.add_argument("--bf16", action="store_true", default=True)
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    train(args)


if __name__ == "__main__":
    main()
