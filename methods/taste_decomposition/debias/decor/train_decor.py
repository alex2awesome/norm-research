#!/usr/bin/env python3
"""Dense-standard reward model trained under DECORRELATION WEIGHTS (per-example
loss weights).  Debias instrument #3 -- importance-reweighting so y is
independent of the named nuisance score in the (reweighted) training
distribution.  No adversary, no representation surgery, no text edits.

Derived from methods/taste_decomposition/debias/train_grl.py (the frozen dense
standard for the N&C responded cell); the GRL head is REMOVED and a per-example
weight multiplies the BCE loss:

    loss_batch = mean_i( w_i * BCE(logit_i, y_i) )

with w mean-1 over the train rows (fit_weights.py output).  Per-example loss
weights were chosen over WeightedRandomSampler deliberately: they are
deterministic, keep the epoch composition / step count / LR schedule IDENTICAL
to the vanilla arms (paired comparison discipline), and give the exact
reweighted expectation without duplicate-row variance.  The implementation is
verified numerically by --gradcheck (gradient linearity in w on the real model
and real batches; see gradcheck_report in the run dir).

Everything else is byte-for-byte the frozen recipe: Llama-3.1-8B, LoRA r16/a32/
dropout .05 on q,k,v,o,gate,up,down, BCEWithLogits, lr 5e-5, wd .01, warmup .1,
batch 16, max_length 1024, 2 epochs, select-on-eval (unweighted eval AUC),
5 checkpoints/epoch.  Config may override `seed` (vanilla seed-band arms).

Run (sk3, ONE ledger-claimed GPU):
  export HOME=/lfs/skampere3/0/alexspan
  CUDA_VISIBLE_DEVICES=<gpu> $HOME/envs/ai_usage/bin/python train_decor.py --config <json>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from peft import LoraConfig, TaskType, get_peft_model

BASE_MODEL = "meta-llama/Llama-3.1-8B"
MAX_LENGTH = 1024
BATCH_SIZE = 16
EVAL_BATCH = 32
EPOCHS = 2
LR = 5e-5
WD = 0.01
WARMUP_RATIO = 0.1
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
TARGETS = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
EVALS_PER_EPOCH = 5
DEFAULT_SEED = 42


class TextDS(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


def build_model():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=1, torch_dtype=torch.bfloat16, device_map="cuda:0",
        low_cpu_mem_usage=True)
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False
    cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=LORA_R, lora_alpha=LORA_ALPHA,
                     lora_dropout=LORA_DROPOUT, target_modules=TARGETS, bias="none")
    model = get_peft_model(model, cfg)
    model.config.pad_token_id = tok.pad_token_id
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    return model, tok


def forward_batch(model, ids, am, device):
    assert bool((am[:, -1] == 1).all()), "left padding violated: last position is PAD"
    out = model(input_ids=ids.to(device), attention_mask=am.to(device), output_hidden_states=True)
    return out.logits[:, 0].float(), out.hidden_states[-1][:, -1, :]


@torch.no_grad()
def score_split(model, tok, texts, device, batch=EVAL_BATCH, want_rep=False, max_length=MAX_LENGTH):
    model.eval()
    probs, reps = [], []
    for i in range(0, len(texts), batch):
        chunk = [str(t) for t in texts[i:i + batch]]
        enc = tok(chunk, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        logit, rep = forward_batch(model, enc["input_ids"], enc["attention_mask"], device)
        if want_rep:
            reps.append(rep.float().cpu().numpy().astype(np.float16))
        probs.append(torch.sigmoid(logit).float().cpu().numpy())
    model.train()
    return np.concatenate(probs), (np.concatenate(reps) if want_rep else None)


def collect_grads(model, names):
    return torch.cat([p.grad.detach().float().flatten().cpu()
                      for n, p in model.named_parameters() if n in names])


def gradcheck(model, tok, texts, y, w, tr_idx, device, out_dir, max_length):
    """Verify the trainer honors per-example weights: the weighted-batch gradient
    must equal sum_i w_i * (gradient of example i alone, one-hot weights), i.e.
    the gradient is LINEAR in w -- checked on 2 real batches of 4 on the real
    model.  Dropout is disabled (model.eval()) so backward passes are
    deterministic; gradient checkpointing stays on (the real code path)."""
    model.eval()          # kill LoRA dropout for determinism; grads still flow
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    # a manageable but real slice of trainable params: score head + first & last LoRA pairs
    trainables = [n for n, p in model.named_parameters() if p.requires_grad]
    probe_names = set([n for n in trainables if "score" in n]
                      + [n for n in trainables if "layers.0." in n and "lora" in n][:4]
                      + [n for n in trainables if "layers.31." in n and "lora" in n][:4])
    rng = np.random.default_rng(0)
    report = {"batches": [], "probe_params": sorted(probe_names)}
    for b in range(2):
        idx = rng.choice(tr_idx, 4, replace=False)
        enc = tok([texts[i] for i in idx], truncation=True, max_length=max_length,
                  padding=True, return_tensors="pt")
        yy = torch.tensor(y[idx], dtype=torch.float32, device=device)
        ww = torch.tensor(w[idx], dtype=torch.float32, device=device)

        def grad_with(weight_vec):
            model.zero_grad(set_to_none=True)
            logit, _ = forward_batch(model, enc["input_ids"], enc["attention_mask"], device)
            loss = (weight_vec * loss_fn(logit, yy)).mean()
            loss.backward()
            return collect_grads(model, probe_names)

        g_full = grad_with(ww)
        g_sum = torch.zeros_like(g_full)
        for j in range(4):
            e = torch.zeros_like(ww)
            e[j] = ww[j]
            g_sum += grad_with(e)
        cos = float(torch.nn.functional.cosine_similarity(g_full, g_sum, dim=0))
        rel = float((g_full - g_sum).norm() / g_full.norm().clamp_min(1e-12))
        report["batches"].append({"idx": idx.tolist(), "w": w[idx].tolist(),
                                  "cosine": cos, "rel_err": rel,
                                  "norm_full": float(g_full.norm()), "norm_sum": float(g_sum.norm())})
        print(f"[gradcheck] batch {b}: cosine={cos:.6f} rel_err={rel:.2e}", flush=True)
    model.zero_grad(set_to_none=True)
    ok = all(bb["cosine"] > 0.999 and bb["rel_err"] < 0.02 for bb in report["batches"])
    report["PASS"] = bool(ok)
    (out_dir / "gradcheck_report.json").write_text(json.dumps(report, indent=2))
    print(f"[gradcheck] {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gradcheck", action="store_true",
                    help="run the 2-batch weight-linearity gradient check and exit")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    tag = cfg["tag"]
    out_dir = Path(cfg["out_dir"]) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    seed = int(cfg.get("seed", DEFAULT_SEED))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda:0"
    id_col = cfg.get("id_col", "doc_id")
    group_col = cfg.get("group_col", "docket")
    split_col = cfg.get("split_col", "split")
    max_length = int(cfg.get("max_length", MAX_LENGTH))

    df = pd.read_csv(cfg["corpus"])
    if cfg.get("subset_ids_from"):
        keep_ids = set(pd.read_csv(cfg["subset_ids_from"])[id_col].astype(str))
        df = df[df[id_col].astype(str).isin(keep_ids)].reset_index(drop=True)
    n_epochs = int(cfg.get("epochs", EPOCHS))

    split = df[split_col].astype(str).values
    tr, ev, te = split == "train", split == "eval", split == "test"
    y = df["judgement"].astype(int).values
    texts = df["text"].astype(str).tolist()
    ids = df[id_col].astype(str).values

    # ---- decorrelation weights ----------------------------------------------
    if cfg.get("weights"):
        wz = np.load(cfg["weights"], allow_pickle=True)
        lut = {str(d): float(v) for d, v in zip(wz["doc_id"], wz["w"])}
        missing = [d for d in ids[tr] if d not in lut]
        assert not missing, f"{len(missing)} train doc_ids missing from weights npz"
        w = np.array([lut.get(d, 1.0) for d in ids], dtype=np.float32)
        m = w[tr].mean()
        assert abs(m - 1.0) < 1e-4, f"train weights not mean-1 (mean={m})"
    else:
        w = np.ones(len(df), dtype=np.float32)

    model, tok = build_model()
    tr_idx, ev_idx = np.flatnonzero(tr), np.flatnonzero(ev)

    if args.gradcheck:
        ok = gradcheck(model, tok, texts, y.astype(float), w, tr_idx, device, out_dir, max_length)
        raise SystemExit(0 if ok else 1)

    def collate(batch_local):
        idx = tr_idx[np.asarray(batch_local)]
        enc = tok([texts[i] for i in idx], truncation=True, max_length=max_length,
                  padding=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "labels": torch.tensor(y[idx], dtype=torch.float32),
                "w": torch.tensor(w[idx], dtype=torch.float32)}

    train_loader = DataLoader(TextDS(len(tr_idx)), batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate)
    params = [{"params": [p for p in model.parameters() if p.requires_grad],
               "lr": LR, "weight_decay": WD}]
    opt = torch.optim.AdamW(params)
    total_steps = max(1, len(train_loader) * n_epochs)
    sched = get_linear_schedule_with_warmup(opt, int(total_steps * WARMUP_RATIO), total_steps)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    trig = {int(np.ceil(i * len(train_loader) / EVALS_PER_EPOCH)) for i in range(1, EVALS_PER_EPOCH + 1)}
    best = {"auc": -1.0, "step": -1}
    best_path = out_dir / "best_state.pt"
    hist, step = [], 0

    print(f"[{tag}] n={len(df)} train={tr.sum()} eval={ev.sum()} test={te.sum()} "
          f"weights={'YES' if cfg.get('weights') else 'none'} seed={seed} "
          f"n_eff={(w[tr].sum()**2 / (w[tr]**2).sum()):.0f}", flush=True)

    for ep in range(n_epochs):
        model.train()
        ep_step = 0
        for batch in train_loader:
            logit, _ = forward_batch(model, batch["input_ids"], batch["attention_mask"], device)
            per_ex = loss_fn(logit, batch["labels"].to(device))
            loss = (batch["w"].to(device) * per_ex).mean()
            task_loss = float(loss.item())
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            ep_step += 1
            if ep_step % 100 == 0:
                print(f"  [{tag}] ep{ep+1} step {ep_step}/{len(train_loader)} loss={task_loss:.4f}", flush=True)
            if ep_step in trig:
                p_ev, _ = score_split(model, tok, [texts[i] for i in ev_idx], device, max_length=max_length)
                auc = float(roc_auc_score(y[ev_idx], p_ev)) if len(set(y[ev_idx])) == 2 else float("nan")
                hist.append({"epoch": ep + 1, "epoch_step": ep_step, "global_step": step,
                             "eval_auc": auc, "task_loss": task_loss})
                print(f"  [{tag}] CKPT ep{ep+1} step{ep_step} eval_auc={auc:.4f}", flush=True)
                if np.isfinite(auc) and auc > best["auc"]:
                    best = {"auc": auc, "step": step, "epoch": ep + 1, "epoch_step": ep_step}
                    sd_ = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                           if "lora_" in k or "score" in k or "modules_to_save" in k}
                    torch.save(sd_, best_path)

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device), strict=False)
    else:
        print(f"[{tag}] WARNING: no checkpoint beat -inf eval AUC; scoring FINAL weights", flush=True)
    probs, reps = score_split(model, tok, texts, device, want_rep=True, max_length=max_length)
    res = {"tag": tag, "config": cfg, "n": int(len(df)),
           "recipe": {"model": BASE_MODEL, "max_length": max_length, "batch": BATCH_SIZE,
                      "epochs": n_epochs, "lr": LR, "lora": [LORA_R, LORA_ALPHA],
                      "selection_split": "eval (unweighted)", "seed": seed,
                      "loss": "per-example decorrelation weights * BCE, batch mean"},
           "lambda_adv": 0.0, "adv_channels": [],
           "weights": cfg.get("weights"),
           "n_eff_train": float(w[tr].sum() ** 2 / (w[tr] ** 2).sum()),
           "best_checkpoint": best, "history": hist}
    for nm, m in (("train", tr), ("eval", ev), ("test", te)):
        res[f"auc_{nm}"] = float(roc_auc_score(y[m], probs[m]))
        res[f"n_{nm}"] = int(m.sum())
    res["auc_evaltest"] = float(roc_auc_score(y[ev | te], probs[ev | te]))

    probs_abl = None
    tok_str = cfg.get("ablate_token")
    if tok_str:
        stripped = [t[len(tok_str) + 1:] if t.startswith(tok_str + " ") else t for t in texts]
        n_str = sum(1 for a, b in zip(texts, stripped) if a != b)
        probs_abl, _ = score_split(model, tok, stripped, device, max_length=max_length)
        res["ablation"] = {"token": tok_str, "n_rows_stripped": int(n_str)}
        for nm, m in (("eval", ev), ("test", te)):
            res["ablation"][f"auc_{nm}_ablated"] = float(roc_auc_score(y[m], probs_abl[m]))
            res["ablation"][f"delta_{nm}"] = res[f"auc_{nm}"] - res["ablation"][f"auc_{nm}_ablated"]
        res["ablation"]["auc_evaltest_ablated"] = float(roc_auc_score(y[ev | te], probs_abl[ev | te]))
        res["ablation"]["delta_evaltest"] = res["auc_evaltest"] - res["ablation"]["auc_evaltest_ablated"]

    np.savez_compressed(out_dir / "reps.npz", doc_id=ids, rep=reps,
                        prob=probs.astype(np.float32),
                        rep_h=np.zeros(0, dtype=np.float16),
                        prob_ablated=(probs_abl.astype(np.float32) if probs_abl is not None
                                      else np.zeros(0, dtype=np.float32)),
                        split=split, y=y)
    slim = pd.DataFrame({"doc_id": ids, "docket": df[group_col].astype(str),
                         "judgement": y, "split": split, "prob": probs, "w": w})
    if probs_abl is not None:
        slim["prob_ablated"] = probs_abl
    slim.to_csv(out_dir / "preds_slim.csv", index=False)
    res["runtime_sec"] = round(time.time() - t0, 1)
    (out_dir / "result.json").write_text(json.dumps(res, indent=2))
    print(f"RESULT {tag} " + json.dumps({k: res[k] for k in res if k not in ("history", "config")}), flush=True)
    print(f"[{tag}] DONE in {res['runtime_sec']}s", flush=True)


if __name__ == "__main__":
    main()
