#!/usr/bin/env python3
"""Dense-standard reward model with an OPTIONAL gradient-reversal (GRL)
adversarial head on the named nuisance channels.  Adversarial-debiasing pilot,
N&C responded cell.  Spec: notes/2026-08-05__taste-decomposition-design.md S9.

Recipe is the frozen dense standard for this cell (verified against
datasets/notice-and-comment/v4/dense_llama/responded/train.log):
  Llama-3.1-8B, LoRA r16 / alpha32 / dropout .05 on q,k,v,o,gate,up,down,
  num_labels=1 + BCEWithLogits, lr 5e-5, wd .01, warmup .1, batch 16,
  grad-accum 1, max_length 1024, 2 epochs, gradient checkpointing, seed 42,
  5 validation checkpoints per epoch.
Two deliberate departures, both declared:
  * selection_split = EVAL (the archived chain selected on TEST; the dense
    standard for fresh trainings is select-on-eval).
  * splits are DOCKET-GROUPED stable-hash (the archived chain used a random
    row split, which leaks dockets across train/eval).

GRL: pooled representation h (last hidden state at the final non-pad position --
the exact vector the model's own scalar head reads) -> GradReverse(lambda) ->
2-layer MLP -> standardized nuisance vector, MSE.  Backward multiplies the
adversary's gradient into the encoder by -lambda; the adversary head itself
minimises normally.  lambda ramps linearly over the first 10% of optimizer steps
(fixed across the sweep) so that lambda=1.0 does not destabilise step 1.

Run (sk3, ONE ledger-claimed GPU):
  export HOME=/lfs/skampere3/0/alexspan
  CUDA_VISIBLE_DEVICES=<gpu> $HOME/envs/ai_usage/bin/python train_grl.py --config <json>
"""
from __future__ import annotations

import argparse
import json
import os
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
ADV_LR = 1e-3
WD = 0.01
WARMUP_RATIO = 0.1
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
TARGETS = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
EVALS_PER_EPOCH = 5
SEED = 42
LAMBDA_RAMP_FRAC = 0.10
ADV_HIDDEN = 256
BN_DIM = 256


class TextDS(Dataset):
    def __init__(self, texts, y, nuis, tok):
        self.texts, self.y, self.nuis, self.tok = texts, y, nuis, tok

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return i


def make_collate(texts, y, nuis, tok):
    def collate(idx):
        idx = np.asarray(idx)
        enc = tok([texts[i] for i in idx], truncation=True, max_length=MAX_LENGTH,
                  padding=True, return_tensors="pt")
        out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
               "labels": torch.tensor(y[idx], dtype=torch.float32),
               "idx": torch.tensor(idx, dtype=torch.long)}
        if nuis is not None:
            out["nuis"] = torch.tensor(nuis[idx], dtype=torch.float32)
        return out
    return collate


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class Adversary(nn.Module):
    def __init__(self, d_in, d_out, hidden=ADV_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, d_out))

    def forward(self, h):
        return self.net(h)


class BottleneckHead(nn.Module):
    """arch="bottleneck": trainable projection between the pooled representation
    and BOTH heads.  The task head and the adversary see ONLY z = proj(h); GRL
    shapes proj directly.

    Rationale (V2-failure root cause, see notes/2026-08-07__debias_audit_fable.md):
    with the scalar head reading h itself, h = frozen-base activations + rank-16
    LoRA deltas, and the frozen substrate carries the planted channel (frozen-base
    probe AUC ~1.0), so GRL-on-LoRA would have to CANCEL frozen-substrate
    information with low-rank deltas -- it never does (plant probe .997-.999 at
    every lambda).  Here the removal target is a fully trainable full-rank map,
    and the debias certificate is scoped to what the score head can see: the
    score is a function of z alone, so "plant unreadable from z" == "the score
    cannot use the plant"."""

    def __init__(self, d_in, d_bn=BN_DIM):
        super().__init__()
        self.proj = nn.Linear(d_in, d_bn)
        self.head = nn.Linear(d_bn, 1)

    def forward(self, h):
        z = self.proj(h)
        return self.head(z)[:, 0], z


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


def forward_batch(model, batch, device):
    """Return (logit, pooled_rep).  Left padding => the final position is always a
    real token, which is exactly the position LlamaForSequenceClassification pools
    from (last non-pad token).  Asserted on every batch."""
    ids = batch["input_ids"].to(device)
    am = batch["attention_mask"].to(device)
    assert bool((am[:, -1] == 1).all()), "left padding violated: last position is PAD"
    out = model(input_ids=ids, attention_mask=am, output_hidden_states=True)
    rep = out.hidden_states[-1][:, -1, :]
    return out.logits[:, 0].float(), rep


@torch.no_grad()
def score_split(model, tok, texts, device, batch=EVAL_BATCH, want_rep=False, bn=None):
    """Returns (probs, reps, reps_h).  reps = the representation the task head
    consumes (z under arch=bottleneck, pooled h otherwise) -- the probe target.
    reps_h = pooled h alongside z (bottleneck only), for the scope check."""
    model.eval()
    if bn is not None:
        bn.eval()
    probs, reps, reps_h = [], [], []
    for i in range(0, len(texts), batch):
        chunk = [str(t) for t in texts[i:i + batch]]
        enc = tok(chunk, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
        b = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        logit, rep = forward_batch(model, b, device)
        if bn is not None:
            logit, z = bn(rep.float())
            if want_rep:
                reps.append(z.float().cpu().numpy().astype(np.float16))
                reps_h.append(rep.float().cpu().numpy().astype(np.float16))
        elif want_rep:
            reps.append(rep.float().cpu().numpy().astype(np.float16))
        probs.append(torch.sigmoid(logit).float().cpu().numpy())
    model.train()
    if bn is not None:
        bn.train()
    return (np.concatenate(probs),
            (np.concatenate(reps) if want_rep else None),
            (np.concatenate(reps_h) if (want_rep and bn is not None) else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    tag = cfg["tag"]
    out_dir = Path(cfg["out_dir"]) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda:0"

    df = pd.read_csv(cfg["corpus"])
    nz = np.load(cfg["nuisance"], allow_pickle=True)
    nz_ids = np.array([str(s) for s in nz["doc_id"]])
    Z_all, names = nz["Z"], [str(s) for s in nz["names"]]
    groups = json.loads(str(nz["groups_json"]))
    extra = {"plant": nz["plant"].astype(float), "realtok": nz["realtok"].astype(float)}

    # restrict to a subsample if asked (V3b)
    if cfg.get("subset_ids_from"):
        keep_ids = set(pd.read_csv(cfg["subset_ids_from"])["doc_id"].astype(str))
        df = df[df["doc_id"].astype(str).isin(keep_ids)].reset_index(drop=True)

    if cfg.get("smoke_rows"):        # smoke test only -- never used for a reported number
        _rs, _sv, _k = np.random.default_rng(0), df["split"].values, int(cfg["smoke_rows"])
        keep = np.concatenate([_rs.choice(np.flatnonzero(_sv == s), min(int((_sv == s).sum()), _k),
                                          replace=False) for s in ("train", "eval", "test")])
        df = df.iloc[np.sort(keep)].reset_index(drop=True)
    n_epochs = int(cfg.get("epochs", EPOCHS))

    order = {d: i for i, d in enumerate(nz_ids)}
    pos = np.array([order[str(d)] for d in df["doc_id"]])
    assert len(set(pos)) == len(pos)

    # ---- assemble the adversary target matrix from the NAMED channels --------
    cols, col_names = [], []
    for ch in cfg["channels"]:
        if ch in groups:
            cols += list(groups[ch])
            col_names += [names[j] for j in groups[ch]]
        elif ch in extra:
            cols.append(-1)
            col_names.append(ch)
        else:
            raise ValueError(f"unknown channel {ch}")
    if cols:
        blocks = []
        for ch in cfg["channels"]:
            if ch in groups:
                blocks.append(Z_all[np.ix_(pos, groups[ch])])
            else:
                blocks.append(extra[ch][pos][:, None])
        Nraw = np.hstack(blocks)
    else:
        Nraw = None

    split = df["split"].values
    tr, ev, te = split == "train", split == "eval", split == "test"
    y = df["judgement"].astype(int).values
    texts = df["text"].astype(str).tolist()

    if Nraw is not None:
        mu, sd = Nraw[tr].mean(0), Nraw[tr].std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        N = (Nraw - mu) / sd
    else:
        N = None
    lam_max = float(cfg.get("lambda_adv", 0.0))
    use_adv = (N is not None) and lam_max > 0
    arch = cfg.get("arch", "pooled")

    model, tok = build_model()
    bn = BottleneckHead(model.config.hidden_size).to(device).float() if arch == "bottleneck" else None
    coll = make_collate(texts, y, N, tok)
    tr_idx, ev_idx = np.flatnonzero(tr), np.flatnonzero(ev)
    train_loader = DataLoader(TextDS([texts[i] for i in tr_idx], y[tr_idx], None, tok),
                              batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda b: coll([tr_idx[i] for i in b]))

    params = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": LR, "weight_decay": WD}]
    if bn is not None:
        params.append({"params": list(bn.parameters()), "lr": LR, "weight_decay": WD})
    adv = None
    if use_adv:
        d_rep = BN_DIM if bn is not None else model.config.hidden_size
        adv = Adversary(d_rep, N.shape[1]).to(device).float()
        params.append({"params": list(adv.parameters()), "lr": ADV_LR, "weight_decay": 0.0})
    opt = torch.optim.AdamW(params)
    total_steps = max(1, len(train_loader) * n_epochs)
    sched = get_linear_schedule_with_warmup(opt, int(total_steps * WARMUP_RATIO), total_steps)
    ramp_steps = max(1, int(LAMBDA_RAMP_FRAC * total_steps))

    loss_fn = nn.BCEWithLogitsLoss()
    trig = {int(np.ceil(i * len(train_loader) / EVALS_PER_EPOCH)) for i in range(1, EVALS_PER_EPOCH + 1)}
    best = {"auc": -1.0, "step": -1}
    best_path = out_dir / "best_state.pt"
    hist, step = [], 0

    print(f"[{tag}] n={len(df)} train={tr.sum()} eval={ev.sum()} test={te.sum()} "
          f"channels={cfg['channels']} lambda={lam_max} arch={arch} "
          f"adv_dim={0 if N is None else N.shape[1]}",
          flush=True)

    for ep in range(n_epochs):
        model.train()
        if bn is not None:
            bn.train()
        ep_step = 0
        for batch in train_loader:
            logit, rep = forward_batch(model, batch, device)
            zrep = rep.float()
            if bn is not None:
                logit, zrep = bn(zrep)
            loss = loss_fn(logit, batch["labels"].to(device))
            task_loss = float(loss.item())
            adv_loss_v = 0.0
            if use_adv:
                lam = lam_max * min(1.0, (step + 1) / ramp_steps)
                a_pred = adv(GradReverse.apply(zrep, lam))
                a_loss = nn.functional.mse_loss(a_pred, batch["nuis"].to(device))
                loss = loss + a_loss
                adv_loss_v = float(a_loss.item())
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            ep_step += 1
            if ep_step % 100 == 0:
                print(f"  [{tag}] ep{ep+1} step {ep_step}/{len(train_loader)} "
                      f"task={task_loss:.4f} adv={adv_loss_v:.4f}", flush=True)
            if ep_step in trig:
                p_ev, _, _ = score_split(model, tok, [texts[i] for i in ev_idx], device, bn=bn)
                auc = float(roc_auc_score(y[ev_idx], p_ev)) if len(set(y[ev_idx])) == 2 else float("nan")
                hist.append({"epoch": ep + 1, "epoch_step": ep_step, "global_step": step,
                             "eval_auc": auc, "task_loss": task_loss, "adv_loss": adv_loss_v})
                print(f"  [{tag}] CKPT ep{ep+1} step{ep_step} eval_auc={auc:.4f}", flush=True)
                if np.isfinite(auc) and auc > best["auc"]:
                    best = {"auc": auc, "step": step, "epoch": ep + 1, "epoch_step": ep_step}
                    sd_ = {k: v.detach().cpu().clone()
                           for k, v in model.state_dict().items() if "lora_" in k or "score" in k or "modules_to_save" in k}
                    torch.save(sd_, best_path)
                    if bn is not None:
                        torch.save(bn.state_dict(), out_dir / "best_bn.pt")
                    if use_adv:
                        torch.save(adv.state_dict(), out_dir / "best_adv.pt")

    # ---- reload best checkpoint, final scoring pass over ALL rows -----------
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device), strict=False)
        if bn is not None and (out_dir / "best_bn.pt").exists():
            bn.load_state_dict(torch.load(out_dir / "best_bn.pt", map_location=device))
    else:
        print(f"[{tag}] WARNING: no checkpoint beat -inf eval AUC; scoring FINAL weights", flush=True)
    probs, reps, reps_h = score_split(model, tok, texts, device, want_rep=True, bn=bn)
    res = {"tag": tag, "config": cfg, "n": int(len(df)),
           "recipe": {"model": BASE_MODEL, "max_length": MAX_LENGTH, "batch": BATCH_SIZE,
                      "epochs": n_epochs, "lr": LR, "adv_lr": ADV_LR, "lora": [LORA_R, LORA_ALPHA],
                      "selection_split": "eval", "seed": SEED, "arch": arch,
                      "bn_dim": (BN_DIM if bn is not None else None),
                      "lambda_ramp_frac": LAMBDA_RAMP_FRAC},
           "adv_channels": col_names, "lambda_adv": lam_max,
           "best_checkpoint": best, "history": hist}
    for nm, m in (("train", tr), ("eval", ev), ("test", te)):
        res[f"auc_{nm}"] = float(roc_auc_score(y[m], probs[m]))
        res[f"n_{nm}"] = int(m.sum())
    res["auc_evaltest"] = float(roc_auc_score(y[ev | te], probs[ev | te]))

    # ---- optional token-ablated scoring pass (V1/V3a contribution readout) ---
    probs_abl = None
    tok_str = cfg.get("ablate_token")
    if tok_str:
        stripped = [t[len(tok_str) + 1:] if t.startswith(tok_str + " ") else t for t in texts]
        n_str = sum(1 for a, b in zip(texts, stripped) if a != b)
        probs_abl, _, _ = score_split(model, tok, stripped, device, bn=bn)
        res["ablation"] = {"token": tok_str, "n_rows_stripped": int(n_str)}
        for nm, m in (("eval", ev), ("test", te)):
            res["ablation"][f"auc_{nm}_ablated"] = float(roc_auc_score(y[m], probs_abl[m]))
            res["ablation"][f"delta_{nm}"] = res[f"auc_{nm}"] - res["ablation"][f"auc_{nm}_ablated"]
        res["ablation"]["auc_evaltest_ablated"] = float(roc_auc_score(y[ev | te], probs_abl[ev | te]))
        res["ablation"]["delta_evaltest"] = res["auc_evaltest"] - res["ablation"]["auc_evaltest_ablated"]

    # ---- adversary diagnostics: per-channel R^2 on held-out rows ------------
    if use_adv and (out_dir / "best_adv.pt").exists():
        adv.load_state_dict(torch.load(out_dir / "best_adv.pt", map_location=device))
        adv.eval()
        with torch.no_grad():
            pr = adv(torch.tensor(reps[ev | te].astype(np.float32), device=device)).cpu().numpy()
        tgt = N[ev | te]
        den = ((tgt - tgt.mean(0)) ** 2).sum(0)
        ss = 1.0 - ((tgt - pr) ** 2).sum(0) / np.where(den > 1e-6, den, np.nan)
        # near-constant columns on the held-out slice (a topic cluster with almost no
        # held-out members) have no defined R^2 -- report null rather than a spurious
        # 1e11-magnitude number.
        res["adversary_heldout_R2"] = {c: (float(v) if np.isfinite(v) and den[i] > 1e-6 else None)
                                       for i, (c, v) in enumerate(zip(col_names, ss))}
        res["adversary_heldout_R2_mean_defined"] = float(np.nanmean(ss[np.isfinite(ss)]))

    np.savez_compressed(out_dir / "reps.npz", doc_id=df["doc_id"].astype(str).values,
                        rep=reps, prob=probs.astype(np.float32),
                        rep_h=(reps_h if reps_h is not None else np.zeros(0, dtype=np.float16)),
                        prob_ablated=(probs_abl.astype(np.float32) if probs_abl is not None
                                      else np.zeros(0, dtype=np.float32)),
                        split=split, y=y)
    slim = pd.DataFrame({"doc_id": df["doc_id"].astype(str), "docket": df["docket"],
                         "judgement": y, "split": split, "prob": probs})
    if probs_abl is not None:
        slim["prob_ablated"] = probs_abl
    slim.to_csv(out_dir / "preds_slim.csv", index=False)
    res["runtime_sec"] = round(time.time() - t0, 1)
    (out_dir / "result.json").write_text(json.dumps(res, indent=2))
    print(f"RESULT {tag} " + json.dumps({k: res[k] for k in res if k not in ("history", "config")}), flush=True)
    print(f"[{tag}] DONE in {res['runtime_sec']}s", flush=True)


if __name__ == "__main__":
    main()
