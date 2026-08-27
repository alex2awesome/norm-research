#!/usr/bin/env python3
"""ROUND-0, part 6: the FAIR ablation -- content swaps that preserve length and format.

`round0_ablation_score.py` deletes parts of the input. That is confounded: a model
trained on ~1024-token element+8-reference blocks is off-distribution when handed a
50-token bare claim element, so its AUC drop mixes "signal lost" with "input shape
changed" (`element_only` mean predicted probability jumped from .585 to .789 -- the
model is not merely less informed, it is miscalibrated by the shift).

The fair test keeps the input SHAPE identical and destroys only the CORRESPONDENCE:

  refs_swapped     this row's 8 references replaced by another row's 8 references.
                   Same format, near-identical length, but the references no longer
                   have anything to do with this claim. **If the model is judging
                   whether the prior art discloses the claim, this must destroy the
                   signal. If the AUC barely moves, it is not doing entailment.**
  element_swapped  this row's claim element replaced by another row's, references kept.
                   The mirror test: if the AUC barely moves, the claim text does not
                   matter either.
  both_swapped     both replaced from the same donor row -- a coherent but wrong
                   (claim, references) pair; the sanity floor.

Donors are chosen by a deterministic stable-hash derangement over the eval rows
(never a seeded shuffle), matched on reference count so lengths stay comparable.

Usage (sk3):  python round0_swap_ablation.py --gpu 3 [--split eval]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path("/lfs/skampere3/0/alexspan/norm-research")
JL = BASE / "datasets/patents/processed/option3_claims_gemma_scale.jsonl"
DS = BASE / "datasets/patents/dense_standard"
OUT = Path(__file__).resolve().parent
csv.field_size_limit(sys.maxsize)
MAXLEN = 1024


def compose(element, refs):
    parts = [f"CLAIM ELEMENT:\n{element}"]
    for i, (doc, spans) in enumerate(refs):
        parts.append(f"REFERENCE {i + 1} (patent {doc}):\n{spans}")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--seed", default="42")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("HF_HOME", "/lfs/skampere3/0/alexspan/.cache/huggingface")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import numpy as np
    import pandas as pd
    import torch
    from peft import PeftModel
    from sklearn.metrics import roc_auc_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    J = [json.loads(l) for l in open(JL) if l.strip()]
    th = defaultdict(list)
    for i, r in enumerate(J):
        th[hashlib.sha1(compose(r["element"] or "", [
            (q.get("doc_id", "?"), " ".join(q.get("spans") or []))
            for q in (r.get("refs") or [])]).encode()).hexdigest()].append(i)
    d = pd.read_csv(DS / "split" / f"{args.split}.csv")
    ptr, idxs = {}, []
    for t in d["text"].astype(str).values:
        h = hashlib.sha1(t.encode()).hexdigest(); lst = th[h]; k = ptr.get(h, 0)
        idxs.append(lst[k] if k < len(lst) else lst[-1]); ptr[h] = k + 1
    y = d["judgement"].to_numpy()
    els = [J[j]["element"] or "" for j in idxs]
    rfs = [[(q.get("doc_id", "?"), " ".join(q.get("spans") or []))
            for q in (J[j].get("refs") or [])] for j in idxs]

    # deterministic derangement within reference-count buckets (stable hash order)
    donor = np.arange(len(idxs))
    by_k = defaultdict(list)
    for i, r in enumerate(rfs):
        by_k[len(r)].append(i)
    for k, members in by_k.items():
        order = sorted(members, key=lambda i: hashlib.sha256(f"patents-swap|{i}".encode()).hexdigest())
        shift = max(1, len(order) // 3)
        for a, b in zip(order, order[shift:] + order[:shift]):
            donor[a] = b
    assert (donor != np.arange(len(idxs))).all() or len(idxs) < 3, "derangement has fixed points"

    texts = {
        "original": [compose(els[i], rfs[i]) for i in range(len(idxs))],
        "refs_swapped": [compose(els[i], rfs[donor[i]]) for i in range(len(idxs))],
        "element_swapped": [compose(els[donor[i]], rfs[i]) for i in range(len(idxs))],
        "both_swapped": [compose(els[donor[i]], rfs[donor[i]]) for i in range(len(idxs))],
    }
    assert texts["original"] == list(d["text"].astype(str).values), "reproduction gate failed"

    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForSequenceClassification.from_pretrained(
        "meta-llama/Llama-3.1-8B", num_labels=1, torch_dtype=torch.bfloat16, device_map="cuda:0")
    base.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(base, f"{DS}/rm_out_seed{args.seed}/best_model").eval()

    res, allp = {}, {}
    for v, tx in texts.items():
        probs = []
        with torch.no_grad():
            for i in range(0, len(tx), args.batch):
                b = tok(tx[i:i + args.batch], truncation=True, max_length=MAXLEN,
                        padding=True, return_tensors="pt").to("cuda:0")
                probs.extend(torch.sigmoid(model(**b).logits.float()[:, 0]).cpu().tolist())
        p = np.array(probs); allp[v] = p
        res[v] = {"auc": round(float(roc_auc_score(y, p)), 4),
                  "mean_prob": round(float(p.mean()), 4), "sd_prob": round(float(p.std()), 4),
                  "mean_chars": int(np.mean([len(t) for t in tx])),
                  "spearman_vs_original": round(float(pd.Series(p).corr(
                      pd.Series(allp["original"]), method="spearman")), 4)}
        print(f"[swap] {args.split} {v:16s} AUC={res[v]['auc']:.4f} "
              f"mean={res[v]['mean_prob']:.4f} chars={res[v]['mean_chars']}", flush=True)
        np.savez_compressed(OUT / f"round0_swap_{args.split}_probs.npz", y=y, **allp)
        json.dump({"split": args.split, "seed": args.seed, "n": int(len(y)), "variants": res},
                  open(OUT / f"round0_swap_{args.split}.json", "w"), indent=2)
    print(json.dumps(res, indent=2), flush=True)
    print("ROUND0_SWAP_ABLATION_DONE", flush=True)


if __name__ == "__main__":
    main()
