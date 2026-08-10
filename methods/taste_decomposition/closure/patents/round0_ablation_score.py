#!/usr/bin/env python3
"""ROUND-0 AUDIT (GPU leg): text-ablation re-scoring of the patents claim-fell dense model.

The construction of this corpus is LABEL-CONDITIONAL (forensic audit 2026-07-07,
memory `patents-prior-art-pipeline`): a POSITIVE row's 8 candidate references are
7 same-CPC FAISS fillers plus the examiner's actual gold reference APPENDED LAST
(gold in the last slot 80.4% of the time); a NEGATIVE row's references contain NO
gold reference at all (0% by design). The dense reader therefore *could* reach a
high AUC by detecting "an examiner-cited document is present in this list" -- or
even just "slot 8 looks different" -- without doing any claim-to-reference
entailment at all.

This script re-scores the EVAL split with the ALREADY-TRAINED seed-42 model under
text variants that dissociate those explanations. No retraining: the question is
what the trained model is keying on.

  full            original text (reproduction gate: must match preds_eval.csv)
  shuffle_refs    references re-ordered by a deterministic per-row permutation
                  -> kills POSITION, keeps content
  drop_last       last reference deleted
  keep_last       claim element + last reference only
  element_only    claim element, no references     -> claim-text-alone channel
  refs_only       references only, no claim element
                  -> THE DECISIVE TEST. Entailment is impossible without the claim.
                     If refs_only ~ full, the model is fingerprinting reference
                     PROVENANCE, not judging disclosure.
  refs_only_shuf  references only, shuffled        -> provenance minus position
  drop_gold       the gold reference removed where present (pos rows only;
                  reported separately since it is label-conditional by definition)

Usage (sk3):
  python round0_ablation_score.py --gpu 3 [--split eval]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

BASE = Path("/lfs/skampere3/0/alexspan/norm-research")
JL = BASE / "datasets/patents/processed/option3_claims_gemma_scale.jsonl"
DS = BASE / "datasets/patents/dense_standard"
OUT = Path(__file__).resolve().parent
csv.field_size_limit(sys.maxsize)
MAXLEN = 1024


def build_text_from(element, refs):
    parts = [f"CLAIM ELEMENT:\n{element}"]
    for i, ref in enumerate(refs):
        parts.append(f"REFERENCE {i + 1} (patent {ref['doc_id']}):\n{ref['spans']}")
    return "\n\n".join(parts)


def perm(key, n):
    """Deterministic per-row permutation (stable hash order, never a seeded shuffle)."""
    return sorted(range(n), key=lambda i: hashlib.sha256(f"{key}|{i}".encode()).hexdigest())


def variants(r):
    el = r["element"] or ""
    refs = [{"doc_id": q.get("doc_id", "?"), "spans": " ".join(q.get("spans") or []),
             "is_gold": bool(q.get("is_gold"))} for q in (r.get("refs") or [])]
    key = f"{r['app_id']}|{r['claim_num']}|{r.get('rejection_type')}"
    p = perm(key, len(refs))
    sref = [refs[i] for i in p]
    v = {
        "full": build_text_from(el, refs),
        "shuffle_refs": build_text_from(el, sref),
        "drop_last": build_text_from(el, refs[:-1]) if len(refs) > 1 else build_text_from(el, refs),
        "keep_last": build_text_from(el, refs[-1:]) if refs else build_text_from(el, refs),
        "element_only": f"CLAIM ELEMENT:\n{el}",
        "refs_only": build_text_from("", refs).replace("CLAIM ELEMENT:\n\n\n", ""),
        "refs_only_shuf": build_text_from("", sref).replace("CLAIM ELEMENT:\n\n\n", ""),
        "drop_gold": build_text_from(el, [q for q in refs if not q["is_gold"]] or refs),
    }
    return v


# decisive variants first (element_only / refs_only), so a truncated run still answers
# the mechanism question; `full` stays first as the reproduction gate.
VARIANTS = ["full", "element_only", "refs_only", "shuffle_refs", "drop_last",
            "keep_last", "refs_only_shuf", "drop_gold"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--seed", default="42")
    ap.add_argument("--batch", type=int, default=16)
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

    print("loading jsonl ...", flush=True)
    jrows = [json.loads(l) for l in open(JL) if l.strip()]
    thash = {}
    for i, r in enumerate(jrows):
        h = hashlib.sha1(build_text_from(r["element"] or "", [
            {"doc_id": q.get("doc_id", "?"), "spans": " ".join(q.get("spans") or [])}
            for q in (r.get("refs") or [])]).encode()).hexdigest()
        thash.setdefault(h, []).append(i)

    d = pd.read_csv(DS / "split" / f"{args.split}.csv")
    ptr, idxs = {}, []
    for t in d["text"].astype(str).values:
        h = hashlib.sha1(t.encode()).hexdigest()
        lst = thash[h]
        k = ptr.get(h, 0)
        idxs.append(lst[k] if k < len(lst) else lst[-1])
        ptr[h] = k + 1
    y = d["judgement"].to_numpy()
    assert all(((1 if jrows[j]["label"] == "pos" else 0) == yy) for j, yy in zip(idxs, y))
    print(f"aligned {len(idxs)} {args.split} rows", flush=True)

    texts = {v: [] for v in VARIANTS}
    has_gold = []
    for j in idxs:
        vv = variants(jrows[j])
        for v in VARIANTS:
            texts[v].append(vv[v])
        has_gold.append(int(any(bool(q.get("is_gold")) for q in (jrows[j].get("refs") or []))))
    has_gold = np.array(has_gold)
    # reproduction gate on the untouched variant
    assert all(a == b for a, b in zip(texts["full"], d["text"].astype(str).values)), \
        "full-variant text does not reproduce the split CSV"

    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    run = DS / f"rm_out_seed{args.seed}"
    base = AutoModelForSequenceClassification.from_pretrained(
        "meta-llama/Llama-3.1-8B", num_labels=1, torch_dtype=torch.bfloat16, device_map="cuda:0")
    base.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(base, f"{run}/best_model").eval()

    res, allp = {}, {}
    for v in VARIANTS:
        probs = []
        with torch.no_grad():
            for i in range(0, len(texts[v]), args.batch):
                b = tok(texts[v][i:i + args.batch], truncation=True, max_length=MAXLEN,
                        padding=True, return_tensors="pt").to("cuda:0")
                probs.extend(torch.sigmoid(model(**b).logits.float()[:, 0]).cpu().tolist())
        p = np.array(probs)
        allp[v] = p
        res[v] = {"auc": round(float(roc_auc_score(y, p)), 4),
                  "mean": round(float(p.mean()), 4), "std": round(float(p.std()), 4),
                  "spearman_vs_full": round(float(pd.Series(p).corr(
                      pd.Series(allp["full"]), method="spearman")), 4)}
        print(f"[abl] {args.split} {v:16s} AUC={res[v]['auc']:.4f} mean={res[v]['mean']:.4f}",
              flush=True)
        # checkpoint after every variant so a truncated run is still usable
        np.savez_compressed(OUT / f"round0_ablation_{args.split}_probs.npz",
                            y=y, has_gold=has_gold, **allp)
        json.dump({"split": args.split, "seed": args.seed, "n": int(len(y)),
                   "variants": res, "PARTIAL": True},
                  open(OUT / f"round0_ablation_{args.split}.json", "w"), indent=2)

    # gold-conditional readouts: is the model detecting gold PRESENCE?
    g = has_gold == 1
    res["_gold_presence"] = {
        "frac_rows_with_gold": round(float(g.mean()), 4),
        "auc_of_has_gold_alone": round(float(roc_auc_score(y, has_gold)), 4),
        "full_auc_within_gold_rows": (round(float(roc_auc_score(y[g], allp["full"][g])), 4)
                                      if len(set(y[g])) > 1 else None),
        "full_auc_within_nogold_rows": (round(float(roc_auc_score(y[~g], allp["full"][~g])), 4)
                                        if len(set(y[~g])) > 1 else None),
        "mean_prob_gold_rows": round(float(allp["full"][g].mean()), 4),
        "mean_prob_nogold_rows": round(float(allp["full"][~g].mean()), 4),
    }
    np.savez_compressed(OUT / f"round0_ablation_{args.split}_probs.npz",
                        y=y, has_gold=has_gold, **allp)
    json.dump({"split": args.split, "seed": args.seed, "n": int(len(y)), "variants": res},
              open(OUT / f"round0_ablation_{args.split}.json", "w"), indent=2)
    print(json.dumps(res, indent=2), flush=True)
    print("ROUND0_ABLATION_DONE", flush=True)


if __name__ == "__main__":
    main()
