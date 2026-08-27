#!/usr/bin/env python3
"""Score ONE dense-standard v3 seed on eval+test, byte-for-byte the same inference
pattern as methods/dense/score_eval_pr_v3.py (which produced the seed-42 numbers):
num_labels=1, bf16, max_length 2048, batch 16, sigmoid(logits[:,0]).

Difference from score_eval_pr_v3.py: it scores ONLY the seed named in $SEED and MERGES
its result into eval_pass_results.json instead of rewriting the file, so the seed-42
entry (eval .6488 / test .7373) and its stored preds are never touched.
"""
import json, os
import pandas as pd, torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

D = "/lfs/skampere3/0/alexspan/norm-research/datasets/code-review/dense_standard_v3"
MAXLEN = 2048
seed = os.environ["SEED"]
run = f"{D}/rm_out_seed{seed}"
assert os.path.exists(f"{run}/best_model"), f"no best_model in {run}"

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

base = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B", num_labels=1, torch_dtype=torch.bfloat16, device_map="cuda:0")
base.config.pad_token_id = tok.pad_token_id
model = PeftModel.from_pretrained(base, f"{run}/best_model").eval()

res = {}
for split in ("eval", "test"):
    df = pd.read_csv(f"{D}/split/{split}.csv")
    probs = []
    with torch.no_grad():
        for i in range(0, len(df), 16):
            b = tok(list(df.text.iloc[i:i + 16].astype(str)), truncation=True,
                    max_length=MAXLEN, padding=True, return_tensors="pt").to("cuda:0")
            probs.extend(torch.sigmoid(model(**b).logits.float()[:, 0]).cpu().tolist())
    auc = roc_auc_score(df.judgement.values, probs)
    res[f"{split}_auc"] = round(float(auc), 4)
    res[f"n_{split}"] = int(len(df))
    pd.DataFrame({"repo": df.repo, "pr_number": df.pr_number,
                  "judgement": df.judgement, "prob": probs}
                 ).to_csv(f"{run}/preds_{split}.csv", index=False)
    print(f"seed{seed} {split} AUC={auc:.4f} n={len(df)}", flush=True)

p = f"{D}/eval_pass_results.json"
allres = json.load(open(p)) if os.path.exists(p) else {}
allres[f"seed{seed}"] = res
json.dump(allres, open(p, "w"), indent=2)
print(json.dumps(allres, indent=2), flush=True)
print(f"CODE_V3_SEED{seed}_SCORED", flush=True)
