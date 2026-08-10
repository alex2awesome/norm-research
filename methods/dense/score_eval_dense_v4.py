#!/usr/bin/env python3
"""Clean-eval pass for the V4 dense-standard remaining cells (HashtagWars
verdict, Style Invitational top-tier, patents claim-fell). Eval split was used
for checkpoint selection (--selection_split eval); this pass re-scores eval AND
test with the selected best_model, same pattern as methods/dense/score_eval_pr_v2.py
(num_labels=1, sigmoid(logits[:,0])).

Usage: python3 score_eval_dense_v4.py --dir <dense_standard dir> [--name NAME]
"""
import argparse
import glob
import json
import os

import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Default 1024 = the frozen dense-standard budget; every existing caller leaves
# the env unset and is therefore unaffected. The override exists for arms that
# deliberately train at a reduced document budget (the V3 raw-matched
# displacement controls), where scoring at 1024 would not match training.
MAXLEN = int(os.environ.get("DENSE_SCORE_MAXLEN", "1024"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dense_standard dir, e.g. .../hashtagwars/dense_standard")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    D = args.dir
    name = args.name or os.path.basename(os.path.dirname(D.rstrip("/")))

    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    out = {}
    for run in sorted(glob.glob(f"{D}/rm_out_seed*")):
        if not os.path.isdir(run) or not os.path.exists(f"{run}/best_model"):
            continue
        seed = os.path.basename(run).replace("rm_out_seed", "")
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
            preds_cols = {"judgement": df.judgement, "prob": probs}
            if "group" in df.columns:
                preds_cols["group"] = df.group
            pd.DataFrame(preds_cols).to_csv(f"{run}/preds_{split}.csv", index=False)
            print(f"[{name}] seed{seed} {split} AUC={auc:.4f} n={len(df)}", flush=True)
        out[f"seed{seed}"] = res
        del model, base
        torch.cuda.empty_cache()

    result_path = f"{D}/eval_pass_results.json"
    json.dump(out, open(result_path, "w"), indent=2)
    print(json.dumps(out, indent=2), flush=True)
    print(f"DENSE_V4_EVAL_PASS_DONE [{name}] -> {result_path}", flush=True)


if __name__ == "__main__":
    main()
