#!/usr/bin/env python3
"""Score a frozen best_model on test.csv EXACTLY ONCE and report AUC (plus per-group AUCs
for any grouping columns present). Companion to train_reward_model.py --selection_split eval:
checkpoint selection happens on eval; this is the single honest test readout.

Run from methods/dense:
  $PY eval_test_once.py --model_dir .../run/best_model --test_csv .../splits/test.csv --max_length 512
"""
import argparse, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from train_reward_model import score_texts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=64)
    a = ap.parse_args()
    df = pd.read_csv(a.test_csv)
    probs = np.array(score_texts(a.model_dir, df.text.tolist(),
                                 max_length=a.max_length, batch_size=a.batch_size))
    y = df.judgement.astype(int).values
    rep = {"n": len(df), "test_auc": round(float(roc_auc_score(y, probs)), 4)}
    for gcol in ("outlet", "rejection_type"):
        if gcol in df:
            per = {}
            for g, s in df.assign(p=probs).groupby(gcol):
                if s.judgement.nunique() == 2 and len(s) >= 80:
                    per[str(g)] = round(float(roc_auc_score(s.judgement, s.p)), 4)
            if per: rep[f"by_{gcol}"] = per
    df.assign(prob=probs).drop(columns=["text"]).to_csv(
        a.test_csv.replace("test.csv", "test_scored.csv"), index=False)
    out = a.test_csv.replace("test.csv", "test_once_report.json")
    json.dump(rep, open(out, "w"), indent=2)
    print("TEST_ONCE", json.dumps(rep))

if __name__ == "__main__":
    main()
