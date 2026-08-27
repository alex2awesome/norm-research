#!/usr/bin/env python3
"""Within-claim paired readout for the patents dense-8B run: for each held-out claim (uid),
does the trained model score the examiner's gold span above the same-claim filler span?
This is the readout comparable to the doctrine bank (.573 @4b / .594 @12b) and the
visible-disclosure oracle ceiling (~.62).

Run from methods/dense (needs train_reward_model.score_texts + the saved best_model):
  $HOME/envs/ai_usage/bin/python eval_patents_within_claim.py \
      --model_dir .../outputs/dense8b/patents/run/best_model \
      --test_csv  .../outputs/dense8b/patents/splits/test.csv
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
    ap.add_argument("--max_length", type=int, default=640)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = pd.read_csv(a.test_csv)
    probs = np.array(score_texts(a.model_dir, df.text.tolist(),
                                 max_length=a.max_length, batch_size=a.batch_size))
    df["prob"] = probs
    y = df.judgement.astype(int).values
    pooled = roc_auc_score(y, probs)
    wins = ties = tot = 0
    for uid, g in df.groupby("uid"):
        pos = g[g.judgement == 1].prob.values
        neg = g[g.judgement == 0].prob.values
        if len(pos) == 0 or len(neg) == 0: continue
        tot += 1
        if pos.max() > neg.max(): wins += 1
        elif pos.max() == neg.max(): ties += 1
    within = (wins + 0.5 * ties) / max(tot, 1)
    rej = {}
    if "rejection_type" in df:
        for rt, g in df.groupby("rejection_type"):
            w = t2 = 0
            for uid, gg in g.groupby("uid"):
                p, n = gg[gg.judgement == 1].prob.values, gg[gg.judgement == 0].prob.values
                if len(p) and len(n):
                    t2 += 1; w += (p.max() > n.max()) + 0.5 * (p.max() == n.max())
            if t2 >= 30: rej[str(rt)] = round(w / t2, 4)
    rep = dict(n_pairs=len(df), n_claims=tot, pooled_auc=round(float(pooled), 4),
               within_claim=round(float(within), 4), by_rejection_type=rej)
    out = a.out or a.test_csv.replace("test.csv", "within_claim_report.json")
    json.dump(rep, open(out, "w"), indent=2)
    df[["uid", "app_id", "judgement", "prob"]].to_csv(
        a.test_csv.replace("test.csv", "test_scored.csv"), index=False)
    print(json.dumps(rep, indent=2))
    print("WITHIN_CLAIM_DONE", flush=True)

if __name__ == "__main__":
    main()
