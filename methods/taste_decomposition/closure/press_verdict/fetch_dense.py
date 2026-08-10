#!/usr/bin/env python3
"""Assemble the press cell's per-row dense (T) predictions from the sk3 dense-standard
run into one local CSV.

Source: `datasets/press-releases/dense_standard_k3/` on sk3 --
  split/{eval,test}.csv        text, judgement, group, row_id
  rm_out_seed{42,1,2}/preds_{eval,test}.csv   judgement, prob, group   (NO row key)

The preds files carry no id, so the join is POSITIONAL against the split file of the
same name.  That is verified, not assumed: for every seed and every split this script
asserts that the preds file has the same length as the split file and that its
`group` and `judgement` columns match the split file's element-for-element.  If the
files were ever reordered the assertion fires.

Run from the repo after the six preds files have been scp'd into --src.
CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
SEEDS = (42, 1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir holding {eval,test}.csv and seed{S}_{split}.csv")
    a = ap.parse_args()
    src = Path(a.src)

    rows = []
    for sp in ("eval", "test"):
        S = pd.read_csv(src / f"{sp}.csv")
        preds = {}
        for s in SEEDS:
            P = pd.read_csv(src / f"seed{s}_{sp}.csv")
            assert len(P) == len(S), f"seed{s}/{sp}: {len(P)} preds vs {len(S)} split rows"
            assert (P["group"].astype(str).values == S["group"].astype(str).values).all(), \
                f"seed{s}/{sp}: group column does not match the split file positionally"
            assert (P["judgement"].values == S["judgement"].values).all(), \
                f"seed{s}/{sp}: judgement column does not match the split file positionally"
            preds[s] = P["prob"].astype(float).values
        for i in range(len(S)):
            rows.append({"id": str(S["row_id"].iloc[i]), "dense_split": sp,
                         "group": str(S["group"].iloc[i]),
                         "judgement": int(S["judgement"].iloc[i]),
                         **{f"p{s}": float(preds[s][i]) for s in SEEDS}})
    D = pd.DataFrame(rows)
    assert D["id"].is_unique
    D.to_csv(HERE / "press_dense_preds_3seed.csv", index=False)

    rep = {"n": len(D), "n_groups": int(D["group"].nunique()),
           "by_split": {sp: int((D.dense_split == sp).sum()) for sp in ("eval", "test")},
           "per_seed_auc": {}, "join": "positional against split/{eval,test}.csv, asserted"}
    for s in SEEDS:
        rep["per_seed_auc"][str(s)] = {
            "eval": float(roc_auc_score(D[D.dense_split == "eval"].judgement,
                                        D[D.dense_split == "eval"][f"p{s}"])),
            "test": float(roc_auc_score(D[D.dense_split == "test"].judgement,
                                        D[D.dense_split == "test"][f"p{s}"])),
            "pooled": float(roc_auc_score(D.judgement, D[f"p{s}"]))}
    rep["T_eval_mean_of_seed_AUCs"] = float(np.mean([rep["per_seed_auc"][str(s)]["eval"] for s in SEEDS]))
    rep["T_test_mean_of_seed_AUCs"] = float(np.mean([rep["per_seed_auc"][str(s)]["test"] for s in SEEDS]))
    rep["T_HONEST_mean_of_seed_AUCs"] = float(np.mean([rep["per_seed_auc"][str(s)]["pooled"] for s in SEEDS]))
    rep["T_HONEST_seed_ensemble_NOT_QUOTED"] = float(
        roc_auc_score(D.judgement, D[[f"p{s}" for s in SEEDS]].mean(axis=1)))
    (HERE / "press_dense_preds_3seed.report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
