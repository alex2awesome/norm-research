#!/usr/bin/env python3
"""Harvest Directions 2+3: compute T_moredata / T_aug_a / T_aug_b for the two
caption cells on the SAME evaluation-valid rows E as the Direction-1 table
(E = dense eval+test rows; identical row sets across all arms by construction:
Direction 2 changed only train, Direction 3 kept every split's rows).

Reads preds_{eval,test}.csv (index-aligned with each dataset dir's
split/{eval,test}.csv) that the sk3 chain's scoring pass wrote, after they are
rsync'd into fusion/dense_data/<name>/rm_out_seed42/.

Usage: python3 harvest_direction23.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
DATA = HERE / "dense_data"
SLIM = HERE.parent / "closure" / "samerows_preds"
CAP = HERE.parents[3] / "datasets/humor/caption_multiy"  # not used; kept for provenance
ORIG = {"cap_crowd": "crowd", "cap_finalist": "finalist"}


def orig_split_frames(cell):
    dl = Path(__file__).resolve().parents[3] / "datasets/humor/caption_multiy/dense_llama" / ORIG[cell]
    return {s: pd.read_csv(dl / "split" / f"{s}.csv") for s in ("eval", "test")}


def arm_auc(cell, name, out):
    d = DATA / name
    res = {}
    frames = []
    for sp in ("eval", "test"):
        pr = pd.read_csv(d / "rm_out_seed42" / f"preds_{sp}.csv")
        sf = pd.read_csv(d / "split" / f"{sp}.csv")
        assert len(pr) == len(sf), (name, sp, len(pr), len(sf))
        assert (pr.judgement.values == sf.judgement.values).all(), f"{name}/{sp} y misaligned"
        res[f"auc_{sp}_only"] = float(roc_auc_score(pr.judgement, pr.prob))
        res[f"n_{sp}"] = int(len(pr))
        frames.append(pr[["judgement", "prob"]])
    both = pd.concat(frames, ignore_index=True)
    res["auc_E_pooled"] = float(roc_auc_score(both.judgement, both.prob))
    res["n_E"] = int(len(both))
    out[name] = res
    return res


def main():
    # gate: the original same-rows T on E, from the slim preds (held-out rows only)
    out = {}
    for cell in ("cap_crowd", "cap_finalist"):
        slim = pd.read_csv(SLIM / f"{cell}_dense_preds_slim.csv")
        ho = slim[~slim.in_dense_train]
        out[f"{cell}_ORIGINAL"] = {
            "auc_E_pooled": float(roc_auc_score(ho.judgement, ho.dense_prob)),
            "auc_eval_only": float(roc_auc_score(*[g for g in [
                ho[ho.dense_split == "eval"].judgement, ho[ho.dense_split == "eval"].dense_prob]])),
            "auc_test_only": float(roc_auc_score(ho[ho.dense_split == "test"].judgement,
                                                 ho[ho.dense_split == "test"].dense_prob)),
            "n_E": int(len(ho)),
        }
        # sanity: the arm eval/test rows must be the same rows as slim's held-out rows
        sf = orig_split_frames(cell)
        assert len(sf["eval"]) + len(sf["test"]) == len(ho), cell

        for arm in ("moredata", "aug_a", "aug_b"):
            name = f"{cell}_{arm}"
            if (DATA / name / "rm_out_seed42" / "preds_eval.csv").exists():
                r = arm_auc(cell, name, out)
                print(name, json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                        for k, v in r.items()}))
            else:
                print(name, "preds not present yet -- skipped")

    (DATA / "harvest_results.json").write_text(json.dumps(out, indent=2))
    print("wrote", DATA / "harvest_results.json")


if __name__ == "__main__":
    main()
