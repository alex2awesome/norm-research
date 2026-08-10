#!/usr/bin/env python3
"""Paired row-level bootstraps for the Direction-3 augmented dense arms on E:
aug vs original dense T, aug vs full-fit bank stack (VA_nl fullfit@E, seed-mean
OOF), and aug_a vs aug_b. Also writes the per-row aligned frame used.

Usage: python3 direction3_boot.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
TD = HERE.parent
DATA = HERE / "dense_data"


def load_module(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


L1 = load_module(TD / "layer1_gemma_cells.py", "l1_fusion_d3b")
D1 = load_module(HERE / "direction1_stack.py", "d1_fusion_d3b")


def aug_preds(cell, arm):
    d = DATA / f"{cell}_aug_{arm}"
    frames = []
    for sp in ("eval", "test"):
        pr = pd.read_csv(d / "rm_out_seed42" / f"preds_{sp}.csv")
        sf = pd.read_csv(d / "split" / f"{sp}.csv")
        assert len(pr) == len(sf) and (pr.judgement.values == sf.judgement.values).all()
        frames.append(pd.DataFrame({"did": sf.did, "y": pr.judgement, f"aug_{arm}": pr.prob}))
    return pd.concat(frames, ignore_index=True)


def main():
    out = {}
    for cell, id_key in (("cap_crowd", "crowd_ids"), ("cap_finalist", "hardneg_ids")):
        dense = D1.caption_dense(cell)
        c = L1._caption_pools()
        ids_all = sorted(x for x in c[id_key] if x in c["X_by_id"])
        # full-population VA_nl OOF (seed-mean), Layer-1 protocol
        _, y_all, groups_all, lin_oof, nl_oof = load_fullpop(cell)
        row_of = {d_: i for i, d_ in enumerate(ids_all)}

        df = aug_preds(cell, "a").merge(aug_preds(cell, "b")[["did", "aug_b"]], on="did")
        df["T_orig"] = [dense[d_][0] for d_ in df.did]
        df["va_nl_fullfit"] = [nl_oof[row_of[d_]] for d_ in df.did]
        assert all(not dense[d_][2] for d_ in df.did)  # all E rows out-of-dense-train
        y_chk = np.array([c["y_crowd" if cell == "cap_crowd" else "y_fin"][d_] for d_ in df.did])
        assert (y_chk == df.y.values).all()

        y = df.y.values
        res = {"n_E": int(len(df))}
        for col in ("T_orig", "va_nl_fullfit", "aug_a", "aug_b"):
            res[f"auc_{col}"] = float(roc_auc_score(y, df[col]))
        for a, b in (("aug_a", "T_orig"), ("aug_a", "va_nl_fullfit"),
                     ("aug_b", "va_nl_fullfit"), ("aug_a", "aug_b")):
            res[f"boot_{a}_minus_{b}"] = D1.paired_boot(y, df[a].values, df[b].values)
        out[cell] = res
        print(cell, json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in res.items() if not k.startswith("boot")}, indent=1))
        for k, v in res.items():
            if k.startswith("boot"):
                print(f"  {k}: {v['estimate']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] "
                      f"P(>0)={v['p_gt_0']:.3f}")
    (HERE / "direction3_boot.json").write_text(json.dumps(out, indent=2))
    print("wrote", HERE / "direction3_boot.json")


def load_fullpop(cell):
    d = L1.CELLS[cell]["loader"]()
    mats, y, groups = d["mats"], d["y"], d["groups"]
    folds = L1.outer_folds(len(y), groups, n_splits=5)
    _, lin_oof = L1.linear_oof_family2(mats["VA"], y, groups, folds)
    nl = np.mean([L1.gbm_oof_raw(mats["VA"], y, groups, folds, s)["oof"] for s in (0, 1, 2)], axis=0)
    return d, y, groups, lin_oof, nl


if __name__ == "__main__":
    main()
