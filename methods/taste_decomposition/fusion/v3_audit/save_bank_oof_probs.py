#!/usr/bin/env python3
"""Save per-row full-population OOF probs for the caption bank controls
(full-bank VA_nl seed-mean + top-20-columns-alone), keyed by did, so the
harvest can bootstrap dense arms against the bank on E. Same protocol as
bank_topk_oof.py (= the fusion note's VA_nl fullfit@E)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TD = HERE.parent.parent

def load_module(path, alias):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod

L1 = load_module(TD / "layer1_gemma_cells.py", "l1_saveoof")
capagg = sys.modules["capagg_taste_decomp"]


def gbm_oof_seedmean(X, y, groups):
    folds = L1.outer_folds(len(y), groups)
    return np.mean([L1.gbm_oof_raw(X, y, groups, folds, s)["oof"]
                    for s in L1.GBM_SEEDS], axis=0)


def main():
    c = L1._caption_pools()
    for cell, id_key, y_key in (("cap_crowd", "crowd_ids", "y_crowd"),
                                ("cap_finalist", "hardneg_ids", "y_fin")):
        rank = json.loads((HERE / f"importance_full_{cell}.json").read_text())["ranking"]
        ids_all = sorted(d for d in c[id_key] if d in c["X_by_id"])
        A = np.array([c["X_by_id"][d] for d in ids_all], dtype=float)
        V = np.array([c["V_by_id"][d] for d in ids_all], dtype=float)
        Ac, _ = capagg.clean_cols(A)
        Vc, _ = capagg.clean_cols(V)
        VA = np.column_stack([Vc, Ac])
        y = np.array([c[y_key][d] for d in ids_all])
        groups = np.array([str(c["contest_by_id"][d]) for d in ids_all])
        out = pd.DataFrame({"did": ids_all, "y": y})
        out["bank_full_nl"] = gbm_oof_seedmean(VA, y, groups)
        cols20 = [r["col"] for r in rank[:20]]
        out["bank_top20_nl"] = gbm_oof_seedmean(VA[:, cols20], y, groups)
        out.to_csv(HERE / f"bank_oof_probs_{cell}.csv", index=False)
        print("saved", cell, len(out))
    print("SAVE_BANK_OOF_DONE")


if __name__ == "__main__":
    main()
