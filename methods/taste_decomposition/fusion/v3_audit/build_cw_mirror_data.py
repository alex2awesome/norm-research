#!/usr/bin/env python3
"""V3 mirror test on the dense-STRONG cell: CW community (7,008-row honest
population, terminal 144-column bank incl. the closure campaign's surviving
mined criteria — round7_state.npz).

Arms:
  cw_raw     text as-is (matched-n control at 4,794 train rows)
  cw_augk20 / cw_augk40   "VA metrics:" block PREPENDED, then the original text
             verbatim. Prepended because 35.6%% of stories exceed max_len 1024
             and the tokenizer truncates the RIGHT side — an appended block
             (caption format) would be silently deleted on exactly the long
             rows. NaN judge scores render as "NA" (0.7%% of entries).

Splits: FIT+MINE -> train (4,794), MONITOR -> eval (1,114), TEST -> test
(1,100), group = prompt_id (the dense split was prompt-grouped and every row
is held out by the ORIGINAL wp_clean dense model, so T_big per-row preds are
honest on all 7,008 rows; our new models are honest on eval+test only).

Importance: FIT+MINE ONLY, frozen protocol (GroupKFold(3), HistGB GRID[1]
seed 0, permutation_importance roc_auc n_repeats=5, mean over folds).
Criterion scores are label-blind Gemma judge outputs; y never in the prompt.

CPU only. Usage: python3 build_cw_mirror_data.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
FUS = HERE.parent
TD = FUS.parent
CW = TD / "closure" / "cw_community"
OUT = FUS / "dense_data"


def load_module(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


L1 = load_module(TD / "layer1_gemma_cells.py", "l1_cw_mirror")


def fmt(v):
    if np.isnan(v):
        return "NA"
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return f"{v:.3f}".rstrip("0").rstrip(".")


def main():
    d = np.load(CW / "round7_state.npz", allow_pickle=True)
    df = pd.read_csv(CW / "cw_population_with_splits.csv")
    VA, y, ids = d["VA"], d["y"].astype(int), [str(x) for x in d["ids"]]
    names = [str(x) for x in d["bank_names"]]
    groups = d["groups"].astype(str)
    split = pd.Series(d["split"]).astype(str).values
    assert (df.id.astype(str).values == np.array(ids)).all()
    assert (df.judgement.values == y).all()
    assert (df.prompt_id.astype(str).values == groups).all()

    # ---- FIT+MINE-only importance (imputed copy for the GBM; prompts keep NA)
    tr = split == "fit_mine"
    col_med = np.nanmedian(VA[tr], axis=0)
    VAi = np.where(np.isnan(VA), col_med[None, :], VA)
    Xtr, ytr, gtr = VAi[tr], y[tr], groups[tr]
    imps = np.zeros(VA.shape[1])
    nf = 0
    for itr, ite in GroupKFold(n_splits=3).split(Xtr, ytr, gtr):
        m = L1._fit_gbm(L1.GRID[1], seed=0)
        m.fit(Xtr[itr], ytr[itr])
        r = permutation_importance(m, Xtr[ite], ytr[ite], scoring="roc_auc",
                                   n_repeats=5, random_state=0, n_jobs=-1)
        imps += r.importances_mean
        nf += 1
    imps /= nf
    order = np.argsort(-imps)
    (HERE / "importance_full_cw_community.json").write_text(json.dumps({
        "n_cols_total": int(VA.shape[1]),
        "protocol": "FIT+MINE-only GroupKFold(3), HistGB GRID[1] seed0, "
                    "permutation_importance roc_auc n_repeats=5, mean over folds",
        "ranking": [{"rank": r + 1, "name": names[j], "col": int(j),
                     "importance": float(imps[j])} for r, j in enumerate(order)],
    }, indent=2))
    print("[cw] top-20 by FIT+MINE permutation importance:")
    for j in order[:20]:
        print(f"    {imps[j]:+.4f}  {names[j]}")

    frames = {"train": df[split == "fit_mine"], "eval": df[split == "monitor"],
              "test": df[split == "test"]}
    row_of = {i: k for k, i in enumerate(df.id.astype(str))}

    def write_arm(arm, k=None):
        ad = OUT / arm
        (ad / "split").mkdir(parents=True, exist_ok=True)
        crits = [] if k is None else [(names[j], int(j)) for j in order[:k]]

        def render(rid, text):
            if k is None:
                return text
            i = row_of[rid]
            lines = ["VA metrics:"]
            for nm, j in crits:
                lines.append(f"    {nm}: {fmt(float(VA[i, j]))}")
            return "\n".join(lines) + "\n" + text

        outs = []
        for s, fr in frames.items():
            o = pd.DataFrame({
                "text": [render(str(r.id), str(r.text)) for r in fr.itertuples()],
                "judgement": fr.judgement.values,
                "group": fr.prompt_id.astype(str).values,
                "did": fr.id.astype(str).values})
            o.to_csv(ad / "split" / f"{s}.csv", index=False)
            outs.append(o)
        pd.concat(outs, ignore_index=True).to_csv(ad / "data.csv", index=False)
        (ad / "manifest.json").write_text(json.dumps({
            "cell": "cw_community", "arm": arm, "k": k,
            "criteria": [nm for nm, _ in crits],
            "format": "metrics block PREPENDED (right-side truncation safety)",
            "splits": {s: int(len(fr)) for s, fr in frames.items()},
            "selection_split": "eval (MONITOR)",
        }, indent=2))
        print(f"[cw] wrote {arm}")

    write_arm("cw_raw")
    write_arm("cw_augk20", 20)
    write_arm("cw_augk40", 40)
    print("BUILD_CW_MIRROR_DONE")


if __name__ == "__main__":
    main()
