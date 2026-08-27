#!/usr/bin/env python3
"""Direction-3 / V3 feature-augmented dense for RoyalRoad (label: v3_aug).

ESTIMAND (binding): v3_aug is a FUSED V+A+T arm for the max-of-variants column
ONLY. It is never an honest T and never enters Delta_beyond. The honest ceiling
remains the plain cross-fitted dense arm.

Spec follows fusion/build_direction23_data.py exactly:
  prompt = full text + a "VA metrics:" block of the TOP-10 criteria
  arm a: "<name>: <score>"      arm b: "<name> (importance W): <score>"
  importance = grouped permutation importance, frozen HistGB (GRID[1]),
  scoring roc_auc, n_repeats=5, inner GroupKFold(3), mean over inner folds.

TWO DEVIATIONS FROM THE CAPTION BUILD, both forced and both documented:
  (1) PER-FOLD IMPORTANCE. The caption cell had one train split; this runs on the
      5-fold cross-fit, so importance is recomputed inside EACH fold's own train
      rows. A fold's held-out tenth never touches its own ranking or weights.
  (2) BLOCK FIRST. RoyalRoad chapters run to a median 2,918 tokens, so chapter +
      block cannot fit any budget -- the block would be truncated away. The VA
      block therefore leads the prompt and the chapter is head/tail-truncated
      (TOKENS, judge code path) into whatever budget remains.

Base view = the A-judge head+tail view, because the head+tail audit showed view
asymmetry is real (T .4986 -> .5846, +.0860 on the same folds).

  python datasets/creative-writing/build_royalroad_v3aug.py
"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

REPO = Path(os.environ.get("NR_REPO", "/lfs/skampere3/0/alexspan/norm-research"))
CELL = REPO / "datasets/creative-writing/royalroad_stubs"
SRC = CELL / "dense_crossfit"
OUT = CELL / "dense_crossfit_v3aug_fulltext"
CWX = REPO / "outputs/va_gemma_banks_cw_expert"
sys.path.insert(0, str(REPO / "methods/taste_decomposition"))
sys.path.insert(0, str(REPO / "datasets/va_gemma_banks"))
import layer1_gemma_cells as L
import cw_expert_layer1 as X
import score_cw_expert_banks as SC
TOP_K, MAXLEN = 10, 16384


def fmt(v):
    if v != v:                      # judge NA -> render as NA, never as a number
        return "NA"
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return f"{v:.3f}".rstrip("0").rstrip(".")


tok = SC._tokenizer()
A, V, ids = [], [], []
si = 0
while (CWX / f"cw_royalroad_verdict_shard{si}.npz").exists():
    z = np.load(CWX / f"cw_royalroad_verdict_shard{si}.npz", allow_pickle=True)
    A.append(z["X"]); V.append(z["V"]); ids += [str(s) for s in z["ids"]]
    a_names = [str(s) for s in z["a_names"]]; v_names = [str(s) for s in z["v_names"]]
    si += 1
A, V = np.vstack(A), np.vstack(V)
VA = np.column_stack([V, A]); names = v_names + a_names
row_of = {d: i for i, d in enumerate(ids)}
pop = pd.read_csv(CELL / "va/population.csv.gz"); pop["row_id"] = pop.row_id.astype(str)
raw = dict(zip(pop.row_id, pop.text.astype(str)))
print(f"[matrix] VA {VA.shape} over {len(row_of)} items")

man = {"design_id": "v3_aug_fulltext", "direction": 3, "base_view": "COMPLETE chapter, no truncation by the builder",
       "estimand": "FUSED V+A+T arm, max-of-variants column ONLY; never an honest T, "
                   "never in Delta_beyond",
       "importance_protocol": "PER FOLD: GroupKFold(3) inside that fold's TRAIN rows, "
                              "frozen HistGB (leaves=31, lr=.06), permutation_importance "
                              "roc_auc n_repeats=5, mean over inner folds; the held-out "
                              "tenth never influences its own ranking or weights",
       "prompt_order": "VA metrics block FIRST, then the COMPLETE chapter. Block-first "
                       "still matters: it guarantees the metrics survive whatever the "
                       "trainer truncates at max_len.",
       "max_len": 16384,
       "residual_truncation": "4/1274 chapters (0.31%) exceed 16384 incl. the 217-token "
                              "overhead and are cut by the trainer; median chapter 2,918, "
                              "p99 9,547, max 19,532",
       "rows_identical_to": "dense_crossfit (byte-identical folds)", "folds": {}}

for k in range(5):
    tr = pd.read_csv(SRC / f"fold{k}/split/train.csv"); tr["row_id"] = tr.row_id.astype(str)
    idx = np.array([row_of[d] for d in tr.row_id])
    Xtr, ytr, gtr = VA[idx], tr.judgement.values, tr.row_id.values.astype(str)
    imps, nf = np.zeros(VA.shape[1]), 0
    for itr, ite in GroupKFold(n_splits=3).split(Xtr, ytr, gtr):
        m = L._fit_gbm(L.GRID[1], seed=0)
        Xi = np.nan_to_num(Xtr, nan=np.nanmedian(Xtr))
        m.fit(Xi[itr], ytr[itr])
        r = permutation_importance(m, Xi[ite], ytr[ite], scoring="roc_auc",
                                   n_repeats=5, random_state=0, n_jobs=-1)
        imps += r.importances_mean; nf += 1
    imps /= nf
    order = np.argsort(-imps)[:TOP_K]
    top = [{"name": names[j], "col": int(j), "importance": float(imps[j])} for j in order]
    ws = sum(max(t["importance"], 0) for t in top) or 1.0
    for t in top:
        t["weight"] = round(max(t["importance"], 0) / ws, 2)

    def block(did, arm):
        i = row_of[did]
        out = ["VA metrics:"]
        for t in top:
            v = fmt(float(VA[i, t["col"]]))
            out.append(f"    {t['name']}: {v}" if arm == "a"
                       else f"    {t['name']} (importance {t['weight']:.2f}): {v}")
        return "\n".join(out)

    for arm in ("a", "b"):
        d = OUT / f"arm_{arm}" / f"fold{k}"; (d / "split").mkdir(parents=True, exist_ok=True)
        for nm in ("train", "eval", "test"):
            s = pd.read_csv(SRC / f"fold{k}/split/{nm}.csv"); s["row_id"] = s.row_id.astype(str)
            texts = []
            for did in s.row_id:
                b = block(did, arm)
                # USER ORDER: the V3/VAT arm sees the FULL text. No builder-side
                # truncation whatsoever; the chapter goes in complete and only the
                # trainer's max_len can cut it (4/1274 rows at 16384).
                texts.append(f"{b}\n\nfull text:\n    {raw[did]}")
            s["text"] = texts
            s[["text", "judgement", "group", "row_id"]].to_csv(d / f"split/{nm}.csv", index=False)
        pd.concat([pd.read_csv(d / f"split/{n}.csv") for n in ("train", "eval", "test")]
                  ).to_csv(d / "data.csv", index=False)
    man["folds"][f"fold{k}"] = {"top_criteria": top}
    print(f"[fold{k}] top3: " + ", ".join(f"{t['name'][:26]}({t['weight']:.2f})" for t in top[:3]))

(OUT).mkdir(parents=True, exist_ok=True)
(OUT / "manifest.json").write_text(json.dumps(man, indent=2))
print("BUILD_V3AUG_DONE")
