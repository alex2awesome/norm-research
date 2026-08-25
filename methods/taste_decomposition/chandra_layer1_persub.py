#!/usr/bin/env python3
"""PER-SUBREDDIT Layer-1 for the Chandrasekharan pooled removal cells (Task A,
user order 2026-08-24). Versioned adaptation of chandra_layer1.py — the pooled
script is untouched. Refits V / A / VA on ONE subreddit's rows with the
canonical machinery (layer1_gemma_cells folds/pipelines + so_votes_layer1
collapse gate recomputed WITHIN the sub).

GROUPING DECLARED: the removal log carries NO timestamps (era undated, see
pooled ledger confound note), so created-month bins are unavailable ->
10 stable-row-hash pseudo-groups (sha256 of "<cell>_persub|<row_id>").

FRAME: v1 populations; era channel open; v2 rescore will supersede.

Usage: chandra_layer1_persub.py --cell chandra_humor --sub funny
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
sys.path.insert(0, str(REPO / "methods/taste_decomposition"))
import layer1_gemma_cells as L
import so_votes_layer1 as SV
import scaleupC_layer1 as SC

VIABLE = {
    "chandra_humor": ["funny", "Showerthoughts", "nottheonion", "me_irl"],  # tifu n=544 skipped
    "chandra_cw": ["nosleep", "books", "asoiaf", "gameofthrones"],
    # v2 (era-uniform kept side, leak-audit rebuild 2026-08-24): tifu restored
    # to 12,650 rows -> viable.
    "chandra_humor_v2": ["funny", "Showerthoughts", "nottheonion", "me_irl", "tifu"],
    "chandra_cw_v2": ["nosleep", "books", "asoiaf", "gameofthrones"],
}
N_PSEUDO = 10

ap = argparse.ArgumentParser()
ap.add_argument("--cell", required=True, choices=list(VIABLE))
ap.add_argument("--sub", required=True)
a = ap.parse_args()
cell, sub = a.cell, a.sub
assert sub in VIABLE[cell], f"{sub} not a viable sub for {cell}: {VIABLE[cell]}"
RESULTS = REPO / "methods/taste_decomposition/results"

meta, A, V, groups, shard, ids = SC.load_scaleupC_bank(
    cell, out=REPO / "outputs/va_gemma_banks_scaleupC")
y_all = np.array(meta["ys"]["removed"], dtype=int)
a_names = list(meta["a_names"])
g_all = np.array([str(x) for x in groups], dtype=object)

m = g_all == sub
A, V, y, ids = A[m], V[m], y_all[m], np.array(ids, dtype=object)[m]

# collapse gate recomputed WITHIN the sub (canonical so_votes gate)
shares = np.array([SV.modal_share(A[:, c]) for c in range(A.shape[1])])
keep_c = shares <= SV.COLLAPSE_MODAL_MAX
dropped = [nm for nm, k in zip(a_names, keep_c) if not k]
A = A[:, keep_c]

# 10 stable-row-hash pseudo-groups (era undated on the removal side)
g = np.array([str(int(hashlib.sha256(f"{cell}_persub|{r}".encode())
                      .hexdigest()[:8], 16) % N_PSEUDO) for r in ids], dtype=object)

print(f"[{cell}/{sub}] n={len(y)} pos={y.mean():.3f} | A kept {A.shape[1]} "
      f"(dropped {len(dropped)}) | pseudo-groups {N_PSEUDO}", flush=True)
folds = L.outer_folds(len(y), g, n_splits=5)

res = {"cell": cell, "sub": sub,
       "frame": "v1 populations; era channel open; v2 rescore will supersede",
       "grouping": f"{N_PSEUDO} stable-row-hash pseudo-groups (removal log undated; "
                   "created-month bins unavailable) — DECLARED",
       "n": int(len(y)), "pos_rate": float(y.mean()),
       "collapse_kept": int(A.shape[1]), "collapse_dropped": dropped}
preds = {}
for k, M in (("V", V), ("A", A), ("VA", np.column_stack([V, A]))):
    lin_auc, _ = L.linear_oof_family1(M, y, g, folds)
    nl = np.mean([L.gbm_oof_family1(M, y, g, folds, s)["oof"] for s in L.GBM_SEEDS], axis=0)
    res[k] = {"linear": float(lin_auc), "nl_mean3": float(roc_auc_score(y, nl))}
    preds[k] = nl
    print(f"[{cell}/{sub}:{k}] linear {res[k]['linear']:.4f} | nl {res[k]['nl_mean3']:.4f}",
          flush=True)
np.savez_compressed(RESULTS / f"{cell}_persub_{sub}_va_oof.npz",
                    ids=np.array([str(i) for i in ids], dtype=object), y=y, groups=g,
                    V_nl=preds["V"], A_nl=preds["A"], VA_nl=preds["VA"])
(RESULTS / f"{cell}_persub_{sub}_ledger.json").write_text(
    json.dumps(res, indent=1, default=float))
print(f"{cell.upper()}_{sub}_PERSUB_L1_DONE", flush=True)
