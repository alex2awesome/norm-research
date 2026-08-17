#!/usr/bin/env python3
"""U4b — so_accepted (VERDICT) Layer-1 ladder on the V6 scored matrix.

Same rows, same shards, same frozen estimators as so_votes_layer1 (everything
IMPORTED from it / layer1_gemma_cells / scaleupC_layer1) — only y changes:
y = accepted_verdict from the shard meta (defined on ALL 16,001 rows; the vote
cell's median-tie drop does NOT apply here).  Collapse gate re-run on this
cell's own analysis rows per the standing ruling.  T is attached later from the
sk1 so_accepted_qtrunc chain (VA-only ladder now — full speed directive).

Run on sk3 (shards live there):  python3 so_accepted_layer1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
import layer1_gemma_cells as L  # noqa: E402
import scaleupC_layer1 as SC  # noqa: E402
import so_votes_layer1 as SV  # noqa: E402

REPO = SC.REPO
RESULTS = SC.RESULTS_DIR
SO_OUT = Path("outputs/va_gemma_banks_so_votes")
SO_DIR = REPO / "datasets/stackoverflow-votes/va"

meta, A, V, groups, shard, ids = SC.load_scaleupC_bank("so_votes", out=REPO / SO_OUT)
y = np.array(meta["ys"]["accepted_verdict"], dtype=float)
assert np.isfinite(y).all(), "accepted_verdict undefined on some rows"
y = y.astype(int)
a_names = list(meta["a_names"])

# collapse gate on THIS cell's analysis rows (all rows here)
shares = np.array([SV.modal_share(A[:, c]) for c in range(A.shape[1])])
keep_c = shares <= SV.COLLAPSE_MODAL_MAX
dropped = [nm for nm, k in zip(a_names, keep_c) if not k]
A = A[:, keep_c]
a_names = [nm for nm, k in zip(a_names, keep_c) if k]
print(f"[collapse gate] dropped {len(dropped)} criteria -> A {A.shape}", flush=True)

pop = pd.read_csv(SO_DIR / "population.csv.gz").set_index("row_id")
pop.index = pop.index.astype(str)
order = [str(i) for i in ids]
position = pop.loc[order, "position"].values.astype(float)
body_len = pop.loc[order, "body"].astype(str).str.len().values.astype(float)

folds = L.outer_folds(len(y), groups, n_splits=5)
res = {"cell": "so_accepted (VERDICT: asker accepted; V6 rows/matrix verbatim)",
       "n": int(len(y)), "n_questions": int(len(set(groups))),
       "pos_rate": float(y.mean()),
       "collapse_gate": {"dropped": dropped, "kept": int(A.shape[1])}}

qmean = pd.Series(y).groupby(pd.Series(groups)).transform("mean").values
res["question_identity_alone_auc"] = float(roc_auc_score(y, qmean))
res["position_line"] = {
    "pooled_auc_first_is_better": float(roc_auc_score(y, -position)),
    "within_question": SV.within_group_auc(y, groups, -position)}
res["charlen_alone_auc"] = float(roc_auc_score(y, body_len))
res["charlen_within_question"] = SV.within_group_auc(y, groups, body_len)

mats = {"V": V, "A": A, "VA": np.column_stack([V, A])}
table, preds = {}, {}
for k, M in mats.items():
    lin_auc, lin_oof = L.linear_oof_family1(M, y, groups, folds)
    gbm_oofs = [L.gbm_oof_family1(M, y, groups, folds, seed=s)[1]
                for s in L.GBM_SEEDS]
    nl = np.mean(gbm_oofs, axis=0)
    table[k] = {"linear": float(lin_auc),
                "nl_mean3": float(roc_auc_score(y, nl)),
                "nl_within_question": SV.within_group_auc(y, groups, nl)}
    preds[k] = nl
    wq = table[k]["nl_within_question"]["pair_weighted"]
    print(f"[{k}] linear {lin_auc:.4f} | nl {table[k]['nl_mean3']:.4f} "
          f"| within-q {wq:.4f}", flush=True)
res["ladder"] = table

np.savez_compressed(RESULTS / "so_accepted_va_oof.npz",
                    ids=np.array(order, dtype=object), y=y,
                    groups=np.array([str(g) for g in groups], dtype=object),
                    V_nl=preds["V"], A_nl=preds["A"], VA_nl=preds["VA"])
(RESULTS / "so_accepted_ledger.json").write_text(json.dumps(res, indent=1, default=float))
print("SO_ACCEPTED_L1_DONE", flush=True)
