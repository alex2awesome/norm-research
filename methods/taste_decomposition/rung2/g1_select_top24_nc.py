#!/usr/bin/env python3
"""ADDENDUM G1, nc_responded — select the top-24 unique articulated criteria
(base A bank + mined rounds; V surface features excluded) by univariate AUC on
the dense TRAIN split only (leakage-safe wrt the E frame), with a modal-share
degeneracy screen matching the v3grid protocol (>=.95 dropped). Definitions
joined from nc_rubrics.jsonl (base) and round proposals_blinded (mined).
Local, CPU. Output: g1_top24_nc.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[3]
NC = REPO / "methods/taste_decomposition/closure/nc_responded"
HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


sys.path.insert(0, str(NC))
L = _load("nc_closure_lib", NC / "nc_closure_lib.py")
R = _load("nc_readout_g1", NC / "readout.py")

pop = L.load_population()
_, split, dsplit, _, _ = L.load_splits()
Xr, nr = R.load_round_scores([1, 2, 3, 4, 5])
A = pop["A"]
assert A.shape[1] == 198 and Xr.shape[1] == 67, (A.shape, Xr.shape)
X = np.column_stack([A, Xr])          # articulated-only: no V block
names = list(pop["a_names"]) + list(nr)
y = pop["y"]
tr = dsplit == "train"
print(f"train rows={tr.sum()}  cols={X.shape[1]}")

# definitions
rub = [json.loads(l) for l in open(
    REPO / "datasets/notice-and-comment/v4/nc_rubrics.jsonl")]
assert [r["name"] for r in rub] == list(pop["a_names"])
defs = {r["name"]: r["description"] for r in rub}
for r in range(1, 6):
    props = json.loads((NC / f"round{r}_proposals_blinded.json").read_text())
    for c in props["criteria"]:
        defs[f'r{r}:{c["id"]}:{c["name"]}'] = c["instruction"]

rows = []
for j, nm in enumerate(names):
    col = X[tr, j]
    ok = ~np.isnan(col)
    if ok.mean() < 0.5:
        continue
    v, yy = col[ok], y[tr][ok]
    if len(np.unique(v)) < 2 or len(np.unique(yy)) < 2:
        continue
    _, cnt = np.unique(v, return_counts=True)
    if cnt.max() / len(v) >= 0.95:
        continue
    a = roc_auc_score(yy, v)
    rows.append({"name": nm, "col_XAmined": j, "auc_dev": round(max(a, 1 - a), 4),
                 "auc_signed": round(a, 4), "definition": defs[nm]})

rows.sort(key=lambda r: -r["auc_dev"])
seen, top = set(), []
for r in rows:
    base = r["name"].split(":", 2)[-1].strip().lower()
    if base in seen:
        continue
    seen.add(base)
    top.append(r)
    if len(top) == 24:
        break

json.dump(top, open(HERE / "g1_top24_nc.json", "w"), indent=1)
for r in top:
    print(f'{r["auc_dev"]:.4f}  {r["name"][:90]}')
print("G1_NC_SELECT_DONE")
