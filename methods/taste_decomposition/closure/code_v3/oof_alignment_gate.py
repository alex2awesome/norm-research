#!/usr/bin/env python3
"""MANDATORY OOF ALIGNMENT GATE (registry 2026-08-10 landmine: "*_va_nl_oof_*.npy arrays
are keyed in bank item_ids order, NOT population/join order -- misaligned joins read
AUC ~ .50").

Gate as specified: AUC(y, oof_SEED0 in the assembled row order) must equal the published
`nonlinear.VA.auc_seeds[0]` to < 1e-9 -- exact identity, seed 0 only, never mean3
(mean3's AUC is not the mean of the per-seed AUCs).

NOTE ON THE READOUT USED HERE: this gate is a POOLED AUC. It is used ONLY as an
alignment identity check and is never quoted as a result on this cell, where pooled AUC
is composition-dominated.
"""
import json, sys
sys.path.insert(0, '.'); sys.path.insert(0, '../maps_hw_si')
import numpy as np
from sklearn.metrics import roc_auc_score
import cells_code as C

pub = json.load(open('abank_rescore/code_v3_enriched_layer1.json'))
d = C.load()
y = d["y"]
out = {"gate": "registry 2026-08-10 OOF alignment gate",
       "rule": "AUC(y, oof_seed0 in assembled row order) == published nonlinear.VA.auc_seeds[0] to <1e-9",
       "note": "pooled AUC used ONLY as an alignment identity; never quoted as a result",
       "splits": {}}
ok = True
for sp in ("eval", "test"):
    m = d["split"] == sp
    published = pub["splits"][sp]["nonlinear"]["VA"]["auc_seeds"]
    for s in (0, 1, 2):
        oof = np.load(f"abank_rescore/code_v3_{sp}_va_nl_oof_seed{s}.npy")
        assert len(oof) == int(m.sum()), f"{sp} seed{s}: length {len(oof)} != {int(m.sum())}"
        got = float(roc_auc_score(y[m], oof))
        delta = abs(got - published[s])
        passed = delta < 1e-9
        ok &= (passed if s == 0 else True)          # gate binds on seed 0
        out["splits"].setdefault(sp, {})[f"seed{s}"] = {
            "assembled_order_pooled_auc": got, "published_auc_seeds": published[s],
            "abs_diff": delta, "pass": passed,
            "binding": s == 0}
    # counter-check: what a bank-item_ids-order (i.e. WRONG) join would look like
    wrong = np.load(f"abank_rescore/code_v3_{sp}_va_nl_oof_seed0.npy").copy()
    rng = np.random.default_rng(0); rng.shuffle(wrong)
    out["splits"][sp]["shuffled_counterfactual_auc"] = float(roc_auc_score(y[m], wrong))
out["GATE_PASS"] = bool(ok)
json.dump(out, open("oof_alignment_gate.json", "w"), indent=1)
print(json.dumps(out, indent=1))
