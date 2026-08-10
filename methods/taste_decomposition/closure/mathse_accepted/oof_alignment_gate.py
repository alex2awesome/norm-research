#!/usr/bin/env python3
"""MANDATORY OOF ALIGNMENT GATE for the math.SE VOTE-SCORE closure cell.

Registry 2026-08-10 landmine: `*_va_nl_oof_*.npy` arrays are keyed in BANK
item_ids order, NOT population/join order -- a misaligned join reads AUC ~= .50.
For the two math.SE cells the key is the KEPT-SUBSET order: bank `item_ids`
filtered by the finite mask of that cell's y.  On THIS cell the mask is a no-op
(13,001 of 13,001) -- the sibling VOTE cell keeps only 11,629.

GATE (binding, seed 0 only, never mean3 -- mean3's AUC is not the mean of the
per-seed AUCs):

    AUC(y, mathse_accepted_verdict_va_nl_oof_seed0.npy in the assembled row order)
        == ledger nonlinear.VA["0"].auc                to < 1e-9

The published value is .6317782528854139.  This gate is a POOLED AUC used ONLY as
an alignment identity; it is never quoted as a result.

Also asserts the SHARED-MATRIX separation: the VOTE-SCORE cell's OOF array has a
different length (11,629 kept rows) and must never be loaded here.

Run standalone:  python3 oof_alignment_gate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PUBLISHED = 0.6317782528854139          # ledger nonlinear.VA["0"].auc


def assert_aligned(d, write=True):
    import cells as C
    led = d["layer1"]
    published = float(led["nonlinear"]["VA"]["0"]["auc"])
    oof0 = np.load(C.OOF_SEED0)
    oof3 = np.load(C.OOF_MEAN3)
    y = d["y"]
    assert len(oof0) == len(y), (
        f"OOF length {len(oof0)} != population {len(y)}: wrong cell's array "
        "(the vote-score cell shares the matrix but keeps only 11,629 rows)")
    got = float(roc_auc_score(y, oof0))
    diff = abs(got - published)
    rng = np.random.default_rng(0)
    shuf = oof0.copy(); rng.shuffle(shuf)
    out = {
        "gate": "registry 2026-08-10 OOF alignment gate",
        "rule": ("AUC(y, oof_seed0 in assembled row order) == ledger "
                 "nonlinear.VA['0'].auc to <1e-9 (seed 0 only, never mean3)"),
        "row_order": "bank item_ids VERBATIM (isfinite(ys['accepted_verdict']) is a no-op: 13,001/13,001)",
        "n": int(len(y)),
        "published_seed0_auc": published,
        "published_constant_in_brief": PUBLISHED,
        "assembled_order_pooled_auc": got,
        "abs_diff": diff,
        "GATE_PASS": bool(diff < 1e-9),
        "mean3_auc_NOT_A_GATE": float(roc_auc_score(y, oof3)),
        "shuffled_counterfactual_auc": float(roc_auc_score(y, shuf)),
        "note": "pooled AUC used ONLY as an alignment identity; never quoted as a result",
    }
    assert out["GATE_PASS"], json.dumps(out, indent=1)
    assert abs(published - PUBLISHED) < 1e-12, "ledger moved under the brief's constant"
    if write:
        (HERE / "oof_alignment_gate.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    import cells as C
    d = C.load(gate=False)
    print(json.dumps(assert_aligned(d), indent=1))
