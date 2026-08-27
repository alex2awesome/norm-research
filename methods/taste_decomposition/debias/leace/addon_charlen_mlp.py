#!/usr/bin/env python3
"""Add-on readout: nonlinear (MLP) residue of the LENGTH channel after the V4
length eraser on B00 -- completes the battery table's nonlinear-residue column
with a continuous, non-token channel (is the nonlinear residue specific to
discrete planted tokens, or generic?).  Writes results/addon_charlen_mlp.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from leace import LeaceEraser, precompute_eig     # noqa: E402
from run_battery_leace import mlp_probe, probe_splits, linear_probe, onehot_bins  # noqa: E402


def main():
    reps_root = Path(sys.argv[1])
    nz = np.load(HERE.parent / "build/nuisance.npz", allow_pickle=True)
    nz_ids = np.array([str(s) for s in nz["doc_id"]])
    names = [str(s) for s in nz["names"]]
    groups = json.loads(str(nz["groups_json"]))
    z = np.load(reps_root / "B00_vanilla_real/reps.npz", allow_pickle=True)
    ids = np.array([str(s) for s in z["doc_id"]])
    pos = np.array([{d: i for i, d in enumerate(nz_ids)}[d] for d in ids])
    X = z["rep_h"].astype(np.float64)
    split = np.array([str(s) for s in z["split"]])
    tr = split == "train"

    Zl = nz["Z"].astype(np.float64)[pos][:, groups["length"]]
    eig = precompute_eig(X[tr])
    Zl_oh, _ = onehot_bins(Zl, tr)
    er = LeaceEraser().fit(X[tr], Zl_oh[tr], eig=eig)          # categorical PRIMARY
    XP = er.apply(X)
    erc = LeaceEraser().fit(X[tr], Zl[tr], eig=eig)            # continuous secondary
    XPc = erc.apply(X)
    cl = (Zl[:, 0] >= np.median(Zl[tr, 0])).astype(int)   # char_len median split
    tr_i, va_i, te_i = probe_splits(split)
    out = {
        "target": "char_len (train-median split), B00 rep_h, length-2 eraser",
        "certificate_form_primary": "categorical (train-decile one-hot)",
        "linear_raw": linear_probe(X, cl, tr, split == "eval"),
        "linear_after": linear_probe(XP, cl, tr, split == "eval"),
        "mlp_raw": mlp_probe(X, cl, tr_i, va_i, te_i),
        "mlp_after": mlp_probe(XP, cl, tr_i, va_i, te_i),
        "continuous_Z_secondary": {
            "linear_after": linear_probe(XPc, cl, tr, split == "eval"),
            "mlp_after": mlp_probe(XPc, cl, tr_i, va_i, te_i),
        },
    }
    (HERE / "results/addon_charlen_mlp.json").write_text(json.dumps(out, indent=2))
    slim = {k: (v if isinstance(v, float) else v.get("auc_eval_mean"))
            for k, v in out.items() if isinstance(v, (float, dict)) and k != "continuous_Z_secondary"}
    slim["cont_linear_after"] = out["continuous_Z_secondary"]["linear_after"]
    slim["cont_mlp_after"] = out["continuous_Z_secondary"]["mlp_after"]["auc_eval_mean"]
    print(json.dumps(slim, indent=1))


if __name__ == "__main__":
    main()
