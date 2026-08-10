#!/usr/bin/env python3
"""CROSS-CELL TRANSPLANT READOUT -- decides the registered P1/P2/P3 predictions.

25 cap_finalist criteria, verbatim text, scored on cap_crowd by the same judge on the same
cartoon+caption item view.  The ONLY difference between the two alone-AUCs is the label:
editor finalist selection versus crowd vote.  TIER D -- no bank entry, no Good-Turing.

alone-AUC on FIT+MINE only (MONITOR never read), exactly as gepa_phrasing computes it on
each cell.  CPU only.  Usage: python transplant_readout.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

import cells as C
import closure_core as L

HERE = Path(__file__).resolve().parent
FIN = HERE.parent / "cap_finalist"


def main():
    d = C.load("cap_crowd")
    sp = json.loads((HERE / "cap_crowd_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    fit = split == "fit_mine"
    held = np.isin(d["dense_split"], ["eval", "test"])
    y = d["y"]

    z = np.load(HERE / "cap_crowd_x1_scores.npz", allow_pickle=True)
    assert (z["i"] == np.arange(len(y))).all(), "transplant rows out of alignment"
    X = z["X"]
    ids = [str(s) for s in z["crit_ids"]]
    spec = json.loads((HERE / "cap_crowd_x1_species.json").read_text())
    prov = {p["blind_id"]: p for p in spec["provenance"]}

    keep, med = L.clean_fit(X[fit])
    Xc = L.clean_apply(X, keep, med)

    rows = []
    for k, j in enumerate(keep):
        bid = ids[j]
        p = prov[bid]
        rows.append({
            "blind_id": bid, "name": p["name"],
            "origin_track_on_cap_finalist": p["origin_track"],
            "selection_reason": p["selection_reason"],
            "alone_AUC_cap_finalist_fitmine": p["alone_AUC_on_cap_finalist_fitmine"],
            "alone_AUC_cap_crowd_fitmine": float(L.auc(y[fit], Xc[fit, k])),
            "alone_AUC_cap_crowd_HONEST": float(L.auc(y[held], Xc[held, k])),
        })
    n_dropped = int(X.shape[1] - len(keep))

    sign = [r for r in rows if r["selection_reason"].startswith("sign")]
    n_flip = sum(1 for r in sign if r["alone_AUC_cap_crowd_fitmine"] > 0.5)

    xa = [r["alone_AUC_cap_finalist_fitmine"] for r in rows
          if r["alone_AUC_cap_finalist_fitmine"] is not None]
    ya = [r["alone_AUC_cap_crowd_fitmine"] for r in rows
          if r["alone_AUC_cap_finalist_fitmine"] is not None]
    rho = spearmanr(xa, ya)
    pear = pearsonr(xa, ya)

    # Hanley-McNeil two-sided band at the null on THIS cell, for "significantly > .5"
    n_pos = int(y[fit].sum())
    n_neg = int((1 - y[fit]).sum())
    se = float(np.sqrt((1 / 12.0) * (1 / n_pos + 1 / n_neg + 1 / (n_pos * n_neg))))
    n_sig_pos = sum(1 for r in sign if r["alone_AUC_cap_crowd_fitmine"] > 0.5 + 2 * se)

    out = {
        "design": "cap_finalist criterion TEXT scored on cap_crowd; same judge, same bank, "
                  "same cartoon+caption item view; the LABEL is the only difference",
        "tier": "D (directed transplant) -- excluded from Good-Turing, does not join the bank",
        "n_transplanted": int(X.shape[1]),
        "n_dropped_by_gate": n_dropped,
        "sign_band": {"n_pos_fitmine": n_pos, "n_neg_fitmine": n_neg,
                      "hanley_mcneil_SE": se, "two_sided_2SE": 2 * se},
        "P1": {
            "statement": "a MAJORITY of cap_finalist's sign-contradicting criteria have "
                         "alone-AUC > .5 on cap_crowd",
            "n_tested": len(sign), "n_above_half": n_flip,
            "n_significantly_above_half": n_sig_pos,
            "share": (n_flip / len(sign)) if sign else None,
            "PASS": bool(sign and n_flip > len(sign) / 2),
        },
        "P2": {
            "statement": "rank correlation of alone-AUC across the two cells is NEGATIVE",
            "n": len(xa),
            "spearman": float(rho.statistic), "spearman_p": float(rho.pvalue),
            "pearson": float(pear.statistic), "pearson_p": float(pear.pvalue),
            "PASS": bool(rho.statistic < 0),
        },
        "P3_falsifier_triggered": bool(sign and n_flip <= len(sign) / 2),
        "means": {
            "cap_finalist_fitmine": float(np.mean(xa)),
            "cap_crowd_fitmine": float(np.mean(ya)),
        },
        "criteria": sorted(rows, key=lambda r: -r["alone_AUC_cap_crowd_fitmine"]),
    }
    (HERE / "cap_crowd_transplant_results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "criteria"}, indent=1))
    print("\n  crowd  finalist  reason        criterion")
    for r in out["criteria"]:
        f = r["alone_AUC_cap_finalist_fitmine"]
        print(f"  {r['alone_AUC_cap_crowd_fitmine']:.3f}  "
              f"{(f if f is not None else float('nan')):.3f}     "
              f"{'SIGN' if r['selection_reason'].startswith('sign') else 'TOP '}  "
              f"{r['name'][:50]}")


if __name__ == "__main__":
    main()
