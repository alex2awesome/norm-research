#!/usr/bin/env python3
"""CROSS-CELL SIGN-FLIP TEST — the registered prediction P1/P2/P3 of
notes/2026-08-09__closure_cap_crowd.md §4, written BEFORE the first cap_crowd fit.

THE CLAIM UNDER TEST.  cap_finalist (EDITOR label) found 16 of 62 mined Track-A craft
criteria significantly ANTI-predictive, and attributed it to its hard-negative pool:
negatives were drawn to be crowd-plausible, so within-contest crowd rank ran .335 against
the editor label and anything measuring generic comic appeal separated the wrong way.

If that mechanism is right, the SAME criteria must read POSITIVE on cap_crowd, where the
crowd rating IS the label.

  P1  a MAJORITY of cap_finalist's sign-triggered criteria have alone-AUC > .5 here.
  P2  the rank correlation of alone-AUC across shared criterion NAMES is NEGATIVE
      (the sharper test: P1 can pass on a level shift, P2 needs the ordering to invert).
  P3  falsifier -- if the same criteria are anti-predictive on BOTH cells, the
      hard-negative explanation is WRONG and cap_finalist section 10.7's mechanism is
      retracted in favour of "the judge's craft scores are mis-signed on captions".

Both cells share the judge, the bank and (after each cell's TIER-R repair) the item view,
so the comparison is like-for-like.  alone-AUC is computed on FIT+MINE only, on each
cell's own splits, exactly as gepa_phrasing.py computes it -- MONITOR is never read.

CPU only.  Usage: python sign_flip_test.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import cells as C
import closure_core as L

HERE = Path(__file__).resolve().parent
FIN = HERE.parent / "cap_finalist"

# Verbatim from notes/2026-08-09__closure_cap_finalist.md section 10.7, registered
# before any cap_crowd number was computed.
SIGN_TRIGGERED_ON_FINALIST = [
    "Cultural allusion repurposing", "Distinctive speaker voice", "Vivid scene specificity",
    "Exact wordplay fit", "Late twist or escalation in the final clause",
    "Idiomatic frame transformation", "Directness on taboo or transgressive subject matter",
    "Obvious first-thought pun", "Metaphorical or Satirical Mapping",
    "Phonetic pun precision", "Adapted familiar phrase", "Off-screen narrative",
    "Idiom-to-Visual Subversion", "Layered Double Entendre", "Character-voiced perspective",
]


def alone_aucs(cell_dir, cell, rounds):
    """alone-AUC on FIT+MINE for every scored criterion of the named rounds."""
    import sys
    sys.path.insert(0, str(cell_dir))
    sp = json.loads((cell_dir / f"{cell}_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    fit = split == "fit_mine"
    y = np.array([int(r["judgement"]) for r in sp["rows"]]) if "judgement" in sp["rows"][0] else None
    if y is None:
        d = C.load(cell) if cell == "cap_crowd" else None
        y = d["y"]
    out = {}
    for r in rounds:
        f = cell_dir / f"{cell}_r{r}_scores.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        X, names = z["X"], [str(s) for s in z["crit_names"]]
        keep, med = L.clean_fit(X[fit])
        Xc = L.clean_apply(X, keep, med)
        for k, j in enumerate(keep):
            out[names[j]] = float(L.auc(y[fit], Xc[fit, k]))
    return out


def main():
    d = C.load("cap_crowd")
    sp = json.loads((HERE / "cap_crowd_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    fit = split == "fit_mine"
    y = d["y"]

    crowd = {}
    for r in ("1", "2", "4", "5"):
        f = HERE / f"cap_crowd_r{r}_scores.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        X, names = z["X"], [str(s) for s in z["crit_names"]]
        keep, med = L.clean_fit(X[fit])
        Xc = L.clean_apply(X, keep, med)
        for k, j in enumerate(keep):
            crowd[names[j]] = float(L.auc(y[fit], Xc[fit, k]))

    fin = json.loads((FIN / "cap_finalist_gepa_targets.json").read_text())
    fin_auc = {c["name"]: c["alone_AUC_fitmine"] for c in fin["criteria"]
               if c.get("alone_AUC_fitmine") is not None}

    # ---- P1 -----------------------------------------------------------------
    p1 = []
    for nm in SIGN_TRIGGERED_ON_FINALIST:
        a = crowd.get(nm)
        p1.append({"name": nm, "cap_crowd_alone_AUC": a,
                   "cap_finalist_alone_AUC": fin_auc.get(nm),
                   "flipped_positive": (a is not None and a > 0.5)})
    found = [r for r in p1 if r["cap_crowd_alone_AUC"] is not None]
    n_flip = sum(r["flipped_positive"] for r in found)

    # ---- P2 -----------------------------------------------------------------
    shared = sorted(set(crowd) & set(fin_auc))
    xa = [fin_auc[n] for n in shared]
    ya = [crowd[n] for n in shared]
    rho = spearmanr(xa, ya) if len(shared) >= 6 else None

    out = {
        "prediction_registered_in": "notes/2026-08-09__closure_cap_crowd.md section 4 "
                                    "(written before the first cap_crowd fit)",
        "readout": "alone-AUC on FIT+MINE only; MONITOR never read",
        "P1": {
            "n_named": len(SIGN_TRIGGERED_ON_FINALIST),
            "n_found_on_cap_crowd": len(found),
            "n_flipped_positive": n_flip,
            "share_flipped": (n_flip / len(found)) if found else None,
            "PASS": bool(found and n_flip > len(found) / 2),
            "detail": p1,
        },
        "P2": {
            "n_shared_criteria": len(shared),
            "spearman_finalist_vs_crowd": (float(rho.statistic) if rho else None),
            "p_value": (float(rho.pvalue) if rho else None),
            "PASS": bool(rho and rho.statistic < 0),
        },
        "P3_falsifier_triggered": bool(found and n_flip <= len(found) / 2),
        "mean_alone_AUC": {
            "cap_crowd_all_mined": float(np.mean(list(crowd.values()))) if crowd else None,
            "cap_finalist_all_mined": float(np.mean(list(fin_auc.values()))) if fin_auc else None,
        },
    }
    (HERE / "cap_crowd_sign_flip_test.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "P1"}, indent=1))
    print(f"\nP1 {n_flip}/{len(found)} flipped positive  PASS={out['P1']['PASS']}")
    for r in sorted(found, key=lambda r: -(r["cap_crowd_alone_AUC"] or 0)):
        print(f"  crowd {r['cap_crowd_alone_AUC']:.3f}  finalist "
              f"{(r['cap_finalist_alone_AUC'] if r['cap_finalist_alone_AUC'] is not None else float('nan')):.3f}"
              f"  {r['name'][:52]}")


if __name__ == "__main__":
    main()
