#!/usr/bin/env python3
"""FREEZE ADDENDUM 3, step 1 -- select the MIXED parents to decompose.

A parent here is an EXISTING Layer-1 bank criterion (an A-bank rubric) whose
contribution to the nonlinear scorecard is entangled with a programmatic SURFACE
feature: exactly the "length/format-mediated interaction" the Layer-1 SHAP screen
flagged on Style Invitational (Linguistic polish x v_punctuation_density,
v_char_count x v_punctuation_density, v_char_count x Linguistic polish;
off-diagonal SHAP mass .549) and, on HashtagWars, the screen run for this campaign
(Rhythm supports the payoff x v_uppercase_letter_ratio, v_char_count x
v_uppercase_letter_ratio; off-diagonal mass .589).

Such a criterion is MIXED in the Addendum-2 sense: its judged score plausibly
carries both real craft AND a surface habit that a judge can see, so the score's
predictive power cannot be attributed to either without splitting it.

SELECTION IS READ ON FIT+MINE ONLY (never MONITOR, never the honest rows), the
same rule the N&C campaign used for its round-5 parents.  Ranking = total
mean|SHAP interaction| between an A-bank criterion and any programmatic V feature.

CPU only.  Usage: python select_parents.py [cell ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import cells as C
import closure_core as L

HERE = Path(__file__).resolve().parent
N_PARENTS = 8


def shap_pairs_fitmine(cell):
    import shap
    from sklearn.ensemble import HistGradientBoostingClassifier

    d = C.load(cell)
    sp = json.loads((HERE / f"{cell}_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    fit = split == "fit_mine"
    y = d["y"][fit]
    Vn, An = d["v_names"], d["a_names"]
    M = np.column_stack([d["V"], d["A"]])
    names = list(Vn) + list(An)
    keep, med = L.clean_fit(M[fit])
    X = L.clean_apply(M[fit], keep, med)
    kn = [names[j] for j in keep]
    is_v = np.array([n.startswith("v_") for n in kn])

    m = HistGradientBoostingClassifier(max_leaf_nodes=15, learning_rate=0.06,
                                       max_iter=400, early_stopping=True,
                                       validation_fraction=0.1, n_iter_no_change=20,
                                       random_state=0)
    m.fit(X, y)
    ex = shap.TreeExplainer(m)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(y), size=min(300, len(y)), replace=False)
    sv = np.abs(ex.shap_values(X[sub])).mean(axis=0)
    top = np.argsort(-sv)[:15]
    m2 = HistGradientBoostingClassifier(max_leaf_nodes=15, learning_rate=0.06,
                                        max_iter=400, early_stopping=True,
                                        validation_fraction=0.1, n_iter_no_change=20,
                                        random_state=0)
    m2.fit(X[:, top], y)
    iv = np.abs(shap.TreeExplainer(m2).shap_interaction_values(X[sub][:, top])).mean(axis=0)
    tn = [kn[j] for j in top]
    tv = is_v[top]

    rows = []
    for i, ni in enumerate(tn):
        if tv[i]:
            continue
        partners = sorted(((float(iv[i, j]), tn[j]) for j in range(len(tn))
                           if tv[j] and j != i), reverse=True)
        rows.append({"criterion": ni,
                     "mean_abs_shap": float(sv[top[i]]),
                     "surface_interaction_mass": float(sum(p[0] for p in partners)),
                     "top_surface_partners": [{"feature": p[1], "mass": p[0]}
                                              for p in partners[:3]],
                     "alone_AUC_FITMINE": L.auc(y, X[:, top[i]])})
    rows.sort(key=lambda r: -r["surface_interaction_mass"])
    return {"cell": cell, "n_fit_mine": int(fit.sum()),
            "note": "SHAP interaction screen refit on FIT+MINE ONLY; descriptive screen, "
                    "used only to choose which bank criteria get decomposed",
            "offdiag_note": "ranked by total mean|SHAP interaction| with programmatic V features",
            "candidates": rows}


if __name__ == "__main__":
    todo = sys.argv[1:] or C.CELLS
    out = {}
    for c in todo:
        r = shap_pairs_fitmine(c)
        r["selected_parents"] = [x["criterion"] for x in r["candidates"][:N_PARENTS]]
        out[c] = r
        (HERE / f"{c}_parents.json").write_text(json.dumps(r, indent=1))
        print(f"=== {c}  (FIT+MINE n={r['n_fit_mine']})")
        for x in r["candidates"][:N_PARENTS]:
            tp = ", ".join(f"{p['feature']} {p['mass']:.3f}" for p in x["top_surface_partners"])
            print(f"  mass={x['surface_interaction_mass']:.3f} aloneAUC={x['alone_AUC_FITMINE']:.3f}"
                  f"  {x['criterion']}   <- {tp}")
    (HERE / "parents_all.json").write_text(json.dumps(out, indent=1))
