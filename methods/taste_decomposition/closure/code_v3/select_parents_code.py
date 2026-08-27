#!/usr/bin/env python3
"""FREEZE ADDENDUM 3, step 1 for the code_v3 cell -- choose the MIXED parents to
decompose BEFORE round 1.

Two sources, both recorded:

  (a) SHAP interaction screen, refit on FIT+MINE ONLY: A-bank criteria whose
      contribution to the nonlinear scorecard is entangled with a programmatic
      SURFACE feature (V_text geometry) or with an execution feature (V_exec).
      Ranked by total mean|SHAP interaction| with any V feature.

  (b) The parent the campaign brief names outright: the cell's dominant articulated
      channel, "change/commit/PR communication quality" (a78) -- which the round-0
      concept census showed is ONE concept with "change description clarity,
      completeness and rationale" (a105), i.e. the bank's #1 and #2 univariate
      criteria on eval are a single concept measured twice.  Adding it is recorded
      here, not silent.

Descriptive screen only; it chooses what gets decomposed and nothing else.
CPU only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "maps_hw_si"))

import cells_code as C                                       # noqa: E402
import closure_core as L                                     # noqa: E402

N_PARENTS = 6
BRIEF_PARENTS = ["a78", "a123"]     # a105 is a78's census-identical twin; see census_dedup_note


def main():
    import shap
    from sklearn.ensemble import HistGradientBoostingClassifier

    d = C.load()
    z = np.load(HERE / "splits.npz", allow_pickle=True)
    fit = z["fitmask"]
    y = d["y"][fit]

    names = list(d["v_names"]) + [f"{a}|{n}" for a, n in zip(d["a_ids"], d["a_names"])]
    is_v = np.array([True] * len(d["v_names"]) + [False] * len(d["a_ids"]))
    M = np.column_stack([d["V"], d["A"]])
    keep, med = L.clean_fit(M[fit])
    X = L.clean_apply(M[fit], keep, med)
    kn = [names[j] for j in keep]
    kv = is_v[keep]

    m = HistGradientBoostingClassifier(max_leaf_nodes=15, learning_rate=0.06,
                                       max_iter=400, early_stopping=True,
                                       validation_fraction=0.1, n_iter_no_change=20,
                                       random_state=0)
    m.fit(X, y)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(y), size=min(300, len(y)), replace=False)
    sv = np.abs(shap.TreeExplainer(m).shap_values(X[sub])).mean(axis=0)
    top = np.argsort(-sv)[:15]

    m2 = HistGradientBoostingClassifier(max_leaf_nodes=15, learning_rate=0.06,
                                        max_iter=400, early_stopping=True,
                                        validation_fraction=0.1, n_iter_no_change=20,
                                        random_state=0)
    m2.fit(X[:, top], y)
    iv = np.abs(shap.TreeExplainer(m2).shap_interaction_values(X[sub][:, top])).mean(axis=0)
    tn, tv = [kn[j] for j in top], kv[top]

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

    sel = [r["criterion"] for r in rows[:N_PARENTS]]
    by_id = {n.split("|")[0]: n for n in kn if not n.startswith(("ve_", "vt_", "n_review"))}
    brief = [by_id[a] for a in BRIEF_PARENTS if a in by_id and by_id[a] not in sel]

    out = {"cell": "code_v3", "n_fit_mine": int(fit.sum()),
           "note": "SHAP interaction screen refit on FIT+MINE ONLY; descriptive screen, "
                   "used only to choose which bank criteria get decomposed",
           "top_shap_features_overall": [{"name": kn[j], "mean_abs_shap": float(sv[j])}
                                         for j in top],
           "candidates": rows,
           "selected_parents_shap": sel,
           "selected_parents_brief": brief,
           "selected_parents": sel + brief}
    (HERE / "parents_code.json").write_text(json.dumps(out, indent=1))
    print(f"FIT+MINE n={out['n_fit_mine']}  |  SHAP-selected {len(sel)} + brief {len(brief)}")
    for x in rows[:N_PARENTS]:
        tp = ", ".join(f"{p['feature']} {p['mass']:.3f}" for p in x["top_surface_partners"])
        print(f"  mass={x['surface_interaction_mass']:.4f} alone={x['alone_AUC_FITMINE']:.3f}"
              f"  {x['criterion']}  <- {tp}")
    print("  brief parents added:", brief)


if __name__ == "__main__":
    main()
