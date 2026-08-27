#!/usr/bin/env python3
"""Round-r MIXED-parent selection over the CURRENT bank (FREEZE ADDENDUM 3, continued).

The decomposition-first pass at round 0 took its parents from the frozen Layer-1
bank.  From round 3 onward the bank also contains mined criteria, and those can be
MIXED in exactly the same way -- a criterion whose judged score is entangled with a
programmatic length/format feature.  This re-runs the FIT+MINE SHAP interaction
screen over the WHOLE current bank (V + A + every A-routed criterion accepted so
far), excludes parents already decomposed, and returns the top-N new MIXED parents.

Selection reads FIT+MINE only -- never MONITOR, never the honest rows.

CPU only.  Usage: python new_parents.py --cell hashtagwars_verdict --round 3 --n 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import cells as C
import closure_core as L
import stage1_slice as S1

HERE = Path(__file__).resolve().parent


def main():
    import shap
    from sklearn.ensemble import HistGradientBoostingClassifier

    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()

    d = C.load(a.cell)
    sp = json.loads((HERE / f"{a.cell}_splits.json").read_text())
    fit = np.array([r["split"] for r in sp["rows"]]) == "fit_mine"
    y = d["y"][fit]

    blocks, tags = S1.current_blocks(d, a.round)
    names, mats, source = [], [], {}
    names += list(d["v_names"]); mats.append(d["V"])
    names += list(d["a_names"]); mats.append(d["A"])
    for n in d["a_names"]:
        source[n] = "frozen_bank"
    for t, B in zip(tags[2:], blocks[2:]):
        rtag = t.replace("A_round", "")
        z = np.load(HERE / f"{a.cell}_r{rtag}_scores.npz", allow_pickle=True)
        cids = [str(s) for s in z["crit_ids"]]
        cnames = [str(s) for s in z["crit_names"]]
        rt = json.loads((HERE / f"{a.cell}_r{rtag}_routing_final.json").read_text())
        a_ids = [x["blind_id"] for x in rt["final"] if x["final_route"] == "A"]
        keep = [cids.index(i) for i in a_ids if i in cids]
        for k in keep:
            names.append(cnames[k])
            source[cnames[k]] = f"mined_round_{rtag}"
        mats.append(B)
    M = np.column_stack(mats)
    assert M.shape[1] == len(names), (M.shape, len(names))

    keep, med = L.clean_fit(M[fit])
    X = L.clean_apply(M[fit], keep, med)
    kn = [names[j] for j in keep]
    is_v = np.array([n.startswith("v_") for n in kn])

    m = HistGradientBoostingClassifier(max_leaf_nodes=15, learning_rate=0.06,
                                       max_iter=400, early_stopping=True,
                                       validation_fraction=0.1, n_iter_no_change=20,
                                       random_state=0).fit(X, y)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(y), size=min(300, len(y)), replace=False)
    sv = np.abs(shap.TreeExplainer(m).shap_values(X[sub])).mean(axis=0)
    top = np.argsort(-sv)[:15]
    m2 = HistGradientBoostingClassifier(max_leaf_nodes=15, learning_rate=0.06,
                                        max_iter=400, early_stopping=True,
                                        validation_fraction=0.1, n_iter_no_change=20,
                                        random_state=0).fit(X[:, top], y)
    iv = np.abs(shap.TreeExplainer(m2).shap_interaction_values(X[sub][:, top])).mean(axis=0)
    tn = [kn[j] for j in top]
    tv = is_v[top]

    done = set()
    for prev in ("d",) + tuple(range(3, a.round)):
        f = HERE / f"{a.cell}_r{prev}_parents_used.json"
        if f.exists():
            done |= set(json.loads(f.read_text())["parents"])

    rows = []
    for i, ni in enumerate(tn):
        if tv[i] or ni in done:
            continue
        partners = sorted(((float(iv[i, j]), tn[j]) for j in range(len(tn))
                           if tv[j] and j != i), reverse=True)
        rows.append({"criterion": ni, "source": source.get(ni, "?"),
                     "mean_abs_shap": float(sv[top[i]]),
                     "surface_interaction_mass": float(sum(p[0] for p in partners)),
                     "top_surface_partners": [{"feature": p[1], "mass": p[0]}
                                              for p in partners[:3]],
                     "alone_AUC_FITMINE": L.auc(y, X[:, top[i]])})
    rows.sort(key=lambda r: -r["surface_interaction_mass"])
    out = {"cell": a.cell, "round": a.round, "n_fit_mine": int(fit.sum()),
           "bank_blocks": tags, "n_bank_features": int(X.shape[1]),
           "already_decomposed": sorted(done),
           "candidates": rows, "selected_parents": [r["criterion"] for r in rows[:a.n]]}
    (HERE / f"{a.cell}_r{a.round}_newparents.json").write_text(json.dumps(out, indent=1))
    print(f"=== {a.cell} r{a.round}: bank {X.shape[1]} feats, {len(done)} already decomposed")
    for r in rows[:a.n + 3]:
        tp = ", ".join(f"{p['feature']} {p['mass']:.3f}" for p in r["top_surface_partners"])
        mark = "SELECTED" if r["criterion"] in out["selected_parents"] else "        "
        print(f"  {mark} mass={r['surface_interaction_mass']:.3f} auc={r['alone_AUC_FITMINE']:.3f} "
              f"[{r['source']}] {r['criterion'][:44]}  <- {tp}")


if __name__ == "__main__":
    main()
