#!/usr/bin/env python3
"""Academia three-y VAT aggregation (laptop / base env with sklearn).

Reads the label-independent A/V score matrix (union_scores.npz) + the three cell
jsonls, attaches y per rung, and reports V / A / V+A grouped-CV AUC per rung, plus
the apples-to-apples curation-vs-revealed contrast on the shared-4296 set.

Run AFTER union_scores.npz has been fetched from sk3:
  python aggregate_3y.py            # writes vat_3y_results.json + prints markdown table

Design (frozen so no real-time decisions needed):
  - A = 154 rubric scores; NA -> column median over non-NA (drop all-NA cols).
  - Degeneracy guard: drop any A/V column with <5 off-modal values or zero variance.
  - Model: StandardScaler + LogisticRegression(C=1, max_iter=2000).
  - AUC = roc_auc on out-of-fold cross_val_predict, GroupKFold(5), group = ntitle.
  - V, A, V+A each fit on the SAME rows so V/A/VA are apples-to-apples within a rung.
  - Apples-to-apples pair: restrict to shared-4296, fit curation-y and revealed-y on
    the identical papers + identical A/V features -> isolates preference-variable effect.
"""
import json, os
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

D = Path(__file__).resolve().parent
NPZ = D / "union_scores.npz"

def load_cells():
    cells = {}
    for name in ["verdict", "curation", "revealed"]:
        rows = [json.loads(l) for l in open(D / f"{name}.jsonl") if l.strip()]
        cells[name] = rows
    return cells

def clean_cols(M):
    """Drop degenerate columns; median-impute NA. Returns cleaned matrix (>=20 rows assumed)."""
    keep = []
    out = []
    for j in range(M.shape[1]):
        col = M[:, j].astype(float)
        nonna = col[~np.isnan(col)]
        if len(nonna) == 0:
            continue
        med = np.median(nonna)
        c = np.where(np.isnan(col), med, col)
        vals, counts = np.unique(c, return_counts=True)
        modal = counts.max()
        offmodal = len(c) - modal
        if offmodal < 5 or c.std() == 0:      # degeneracy guard
            continue
        keep.append(j); out.append(c)
    if not out:
        return np.zeros((M.shape[0], 0)), keep
    return np.column_stack(out), keep

def auc_cv(Xf, y, groups):
    if Xf.shape[1] == 0 or len(np.unique(y)) < 2:
        return float("nan")
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    gkf = GroupKFold(n_splits=n_splits)
    proba = cross_val_predict(clf, Xf, y, cv=gkf, groups=groups, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))

def _valid_y(r):
    j = r.get("judgement")
    try:
        return int(float(j)) in (0, 1)
    except (TypeError, ValueError):
        return False

def rung_row(rows, X_by_nt, V_by_nt, restrict_shared=False, only_nt=None):
    R = [r for r in rows if r.get("ntitle") in X_by_nt and _valid_y(r)
         and (not restrict_shared or r.get("in_shared"))
         and (only_nt is None or r.get("ntitle") in only_nt)]
    if len(R) < 20:
        return None
    nt = [r["ntitle"] for r in R]
    y = np.array([int(float(r["judgement"])) for r in R])
    A = np.array([X_by_nt[k] for k in nt], dtype=float)
    V = np.array([V_by_nt[k] for k in nt], dtype=float)
    groups = np.array(nt)
    Ac, _ = clean_cols(A); Vc, _ = clean_cols(V)
    VA = np.column_stack([Vc, Ac]) if Vc.shape[1] and Ac.shape[1] else (Vc if Vc.shape[1] else Ac)
    out = {"n": len(R), "pos": float(y.mean()),
           "V": auc_cv(Vc, y, groups), "A": auc_cv(Ac, y, groups),
           "VA": auc_cv(VA, y, groups)}
    out["A_minus_V"] = out["A"] - out["V"]
    return out

def main():
    assert NPZ.exists(), f"missing {NPZ} — fetch from sk3 first (see runbook)"
    z = np.load(NPZ, allow_pickle=True)
    X, V, nt = z["X"], z["V"], z["ntitle"]
    na_rate = float(z["na_rate"]) if "na_rate" in z else float(np.isnan(X).mean())
    X_by_nt = {nt[i]: X[i] for i in range(len(nt))}
    V_by_nt = {nt[i]: V[i] for i in range(len(nt))}
    cells = load_cells()

    res = {"na_rate": na_rate, "rungs": {}, "apples_to_apples": {}}
    for name in ["verdict", "curation", "revealed"]:
        res["rungs"][name] = rung_row(cells[name], X_by_nt, V_by_nt)
    # apples-to-apples: curation-y vs revealed-y on the COMMON set — papers that are
    # shared, scored, AND carry a valid y in BOTH cells (citations drops mid-tercile).
    def labeled_nt(name):
        return {r["ntitle"] for r in cells[name]
                if r.get("in_shared") and _valid_y(r) and r.get("ntitle") in X_by_nt}
    common = labeled_nt("curation") & labeled_nt("revealed")
    res["apples_to_apples"]["common_n"] = len(common)
    res["apples_to_apples"]["curation"] = rung_row(cells["curation"], X_by_nt, V_by_nt, only_nt=common)
    res["apples_to_apples"]["revealed"] = rung_row(cells["revealed"], X_by_nt, V_by_nt, only_nt=common)

    (D / "vat_3y_results.json").write_text(json.dumps(res, indent=2))
    # markdown
    print(f"\nA-bank NA rate: {na_rate:.3f}\n")
    print("### Nested selectivity ladder (all ICLR papers per rung)")
    print("| rung | n | pos | V | A | V+A | A−V |")
    print("|---|---|---|---|---|---|---|")
    for name in ["verdict", "curation", "revealed"]:
        r = res["rungs"][name]
        if r: print(f"| {name} | {r['n']} | {r['pos']:.3f} | {r['V']:.3f} | {r['A']:.3f} | {r['VA']:.3f} | {r['A_minus_V']:+.3f} |")
    print(f"\n### Apples-to-apples (identical {res['apples_to_apples']['common_n']} papers labeled in BOTH cells, same A/V features)")
    print("| preference y | n | pos | V | A | V+A | A−V |")
    print("|---|---|---|---|---|---|---|")
    for name in ["curation", "revealed"]:
        r = res["apples_to_apples"][name]
        if r: print(f"| {name} | {r['n']} | {r['pos']:.3f} | {r['V']:.3f} | {r['A']:.3f} | {r['VA']:.3f} | {r['A_minus_V']:+.3f} |")
    print(f"\nwrote {D/'vat_3y_results.json'}")

if __name__ == "__main__":
    main()
