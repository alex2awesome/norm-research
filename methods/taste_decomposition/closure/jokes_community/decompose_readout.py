#!/usr/bin/env python3
"""Readout for the FREEZE-ADDENDUM-3 decomposition pass (round tag `rd`).

The pass asks one question per parent: is the parent bank criterion's predictive
power CRAFT or the SURFACE CARRIER it is entangled with?  Five readouts answer it,
all on populations where both instruments are out-of-sample.

  1. alone-AUC of the parent P, its candidate-real component R and its surface
     component F, on FIT+MINE and on the HONEST (dense-held-out) rows.
  2. separability: Spearman(R, F), Spearman(P, R), Spearman(P, F).
  3. THE DECISIVE TEST -- P's AUC stratified on deciles of F, and R's AUC
     stratified on deciles of F.  A parent whose signal is the carrier collapses
     to chance inside F strata; a component that isolates craft does not.
  4. the blind audit's INDEPENDENT routing of each component, and the resulting
     verdict per parent (both components survived / the candidate-real component
     was routed to B, i.e. the decomposition failed and the parent was surface
     all along -- the N&C round-5 pattern).
  5. bank sensitivity: VA_nl for the frozen Layer-1 bank, for bank + A-routed
     components, and for (bank - parents) + A-routed components.  The frozen bank
     is never modified in place; this is a reported sensitivity, not a redefinition.

CPU only.  Usage: python decompose_readout.py --cell hashtagwars_verdict
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import cells as C
import closure_core as L

HERE = Path(__file__).resolve().parent


def strat_auc(y, s, by, q=10, min_n=20):
    st = L.decile_strata(by, q=q)
    a, info = L.stratified_auc(y, s, st, min_n=min_n)
    return a, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    a = ap.parse_args()
    cell = a.cell
    tag = f"{cell}_rd"

    d = C.load(cell)
    sp = json.loads((HERE / f"{cell}_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    y, g, dense = d["y"], d["groups"], d["dense"]
    fitm, monm = split == "fit_mine", split == "monitor"
    held = np.isin(d["dense_split"], ["eval", "test"])

    z = np.load(HERE / f"{tag}_scores.npz", allow_pickle=True)
    assert (z["i"] == np.arange(len(y))).all()
    Xnew, cids = z["X"], [str(s) for s in z["crit_ids"]]
    sel = {c["blind_id"]: c for c in json.loads((HERE / f"{tag}_species.json").read_text())["selected"]}
    routing = {r["blind_id"]: r for r in
               json.loads((HERE / f"{tag}_routing_final.json").read_text())["final"]}

    # parent columns from the frozen A bank
    a_names = d["a_names"]
    parents = json.loads((HERE / f"{cell}_rd_parents_used.json").read_text())["parents"]

    def col_of(bid):
        return Xnew[:, cids.index(bid)]

    def clean1(v):
        v = v.astype(float).copy()
        m = np.isfinite(v)
        v[~m] = float(np.nanmedian(v[m])) if m.any() else 0.0
        return v

    rows = []
    for p in parents:
        P = clean1(d["A"][:, a_names.index(p)])
        kids = [b for b, c in sel.items() if c.get("parent") == p]
        R = [b for b in kids if sel[b]["component_kind"] == "candidate_real"]
        F = [b for b in kids if sel[b]["component_kind"] == "surface"]
        if not (R and F):
            continue
        rb, fb = R[0], F[0]
        Rv, Fv = clean1(col_of(rb)), clean1(col_of(fb))
        rec = {
            "parent": p,
            "component_real": {"blind_id": rb, "name": sel[rb]["name"],
                               "audit_label": routing[rb]["audit_label"],
                               "final_route": routing[rb]["final_route"],
                               "route_source": routing[rb]["route_source"]},
            "component_surface": {"blind_id": fb, "name": sel[fb]["name"],
                                  "audit_label": routing[fb]["audit_label"],
                                  "final_route": routing[fb]["final_route"],
                                  "route_source": routing[fb]["route_source"]},
            "alone_AUC": {
                "parent_FITMINE": L.auc(y[fitm], P[fitm]),
                "parent_HONEST": L.auc(y[held], P[held]),
                "real_FITMINE": L.auc(y[fitm], Rv[fitm]),
                "real_HONEST": L.auc(y[held], Rv[held]),
                "surface_FITMINE": L.auc(y[fitm], Fv[fitm]),
                "surface_HONEST": L.auc(y[held], Fv[held]),
            },
            "separability": {
                "spearman_real_vs_surface": float(spearmanr(Rv, Fv).statistic),
                "spearman_parent_vs_real": float(spearmanr(P, Rv).statistic),
                "spearman_parent_vs_surface": float(spearmanr(P, Fv).statistic),
            },
        }
        for lab, vec in (("parent", P), ("real", Rv)):
            au, info = strat_auc(y[held], vec[held], Fv[held])
            rec.setdefault("stratified_on_surface_component_HONEST", {})[lab] = {
                "AUC_adj": au, **info}
        # also stratify on the programmatic carrier the SHAP screen named
        rec["decomposition_verdict"] = (
            "BOTH COMPONENTS SURVIVED INDEPENDENT ROUTING"
            if routing[rb]["final_route"] == "A" and routing[fb]["final_route"] == "B"
            else "DECOMPOSITION FAILED -- candidate-real component routed to B "
                 "(the parent was surface all along)"
            if routing[rb]["final_route"] == "B"
            else "SURFACE COMPONENT ROUTED TO A (auditor judged the carrier itself "
                 "quality-relevant)")
        rows.append(rec)

    # ------------------------------------------------- bank sensitivity -----
    A_ids = [b for b, r in routing.items() if r["final_route"] == "A"]
    XA = Xnew[:, [cids.index(i) for i in A_ids]] if A_ids else np.zeros((len(y), 0))
    par_idx = [a_names.index(p) for p in parents if p in a_names]
    keep_cols = [j for j in range(d["A"].shape[1]) if j not in par_idx]

    def block(blocks):
        r = L.fit_block(blocks, fitm, monm, y, g)
        v = np.full(len(y), np.nan)
        v[fitm] = r["oof_nl_fitmine"]
        v[monm] = r["nl_mon"]
        return {"n_features": r["n_features"],
                "VA_nl_HONEST": L.auc(y[held], v[held]),
                "VA_nl_MONITOR": L.auc(y[monm], r["nl_mon"]),
                "VA_lin_MONITOR": L.auc(y[monm], r["lin_mon"])}, v

    base, v_base = block([d["V"], d["A"]])
    plus, v_plus = block([d["V"], d["A"], XA])
    repl, v_repl = block([d["V"], d["A"][:, keep_cols], XA])
    drop, v_drop = block([d["V"], d["A"][:, keep_cols]])

    T_h = L.auc(y[held], dense[held])
    out = {"cell": cell, "tag": tag, "n_parents": len(rows),
           "T_HONEST_ensemble": T_h,
           "routing": {"misrouting_rate": json.loads(
               (HERE / f"{tag}_routing_final.json").read_text())["misrouting_rate"],
               "probe_pass": json.loads(
                   (HERE / f"{tag}_routing_final.json").read_text())["probe_pass"],
               "n_final_A": len(A_ids)},
           "score_report": json.loads((HERE / f"{tag}_score_report.json").read_text()),
           "parents": rows,
           "bank_sensitivity": {
               "frozen_layer1_bank": base,
               "bank_plus_A_routed_components": plus,
               "bank_minus_parents_plus_components": repl,
               "bank_minus_parents_only": drop,
               "Delta_beyond_HONEST": {
                   "frozen": T_h - base["VA_nl_HONEST"],
                   "plus_components": T_h - plus["VA_nl_HONEST"],
                   "replaced": T_h - repl["VA_nl_HONEST"],
                   "parents_dropped": T_h - drop["VA_nl_HONEST"]},
               "note": "the frozen Layer-1 bank is never modified in place; these are "
                       "reported sensitivities"},
           }
    np.savez(HERE / f"{cell}_rd_preds.npz", va_base=v_base, va_plus=v_plus,
             va_repl=v_repl, va_drop=v_drop)
    (HERE / f"{tag}_results.json").write_text(json.dumps(out, indent=1, default=float))

    print(f"=== {cell} decomposition pass ({len(rows)} parents)")
    for r in rows:
        au = r["alone_AUC"]
        print(f"  {r['parent'][:38]:38s} P {au['parent_HONEST']:.3f} | "
              f"R {au['real_HONEST']:.3f} ({r['component_real']['final_route']}) | "
              f"F {au['surface_HONEST']:.3f} ({r['component_surface']['final_route']}) | "
              f"P|F {r['stratified_on_surface_component_HONEST']['parent']['AUC_adj']:.3f} "
              f"R|F {r['stratified_on_surface_component_HONEST']['real']['AUC_adj']:.3f} | "
              f"rho(R,F) {r['separability']['spearman_real_vs_surface']:+.2f}")
    bs = out["bank_sensitivity"]
    print(f"  bank VA_nl HONEST: frozen {bs['frozen_layer1_bank']['VA_nl_HONEST']:.4f} "
          f"| +components {bs['bank_plus_A_routed_components']['VA_nl_HONEST']:.4f} "
          f"| replaced {bs['bank_minus_parents_plus_components']['VA_nl_HONEST']:.4f} "
          f"| parents dropped {bs['bank_minus_parents_only']['VA_nl_HONEST']:.4f}")
    print(f"  T {T_h:.4f}; Delta_beyond {json.dumps({k: round(v,4) for k,v in bs['Delta_beyond_HONEST'].items()})}")


if __name__ == "__main__":
    main()
