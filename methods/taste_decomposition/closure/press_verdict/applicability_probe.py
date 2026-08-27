#!/usr/bin/env python3
"""ROUND-0 diagnostic the census forced: where does the A bank's .67 actually come from?

The census found that all 37 surviving rubric columns are individually at chance
(max alone-AUC .527, median .496, MAD .0069, none at .55) while Layer 1 reports
A_lin .669 / A_nl .674.  A bank of near-null columns that aggregates to .67 has to be
carrying the signal somewhere other than in its judged LEVELS, and this cell's A matrix
has an obvious candidate: it is APPLICABILITY-GATED.  Each rubric comes with an
`applicable` bit, and Layer 1 turns an inapplicable cell into the constant 0.5, so the
MISSINGNESS PATTERN -- which of 40 news-and-science-communication rubrics a judge
thinks even apply to this release -- is silently a 40-bit feature vector describing the
release's genre.

This script decomposes the A block into
    MASK   the 40 applicability bits, nothing else
    LEVELS the judged 0-10 values, with the mask's information removed by imputing
           inapplicable cells at the FIT+MINE median of the applicable ones
    BOTH   Layer 1's own constant-0.5 matrix, which carries both
and fits the frozen closure spec to each.  It also reports the alone-AUC of every
single applicability bit.

If MASK alone approaches the full A number, the press A bank is largely a genre
detector, and that is a Track-B finding sitting inside the A block -- which is exactly
what the dual-track design exists to catch.

CPU only.  Usage: python applicability_probe.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import cells as C
import closure_core as L

HERE = Path(__file__).resolve().parent


def main():
    C.sklearn_guard()
    d = C.load()
    sp = json.loads((HERE / "press_verdict_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    fitm, monm = split == "fit_mine", split == "monitor"
    held = np.isin(d["dense_split"], ["eval", "test"])
    y, g, names = d["y"], d["groups"], d["a_names"]

    MASK = np.isfinite(d["A_median_impute"]).astype(float)
    LEV = d["A_median_impute"].copy()          # NaN where inapplicable; clean_fit median-imputes
    BOTH = d["A_const05"]

    out = {"cell": "press_verdict",
           "question": "does the A bank's aggregate AUC come from the judged LEVELS or from "
                       "the APPLICABILITY MASK (a genre fingerprint)?",
           "blocks": {}}

    def fit(label, blocks):
        r = L.fit_block(blocks, fitm, monm, y, g)
        v = np.full(len(y), np.nan)
        v[fitm] = r["oof_nl_fitmine"]
        v[monm] = r["nl_mon"]
        out["blocks"][label] = {
            "n_features": r["n_features"],
            "lin_MONITOR": L.auc(y[monm], r["lin_mon"]),
            "nl_MONITOR": L.auc(y[monm], r["nl_mon"]),
            "nl_HONEST": L.auc(y[held], v[held]),
            "lin_OOF_fitmine": L.auc(y[fitm], r["oof_lin_fitmine"]),
            "nl_OOF_fitmine": L.auc(y[fitm], r["oof_nl_fitmine"])}
        print(label, json.dumps(out["blocks"][label]), flush=True)
        return v

    fit("A_mask_only", [MASK])
    fit("A_levels_only_median_imputed", [LEV])
    fit("A_layer1_const05", [BOTH])
    fit("A_mask_plus_levels", [MASK, LEV])
    fit("V_only", [d["V"]])
    fit("V_plus_mask", [d["V"], MASK])

    per = []
    for j, nm in enumerate(names):
        col = MASK[:, j]
        if col[fitm].std() == 0:
            continue
        per.append({"rubric": nm, "applicability_rate": float(col.mean()),
                    "mask_bit_alone_AUC_FITMINE": L.auc(y[fitm], col[fitm])})
    per.sort(key=lambda r: -abs(r["mask_bit_alone_AUC_FITMINE"] - .5))
    out["per_rubric_mask_bit"] = per
    out["n_applicable_alone_AUC_FITMINE"] = L.auc(y[fitm], MASK[fitm].sum(axis=1))
    out["mean_n_applicable"] = float(MASK.sum(axis=1).mean())

    (HERE / "applicability_probe.json").write_text(json.dumps(out, indent=1, default=float))
    print("\nn_applicable count alone:", round(out["n_applicable_alone_AUC_FITMINE"], 4))
    print("TOP applicability bits (FIT+MINE alone-AUC):")
    for p in per[:10]:
        print(f"  {p['mask_bit_alone_AUC_FITMINE']:.3f}  rate={p['applicability_rate']:.2f}  {p['rubric'][:58]}")


if __name__ == "__main__":
    main()
