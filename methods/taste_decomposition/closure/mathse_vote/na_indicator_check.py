#!/usr/bin/env python3
"""DEVIATION CHECK: the closure protocol's imputation drops the APPLICABILITY PATTERN.

`closure_core.clean_fit` median-imputes an inapplicable rubric cell and stops there.
This cell's Layer-1 linear leg used `SimpleImputer(median, add_indicator=True)`, i.e. it
ALSO gave the model a 0/1 "the judge said this criterion does not apply here" column per
rubric.  With an A-block NA rate of .2396 that pattern is not a rounding detail, and
dropping it can only make the articulated instrument WEAKER -- which would inflate
Delta_beyond.

This script measures the size of that effect on MONITOR by refitting the bank with the
NA indicators appended, under the otherwise-identical closure protocol.  It is a
SENSITIVITY, not a change of protocol: the frozen closure spec stays as it is and every
round's curve is read under it.  What this establishes is how much of any residual could
be an artifact of the protocol's own imputation.

CPU only.  Usage: OMP_NUM_THREADS=6 python3 na_indicator_check.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import cells as C
import closure_core as L
from position_line import within_question_auc

HERE = Path(__file__).resolve().parent


def main():
    d = C.load()
    sp = json.loads((HERE / "mathse_vote_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    fitm, monm = split == "fit_mine", split == "monitor"
    y, g = d["y"], d["groups"]
    gm = np.array([str(x) for x in g[monm]])

    NA = np.isnan(d["A"]).astype(float)
    r_plain = L.fit_block([d["V"], d["A"]], fitm, monm, y, g, want_oof=False)
    r_ind = L.fit_block([d["V"], d["A"], NA], fitm, monm, y, g, want_oof=False)
    r_naonly = L.fit_block([NA], fitm, monm, y, g, want_oof=False)

    T = float(roc_auc_score(y[monm], d["dense"][monm]))
    out = {
        "cell": "mathse_vote", "sklearn": C.sklearn_guard(),
        "A_na_rate": float(np.isnan(d["A"]).mean()),
        "n_na_indicator_cols_after_screen": r_ind["n_features"] - r_plain["n_features"],
        "MONITOR": {
            "T": T,
            "VA_nl_frozen_protocol": float(roc_auc_score(y[monm], r_plain["nl_mon"])),
            "VA_nl_plus_NA_indicators": float(roc_auc_score(y[monm], r_ind["nl_mon"])),
            "NA_pattern_alone_nl": float(roc_auc_score(y[monm], r_naonly["nl_mon"])),
            "VA_lin_frozen_protocol": float(roc_auc_score(y[monm], r_plain["lin_mon"])),
            "VA_lin_plus_NA_indicators": float(roc_auc_score(y[monm], r_ind["lin_mon"])),
        },
        "MONITOR_within_question": {
            "T": within_question_auc(y[monm], d["dense"][monm], gm)[0],
            "VA_nl_frozen_protocol": within_question_auc(y[monm], r_plain["nl_mon"], gm)[0],
            "VA_nl_plus_NA_indicators": within_question_auc(y[monm], r_ind["nl_mon"], gm)[0],
        },
    }
    m = out["MONITOR"]
    out["Delta_frozen_protocol"] = m["T"] - m["VA_nl_frozen_protocol"]
    out["Delta_with_NA_indicators"] = m["T"] - m["VA_nl_plus_NA_indicators"]
    out["imputation_cost_to_Delta"] = out["Delta_frozen_protocol"] - out["Delta_with_NA_indicators"]
    out["reading"] = ("positive imputation_cost means the frozen protocol's median-impute "
                      "understates the articulated bank and therefore OVERSTATES the "
                      "residual by that amount; this is a sensitivity, the frozen spec is "
                      "not changed and every round's curve is read under it.")
    (HERE / "na_indicator_check.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
