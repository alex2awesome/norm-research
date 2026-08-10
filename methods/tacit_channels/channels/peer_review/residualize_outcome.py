"""5.1 residualizer - isolate the tacit component of the distal outcome, with the stop-rule.

Regresses the target model's HOLISTIC outcome judgment on its ARTICULABLE componential
judgments (train split only); the residual is "the part of the outcome the articulable
metrics cannot explain" - the training signal for the reward channel. Emits residual soft
labels + fitted values (the fitted-values arm is the double-dissociation control) +
diagnostics.

STOP-RULE (gate G3, audit): if residual variance is not credibly above the noise floor
implied by the outcome's own test-retest reliability, there is NOTHING TO INSTALL - the run
reports and exits nonzero rather than training on noise.

Input rows (jsonl): {"item_id": ..., "holistic": float, "components": {metric_id: float}}
All values are TARGET-MODEL judgments (no human labels).

CPU-only, numpy least squares - no sklearn dependency.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from methods.tacit_channels.channels.common import read_jsonl, stable_split, write_jsonl


def fit_residuals(rows: list[dict], component_ids: list[str]):
    for r in rows:
        r["split"] = r.get("split") or stable_split(str(r["item_id"]))
    train = [r for r in rows if r["split"] == "train"]
    if len(train) < len(component_ids) + 2:
        raise SystemExit(f"too few train rows ({len(train)}) for "
                         f"{len(component_ids)} components")

    def design(subset):
        X = np.array([[r["components"][c] for c in component_ids] for r in subset], float)
        return np.hstack([X, np.ones((len(subset), 1))])  # intercept

    y_tr = np.array([r["holistic"] for r in train], float)
    beta, *_ = np.linalg.lstsq(design(train), y_tr, rcond=None)

    X_all = design(rows)
    y_all = np.array([r["holistic"] for r in rows], float)
    fitted = X_all @ beta
    resid = y_all - fitted

    ss_res = float(np.sum((y_tr - design(train) @ beta) ** 2))
    ss_tot = float(np.sum((y_tr - y_tr.mean()) ** 2))
    r2_train = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return beta, fitted, resid, r2_train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True,
                    help="jsonl of {item_id, holistic, components{...}} target judgments")
    ap.add_argument("--components", required=True,
                    help="comma list of ARTICULABLE metric ids to residualize on")
    ap.add_argument("--outcome-reliability", type=float, default=None,
                    help="test-retest reliability of the holistic outcome "
                         "(from reliability_ceilings.py) - enables the stop-rule")
    ap.add_argument("--noise-margin", type=float, default=1.25,
                    help="residual variance must exceed noise-floor variance by this factor")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.rows)
    component_ids = args.components.split(",")
    beta, fitted, resid, r2_train = fit_residuals(rows, component_ids)

    var_y = float(np.var([r["holistic"] for r in rows]))
    var_resid = float(np.var(resid))
    diag = {
        "n_rows": len(rows), "components": component_ids,
        "beta": {c: round(float(b), 4) for c, b in zip(component_ids, beta[:-1])},
        "intercept": round(float(beta[-1]), 4),
        "r2_train": round(r2_train, 4),
        "var_holistic": round(var_y, 6), "var_residual": round(var_resid, 6),
        "residual_share": round(var_resid / var_y, 4) if var_y > 0 else None,
    }

    # STOP-RULE: noise variance in the outcome = (1 - reliability) * var(y). If the residual
    # is not credibly larger than that, it is measurement noise, not a tacit component.
    if args.outcome_reliability is not None:
        noise_var = (1 - args.outcome_reliability) * var_y
        diag["noise_floor_var"] = round(noise_var, 6)
        diag["residual_over_noise"] = round(var_resid / noise_var, 3) if noise_var > 0 else None
        diag["stop_rule_pass"] = bool(var_resid > args.noise_margin * noise_var)
    else:
        diag["stop_rule_pass"] = None
        diag["warning"] = ("no --outcome-reliability supplied: stop-rule NOT evaluated; "
                           "do not train on this residual for a confirmatory run")

    out_rows = [{"item_id": r["item_id"], "split": r["split"],
                 "holistic": r["holistic"],
                 "fitted_articulable": round(float(f), 6),
                 "residual": round(float(e), 6)}
                for r, f, e in zip(rows, fitted, resid)]
    write_jsonl(args.out, out_rows)
    diag_path = args.out.replace(".jsonl", "_diagnostics.json")
    json.dump(diag, open(diag_path, "w"), indent=2)
    print(json.dumps(diag, indent=2))

    if diag["stop_rule_pass"] is False:
        raise SystemExit("STOP-RULE (G3): residual variance is not credibly above the "
                         "outcome's noise floor - there is nothing to install. "
                         "Report this; do NOT train the reward channel on this residual.")


if __name__ == "__main__":
    main()
