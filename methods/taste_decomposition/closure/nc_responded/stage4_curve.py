#!/usr/bin/env python3
"""Recompute EVERY bank state of the closure curve in one pass, identical estimator,
identical split (the pilot's `round4_results.json` pattern).

  python stage4_curve.py --upto 0      # round-0 baseline only
  python stage4_curve.py --upto 2      # states 0,1,2

Writes roundN_results.json (N = --upto) carrying the whole curve so far, plus
round-over-round docket-level paired bootstrap gains and the frozen saturation flags.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import nc_closure_lib as L
from readout import fit_state, load_dense

HERE = Path(__file__).resolve().parent
EPS = 0.005


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upto", type=int, required=True)
    a = ap.parse_args()

    dense = load_dense()
    pop = L.load_population()
    summary, split, dsplit, mining, monitor_full = L.load_splits()
    y, docket = pop["y"], pop["docket"]

    states, preds = [], {}
    for r in range(0, a.upto + 1):
        t0 = time.time()
        res, P = fit_state(list(range(1, r + 1)), dense_prob=dense, tag=f"round{r}",
                           save_preds=f"state{r}_preds.npz")
        res["seconds"] = round(time.time() - t0)
        states.append(res)
        preds[r] = P
        print(f"[state {r}] feats={res['n_features']} "
              f"MONITOR_FULL VA_nl={res['monitor_full']['VA_nl']:.4f} "
              f"MONITOR VA_nl={res['monitor']['VA_nl']:.4f} "
              f"honest Delta={res['honest'].get('Delta', float('nan')):.4f} "
              f"({res['seconds']}s)", flush=True)

    gains = []
    for r in range(1, a.upto + 1):
        row = {"round": r}
        for label, mask in (("monitor_full", monitor_full),
                            ("monitor", split == "monitor"),
                            ("honest", np.isin(dsplit, ["eval", "test"]))):
            bs = L.group_bootstrap_delta(
                y[mask], preds[r - 1]["nl_mean"][mask], preds[r]["nl_mean"][mask],
                docket[mask])
            row[label] = {
                "prev": states[r - 1][label]["VA_nl"], "cur": states[r][label]["VA_nl"],
                "gain": states[r][label]["VA_nl"] - states[r - 1][label]["VA_nl"],
                "boot": bs,
            }
        g_ = row["monitor_full"]["gain"]
        # FROZEN rule, literal text: "2 consecutive rounds with MONITOR VA_nl gain
        # < eps".  `gain < eps` is SIGNED, so a NEGATIVE gain is sub-eps -- a round
        # whose criteria made the monitor readout worse is certainly not evidence
        # that mining is still buying anything.  The magnitude variant
        # (|gain| < eps) is the conservative reading, which keeps mining longer;
        # it is reported alongside but is NOT the declared rule.
        row["sub_eps_signed_FROZEN"] = bool(g_ < EPS)
        row["sub_eps_magnitude_variant"] = bool(abs(g_) < EPS)
        gains.append(row)

    def trail(flags):
        t = 0
        for f in reversed(flags):
            if f:
                t += 1
            else:
                break
        return t

    flags = [g["sub_eps_signed_FROZEN"] for g in gains]
    flags_mag = [g["sub_eps_magnitude_variant"] for g in gains]
    trailing = trail(flags)
    trailing_mag = trail(flags_mag)

    out = {
        "cell": "nc_responded",
        "protocol": "notes/2026-08-05__layer3-closure-prereg.md FREEZE DECLARATION 2026-08-06",
        "eps": EPS,
        "saturation_statistic": "VA_nl gain on MONITOR_FULL (n=1,892, VA-honest)",
        "stopping_rule_FROZEN": "2 consecutive rounds with gain < eps (signed, prereg text)",
        "splits": summary,
        "states": states,
        "gains": gains,
        "sub_eps_flags": flags,
        "trailing_sub_eps_run": trailing,
        "saturation_declared": bool(trailing >= 2),
        "sub_eps_flags_magnitude_variant": flags_mag,
        "trailing_sub_eps_run_magnitude_variant": trailing_mag,
        "saturation_magnitude_variant": bool(trailing_mag >= 2),
    }
    (HERE / f"round{a.upto}_results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"flags": flags, "trailing": trailing,
                      "saturation": out["saturation_declared"]}, indent=1))


if __name__ == "__main__":
    main()
