#!/usr/bin/env python3
"""TERMINAL LEDGER for cap_crowd.

jokes_community/terminal_ledger.py is written for a cell with TWO T conventions
(mean-over-seeds-of-the-AUC vs AUC-of-the-seed-mean) and calls cells.T_by_seed.  This
cell has ONE T convention -- a single dense probability column that reproduces the master
ledger's .6124 exactly -- so that machinery does not apply and this assembles the ledger
from the round artifacts instead.  Nothing is recomputed here that a round readout
already computed under the same gate and the same view; the point of the file is that the
terminal numbers and their provenance live in one place.

CPU only.  Usage: python terminal_ledger_cap.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    vr = json.loads((HERE / "cap_crowd_r3_viewrepair_results.json").read_text())
    rep = vr["views"]["REPAIRED (cartoon+caption)"]
    r4 = json.loads((HERE / "cap_crowd_r4_results.json").read_text())
    r5 = json.loads((HERE / "cap_crowd_r5_results.json").read_text())
    disc = json.loads((HERE / "cap_crowd_r5_cumulative_discount.json").read_text())
    gepa = json.loads((HERE / "cap_crowd_gepa_targets.json").read_text())
    line = json.loads((HERE / "cap_crowd_contest_line.json").read_text())
    mj4 = json.loads((HERE / "cap_crowd_r4_merged_jackknife.json").read_text())
    mj5 = json.loads((HERE / "cap_crowd_r5_merged_jackknife.json").read_text())

    def rnd(r, key):
        tA = r["track_A"]
        after = tA[f"state_after_round_{r['round']}"]
        return {
            "kind": key,
            "n_features_after": after["n_features"],
            "VA_nl_MONITOR": after["VA_nl_MONITOR"],
            "VA_nl_HONEST": after["VA_nl_HONEST"],
            "gain_MONITOR": tA["gain_MONITOR"],
            "gain_ci_MONITOR_group": tA["gain_ci_MONITOR"],
            "gain_HONEST": tA["gain_HONEST"],
            "gain_ci_HONEST_group": tA["gain_ci_HONEST"],
            "gain_MONITOR_within_contest": tA["gain_MONITOR_within_contest"],
            "sub_epsilon": bool(tA["gain_MONITOR"] < 0.005),
            "Delta_beyond_MONITOR": tA[f"Delta_beyond_MONITOR_new"],
            "Delta_beyond_HONEST": tA[f"Delta_beyond_HONEST_new"],
            "probe_pass": r["routing"]["probe_pass"],
            "misrouting_rate": r["routing"]["misrouting_rate"],
            "swap": r["swap_pair_HONEST"]["delta"],
            "stacked_dense_increment_over_B_plus_bank_HONEST":
                r["stacked_increment_HONEST"]["dense_increment_over_B_plus_bank"],
        }

    out = {
        "cell": "cap_crowd",
        "y": "EDITOR finalist selection (3 finalists vs ~20 hard negatives per contest)",
        "populations": vr["n"],
        "T": vr["T"],
        "T_convention": "ONE convention on this cell (single dense probability column; "
                        "reproduces master-ledger T = .6124 on the 1,055 dense-held-out rows)",
        "collapse_gate": {
            "rule": "modal_frac > .98 dropped inside clean_fit, enforced on EVERY refit",
            "effect_on_bank_0": "345 features under the historic off-modal<5 screen -> "
                                "289 under the enforced gate (55 A criteria + 1 V column "
                                "dropped); VA_nl MONITOR .6919 -> .6767",
        },
        "item_view": {
            "rule": 'CARTOON: <desc>\\n\\nCAPTION: "<text>" -- matched to the A bank',
            "rounds_1_2": "scored WITHOUT the cartoon (defect); re-scored on the matched "
                          "view by the round-3 TIER-R view-repair pass",
        },
        "rounds": {
            "0": {"kind": "baseline", "n_features_after": 289,
                  "VA_nl_MONITOR": rep["round1"]["VA_nl_MONITOR_before"],
                  "VA_nl_HONEST": rep["round1"]["VA_nl_HONEST_before"]},
            "1": {"kind": "proposing (DEGRADED P=4/2fam, view-repaired)",
                  "n_features_after": rep["round1"]["n_features_after"],
                  "VA_nl_MONITOR": rep["round1"]["VA_nl_MONITOR_after"],
                  "VA_nl_HONEST": rep["round1"]["VA_nl_HONEST_after"],
                  "gain_MONITOR": rep["round1"]["gain_MONITOR"],
                  "gain_ci_MONITOR_group": rep["round1"]["gain_ci_MONITOR_group"],
                  "gain_HONEST": rep["round1"]["gain_HONEST"],
                  "counts_toward_clock": False,
                  "swap": rep["round1"]["swap"]["delta"]},
            "2": {"kind": "proposing (DEGRADED P=4/2fam, view-repaired)",
                  "n_features_after": rep["round2"]["n_features_after"],
                  "VA_nl_MONITOR": rep["round2"]["VA_nl_MONITOR_after"],
                  "VA_nl_HONEST": rep["round2"]["VA_nl_HONEST_after"],
                  "gain_MONITOR": rep["round2"]["gain_MONITOR"],
                  "gain_ci_MONITOR_group": rep["round2"]["gain_ci_MONITOR_group"],
                  "gain_HONEST": rep["round2"]["gain_HONEST"],
                  "counts_toward_clock": False,
                  "swap": rep["round2"]["swap"]["delta"]},
            "3": {"kind": "VIEW REPAIR (TIER R) -- no fleet, exempt from clock and mass"},
            "4": {**rnd(r4, "proposing (FULL P=8 / 3 families)"),
                  "counts_toward_clock": True},
            "5": {**rnd(r5, "proposing (DEGRADED P=5 / 2 families -- Claude subagent "
                             "budget exhausted; CAP round)"),
                  "counts_toward_clock": False},
        },
        "stopping": {
            "epsilon": 0.005,
            "rule": "2 consecutive sub-epsilon PROPOSING rounds at current standard, or cap 5 rounds",
            "clock_trace": "r1 n/a (degraded, registered) -> r2 n/a (degraded, registered) "
                           "-> r3 exempt (TIER R) -> r4 SUB-EPSILON, clock 1 "
                           "-> r5 +.0204 NOT sub-epsilon, clock RESET to 0",
            "clock_at_termination": 0,
            "terminated_on": "CAP (5 rounds). The stopping rule did NOT fire; the closure "
                             "curve is still gaining at the cap.",
        },
        "terminal_state": {
            "n_bank_features": disc["n_bank_features"],
            "VA_nl_MONITOR": disc["VA_nl_MONITOR"],
            "VA_nl_HONEST": disc["VA_nl_HONEST"],
            "Delta_beyond_MONITOR": disc["Delta_MONITOR"],
            "Delta_beyond_HONEST": disc["Delta_HONEST"],
            "total_gain_MONITOR_from_bank0": disc["VA_nl_MONITOR"]
            - rep["round1"]["VA_nl_MONITOR_before"],
            "total_gain_HONEST_from_bank0": disc["VA_nl_HONEST"]
            - rep["round1"]["VA_nl_HONEST_before"],
        },
        "discount": {
            "n_nuisance_channels": disc["n_B_channels"],
            "ALL_B": {p: {k: v for k, v in disc["ALL_B"][p].items()
                          if not isinstance(v, dict)} for p in ("HONEST", "MONITOR")},
            "STRICT_no_mixed": {p: {k: v for k, v in disc["STRICT_no_mixed"][p].items()
                                    if not isinstance(v, dict)}
                                for p in ("HONEST", "MONITOR")},
            "stacked_increment_of_record": {
                band: {p: disc[band][p]["stacked"] for p in ("HONEST", "MONITOR")}
                for band in ("ALL_B", "STRICT_no_mixed")},
        },
        "missing_mass_strict_merged": {
            "r4_FULL_FLEET_P8_3fam": {"A": mj4["A"] if "A" in mj4 else mj4.get("tracks", {}).get("A"),
                                      "B": mj4["B"] if "B" in mj4 else mj4.get("tracks", {}).get("B")},
            "r5_DEGRADED_P5_2fam": {"A": mj5["A"] if "A" in mj5 else mj5.get("tracks", {}).get("A"),
                                    "B": mj5["B"] if "B" in mj5 else mj5.get("tracks", {}).get("B")},
            "figure_of_record": "r4 (the only round at full P=8 / 3 families)",
        },
        "gepa_stage1_screen": {k: v for k, v in gepa.items() if not isinstance(v, list)},
        "gepa_stages_2_4": "NOT RUN -- see the campaign note. Stage 1 identified 12 "
                           "rephrasing targets; stages 2-4 would raise the terminal bank "
                           "level further and cannot change the sign of any conclusion "
                           "here, since the bank already exceeds the dense standard.",
        "observed_covariate_line": line["covariates"],
    }
    (HERE / "cap_crowd_TERMINAL_LEDGER.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["stopping"], indent=1))
    print(json.dumps(out["terminal_state"], indent=1))
    print("wrote cap_crowd_TERMINAL_LEDGER.json")


if __name__ == "__main__":
    main()
