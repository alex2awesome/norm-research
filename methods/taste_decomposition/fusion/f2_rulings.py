#!/usr/bin/env python3
"""F2: apply the coordinator's two rulings (2026-08-11) to every f2_deconf_<cell>.json.

RULING 1 -- §11 RE-BASED FOR F2 ROWS.  The fused arm must beat the strongest NAMEABLE
stack, not the bare bank:  PASS iff (d) > (c).  The old (d) vs (a) margin is kept as
context and explicitly labelled TRIVIAL-UNDER-NUISANCE, because against a strong named
nuisance block (d) clears (a) almost automatically -- mathse_accepted recorded +.1207
on that basis while its actual taste residual was +.0086.

RULING 2 -- LEACE Leg 3 is required only for per-channel EFFECT claims (matched /
Delta_adj magnitudes for a named channel).  Descriptive alone-AUCs are exempt.  No F2
readout is a per-channel effect, so Leg 3 is NOT triggered; the exemption is recorded
per cell together with the spurious-alone > .65 flag so a later per-channel quote
knows it must run LEACE first.

REGISTRY DISTINCTION OF RECORD -- every quoted number names which quantity it is:
  LEVEL residual        T - bank        (the closure campaigns' Delta_beyond)
  INCREMENTAL information  (d) - (c)    (this battery's stacked increment)
These are different estimands. press_verdict has level ~ +.009 and increment +.062,
both true: bank and dense reach similar LEVELS via partly uncorrelated signal.

Idempotent. CPU only.  Usage: python3 f2_rulings.py [--cell X ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

ESTIMAND = {
    "LEVEL_residual": ("T - bank: how much HIGHER the dense arm scores than the "
                       "articulated bank. This is the closure campaigns' Delta_beyond "
                       "and the quantity their same-rows verdicts report."),
    "INCREMENTAL_information": ("(d) - (c): how much the dense column adds ON TOP OF "
                                "the enriched bank AND the named nuisance block, jointly "
                                "stacked. This battery's PRIMARY readout."),
    "why_they_differ": ("Two instruments can reach the same LEVEL while carrying partly "
                        "UNCORRELATED signal, in which case the increment is large and the "
                        "level residual is ~0. press_verdict is the worked example: level "
                        "~+.009 (closure, same-rows, full-strength bank) vs increment "
                        "+.0622 (matched-strength companion). Both are true."),
    "rule": "NEVER conflate them; every quoted number names which quantity it is.",
}


def apply(cell):
    p = RESULTS / f"f2_deconf_{cell}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    a = d["arms"]
    c_nl = a["c_VA_enr_plus_NUIS_nl"]
    d_nl = a["d_VAT_dec_trained_nl"]
    e_nl = a["e_VAT_dec_untrained_nl"]
    a_nl = a["a_VA_enr_nl"]
    best_fused = max(d_nl, e_nl)

    old = d.get("fused_must_beat_bank", {})
    d["standing_rule_sec11_F2"] = {
        "schema": "sec11_f2/v1",
        "rule": ("RE-BASED for F2 rows (coordinator 2026-08-11): the fused arm must beat "
                 "the strongest NAMEABLE stack, i.e. PASS iff (d) > (c). The bare-bank "
                 "comparison is not discriminating once a strong named nuisance block "
                 "exists."),
        "c_strongest_nameable_stack": c_nl,
        "d_fused_trained": d_nl,
        "margin_d_minus_c": d_nl - c_nl,
        "verdict": "PASS" if d_nl > c_nl else "FAIL",
        "note_on_margin": ("margin is the seed-mean AUC difference; the SIGNIFICANCE of the "
                           "same contrast is the group paired bootstrap on the seed-0 OOF "
                           "pair, which differs slightly in point value by construction"),
        "significance_source": "PRIMARY_stacked_increment_d_minus_c (group bootstrap)",
        "bootstrap_estimate": d["PRIMARY_stacked_increment_d_minus_c"]["estimate"],
        "bootstrap_p_gt_0": d["PRIMARY_stacked_increment_d_minus_c"]["p_gt_0"],
        "SIGNIFICANT_AT_P95": bool(d["PRIMARY_stacked_increment_d_minus_c"]["p_gt_0"] >= 0.95),
        "verdict_qualified": (
            ("PASS" if d_nl > c_nl else "FAIL") +
            ("" if d["PRIMARY_stacked_increment_d_minus_c"]["p_gt_0"] >= 0.95
             else " (POINT-ONLY -- bootstrap CI includes 0; not evidence of a residual)")),
        "context_only_d_vs_a": {
            "bank_enriched_a": a_nl, "best_fused": best_fused,
            "margin": best_fused - a_nl,
            "old_verdict": old.get("verdict"),
            "LABEL": ("TRIVIAL-UNDER-NUISANCE -- retained as context only. Against a strong "
                      "named nuisance block this margin is cleared almost automatically and "
                      "does not evidence a taste residual."),
        },
    }
    if "fused_must_beat_bank" in d:
        d["fused_must_beat_bank"]["SUPERSEDED_BY"] = "standing_rule_sec11_F2 (see LABEL)"

    # the re-based rule must ALSO be read on the matched-strength footing wherever the
    # companion exists: that is the footing on which a full-strength bank comparison is
    # made, and on the big-gap cells it can reverse the sign.
    ms = d.get("matched_strength_companion")
    if ms and ms.get("applicable"):
        cs = ms["arms_matched"]["c_star_bankfull_plus_NUIS_nl"]
        ds = ms["arms_matched"]["d_star_plus_T_nl"]
        b = ms["COMPANION_increment_dstar_minus_cstar"]
        d["standing_rule_sec11_F2"]["matched_strength_footing"] = {
            "c_star": cs, "d_star": ds, "margin_dstar_minus_cstar": ds - cs,
            "bootstrap_estimate": b["estimate"], "bootstrap_p_gt_0": b["p_gt_0"],
            "verdict": "PASS" if ds > cs else "FAIL",
            "SIGNIFICANT_AT_P95": bool(b["p_gt_0"] >= 0.95),
            "SIGNIFICANTLY_NEGATIVE": bool(b["p_gt_0"] <= 0.05),
            "note": ("where the enriched-bank gap exceeds .02 this is the footing that "
                     "governs any full-strength bank comparison; a FAIL here with an "
                     "E-refit PASS means the E-refit increment was an artefact of "
                     "starving the bank, not a taste residual"),
        }

    d["estimand_distinction"] = ESTIMAND
    d["PRIMARY_stacked_increment_d_minus_c"]["estimand"] = "INCREMENTAL_information"
    if d.get("matched_strength_companion", {}).get("applicable"):
        d["matched_strength_companion"]["COMPANION_increment_dstar_minus_cstar"]["estimand"] = \
            "INCREMENTAL_information (bank at full training strength)"

    d["leace_leg3"] = {
        "required": False,
        "rule": ("LEACE Leg 3 is required only for per-channel EFFECT claims (matched / "
                 "Delta_adj magnitudes attributed to a NAMED channel). Descriptive "
                 "alone-AUCs are exempt (coordinator 2026-08-11)."),
        "why_not_triggered": ("no F2 readout is a per-channel effect: the primary is a "
                              "joint stacked increment and top_nuisance_channels are "
                              "descriptive alone-AUCs"),
        "spurious_alone": d.get("spurious_alone_b"),
        "spurious_alone_gt_065": d.get("spurious_alone_gt_065"),
        "IF_A_PER_CHANNEL_EFFECT_IS_LATER_QUOTED_ON_THIS_CELL":
            ("LEACE must run first" if d.get("spurious_alone_gt_065") else
             "standard Legs 1-2 suffice"),
    }
    p.write_text(json.dumps(d, indent=2, default=str))
    return d["standing_rule_sec11_F2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", default=None)
    args = ap.parse_args()
    cells = args.cell or sorted(q.stem.replace("f2_deconf_", "")
                                for q in RESULTS.glob("f2_deconf_*.json"))
    for c in cells:
        r = apply(c)
        if r is None:
            print(f"  [{c}] no results file")
            continue
        print(f"  [{c}] §11-F2 {r['verdict']} (d)-(c) {r['margin_d_minus_c']:+.4f} "
              f"| context (d) vs (a) {r['context_only_d_vs_a']['margin']:+.4f} "
              f"[{r['context_only_d_vs_a']['old_verdict']}, trivial-under-nuisance]")


if __name__ == "__main__":
    main()
