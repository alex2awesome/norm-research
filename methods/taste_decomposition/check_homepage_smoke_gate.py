#!/usr/bin/env python3
"""Smoke gate for the homepage v2 A-bank rebuild -- the GEPA iteration checkpoint.

The rebuild exists because the census bank scored word salad ABOVE real headlines
(coherent-vs-scrambled .387). It would be worthless to spend 377K judge calls on a
replacement without first checking that the replacement fixes that. This script reads
the pilot artifacts written by `score_homepage_v2_bank.py --smoke` and decides, on
LABEL-BLIND criteria fixed in advance, whether the full run may proceed.

The gate NEVER looks at y. Every quantity below is either a coherence contrast between
real and scrambled text, or a distribution-shape statistic.

  PASS (exit 0)  -> the chain runs the full 12,998-item scoring
  FAIL (exit 7)  -> the chain stops; the criteria are revised and the pilot is re-run.
                    That revise-and-re-pilot loop IS the GEPA iteration for this bank,
                    and every iteration is recorded in the build note.

THRESHOLDS, fixed before the first pilot was scored:
  1. row-mean coherent-vs-scrambled AUC >= 0.60   (census bank: .387)
  2. at most 3 of 29 criteria individually below chance on coherence
  3. all five coherence-backbone criteria (b01-b05) at or above 0.60 individually
  4. no distribution collapse: NA rate < .10, fewer than half the criteria pinned
     above .90 modal share, at most 3 near-constant criteria
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

OUT = Path(os.environ.get("VA_OUT_HP2",
                          "/lfs/skampere3/0/alexspan/norm-research/"
                          "outputs/va_gemma_banks_homepage_v2"))
BANK = "homepage_curation_v2"
BACKBONE = {"b01", "b02", "b03", "b04", "b05"}

MIN_ROW_COHERENCE = 0.60
MAX_BELOW_CHANCE = 3
MIN_BACKBONE_COHERENCE = 0.60
MAX_NA_RATE = 0.10
MAX_NEAR_CONSTANT = 3


def main():
    fails, notes = [], []

    bp = OUT / "anchor_battery.json"
    if not bp.exists():
        print("[gate] MISSING anchor_battery.json"); sys.exit(7)
    row = json.loads(bp.read_text()).get(BANK)
    if row is None:
        print(f"[gate] MISSING {BANK} in anchor_battery.json"); sys.exit(7)
    coh = row.get("coherent_vs_scrambled_auc")
    pvn = row.get("pos_vs_neg_auc")
    notes.append(f"row-mean coherent-vs-scrambled AUC {coh:.4f} "
                 f"(census bank .3869); pos-vs-neg {pvn:.4f}")
    if not (coh is not None and coh >= MIN_ROW_COHERENCE):
        fails.append(f"coherence {coh} < {MIN_ROW_COHERENCE}")

    pcp = OUT / "anchor_battery_percriterion_smoke.json"
    if not pcp.exists():
        pcp = OUT / "anchor_battery_percriterion.json"
    if not pcp.exists():
        print("[gate] MISSING per-criterion battery"); sys.exit(7)
    pc = json.loads(pcp.read_text()).get(BANK)
    if pc is None:
        print(f"[gate] MISSING {BANK} in per-criterion battery"); sys.exit(7)
    below = [d["rubric_id"] for d in pc["per_criterion"] if d["is_entity_detector"]]
    notes.append(f"criteria below chance on coherence: {len(below)}/"
                 f"{pc['n_criteria']} {below}")
    if len(below) > MAX_BELOW_CHANCE:
        fails.append(f"{len(below)} criteria below chance > {MAX_BELOW_CHANCE}")
    bb = {d["rubric_id"]: d["coherent_vs_scrambled_auc"]
          for d in pc["per_criterion"] if d["rubric_id"] in BACKBONE}
    notes.append("backbone coherence " + " ".join(f"{k}={v:.3f}"
                                                  for k, v in sorted(bb.items())))
    weak_bb = [k for k, v in bb.items() if not (v >= MIN_BACKBONE_COHERENCE)]
    if len(bb) != len(BACKBONE):
        fails.append(f"backbone criteria missing from battery: "
                     f"{sorted(BACKBONE - set(bb))}")
    if weak_bb:
        fails.append(f"backbone criteria below {MIN_BACKBONE_COHERENCE}: "
                     f"{sorted(weak_bb)}")
    scram_na = [d["rubric_id"] for d in pc["per_criterion"]
                if d["na_rate_on_scrambled"] > 0.5]
    if scram_na:
        notes.append(f"criteria answering NA on >50% of scrambled anchors "
                     f"(the old failure route): {scram_na}")

    # distribution shape: from the smoke print we cannot read it, so use the battery
    # matrix stand-in when the full distribution_check is absent (pilot stage).
    dp = OUT / "distribution_check.json"
    if dp.exists() and json.loads(dp.read_text()).get(BANK):
        dc = json.loads(dp.read_text())[BANK]
        notes.append(f"distribution: NA {dc['na_rate_overall']:.4f} near-constant "
                     f"{dc['n_near_constant']} all-min {dc['all_min_collapse']} "
                     f"half-pinned {dc['half_pinned_to_one_value']}")
        if dc["na_rate_overall"] > MAX_NA_RATE:
            fails.append(f"NA rate {dc['na_rate_overall']:.3f} > {MAX_NA_RATE}")
        if dc["n_near_constant"] > MAX_NEAR_CONSTANT:
            fails.append(f"{dc['n_near_constant']} near-constant criteria")
        if dc["all_min_collapse"] or dc["half_pinned_to_one_value"]:
            fails.append("judge score distribution collapsed")
    else:
        na_anchor = max(d["na_rate_on_anchors"] for d in pc["per_criterion"])
        notes.append(f"no full distribution_check yet (pilot); max per-criterion NA "
                     f"rate on anchors {na_anchor:.4f}")
        if na_anchor > 0.5:
            fails.append(f"a criterion answers NA on {na_anchor:.2f} of anchors")

    print("[gate] " + "\n[gate] ".join(notes))
    if fails:
        print("[gate] FAIL:")
        for f in fails:
            print(f"[gate]   - {f}")
        sys.exit(7)
    print("[gate] PASS -- proceeding to the full scoring run")
    sys.exit(0)


if __name__ == "__main__":
    main()
