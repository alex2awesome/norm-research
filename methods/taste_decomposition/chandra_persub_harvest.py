#!/usr/bin/env python3
"""Harvest the per-subreddit chandra ladders (Task A, 2026-08-24) into the two
deliverable tables (sub x V/A/VA/T/VAT + n). Columns mirror the pooled quotes:
V/A/VA = per-sub refit layer-1 GBM 3-seed-mean OOF AUC; T = pooled-trained
dense, sub-restricted TEST readout (eval in json); VAT = per-sub bake-off
winner (eval-selected fusion variant) TEST readout.
FRAME: v1 populations; era channel open; v2 rescore will supersede."""
import json
from pathlib import Path

RESULTS = Path("/lfs/skampere3/0/alexspan/norm-research/methods/taste_decomposition/results")
VIABLE = {
    "chandra_humor": ["funny", "Showerthoughts", "nottheonion", "me_irl"],
    "chandra_cw": ["nosleep", "books", "asoiaf", "gameofthrones"],
}
POOLED = {"chandra_humor": dict(V=.551, A=.689, VA=.694, T=.849, VAT=.829),
          "chandra_cw": dict(V=.543, A=.583, VA=.579, T=.911, VAT=.906)}

for cell, subs in VIABLE.items():
    print(f"\n### {cell}\n")
    print("| sub | n | V | A | VA | T (pooled-dense, sub-restricted, test) | "
          "VAT (bakeoff winner, test) | winner |")
    print("|---|---|---|---|---|---|---|---|")
    for sub in subs:
        led = json.loads((RESULTS / f"{cell}_persub_{sub}_ledger.json").read_text())
        bak = json.loads((RESULTS / f"{cell}_persub_{sub}_vat_bakeoff.json").read_text())
        t_test = bak["table"]["T_alone"]["test"]
        print(f"| {sub} | {led['n']} | {led['V']['nl_mean3']:.3f} | "
              f"{led['A']['nl_mean3']:.3f} | {led['VA']['nl_mean3']:.3f} | "
              f"{t_test:.3f} | {bak['VAT_bakeoff_test']:.3f} | {bak['winner_by_eval']} |")
    p = POOLED[cell]
    print(f"| POOLED (ref) | — | {p['V']:.3f} | {p['A']:.3f} | {p['VA']:.3f} | "
          f"{p['T']:.3f} | {p['VAT']:.3f} | — |")
