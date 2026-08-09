#!/usr/bin/env python3
"""T0 (UNTRAINED-T) ARM, step 4: merge the per-box per-cell fusion outputs into
ONE results file + emit the 16-row markdown table.

For every cell, `t0_fuse.py` was run on BOTH boxes (mac / sk3) because the
ledger's own landmine says GroupKFold fold MEMBERSHIP is sklearn-version AND
architecture dependent.  The box kept for a cell is the one that REPRODUCES that
cell's published VA_nl and VAT_nl; if both reproduce, the one with the smaller
total absolute deviation; if neither, the closer one, flagged.

Writes:  methods/taste_decomposition/results/t0_untrained_arm.json
Prints:  the 16-row table for notes/2026-08-08__t0_untrained_arm.md
"""
from __future__ import annotations

import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
IN = HERE / "t0_results"

ROW_ORDER = [
    ("Peer review", ["peer_verdict", "peer_curation", "peer_revealed"]),
    ("Regulatory (N&C)", ["nc_responded", "nc_outcome", "nc_agree"]),
    ("Creative writing", ["cw_community"]),
    ("Humor", ["hashtagwars_verdict", "cap_finalist", "cap_crowd", "jokes_community"]),
    ("Math", ["mathse_accepted_verdict", "mathse_vote_score", "aops_curation"]),
    ("Software code", ["code_v3"]),
    ("Journalism/press", ["press_verdict"]),
]
BOXES = ["mac", "sk3"]


def dev(d):
    """total |reproduced - published| over the two gated quantities."""
    g = d["ledger_gate"]
    tot = 0.0
    for k in ("VA_nl", "VAT_nl"):
        a = g[k].get("abs_diff")
        tot += 1e9 if a is None else a
    return tot


def pick(cell):
    have = {}
    for b in BOXES:
        p = IN / f"{cell}.{b}.json"
        if p.exists():
            have[b] = json.loads(p.read_text())
    if not have:
        return None, {}
    passing = {b: d for b, d in have.items() if d["ledger_gate"]["pass"]}
    pool = passing or have
    best = min(pool, key=lambda b: dev(pool[b]))
    return best, have


def f4(x):
    return "—" if x is None else f"{x:.4f}".lstrip("0")


def sg(x):
    if x is None:
        return "—"
    return ("+" if x >= 0 else "−") + f"{abs(x):.4f}".lstrip("0")


def boot(b):
    if not b:
        return "—"
    return f"{sg(b['estimate'])} [{sg(b['ci95'][0])},{sg(b['ci95'][1])}] {b['p_gt_0']:.2f}"


def main():
    out = {
        "arm": "T0 / UNTRAINED-T FUSION ARM",
        "design_source": ("notes/2026-07-27__vat-run-registry.md, entry '2026-08-08 -- "
                          "FROZEN DESIGN (before any scoring): UNTRAINED-T FUSION ARM'"),
        "templates": "methods/taste_decomposition/fusion/t0_templates.json",
        "T0_definition": ("base meta-llama/Llama-3.1-8B (the checkpoint the program's LoRA "
                          "dense T trains from), ZERO-SHOT, offline batch vLLM; one frozen "
                          "Yes/No question per cell + the document truncated to 1024 tokens "
                          "(T's own --max_length) + 'Answer Yes or No.'; score = P(Yes) from "
                          "the first token with logits masked to the {Yes,No} variant set"),
        "fusion": ("VAT0 = the IDENTICAL frozen Layer-1 stack as VAT "
                   "(direction1_mirror.fit_arm), the trained dense column swapped for the "
                   "T0 column; grouped OOF GroupKFold(5) on the cell's canonical grouping "
                   "unit, HistGB seeds {0,1,2} mean, same bank family per cell"),
        "bootstraps": "group-level paired, 2,000 draws, on the cell's grouping unit",
        "merged_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cells": {}, "box_choice": {}, "missing": [],
    }
    rows = []
    for field, cells in ROW_ORDER:
        for cell in cells:
            box, have = pick(cell)
            if box is None:
                out["missing"].append(cell)
                rows.append((field, cell, None))
                continue
            d = have[box]
            out["cells"][cell] = d
            out["box_choice"][cell] = {
                "kept": box,
                "reason": ("reproduces the published VA_nl and VAT_nl within 1e-4"
                           if d["ledger_gate"]["pass"] else
                           "NEITHER box reproduced the ledger within 1e-4; closest kept"),
                "per_box_total_abs_dev": {b: dev(v) for b, v in have.items()},
                "gate_pass": {b: v["ledger_gate"]["pass"] for b, v in have.items()},
            }
            rows.append((field, cell, d))

    lines = ["| field | cell | n_E | T₀ | T | VA_nl | VAT₀_nl | VAT_nl | (VAT₀−VA) est [CI] P | (VAT−VAT₀) est [CI] P | (T₀−T) est [CI] P |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|"]
    for field, cell, d in rows:
        if d is None:
            lines.append(f"| {field} | `{cell}` | — | — | — | — | — | — | — | — | — |")
            continue
        flag = " ‖" if d.get("POOLED_DO_NOT_QUOTE") else ""
        lines.append("| {} | `{}`{} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            field, cell, flag, d["n_E"], f4(d["T0"]), f4(d["T"]), f4(d["VA_nl"]),
            f4(d["VAT0_nl"]), f4(d["VAT_nl"]),
            boot(d["boot_VAT0_nl_minus_VA_nl"]), boot(d["boot_VAT_nl_minus_VAT0_nl"]),
            boot(d["boot_T0_minus_T"])))
    table = "\n".join(lines)

    d2 = ["", "| cell | VAT₀−VA_nl | VAT−VA_nl | share of the fused gain reached WITHOUT training | T₀ score collapse? | box |",
          "|---|---:|---:|---:|---|---|"]
    for _, cells in ROW_ORDER:
        for cell in cells:
            d = out["cells"].get(cell)
            if not d:
                d2.append(f"| `{cell}` | — | — | — | — | — |")
                continue
            g0 = d["VAT0_nl"] - d["VA_nl"]
            gT = d["VAT_nl"] - d["VA_nl"]
            share = None if abs(gT) < 1e-9 else g0 / gT
            d["gain_VAT0_over_VA"] = g0
            d["gain_VAT_over_VA"] = gT
            d["untrained_share_of_fusion_gain"] = share
            d2.append("| `{}` | {} | {} | {} | {} | {} |".format(
                cell, sg(g0), sg(gT),
                "—" if share is None else f"{share:+.0%}",
                "YES" if d["t0_score_distribution"].get("COLLAPSE_FLAG") else "no",
                out["box_choice"][cell]["kept"]))
    table2 = "\n".join(d2)

    out["table_markdown"] = table
    out["table2_markdown"] = table2
    (RESULTS / "t0_untrained_arm.json").write_text(json.dumps(out, indent=2, default=str))
    print(table)
    print(table2)
    if out["missing"]:
        print("\nMISSING CELLS:", out["missing"])
    bad = [c for c, v in out["box_choice"].items() if not any(v["gate_pass"].values())]
    print("\nLEDGER-GATE FAIL ON BOTH BOXES:", bad or "none")
    print("wrote", RESULTS / "t0_untrained_arm.json")


if __name__ == "__main__":
    main()
