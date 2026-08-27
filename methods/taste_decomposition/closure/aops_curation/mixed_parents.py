#!/usr/bin/env python3
"""FREEZE ADDENDUM 3 -- MIXED-channel parent selection + sealed decomposer brief,
for a campaign that already has mined Track-B rounds on file.

Addendum 3 (2026-08-07): "MIXED channels ... get a DECOMPOSITION PASS: author >= 2
refined criteria isolating the components ..., score each separately, route each
through the blind audit independently.  The parent channel is retired from readouts
once its components are scored (recorded, not deleted).  Decomposed components count
toward their round's k budgets."

Difference from `maps_hw_si/new_parents.py`: there the parents were BANK criteria
picked off a SHAP interaction screen.  Here the parents are the accumulated
**Track-B channels the audit routed to B with `mixed=true`** in the rounds already
run -- which is the literal object Addendum 3 names, and the object this cell has a
lot of (peer revealed: 9 of 11 mixed in r1, 10 of 13 in r2).

PARENT SELECTION IS A DESIGN DECISION AND READS FIT+MINE ONLY (the M3 precedent
carried by `nc_responded/mixed_decomposition.py`: "MONITOR is never read for a design
decision").  Parents are ranked by |alone-AUC - .5| on FIT+MINE -- by how much signal
the channel actually carries, not by how interesting it sounds -- with the frozen
per-criterion collapse gate applied first.

Output is byte-compatible with `decompose_round.py --cmd merge`, which composes the
round's scored set as
    k_A = 15 = 12 fleet species + 3 candidate-real components
    k_B = 10 =  7 fleet species + 3 surface components

  rank  -> <cell>_r<r>_newparents.json, <cell>_r<r>_parents_used.json,
           and the sealed decomposer prompt in the scratch round directory
CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import cells as C
import closure_core as L

HERE = Path(__file__).resolve().parent
SCRATCH = Path("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
               "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad") / HERE.name

HEAD = """You are the DECOMPOSER in a preregistered measurement protocol. You are working
label-blind: you have not been shown any outcome, citation count, decision, vote or score
for any item, and nothing you write may refer to one.

CORPUS: {corpus}
ITEM: a {item}
CONSTRUCT the scorecard is trying to measure: {construct}

BACKGROUND. A separate miner proposed the channels below as SUSPECTED-SPURIOUS: features
that plausibly predict how such an item fares but are not {construct} itself. Each was
then flagged MIXED, meaning the unseen upstream cause conjectured behind it plausibly
produces REAL {construct} as well as a surface fingerprint (a well-resourced group may
both be favoured for irrelevant reasons AND actually do better work). A MIXED channel is
useless as a nuisance control as it stands: discounting it would also discount genuine
merit, and NOT discounting it leaves a live confound. So each one must be pulled apart.

YOUR TASK. For EACH parent channel below, author EXACTLY TWO replacement criteria:

  * a CANDIDATE-REAL component: the substantive merit that the same upstream cause
    produces, written so a judge is forced to judge the substance IN CONTEXT and is
    explicitly told NOT to reward the surface carrier. It must be possible for a short,
    plainly written, unfashionable item to score 10, and for an item carrying every
    surface marker but no substance to score 0.
  * a SURFACE component: the fingerprint alone, with every trace of merit removed, put as
    a pure EXTENT question a judge can count or estimate, with an explicit instruction
    NOT to decide whether the feature is good. An item with the marker but no substance
    must be able to score 10 on this one.

The two must be SEPARABLE: in each rationale, describe an item that scores high on one and
low on the other, in BOTH directions.

PARENTS TO DECOMPOSE:

{parents}

HARD CONSTRAINTS. (1) Label-blind: no reference to any outcome, citation, decision,
selection, rank or score. (2) Each criterion is scored 0-10 by an independent judge
reading only the item text; say what a 10 and a 0 look like. (3) Do not write a criterion
whose high end can be reached by formatting, length or vocabulary fashion alone unless you
are putting it on the surface side. (4) Names <= 12 words.

OUTPUT. Emit exactly one JSON object and nothing else:

{{"components": [
  {{"id": "D01", "parent": "<parent name, copied exactly>", "kind": "candidate_real",
    "name": "<name>", "instruction": "<0-10 scoring instruction>",
    "rationale": "<why this isolates the merit; state the two-way dissociation>"}},
  {{"id": "D02", "parent": "<same parent>", "kind": "surface",
    "name": "<name>", "instruction": "<0-10 extent instruction, explicitly not a quality judgement>",
    "rationale": "<why this is the carrier; state the two-way dissociation>"}},
  ... two entries per parent, in the order the parents are listed ...
]}}
"""


def accumulated_mixed(cell, upto):
    """Every B-routed, mixed, non-collapsed channel of rounds 1..upto, with its
    FIT+MINE alone-AUC."""
    d = C.load(cell)
    sp = json.loads((HERE / f"{cell}_splits.json").read_text())
    fit = np.array([r["split"] for r in sp["rows"]]) == "fit_mine"
    y = d["y"]
    out = []
    for r in range(1, upto + 1):
        f = HERE / f"{cell}_r{r}_scores.npz"
        rt = HERE / f"{cell}_r{r}_routing_final.json"
        if not (f.exists() and rt.exists()):
            continue
        z = np.load(f, allow_pickle=True)
        cids = [str(s) for s in z["crit_ids"]]
        gate = json.loads((HERE / f"{cell}_r{r}_score_report.json").read_text())
        sel = {s["blind_id"]: s for s in
               json.loads((HERE / f"{cell}_r{r}_species.json").read_text())["selected"]}
        for x in json.loads(rt.read_text())["final"]:
            if x["final_route"] != "B" or not x.get("mixed"):
                continue
            bid = x["blind_id"]
            if bid not in cids or gate["per_criterion"][bid]["collapsed"]:
                continue
            col = z["X"][:, cids.index(bid)].astype(float)
            med = np.nanmedian(col[fit])
            colf = np.where(np.isnan(col), med, col)
            auc_fit = L.auc(y[fit], colf[fit])
            out.append({"round": r, "blind_id": bid, "criterion": x["name"],
                        "instruction": sel.get(bid, {}).get("instruction", ""),
                        "upstream_parent": x.get("upstream_parent", "surface-only"),
                        "alone_AUC_FITMINE": auc_fit,
                        "signal": abs(auc_fit - 0.5)})
    out.sort(key=lambda r: -r["signal"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--round", required=True)
    ap.add_argument("--upto", type=int, required=True, help="last round already scored")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()

    cand = accumulated_mixed(a.cell, a.upto)
    # never decompose the same parent twice (Addendum 3: the parent is RETIRED, recorded)
    already = set()
    for p in sorted(HERE.glob(f"{a.cell}_r*_parents_used.json")):
        for x in json.loads(p.read_text())["parents"]:
            already.add((x["round"], x["blind_id"]))
    fresh = [c for c in cand if (c["round"], c["blind_id"]) not in already]
    sel = fresh[:a.n]
    meta = C.CELL_META[a.cell]
    blocks = []
    for j, c in enumerate(sel):
        blocks.append(
            f"[P{j+1:02d}] PARENT: {c['criterion']}\n"
            f"      INSTRUCTION AS WRITTEN: {c['instruction']}\n"
            f"      CONJECTURED UPSTREAM CAUSE: {c['upstream_parent']}")
    prompt = HEAD.format(corpus=meta["corpus"], item=meta["item"],
                         construct=meta["construct"], parents="\n\n".join(blocks))
    dd = SCRATCH / f"{a.cell}_r{a.round}"
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "prompt_decomposer.txt").write_text(prompt)

    # decompose_round.cmd_build's format, so `merge` needs no change
    (HERE / f"{a.cell}_r{a.round}_newparents.json").write_text(json.dumps(
        {"cell": a.cell, "round": a.round, "rule": "accumulated MIXED Track-B channels, "
         "ranked by |alone-AUC - .5| on FIT+MINE only",
         "selected_parents": [c["criterion"] for c in sel],
         "candidates": cand}, indent=1))
    (HERE / f"{a.cell}_r{a.round}_parents_used.json").write_text(json.dumps(
        {"cell": a.cell, "round": a.round, "parents": sel,
         "n_mixed_available": len(cand),
         "composition": {"fleet_A": 12, "fleet_B": 7,
                         "decomposition_candidate_real": len(sel),
                         "decomposition_surface": len(sel)}}, indent=1))
    # FREEZE ADDENDUM 3: the parent is RETIRED from the nuisance readouts once its
    # components are scored -- recorded here, never deleted from the score matrices.
    rp = HERE / f"{a.cell}_retired_channels.json"
    prev = json.loads(rp.read_text())["retired"] if rp.exists() else []
    keys = {(x["round"], x["blind_id"]) for x in prev}
    for c in sel:
        if (c["round"], c["blind_id"]) not in keys:
            prev.append({"round": c["round"], "blind_id": c["blind_id"],
                         "name": c["criterion"], "retired_at_round": a.round,
                         "reason": "FREEZE ADDENDUM 3 decomposition: replaced by its "
                                   "candidate-real and surface components"})
    rp.write_text(json.dumps({"cell": a.cell, "retired": prev}, indent=1))

    print(f"{a.cell} r{a.round}: {len(cand)} accumulated MIXED channels "
          f"({len(already)} already decomposed), top {len(sel)} selected "
          f"-> {2*len(sel)} components; {len(prev)} channels now retired")
    for c in fresh[:10]:
        mark = "*" if c in sel else " "
        print(f" {mark} r{c['round']} {c['blind_id']} aloneAUC(FIT+MINE)="
              f"{c['alone_AUC_FITMINE']:.3f}  {c['criterion'][:58]}")
    print("prompt ->", dd / "prompt_decomposer.txt")


if __name__ == "__main__":
    main()
