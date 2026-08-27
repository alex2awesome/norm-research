#!/usr/bin/env python3
"""FREEZE ADDENDUM 3 (2026-08-07): MIXED-channel DECOMPOSITION PASS.

A Track-B channel tagged `mixed=true` is one whose conjectured upstream parent
plausibly causes REAL quality as well as a surface fingerprint (e.g. law-firm drafting
resources both produce letterhead conventions AND produce more rigorous argument).
Addendum 2 dual-reported those channels as a sensitivity band. Addendum 3 replaces
that with an actual decomposition:

  * for each selected MIXED parent, author >= 2 refined criteria that isolate its
    components -- a CANDIDATE-REAL component (the merit the upstream cause also
    produces) and a SURFACE component (the fingerprint with no merit content);
  * score each component separately over the full population;
  * route each component through the blind audit INDEPENDENTLY (the audit, not the
    author, decides which side each lands on);
  * RETIRE the parent from the readouts once its components are scored -- recorded in
    `retired_channels.json`, never deleted.

Parent selection is a DESIGN DECISION and therefore reads FIT+MINE ONLY (the M3
precedent: "alone-AUC computed on FIT+MINE only. MONITOR is never read for a design
decision"). Parents are ranked by |alone-AUC - .5| on FIT+MINE, i.e. by how much
signal the channel actually carries, not by how interesting it sounds.

  rank    -> mixed_parents_ranked.json  + the sealed decomposer's brief
  collect -> round5_decomposition.json  (components, ready for the phrasing pass)

Usage:
  python mixed_decomposition.py rank --upto 4 --n-parents 10
  python mixed_decomposition.py collect --round 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import nc_closure_lib as L

HERE = Path(__file__).resolve().parent

DECOMPOSE_INSTRUCTION = """You are decomposing SUSPECTED-SPURIOUS channels used to score PUBLIC
COMMENTS submitted to United States federal agencies on proposed rules.

Each channel below was proposed as "predictive but not merit", and each was flagged
MIXED: the unseen cause the proposer conjectured behind it plausibly produces REAL
comment quality as well as a surface fingerprint. A channel like that is useless as a
nuisance control, because discounting it would also discount genuine merit.

YOUR TASK. For EACH parent channel below, author exactly {k} refined criteria that
pull the parent apart:

  * at least ONE **candidate-real** component: the substantive merit that the same
    upstream cause produces. It must be scorable ONLY by reading what the comment
    establishes -- a comment with the surface markers but no substance must score 0.
  * at least ONE **surface** component: the fingerprint alone, with every trace of
    merit removed. A comment with the marker but no substance must score 10 on this
    one. Say plainly that it scores the surface feature; do not dress it as merit.

Worked example. Parent = "professional regulatory drafting" (upstream cause: law-firm
or regulatory-affairs drafting resources).
  candidate-real -> "Argument constructed with legal rigor": score 10 when the comment
      builds a chain from statutory or regulatory text to a conclusion about this rule,
      each step stated; score 0 when no such chain is built, however formal the prose.
  surface        -> "Law-firm filing conventions": score 10 for letterhead, counsel
      signature blocks, RE: lines, numbered headings and formal transmittal language;
      score 0 for plain unformatted prose. Score only the filing conventions, never
      what the document argues.

HARD CONSTRAINTS.
1. LABEL-BLIND. No component may reference agency responses, rule changes, or any
   other outcome variable.
2. Each component must be SCORABLE 0-10 by an independent judge reading only the
   comment text, with a clear 10 and a clear 0.
3. The components must be SEPARABLE: it must be possible for a comment to score high
   on one and low on the other. If you cannot separate a parent, say so in that
   parent's entry by emitting a component named "NOT SEPARABLE" with an explanation --
   that is a legitimate and useful answer.
4. Do NOT decide which track a component belongs to. An independent blind auditor
   decides that. Just write the criterion.

OUTPUT FORMAT. Emit exactly one JSON object and nothing else:

{{"decompositions": [
  {{"parent_id": "<the parent id given below, unchanged>",
    "components": [
      {{"kind": "candidate_real" or "surface",
        "name": "<short name, <= 12 words>",
        "instruction": "<0-10 scoring instruction; say what a 10 and a 0 look like>",
        "why_separable": "<one sentence: what kind of comment scores high here and low on the sibling>"}},
      ... exactly {k} components ...
    ]}},
  ... one entry for EVERY parent below ...
]}}
"""


def load_b_channels(upto):
    """Every B-routed, non-collapsed channel of rounds 1..upto with its tags."""
    cols, meta = [], []
    for r in range(1, upto + 1):
        p = HERE / f"round{r}_scores.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        routed = json.loads((HERE / f"round{r}_routing_final.json").read_text())
        gate = json.loads((HERE / f"round{r}_score_report.json").read_text())
        bmap = {c["id"]: c for c in routed["B"]}
        cids = [str(s) for s in z["crit_ids"]]
        cnames = [str(s) for s in z["crit_names"]]
        fin = json.loads((HERE / f"round{r}_criteria_final.json").read_text())
        instr = {c["id"]: c["instruction"] for t in ("A", "B") for c in fin[t]}
        for k, cid in enumerate(cids):
            if cid not in bmap or gate["per_criterion"][cid]["collapsed"]:
                continue
            cols.append(z["X"][:, k])
            meta.append({"round": r, "blind_id": cid, "uid": f"r{r}:{cid}",
                         "src_id": bmap[cid].get("src_id"),
                         "name": cnames[k],
                         "instruction": instr.get(bmap[cid].get("src_id"), ""),
                         "upstream_parent": bmap[cid].get("upstream_parent"),
                         "mixed": bool(bmap[cid].get("mixed"))})
    return (np.column_stack(cols) if cols else None), meta


def cmd_rank(a):
    pop = L.load_population()
    summary, split, dsplit, mining, monitor_full = L.load_splits()
    y = pop["y"]
    fm = split == "fit_mine"

    X, meta = load_b_channels(a.upto)
    assert X is not None
    rows = []
    for j, m in enumerate(meta):
        col = X[fm, j]
        ok = ~np.isnan(col)
        try:
            auc = float(roc_auc_score(y[fm][ok], col[ok]))
        except ValueError:
            auc = float("nan")
        rows.append({**m, "alone_auc_fitmine": auc, "signal": abs(auc - 0.5)})

    mixed = sorted([r for r in rows if r["mixed"] and not np.isnan(r["signal"])],
                   key=lambda r: -r["signal"])
    out = {
        "upto_round": a.upto,
        "note": "alone-AUC on FIT+MINE ONLY -- MONITOR and the honest population are "
                "never read for a design decision (M3 precedent).",
        "n_b_channels": len(rows), "n_mixed": len(mixed),
        "all_channels": rows,
        "selected_parents": mixed[:a.n_parents],
    }
    (HERE / "mixed_parents_ranked.json").write_text(json.dumps(out, indent=1))

    sel = out["selected_parents"]
    body = DECOMPOSE_INSTRUCTION.format(k=a.k_components) + "\n\n" + "\n\n".join(
        f"--- PARENT {p['uid']} ---\n"
        f"NAME: {p['name']}\nINSTRUCTION: {p['instruction']}\n"
        f"CONJECTURED UPSTREAM CAUSE: {p['upstream_parent']}"
        for p in sel) + "\n"
    (HERE / "round5_decomposition_prompt.txt").write_text(body)
    print(f"{len(rows)} B channels, {len(mixed)} MIXED; selected {len(sel)} parents")
    for p in sel:
        print(f"  {p['uid']} alone={p['alone_auc_fitmine']:.4f} "
              f"|signal|={p['signal']:.4f}  {p['name']}")
    print(f"-> round5_decomposition_prompt.txt ({len(body)} chars)")


def cmd_collect(a):
    d = json.loads((HERE / "round5_decomposition_raw.json").read_text())
    ranked = {p["uid"]: p for p in
              json.loads((HERE / "mixed_parents_ranked.json").read_text())["selected_parents"]}
    comps, retired = [], []
    for entry in d["decompositions"]:
        pid = entry["parent_id"]
        parent = ranked.get(pid)
        usable = [c for c in entry["components"] if c["name"].strip().upper() != "NOT SEPARABLE"]
        if not usable:
            print(f"  {pid}: NOT SEPARABLE -- parent kept, not retired")
            continue
        for c in usable:
            comps.append({"parent_blind_id": pid,
                          "parent_name": parent["name"] if parent else None,
                          "parent_upstream": parent["upstream_parent"] if parent else None,
                          "kind": c["kind"], "name": c["name"].strip(),
                          "instruction": c["instruction"].strip(),
                          "why_separable": c.get("why_separable", "").strip()})
        retired.append({"uid": pid, "blind_id": parent["blind_id"] if parent else None,
                        "round": parent["round"] if parent else None,
                        "name": parent["name"] if parent else None,
                        "alone_auc_fitmine": parent["alone_auc_fitmine"] if parent else None,
                        "n_components": len(usable)})
    (HERE / "round5_decomposition.json").write_text(json.dumps(
        {"round": a.round, "n_parents": len(retired), "n_components": len(comps),
         "components": comps, "retired_parents": retired}, indent=1))
    (HERE / "retired_channels.json").write_text(json.dumps(
        {"rule": "FREEZE ADDENDUM 3: a MIXED parent is retired from the Track-B readouts "
                 "once its decomposed components are scored. Recorded, never deleted.",
         "retired": retired}, indent=1))
    print(f"{len(retired)} parents decomposed -> {len(comps)} components")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rank")
    r.add_argument("--upto", type=int, required=True)
    r.add_argument("--n-parents", type=int, default=10)
    r.add_argument("--k-components", type=int, default=2)
    c = sub.add_parser("collect"); c.add_argument("--round", type=int, default=5)
    a = ap.parse_args()
    {"rank": cmd_rank, "collect": cmd_collect}[a.cmd](a)
