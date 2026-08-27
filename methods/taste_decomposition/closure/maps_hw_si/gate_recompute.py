#!/usr/bin/env python3
"""HASHTAGWARS DEEP AUDIT, follow-up -- RECOMPUTE THE SCRAMBLED GATE ON THE A-ROUTED SUBSET.

WHY THIS RUNS BEFORE ANY GPU IS CLAIMED (reuse-before-rebuild).  The coordinator authorised
a full rescore of the nine decomposition components under a fresh passing anchor battery.
Before spending a card, note that the batch's anchor scores are ALREADY ON DISK
(`hashtagwars_verdict_rd_scores.npz` carries `Xanchor`, 150 anchors x 23 criteria), and the
campaign's own diagnosis says the pooled gate is misapplied:

    "nine of the twenty-three scored criteria are deliberate extent-of-surface channels
     ... and a word-salad built from two tweets is roughly twice the length of a real
     tweet and keeps its capitals and hashtags -- so it legitimately scores high on them.
     Carry-forward: the scrambled gate must be computed on the A-routed subset, not on a
     batch that is half surface channels by design."
    -- notes/2026-08-08__maps_hw_si.md, instrument-health flag

A scrambled control is a test of whether the judge is READING; it is only valid on criteria
whose true value is destroyed by scrambling.  For an "extent of capitals, length and hashtag
count" channel, scrambling destroys nothing -- the extent is still there -- so a high
scrambled score is the correct answer, not a judge failure.  Pooling those channels into the
gate makes the gate measure the batch's composition instead of the judge's reading.

So the gate is recomputed three ways on the SAME stored anchors:
  * ALL 23 criteria           -- reproduces the published .5876 FAIL (an identity check)
  * the 9 A-ROUTED components -- the subset that actually joined the bank and produced the
                                 +.0241; this is the gate that governs the closure claim
  * the 14 B-routed channels  -- reported so the composition effect is visible, not asserted

If the A-routed subset PASSES, the +.0241 needs no rescore and the audit's provisional
reading ("84% of the closure came from a batch that failed the gate") must be corrected in
the direction of the campaign.  If it FAILS, the rescore is warranted and the GPU job runs.

Per-criterion gates are also emitted, because a subset gate can pass on average while one
member is broken.

CPU only.  Usage: python3 gate_recompute.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
CELL = "hashtagwars_verdict"
GATE = 0.70


def gate_block(Xa, tags, cols, label):
    """Item-level gate over a chosen column subset, exactly as score_gemma_maps computes it
    (mean over criteria per anchor item, then AUC), plus the per-criterion breakdown."""
    sub = Xa[:, cols]
    item = np.nanmean(sub, axis=1)
    pv, nv, sv = item[tags == "anchor_pos"], item[tags == "anchor_neg"], item[tags == "anchor_scram"]
    cvs = float(roc_auc_score([1] * (len(pv) + len(nv)) + [0] * len(sv),
                              np.concatenate([pv, nv, sv])))
    pvn = float(roc_auc_score([1] * len(pv) + [0] * len(nv), np.concatenate([pv, nv])))
    return {
        "subset": label, "n_criteria": len(cols),
        "anchor_pos_mean": float(np.nanmean(pv)), "anchor_neg_mean": float(np.nanmean(nv)),
        "anchor_scram_mean": float(np.nanmean(sv)),
        "coherent_vs_scrambled_auc": cvs, "pos_vs_neg_auc": pvn,
        "PASS_scrambled": bool(cvs >= GATE),
        "scram_above_neg": bool(np.nanmean(sv) > np.nanmean(nv)),
    }


def main():
    z = np.load(HERE / f"{CELL}_rd_scores.npz", allow_pickle=True)
    Xa = np.asarray(z["Xanchor"], dtype=float)
    tags = np.array([str(s) for s in z["anchor_tags"]])
    cids = [str(s) for s in z["crit_ids"]]
    names = {c: str(n) for c, n in zip(cids, z["crit_names"])}
    routing = json.loads((HERE / f"{CELL}_rd_routing_final.json").read_text())["final"]
    route = {x["blind_id"]: x["final_route"] for x in routing}
    kind = {x["blind_id"]: x.get("component_kind") for x in routing}

    A = [i for i, c in enumerate(cids) if route.get(c) == "A"]
    B = [i for i, c in enumerate(cids) if route.get(c) == "B"]
    SURF = [i for i, c in enumerate(cids) if kind.get(c) == "surface"]

    out = {"schema": "hw_gate_recompute/v1",
           "source": f"{CELL}_rd_scores.npz (stored anchors, no rescore)",
           "gate_threshold": GATE,
           "k_per_class": int((tags == "anchor_pos").sum()),
           "campaign_published": {"coherent_vs_scrambled_auc": 0.5876,
                                  "pos_vs_neg_auc": 0.7282, "pass": False},
           "blocks": []}
    for cols, label in ((list(range(len(cids))), "ALL_23_published_gate"),
                        (A, "A_ROUTED_9_components_GOVERNING"),
                        (B, "B_ROUTED_14_channels"),
                        (SURF, "surface_extent_channels_only")):
        if cols:
            out["blocks"].append(gate_block(Xa, tags, cols, label))

    # per-criterion, A-routed only
    per = []
    for i in A:
        b = gate_block(Xa, tags, [i], cids[i])
        b["name"] = names[cids[i]]
        per.append(b)
    per.sort(key=lambda r: r["coherent_vs_scrambled_auc"])
    out["per_criterion_A_routed"] = per

    for b in out["blocks"]:
        print(f"{b['subset']:34s} n={b['n_criteria']:2d}  coherent-vs-scram "
              f"{b['coherent_vs_scrambled_auc']:.4f}  {'PASS' if b['PASS_scrambled'] else 'FAIL'}"
              f"   pos-vs-neg {b['pos_vs_neg_auc']:.4f}"
              f"   scram {b['anchor_scram_mean']:.2f} vs neg {b['anchor_neg_mean']:.2f}")
    print("\nper-criterion, A-routed (the 9 components that joined the bank):")
    for r in per:
        print(f"   {r['subset']:4s} cvs {r['coherent_vs_scrambled_auc']:.4f} "
              f"{'PASS' if r['PASS_scrambled'] else 'FAIL'}  "
              f"scram {r['anchor_scram_mean']:.2f} neg {r['anchor_neg_mean']:.2f}  {r['name'][:44]}")
    n_fail = sum(1 for r in per if not r["PASS_scrambled"])
    out["n_A_components_failing_individually"] = n_fail
    gov = [b for b in out["blocks"] if b["subset"].startswith("A_ROUTED")][0]
    out["VERDICT"] = ("A-routed subset PASSES the scrambled gate -- the published FAIL is a "
                      "composition artifact of pooling surface-extent channels into the gate; "
                      "the +.0241 needs no rescore"
                      if gov["PASS_scrambled"] else
                      "A-routed subset FAILS -- rescore of the nine components is warranted")
    print("\nVERDICT:", out["VERDICT"])
    (HERE / "hashtagwars_gate_recompute.json").write_text(json.dumps(out, indent=1, default=float))
    print("wrote", HERE / "hashtagwars_gate_recompute.json")


if __name__ == "__main__":
    main()
