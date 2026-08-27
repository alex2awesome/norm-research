#!/usr/bin/env python3
"""Readout for the Wigleaf PAIRWISE probe.

THE QUESTION: absolute scoring could not certify this cell -- the K=50 battery
inverted (pos .8798 < neg .9016, pos-vs-neg AUC .498) and the bank saturated (83%
of responses 1.0, mean .899), giving A_lin .5407. Does asking the SAME criteria
COMPARATIVELY recover the editor's cut?

READOUTS
  per-criterion pairwise AUC = mean over pairs of [1 pos picked / .5 tie / 0 neg]
      (for a forced-choice paired design this indicator mean IS the AUC)
  composite (majority vote over the 45 criteria, ties at .5)
  holistic  (the single 'overall stronger piece' question)
  all three stratified by match kind (same_magazine vs same_year)

VALIDITY GATES, read BEFORE any AUC:
  anchors            real-vs-scrambled must be ~1.0 (blinded, mixed into the batch)
  order consistency  the 40 flipped replicates must pick the SAME PIECE, not the
                     same SIDE -- this is the pairwise analogue of retest reliability
  position bias      overall A-choice rate; assignment was hash-randomised so bias
                     adds noise rather than bias, but a large skew invalidates ties

  python datasets/creative-writing/readout_wigleaf_pairwise.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PW = REPO / "datasets/creative-writing/wigleaf/pairwise"
ABS_BANK = {"A_lin": 0.5407, "battery_pos_vs_neg_auc": 0.498,
            "bank_mean": 0.8991, "frac_responses_at_1.0": 0.827}


def ind(choice, target):
    """1 if the judge picked the target side, .5 for a tie, 0 otherwise."""
    if choice == "TIE":
        return 0.5
    return 1.0 if choice == target else 0.0


def boot_ci(vals, n_boot=5000, seed=11):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return [float("nan")] * 2
    m = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def main():
    pk, V, want = {"meta": {}}, {}, 0
    for pf, vf in (("packet.json", "verdicts_sol.json"),
                   ("packet_wave2.json", "verdicts_sol_wave2.json")):
        if not ((PW / pf).exists() and (PW / vf).exists()):
            continue
        P = json.loads((PW / pf).read_text()); D = json.loads((PW / vf).read_text())
        pk.setdefault("criteria", P["criteria"]); pk.setdefault("match_composition", {})
        for k, v in P["match_composition"].items():
            pk["match_composition"][k] = pk["match_composition"].get(k, 0) + v
        pk["meta"].update(P["meta"]); V.update(D["verdicts"]); want += D["n_want"]
        print(f"  loaded {pf}: {len(D['verdicts'])} verdicts")
    vd = {"judge": "gpt-5.6-sol", "n_want": want}
    meta = pk["meta"]
    crit = {m["id"]: m["name"] for m in pk["criteria"]}
    print(f"coverage {len(V)}/{want}")

    real = {p: v for p, v in V.items() if meta[p]["kind"] == "real"}
    flips = {p: v for p, v in V.items() if meta[p]["kind"] == "flip_replicate"}
    ancs = {p: v for p, v in V.items() if meta[p]["kind"] == "anchor"}

    # ---------------- validity gates ----------------
    a_scores, a_overall = [], []
    for p, v in ancs.items():
        t = meta[p]["real_side"]
        a_scores += [ind(c, t) for c in v["criteria"].values()]
        if v.get("overall"):
            a_overall.append(ind(v["overall"], t))
    anchors = {"n_pairs": len(ancs), "n_judgements": len(a_scores),
               "pick_real_rate": round(float(np.mean(a_scores)), 4) if a_scores else None,
               "overall_pick_real_rate": round(float(np.mean(a_overall)), 4) if a_overall else None}

    side_counts = Counter()
    for v in real.values():
        side_counts.update(v["criteria"].values())
    tot = sum(side_counts.values())
    position = {"A_rate": round(side_counts["A"] / tot, 4),
                "B_rate": round(side_counts["B"] / tot, 4),
                "tie_rate": round(side_counts["TIE"] / tot, 4), "n": tot}

    cons, cons_overall = [], []
    for p, v in flips.items():
        base = meta[p]["replicate_of"]
        if base not in real:
            continue
        b = real[base]
        for cid, ch in v["criteria"].items():
            if cid not in b["criteria"]:
                continue
            # same PIECE = same pos/neg verdict, expressed on flipped sides
            same = ind(ch, meta[p]["pos_side"]) == ind(b["criteria"][cid], meta[base]["pos_side"])
            cons.append(1.0 if same else 0.0)
        if v.get("overall") and b.get("overall"):
            cons_overall.append(1.0 if ind(v["overall"], meta[p]["pos_side"])
                                == ind(b["overall"], meta[base]["pos_side"]) else 0.0)
    consistency = {"n_replicate_pairs": len(flips), "n_judgements": len(cons),
                   "criterion_order_consistency": round(float(np.mean(cons)), 4) if cons else None,
                   "overall_order_consistency": round(float(np.mean(cons_overall)), 4) if cons_overall else None}

    # ---------------- per-criterion pairwise AUC ----------------
    by_c = defaultdict(list)
    for p, v in real.items():
        t = meta[p]["pos_side"]
        for cid, ch in v["criteria"].items():
            by_c[cid].append(ind(ch, t))
    per_crit = []
    for cid, vals in by_c.items():
        per_crit.append({"id": cid, "name": crit.get(cid, cid), "n": len(vals),
                         "pairwise_auc": round(float(np.mean(vals)), 4),
                         "ci95": boot_ci(vals)})
    per_crit.sort(key=lambda r: -r["pairwise_auc"])

    # ---------------- composite + holistic ----------------
    def composite(subset):
        comp, hol = [], []
        for p, v in subset.items():
            t = meta[p]["pos_side"]
            s = [ind(c, t) for c in v["criteria"].values()]
            net = np.mean(s) - 0.5
            comp.append(1.0 if net > 0 else (0.5 if net == 0 else 0.0))
            if v.get("overall"):
                hol.append(ind(v["overall"], t))
        return comp, hol

    comp, hol = composite(real)
    strat = {}
    for kind in ("same_magazine", "same_year"):
        sub = {p: v for p, v in real.items() if meta[p]["match"] == kind}
        c2, h2 = composite(sub)
        strat[kind] = {"n": len(sub),
                       "composite_auc": round(float(np.mean(c2)), 4) if c2 else None,
                       "holistic_auc": round(float(np.mean(h2)), 4) if h2 else None}

    mean_crit = float(np.mean([r["pairwise_auc"] for r in per_crit]))
    res = {
        "cell": "cw_wigleaf_curation", "probe": "within-pool pairwise comparative judging",
        "judge": vd["judge"], "coverage": f"{len(V)}/{vd['n_want']}",
        "n_real_pairs": len(real), "match_composition": pk["match_composition"],
        "validity_gates": {"anchors": anchors, "order_consistency": consistency,
                           "position_bias": position},
        "composite_auc": round(float(np.mean(comp)), 4), "composite_ci95": boot_ci(comp),
        "holistic_auc": round(float(np.mean(hol)), 4) if hol else None,
        "holistic_ci95": boot_ci(hol) if hol else None,
        "mean_per_criterion_auc": round(mean_crit, 4),
        "n_criteria_above_.55": sum(1 for r in per_crit if r["pairwise_auc"] > .55),
        "n_criteria_ci_excludes_.50": sum(1 for r in per_crit if r["ci95"][0] > .50),
        "stratified": strat,
        "per_criterion": per_crit,
        "absolute_bank_comparison": ABS_BANK,
        "design_caveat":
            "pairwise AUC is measured on MATCHED pairs (130 same-magazine / 70 "
            "same-year) under forced choice; the absolute bank's A_lin .5407 is an "
            "unpaired grouped-OOF AUC over all 1,568 rows. The two are not the same "
            "estimand -- the comparison answers 'does comparative judging separate "
            "the cut', NOT 'is .X bigger than .5407' as a like-for-like delta.",
    }
    (PW / "readout_sol.json").write_text(json.dumps(res, indent=2))

    print("\n=== VALIDITY GATES ===")
    print(f"  anchors        pick-real {anchors['pick_real_rate']} "
          f"(overall {anchors['overall_pick_real_rate']}) on {anchors['n_pairs']} pairs")
    print(f"  order consist. criteria {consistency['criterion_order_consistency']} "
          f"| overall {consistency['overall_order_consistency']}")
    print(f"  position       A {position['A_rate']} / B {position['B_rate']} "
          f"/ TIE {position['tie_rate']}")
    print("\n=== SEPARATION ===")
    print(f"  composite (majority of 45)  {res['composite_auc']}  CI {res['composite_ci95']}")
    print(f"  holistic  (overall)         {res['holistic_auc']}  CI {res['holistic_ci95']}")
    print(f"  mean per-criterion          {res['mean_per_criterion_auc']}")
    print(f"  criteria >.55: {res['n_criteria_above_.55']}/45 | "
          f"CI excludes .50: {res['n_criteria_ci_excludes_.50']}/45")
    print(f"  by match: {strat}")
    print("\n  top 8 criteria:")
    for r in per_crit[:8]:
        print(f"    {r['pairwise_auc']:.4f} CI{r['ci95']} {r['name'][:46]}")
    print("  bottom 4:")
    for r in per_crit[-4:]:
        print(f"    {r['pairwise_auc']:.4f} CI{r['ci95']} {r['name'][:46]}")
    print(f"\n  absolute bank for context: A_lin {ABS_BANK['A_lin']}, "
          f"battery pos-vs-neg {ABS_BANK['battery_pos_vs_neg_auc']}")
    print("WIGLEAF_PAIRWISE_READOUT_DONE")


if __name__ == "__main__":
    main()
