#!/usr/bin/env python3
"""M3 step 4d -- primary adjudicated rediscovery readout from the full-recall instrument.

Reads the sealed judges' per-replicate answers, joins them to the (withheld) target
provenance, and reports:

  * SENSITIVITY  = fraction of HELD-OUT concepts matched by >= 1 fleet proposal
  * SPECIFICITY BASELINE = the same statistic on the stratum-matched RETAINED concepts,
    which are still in the depleted bank.  These are targets the fleet had no reason to
    be pushed toward, so their match rate is the instrument's false-positive floor.
    SENSITIVITY - BASELINE is the informative quantity; a high sensitivity with an
    equally high baseline means the judge is matching generic bank-likeness.
  * breakdowns by alone-AUC stratum, by replicate, by proposer family, by proposer;
    and unique-discovery counts (which family found what nobody else found).

Two judges per replicate; the primary rule is agreement, with either-judge and
both-judges rules reported as the sensitivity band.

CPU only.  Usage: python m3_recall_analyze.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRATCH = Path("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
               "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/robust_mm")
REPS = ("rep1", "rep2", "rep3")
JUDGES = ("A", "B")


def load(rep, j):
    p = SCRATCH / rep / f"recall_out_{j}.json"
    if not p.exists():
        return None
    t = p.read_text().strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n", "", t)
        t = re.sub(r"\n```\s*$", "", t)
    d = json.loads(t)
    return {r["target"]: r for r in d["results"]}


def main():
    man = json.loads((HERE / "m3_recall_manifest.json").read_text())
    prov = {rep: {p["pid"]: p for p in json.loads((HERE / f"proposals_{rep}.json").read_text())["proposals"]}
            for rep in REPS}

    out = {"instrument": "full-recall: every fleet proposal shown for every target; "
                         "8 held-out + 8 stratum-matched retained targets per replicate, "
                         "shuffled and unlabeled",
           "judges": list(JUDGES), "replicates": {}}
    recs = []          # one row per (rep, target)

    for rep in REPS:
        got = {j: load(rep, j) for j in JUDGES}
        got = {j: v for j, v in got.items() if v}
        if not got:
            print(f"{rep}: no judge output, skip")
            continue
        tmap = man[rep]["target_map"]
        cmap = man[rep]["candidate_map"]
        agree = []
        for tid, meta in tmap.items():
            labs, cands = {}, {}
            for j, v in got.items():
                r = v.get(tid)
                if r is None:
                    continue
                labs[j] = bool(r.get("match"))
                cands[j] = [cmap[k] for k in (r.get("candidate_ids") or []) if k in cmap]
            if not labs:
                continue
            agree.append(len(set(labs.values())) == 1)
            both = all(labs.values())
            either = any(labs.values())
            # primary = agreement; on disagreement fall back to strict (both)
            primary = both if len(set(labs.values())) > 1 else either
            pids = sorted({p for j in cands for p in cands[j]}) if either else []
            strict_pids = sorted(set.intersection(*[set(v) for v in cands.values()])) if len(cands) > 1 and both else pids
            fams = sorted({prov[rep][p]["family"] for p in (strict_pids if primary else []) if p in prov[rep]})
            props = sorted({prov[rep][p]["proposer"] for p in (strict_pids if primary else []) if p in prov[rep]})
            recs.append({"rep": rep, "target": tid, "concept": meta["concept"],
                         "kind": meta["kind"], "stratum": meta["stratum"],
                         "alone_auc_fitmine": meta["alone_auc_fitmine"],
                         "match_either": either, "match_both": both, "match_primary": primary,
                         "judge_labels": labs, "matched_pids": pids,
                         "matched_pids_strict": strict_pids,
                         "families": fams, "proposers": props})
        out["replicates"][rep] = {
            "n_candidates": man[rep]["n_candidates"], "n_targets": man[rep]["n_targets"],
            "n_judges": len(got), "judge_agreement": float(np.mean(agree)) if agree else None}
        print(f"{rep}: {len(got)} judges, agreement {out['replicates'][rep]['judge_agreement']}")

    if not recs:
        print("no records"); return

    held = [r for r in recs if r["kind"] == "heldout"]
    ctrl = [r for r in recs if r["kind"] != "heldout"]
    agg = {"n_heldout": len(held), "n_control": len(ctrl)}
    for rule in ("primary", "either", "both"):
        k = f"match_{rule}"
        agg[f"sensitivity_{rule}"] = float(np.mean([r[k] for r in held]))
        agg[f"control_{rule}"] = float(np.mean([r[k] for r in ctrl])) if ctrl else None
        agg[f"lift_{rule}"] = agg[f"sensitivity_{rule}"] - (agg[f"control_{rule}"] or 0.0)
    for s in ("high", "mid", "low"):
        hs = [r["match_primary"] for r in held if r["stratum"] == s]
        cs = [r["match_primary"] for r in ctrl if r["stratum"] == s]
        agg[f"sensitivity_{s}"] = float(np.mean(hs)) if hs else None
        agg[f"n_{s}"] = len(hs)
        agg[f"control_{s}"] = float(np.mean(cs)) if cs else None
    for rep in REPS:
        hs = [r["match_primary"] for r in held if r["rep"] == rep]
        agg[f"sensitivity_{rep}"] = float(np.mean(hs)) if hs else None

    fam_c, fam_u, prop_c = defaultdict(int), defaultdict(int), defaultdict(int)
    for r in held:
        if not r["match_primary"]:
            continue
        for f in r["families"]:
            fam_c[f] += 1
        if len(r["families"]) == 1:
            fam_u[r["families"][0]] += 1
        for p in r["proposers"]:
            prop_c[p] += 1
    agg["per_family_catches"] = dict(fam_c)
    agg["per_family_unique_catches"] = dict(fam_u)
    agg["per_proposer_catches"] = dict(sorted(prop_c.items(), key=lambda kv: -kv[1]))
    agg["rediscovered_concepts"] = {rep: sorted({r["concept"] for r in held
                                                 if r["rep"] == rep and r["match_primary"]})
                                    for rep in REPS}
    agg["missed_concepts"] = {rep: sorted({r["concept"] for r in held
                                           if r["rep"] == rep and not r["match_primary"]})
                              for rep in REPS}
    # ---- concept-level bootstrap on sensitivity and on the sensitivity-minus-control
    # lift.  The lift is the quantity that decides whether rediscovery is DEPLETION-
    # DIRECTED at all; sensitivity alone cannot separate "found because we removed it"
    # from "names this concept anyway".
    rng = np.random.default_rng(0)

    def boot(vals, n=20000):
        v = np.asarray(vals, dtype=float)
        return np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(n)])

    ci = {}
    for tag, hh, cc in [("overall", [r["match_primary"] for r in held],
                         [r["match_primary"] for r in ctrl])] + [
            (s, [r["match_primary"] for r in held if r["stratum"] == s],
             [r["match_primary"] for r in ctrl if r["stratum"] == s])
            for s in ("high", "mid", "low")]:
        if not hh:
            continue
        bs, bc = boot(hh), boot(cc) if cc else None
        e = {"sensitivity": float(np.mean(hh)),
             "sensitivity_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
             "P_sensitivity_ge_0.70": float((bs >= 0.70).mean())}
        if bc is not None:
            d = bs - bc
            e.update({"control": float(np.mean(cc)), "lift": float(np.mean(hh) - np.mean(cc)),
                      "lift_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                      "P_lift_gt0": float((d > 0).mean())})
        ci[tag] = e
    agg["bootstrap"] = ci
    agg["interpretation_contract"] = (
        "design note Sec 8: the closure verdict 'no more articulable signal' is quotable only "
        "at a measured rediscovery sensitivity >= 70% on high-value holdouts. Measured here: "
        f"{ci['high']['sensitivity']:.3f} on the high stratum, 95% CI "
        f"[{ci['high']['sensitivity_ci95'][0]:.3f}, {ci['high']['sensitivity_ci95'][1]:.3f}], "
        f"P(>=.70) = {ci['high']['P_sensitivity_ge_0.70']:.2f}; and the depletion LIFT over "
        f"retained concepts is {ci['overall']['lift']:+.3f} "
        f"[{ci['overall']['lift_ci95'][0]:+.3f}, {ci['overall']['lift_ci95'][1]:+.3f}]. "
        "The floor is NOT met.")

    out["aggregate"] = agg
    out["records"] = recs

    (HERE / "m3_recall.json").write_text(json.dumps(out, indent=2))
    print("\n", json.dumps({k: v for k, v in agg.items()
                            if not isinstance(v, dict) or k.startswith("per_family")}, indent=1))
    print("wrote m3_recall.json")


if __name__ == "__main__":
    main()
