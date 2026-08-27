#!/usr/bin/env python3
"""M3 step 4 -- rediscovery detection, mechanical + adjudicated.

TWO READOUTS, because the mechanical one turns out to be out of range.

(1) MECHANICAL (tau on bge cosine).  A held-out bank concept counts as rediscovered if
    some fleet proposal sits at cosine >= tau.  tau's floor comes from the pilot's
    PLANTED PROBES (lexical look-alikes of a real criterion, conceptually distinct;
    name+definition cosines .739 and .615), so the defensible band is tau >= .78 and
    the pilot used .78-.80.  Reported at .77 / .79 / .81.

    *** RANGE WARNING, measured not assumed: the ENTIRE bank-vs-fleet cosine
    distribution tops out around .75 -- below the threshold band. The pilot's tau was
    calibrated on WITHIN-REGISTER pairs (mined ML-abstract criteria vs mined ML-abstract
    criteria).  The 154-bank is written in general scientific-reporting register
    (CONSORT / PRISMA / STROBE / TIDieR items) while the fleet, reading ML abstracts,
    writes in ML register.  Cross-register cosine sits systematically lower, so the
    mechanical detector has NO dynamic range above tau here and returns ~0 by
    construction.  Quoting it as a rediscovery rate would be an artefact. ***

(2) ADJUDICATED (blind, provenance-stripped, with a matched control and known-label
    anchors).  For every held-out concept the top-N nearest proposals are pooled with
    the top-N nearest proposals of an equal number of RETAINED concepts (still in the
    depleted bank -- the false-positive baseline), plus known-SAME and known-DIFFERENT
    anchor pairs.  Everything is shuffled by hash, X/Y order randomised, provenance
    removed.  A sealed judge decides same-concept vs different-concept.  Sensitivity =
    hit rate on held-out; specificity baseline = hit rate on retained.

CPU only.  Usage: python m3_detect.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CLOSURE = HERE.parent
sys.path.insert(0, str(HERE))
import embed_lib as E  # noqa: E402

TAUS = (0.77, 0.79, 0.81)
TAU_MAIN = 0.79
REPS = ("rep1", "rep2", "rep3")
TOP_N = 3                      # candidates per concept sent to adjudication
SALT = "m3-blind-adjudication-v1"


def h(s):
    return hashlib.sha256(f"{SALT}|{s}".encode()).hexdigest()


def anchor_pairs():
    """Known-label anchors (blinded-anchor-battery rule).

    DIFFERENT: the pilot's two planted probes -- authored to be lexical look-alikes of a
    real criterion and conceptually distinct; a blind auditor caught both.
    SAME: the pilot's two highest cross-round recaptures, which the census read as
    genuine re-namings of one concept.
    """
    prop = {}
    for r in (1, 2, 3, 4):
        for c in json.loads((CLOSURE / f"round{r}_proposals_blinded.json").read_text())["criteria"]:
            prop[f"r{r}:{c['id']}"] = c
    pairs = [
        ("ANCHOR_DIFF_1", "different", "r4:P10", "r4:P05"),
        ("ANCHOR_DIFF_2", "different", "r4:P24", "r4:P17"),
        ("ANCHOR_SAME_1", "same", "r1:P05", "r3:P12"),
        ("ANCHOR_SAME_2", "same", "r2:P02", "r3:P07"),
    ]
    out = []
    for tag, truth, a, b in pairs:
        if a not in prop or b not in prop:
            continue
        out.append({"kind": "anchor", "truth": truth, "anchor_tag": tag,
                    "text_X": E.crit_text(prop[a]["name"], prop[a]["instruction"]),
                    "text_Y": E.crit_text(prop[b]["name"], prop[b]["instruction"])})
    return out


def main():
    cfg = json.loads((HERE / "m3_concepts.json").read_text())
    bank = E.bank_concept_texts()
    conc_rows = {r["concept"]: r for r in cfg["concepts"]}

    out = {"tau_main": TAU_MAIN, "taus_reported": list(TAUS),
           "probe_floor_name_plus_definition": E.PROBE_FLOOR,
           "detection_rule_mechanical": "held-out concept REDISCOVERED if max cosine over "
                                        "the replicate's fleet proposals >= tau",
           "replicates": {}}
    adjudication = list(anchor_pairs())

    for rep in REPS:
        pf = HERE / f"proposals_{rep}.json"
        if not pf.exists():
            print(f"{rep}: proposals missing, skip")
            continue
        props = json.loads(pf.read_text())["proposals"]
        held = cfg["replicates"][rep]
        detail = {d["concept"]: d for d in cfg["replicate_detail"][rep]}
        retained = [c for c in cfg["concept_footprints"] if c not in held]

        ptext = [E.crit_text(p["name"], p["instruction"]) for p in props]
        Ep = E.embed(ptext)
        Ec = E.embed([E.crit_text(c, bank[c]) for c in held])
        S = Ec @ Ep.T

        # stratum-matched control concepts: same stratum mix as the holdout, chosen by
        # stable hash so the control is reproducible and not cherry-picked.
        ctrl_names = []
        for s, want in (("high", 3), ("mid", 3), ("low", 2)):
            pool = sorted([c for c in retained if conc_rows[c]["stratum"] == s],
                          key=lambda c: h(f"{rep}|ctrl|{c}"))
            ctrl_names += pool[:want]
        Ectrl = E.embed([E.crit_text(c, bank[c]) for c in ctrl_names])
        Sctrl = Ectrl @ Ep.T

        rows = []
        for i, c in enumerate(held):
            order = np.argsort(-S[i])
            rec = {"concept": c, "stratum": detail[c]["stratum"],
                   "alone_auc_fitmine": detail[c]["alone_auc_fitmine"],
                   "max_cos": float(S[i, order[0]]),
                   "top5": [{"cos": float(S[i, j]), "pid": props[j]["pid"],
                             "family": props[j]["family"], "name": props[j]["name"]}
                            for j in order[:5]]}
            for t in TAUS:
                hit = S[i] >= t
                rec[f"rediscovered_tau{t}"] = bool(hit.any())
                rec[f"n_matching_proposals_tau{t}"] = int(hit.sum())
                rec[f"families_matching_tau{t}"] = sorted({props[j]["family"] for j in np.where(hit)[0]})
                rec[f"proposers_matching_tau{t}"] = sorted({props[j]["proposer"] for j in np.where(hit)[0]})
            rows.append(rec)
            for j in order[:TOP_N]:
                adjudication.append({
                    "kind": "heldout", "rep": rep, "concept": c, "pid": props[int(j)]["pid"],
                    "proposer": props[int(j)]["proposer"], "family": props[int(j)]["family"],
                    "cos": float(S[i, j]), "stratum": detail[c]["stratum"],
                    "text_X": E.crit_text(c, bank[c]), "text_Y": ptext[int(j)]})

        for i, c in enumerate(ctrl_names):
            order = np.argsort(-Sctrl[i])
            for j in order[:TOP_N]:
                adjudication.append({
                    "kind": "control_retained", "rep": rep, "concept": c,
                    "pid": props[int(j)]["pid"], "proposer": props[int(j)]["proposer"],
                    "family": props[int(j)]["family"], "cos": float(Sctrl[i, j]),
                    "stratum": conc_rows[c]["stratum"],
                    "text_X": E.crit_text(c, bank[c]), "text_Y": ptext[int(j)]})

        ctrl = {"n_retained_concepts": len(retained), "control_concepts": ctrl_names}
        for t in TAUS:
            ctrl[f"retained_match_rate_tau{t}"] = float(np.mean(
                ((Ep @ E.embed([E.crit_text(c, bank[c]) for c in retained], verbose=False).T) >= t
                 ).any(axis=0)))
        ctrl["all_bank54_vs_fleet_max_cos"] = float(
            (E.embed([E.crit_text(c, bank[c]) for c in cfg["concept_footprints"]], verbose=False)
             @ Ep.T).max())

        out["replicates"][rep] = {
            "n_proposals": len(props), "n_proposers": len({p["proposer"] for p in props}),
            "concepts": rows, "specificity_control": ctrl,
            **{f"rediscovery_rate_tau{t}": float(np.mean([r[f"rediscovered_tau{t}"] for r in rows]))
               for t in TAUS}}
        print(f"{rep}: {len(props)} proposals / {out['replicates'][rep]['n_proposers']} proposers; "
              f"mechanical " + " ".join(f"tau{t}={out['replicates'][rep][f'rediscovery_rate_tau{t}']:.2f}"
                                        for t in TAUS)
              + f"; max bank-vs-fleet cos {ctrl['all_bank54_vs_fleet_max_cos']:.3f}")

    allrows = [r for rep in out["replicates"] for r in out["replicates"][rep]["concepts"]]
    if allrows:
        agg = {"n_concepts": len(allrows)}
        for t in TAUS:
            agg[f"overall_rediscovery_tau{t}"] = float(np.mean([r[f"rediscovered_tau{t}"] for r in allrows]))
            for s in ("high", "mid", "low"):
                sub = [r for r in allrows if r["stratum"] == s]
                agg[f"{s}_n"] = len(sub)
                agg[f"{s}_rediscovery_tau{t}"] = float(np.mean([r[f"rediscovered_tau{t}"] for r in sub])) if sub else None
            agg[f"CONTROL_retained_match_rate_tau{t}"] = float(np.mean(
                [out["replicates"][rep]["specificity_control"][f"retained_match_rate_tau{t}"]
                 for rep in out["replicates"]]))
        agg["max_cos_distribution_heldout"] = {
            "min": float(min(r["max_cos"] for r in allrows)),
            "median": float(np.median([r["max_cos"] for r in allrows])),
            "max": float(max(r["max_cos"] for r in allrows))}
        agg["RANGE_WARNING"] = (
            "the maximum cosine between ANY of the 54 bank concepts and ANY fleet proposal is "
            f"{max(out['replicates'][r]['specificity_control']['all_bank54_vs_fleet_max_cos'] for r in out['replicates']):.3f}"
            ", below the tau band; the mechanical rate is 0 by construction and must not be "
            "quoted as a rediscovery rate.  Use the adjudicated readout.")
        out["aggregate"] = agg
        print("\n", json.dumps({k: v for k, v in agg.items() if "tau0.79" in k or "RANGE" in k
                                or k.startswith("max_cos")}, indent=1))

    # -------------------------------------------------- blind adjudication set --
    for a in adjudication:
        base = a.get("anchor_tag") or f"{a['rep']}|{a['concept']}|{a['pid']}"
        a["bid"] = h(base)[:10]
        if int(h(base)[:2], 16) % 2:                       # randomise X/Y order
            a["text_X"], a["text_Y"] = a["text_Y"], a["text_X"]
            a["xy_swapped"] = True
    adjudication.sort(key=lambda a: a["bid"])

    (HERE / "m3_detection.json").write_text(json.dumps(out, indent=1))
    (HERE / "m3_adjudication_key.json").write_text(json.dumps(adjudication, indent=1))
    (HERE / "m3_adjudication_blind.json").write_text(json.dumps({
        "task": ("For each pair decide whether text_X and text_Y name the SAME evaluative "
                 "concept -- i.e. a judge scoring a paper abstract on one would be scoring "
                 "essentially the same property as a judge scoring it on the other -- or "
                 "DIFFERENT concepts. Wording, register and level of specificity will differ; "
                 "judge the underlying property, not the phrasing. Partial overlap where one "
                 "is clearly broader or targets a different property = DIFFERENT."),
        "labels": ["same", "different"],
        "n_pairs": len(adjudication),
        "pairs": [{"bid": a["bid"], "text_X": a["text_X"], "text_Y": a["text_Y"]}
                  for a in adjudication]}, indent=1))
    print(f"\nwrote m3_detection.json + {len(adjudication)} blind pairs "
          f"({sum(1 for a in adjudication if a['kind']=='heldout')} held-out, "
          f"{sum(1 for a in adjudication if a['kind']=='control_retained')} control, "
          f"{sum(1 for a in adjudication if a['kind']=='anchor')} anchors)")


if __name__ == "__main__":
    main()
