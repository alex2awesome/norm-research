#!/usr/bin/env python3
"""B-SIDE step -- primary adjudicated rediscovery readout from the full-recall
instrument, mirroring ../m3_recall_analyze.py.

Two sealed judges per replicate (recall_out_A.json / recall_out_B.json). Where they
AGREE, that is the verdict. Where they DISAGREE (borderline), the target is resolved
by a third, provenance-stripped ADJUDICATION pass (adjudicate_out.json) that is the
FINAL word for the primary sensitivity statistic -- this is what the task spec calls
"borderline adjudicated in a provenance-stripped pass", and is a tighter design than
the A-side's implicit strict-on-disagreement fallback (A-side never actually ran a
third pass; this one does, when there are any borderline targets).

Reports SENSITIVITY (held-out match rate), the RETAINED-CHANNEL CONTROL (false-
positive floor -- channels this replicate never held out), the LIFT (sensitivity -
control, the zero-lift test), breakdowns by alone-AUC stratum / replicate / proposer
family, and unique-discovery counts.

CPU only. Usage: python recall_analyze_b.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRATCH = Path("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
               "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/robust_mm/bside")
REPS = ("bside_rep1", "bside_rep2", "bside_rep3")
JUDGES = ("A", "B")


def load(rep, fname):
    p = SCRATCH / rep / fname
    if not p.exists():
        return None
    t = p.read_text().strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n", "", t)
        t = re.sub(r"\n```\s*$", "", t)
    d = json.loads(t)
    return {r["target"]: r for r in d["results"]}


def main():
    man = json.loads((HERE / "bside_recall_manifest.json").read_text())
    prov = {rep: {p["pid"]: p for p in json.loads((HERE / f"proposals_{rep}.json").read_text())["proposals"]}
            for rep in REPS if (HERE / f"proposals_{rep}.json").exists()}

    out = {"instrument": "full-recall: every fleet proposal shown for every target; "
                         "6 held-out + 6 stratum-matched retained targets per replicate, "
                         "shuffled and unlabeled; disagreements resolved by a third "
                         "provenance-stripped adjudication pass",
           "judges": list(JUDGES), "replicates": {}}
    recs = []
    borderline_log = []

    for rep in REPS:
        if rep not in man:
            continue
        got = {j: load(rep, f"recall_out_{j}.json") for j in JUDGES}
        got = {j: v for j, v in got.items() if v}
        if not got:
            print(f"{rep}: no judge output, skip")
            continue
        adjud = load(rep, "adjudicate_out.json")  # may be None if no borderline targets
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
            is_agree = len(set(labs.values())) == 1
            agree.append(is_agree)
            if is_agree:
                final = list(labs.values())[0]
                final_pids = sorted({p for c in cands.values() for p in c})
            else:
                # borderline -- resolved by the adjudication pass if it ran;
                # otherwise fall back to strict (both-must-agree) as a conservative default
                borderline_log.append({"rep": rep, "target": tid, "judge_labels": labs})
                if adjud is not None and tid in adjud:
                    ar = adjud[tid]
                    final = bool(ar.get("match"))
                    final_pids = sorted({cmap[k] for k in (ar.get("candidate_ids") or []) if k in cmap})
                else:
                    final = False
                    final_pids = []
            fams = sorted({prov[rep][p]["family"] for p in final_pids if p in prov[rep]})
            props = sorted({prov[rep][p]["proposer"] for p in final_pids if p in prov[rep]})
            recs.append({"rep": rep, "target": tid, "channel": meta["channel"],
                         "kind": meta["kind"], "stratum": meta["stratum"],
                         "alone_auc_fitmine": meta["alone_auc_fitmine"],
                         "match_final": final, "was_borderline": not is_agree,
                         "judge_labels": labs, "matched_pids": final_pids,
                         "families": fams, "proposers": props})
        out["replicates"][rep] = {
            "n_candidates": man[rep]["n_candidates"], "n_targets": man[rep]["n_targets"],
            "n_judges": len(got), "judge_agreement": float(np.mean(agree)) if agree else None,
            "n_borderline": sum(1 for a in agree if not a),
            "adjudication_ran": adjud is not None}
        print(f"{rep}: {len(got)} judges, agreement {out['replicates'][rep]['judge_agreement']}, "
              f"borderline {out['replicates'][rep]['n_borderline']}")

    if not recs:
        print("no records"); (HERE / "bside_recall.json").write_text(json.dumps(out, indent=2)); return

    held = [r for r in recs if r["kind"] == "heldout"]
    ctrl = [r for r in recs if r["kind"] != "heldout"]
    agg = {"n_heldout": len(held), "n_control": len(ctrl),
           "n_borderline_total": len(borderline_log)}
    agg["sensitivity"] = float(np.mean([r["match_final"] for r in held])) if held else None
    agg["control"] = float(np.mean([r["match_final"] for r in ctrl])) if ctrl else None
    agg["lift"] = (agg["sensitivity"] - agg["control"]) if held and ctrl else None

    for s in ("high", "mid", "low"):
        hs = [r["match_final"] for r in held if r["stratum"] == s]
        cs = [r["match_final"] for r in ctrl if r["stratum"] == s]
        agg[f"sensitivity_{s}"] = float(np.mean(hs)) if hs else None
        agg[f"n_{s}"] = len(hs)
        agg[f"control_{s}"] = float(np.mean(cs)) if cs else None
    for rep in REPS:
        hs = [r["match_final"] for r in held if r["rep"] == rep]
        agg[f"sensitivity_{rep}"] = float(np.mean(hs)) if hs else None

    fam_c, fam_u, prop_c = defaultdict(int), defaultdict(int), defaultdict(int)
    for r in held:
        if not r["match_final"]:
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
    agg["rediscovered_channels"] = {rep: sorted({r["channel"] for r in held
                                                 if r["rep"] == rep and r["match_final"]})
                                    for rep in REPS}
    agg["missed_channels"] = {rep: sorted({r["channel"] for r in held
                                           if r["rep"] == rep and not r["match_final"]})
                              for rep in REPS}

    rng = np.random.default_rng(0)

    def boot(vals, n=20000):
        v = np.asarray(vals, dtype=float)
        return np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(n)])

    ci = {}
    for tag, hh, cc in [("overall", [r["match_final"] for r in held],
                         [r["match_final"] for r in ctrl])] + [
            (s, [r["match_final"] for r in held if r["stratum"] == s],
             [r["match_final"] for r in ctrl if r["stratum"] == s])
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
    if "high" in ci:
        agg["interpretation_contract"] = (
            "B-side (spurious-track) mirror of the A-side M3 interpretation contract: "
            "a claim of the form 'the map covers the channel space' is quotable only at a "
            "measured rediscovery sensitivity >= 70% on high-value holdouts, with lift over "
            "the retained control distinguishable from zero. Measured here: "
            f"{ci['high']['sensitivity']:.3f} on the high stratum, 95% CI "
            f"[{ci['high']['sensitivity_ci95'][0]:.3f}, {ci['high']['sensitivity_ci95'][1]:.3f}], "
            f"P(>=.70) = {ci['high']['P_sensitivity_ge_0.70']:.2f}; overall lift "
            f"{ci['overall']['lift']:+.3f} [{ci['overall']['lift_ci95'][0]:+.3f}, "
            f"{ci['overall']['lift_ci95'][1]:+.3f}], P(lift>0) = {ci['overall']['P_lift_gt0']:.2f}.")

    out["aggregate"] = agg
    out["records"] = recs
    out["borderline_log"] = borderline_log

    (HERE / "bside_recall.json").write_text(json.dumps(out, indent=2))
    print("\n", json.dumps({k: v for k, v in agg.items()
                            if not isinstance(v, dict) or k.startswith("per_family")}, indent=1))
    print("wrote bside_recall.json")


if __name__ == "__main__":
    main()
