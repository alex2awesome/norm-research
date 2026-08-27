#!/usr/bin/env python3
"""M3 step 4b -- fold the two blind adjudications into the rediscovery readout.

Inputs: adjudication_judgeA.json / adjudication_judgeB.json (two independent sealed
Opus judges, provenance-stripped, X/Y order randomised) and m3_adjudication_key.json
(the provenance the judges never saw).

Reported:
  * ANCHOR PASS RATE on the known-label pairs (blinded-anchor-battery rule).  Two
    known-DIFFERENT anchors are the pilot's planted probes; two known-SAME anchors are
    the pilot's highest cross-round recaptures.
  * inter-judge agreement + Cohen's kappa;
  * REDISCOVERY (adjudicated): a held-out concept is rediscovered if >=1 of its top-N
    candidate proposals is judged "same".  Reported under three rules -- either judge
    (liberal), both judges (strict), and the primary = union with the third-party
    adjudication of disagreements;
  * SPECIFICITY: the identical statistic on the stratum-matched RETAINED concepts,
    which are still in the depleted bank and therefore should NOT be "rediscovered"
    at a high rate.  Sensitivity minus this baseline is the informative quantity;
  * per-stratum, per-family and per-proposer breakdowns.

CPU only.  Usage: python m3_adjudicate_analyze.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRATCH = Path("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
               "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/robust_mm")


def load_judge(name):
    p = SCRATCH / f"adjudication_judge{name}.json"
    if not p.exists():
        return None
    txt = p.read_text().strip()
    if txt.startswith("```"):
        txt = txt.split("\n", 1)[1].rsplit("```", 1)[0]
    d = json.loads(txt)
    return {j["bid"]: j["label"].strip().lower() for j in d["judgements"]}


def kappa(a, b, bids):
    la = [a[k] for k in bids]
    lb = [b[k] for k in bids]
    po = np.mean([x == y for x, y in zip(la, lb)])
    pa_s, pb_s = np.mean([x == "same" for x in la]), np.mean([x == "same" for x in lb])
    pe = pa_s * pb_s + (1 - pa_s) * (1 - pb_s)
    return float(po), float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def main():
    key = {a["bid"]: a for a in json.loads((HERE / "m3_adjudication_key.json").read_text())}
    A, B = load_judge("A"), load_judge("B")
    third = {}
    tp = HERE / "m3_adjudication_third_pass.json"
    if tp.exists():
        third = {k: v for k, v in json.loads(tp.read_text()).items()}

    judges = {k: v for k, v in (("A", A), ("B", B)) if v is not None}
    assert judges, "no judge output found"
    bids = [b for b in key if all(b in j for j in judges.values())]
    print(f"{len(bids)}/{len(key)} pairs judged by all {len(judges)} judges")

    out = {"n_pairs": len(key), "n_judged_by_all": len(bids), "judges": list(judges)}

    # ---------------------------------------------------------------- anchors --
    anch = [b for b in bids if key[b]["kind"] == "anchor"]
    out["anchor_label_strength"] = {
        "ANCHOR_DIFF_1/2": "STRONG label -- the pilot's deliberately PLANTED probes, authored "
                           "to be lexical look-alikes of a real criterion and conceptually "
                           "distinct; the pilot's blind auditor caught both.",
        "ANCHOR_SAME_1/2": "WEAK label -- the pilot's two highest cross-round embedding "
                           "recaptures (cos .813/.812).  They were never human-verified as "
                           "the same concept; a strict judge rejecting them is evidence about "
                           "the LABEL, not only about the judge.  Failures here should be read "
                           "as 'this judge is strict', which makes the reported sensitivity a "
                           "LOWER bound."}
    out["anchors"] = {}
    for jn, j in judges.items():
        hits = [(key[b]["anchor_tag"], key[b]["truth"], j[b]) for b in anch]
        out["anchors"][jn] = {"detail": [{"tag": t, "truth": tr, "judged": jd} for t, tr, jd in hits],
                              "pass_rate": float(np.mean([tr == jd for _, tr, jd in hits])) if hits else None}
        print(f"  anchor pass {jn}: {out['anchors'][jn]['pass_rate']}")

    if len(judges) == 2:
        po, kp = kappa(A, B, bids)
        out["inter_judge"] = {"raw_agreement": po, "cohens_kappa": kp,
                              "n_disagreements": int(sum(A[b] != B[b] for b in bids))}
        print(f"  inter-judge agreement {po:.3f}, kappa {kp:.3f}, "
              f"{out['inter_judge']['n_disagreements']} disagreements")

    # ------------------------------------------------------------ rediscovery --
    def label(b, rule):
        ls = [j[b] for j in judges.values()]
        if rule == "either":
            return "same" in ls
        if rule == "both":
            return all(x == "same" for x in ls)
        # primary: agreement wins; disagreements go to the third pass (default strict)
        if len(set(ls)) == 1:
            return ls[0] == "same"
        return third.get(b, "different") == "same"

    res = {}
    for rule in ("either", "both", "primary"):
        by_concept = defaultdict(lambda: {"hit": False, "pids": [], "families": set(), "proposers": set()})
        for b in bids:
            k = key[b]
            if k["kind"] == "anchor":
                continue
            e = by_concept[(k["kind"], k["rep"], k["concept"], k["stratum"])]
            if label(b, rule):
                e["hit"] = True
                e["pids"].append(k["pid"])
                e["families"].add(k["family"])
                e["proposers"].add(k["proposer"])
        held = {k: v for k, v in by_concept.items() if k[0] == "heldout"}
        ctrl = {k: v for k, v in by_concept.items() if k[0] == "control_retained"}
        r = {"n_heldout": len(held), "n_control": len(ctrl),
             "sensitivity_heldout": float(np.mean([v["hit"] for v in held.values()])),
             "false_positive_control": float(np.mean([v["hit"] for v in ctrl.values()]))}
        r["lift"] = r["sensitivity_heldout"] - r["false_positive_control"]
        for s in ("high", "mid", "low"):
            hs = [v["hit"] for k, v in held.items() if k[3] == s]
            cs = [v["hit"] for k, v in ctrl.items() if k[3] == s]
            r[f"sensitivity_{s}"] = float(np.mean(hs)) if hs else None
            r[f"n_{s}"] = len(hs)
            r[f"control_{s}"] = float(np.mean(cs)) if cs else None
        for rep in ("rep1", "rep2", "rep3"):
            hs = [v["hit"] for k, v in held.items() if k[1] == rep]
            r[f"sensitivity_{rep}"] = float(np.mean(hs)) if hs else None
        fams = defaultdict(int)
        uniq = defaultdict(int)
        props = defaultdict(int)
        for k, v in held.items():
            for f in v["families"]:
                fams[f] += 1
            if len(v["families"]) == 1 and v["hit"]:
                uniq[list(v["families"])[0]] += 1
            for p in v["proposers"]:
                props[p] += 1
        r["per_family_catches"] = dict(fams)
        r["per_family_unique_catches"] = dict(uniq)
        r["per_proposer_catches"] = dict(sorted(props.items(), key=lambda kv: -kv[1]))
        r["rediscovered_by_rep"] = {rep: sorted({k[2] for k, v in held.items() if k[1] == rep and v["hit"]})
                                    for rep in ("rep1", "rep2", "rep3")}
        r["missed_by_rep"] = {rep: sorted({k[2] for k, v in held.items() if k[1] == rep and not v["hit"]})
                              for rep in ("rep1", "rep2", "rep3")}
        res[rule] = r
        print(f"  rule={rule:8s} sensitivity {r['sensitivity_heldout']:.3f} "
              f"control {r['false_positive_control']:.3f} lift {r['lift']:+.3f} "
              f"(high {r['sensitivity_high']}, mid {r['sensitivity_mid']}, low {r['sensitivity_low']})")
    out["adjudicated"] = res

    # bid list consumed by m3_recover.py
    same_bids = [b for b in bids if key[b]["kind"] == "heldout" and label(b, "primary")]
    (HERE / "m3_borderline_adjudication.json").write_text(json.dumps(
        {"rule": "primary (both judges agree; disagreements -> third pass, default different)",
         "same_concept_bids": same_bids}, indent=1))
    if len(judges) == 2:
        out["disagreement_bids"] = [b for b in bids if A[b] != B[b] and key[b]["kind"] != "anchor"]
    (HERE / "m3_adjudicated.json").write_text(json.dumps(out, indent=2))
    print("\nwrote m3_adjudicated.json")


if __name__ == "__main__":
    main()
