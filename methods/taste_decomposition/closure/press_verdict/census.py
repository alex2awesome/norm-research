#!/usr/bin/env python3
"""ROUND-0 CONCEPT CENSUS of the incoming A bank (freeze: "concept census of the
incoming bank at round 0"), press/journalism verdict cell.

Same ladder as the N&C campaign's census (notes/2026-08-06__closure_nc_responded.md
2.1), same discipline: the embedding is used ONLY to shortlist candidate duplicate
pairs, and only WITHIN one register (bank rubric vs bank rubric); identity is decided
by two sealed blind judges, never by a cosine threshold.

  L0  rubrics delivered
  L1  distinct names (normalised, exact)
  L2  columns surviving the frozen degeneracy screen, fit on FIT+MINE only
  L3  value clusters after collapsing |Pearson r| >= .98 columns
  L5  effective concepts after blind pairwise adjudication (strict: both judges SAME)
  L5' loose rule (either judge says SAME)

Also reports per-column alone-AUC on FIT+MINE only (never MONITOR), the NA /
applicability rate per rubric (this bank is applicability-gated and several rubrics are
plainly off-register for a corporate press release), and the register statement the
freeze requires.

  stage1  -> census_stage1.json + census_blind_packet.json (the judge packet)
  finish  -> census.json          (after both judges' verdicts are on disk)

CPU only.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import cells as C
import closure_core as L

HERE = Path(__file__).resolve().parent
RUBRICS = HERE.parents[3] / "datasets" / "press-releases" / "rubrics.jsonl"
TAU_SHORTLIST = 0.62      # in-register shortlist only; identity decided by judges


def norm_name(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def cmd_stage1(_):
    C.sklearn_guard()
    d = C.load()
    sp = json.loads((HERE / "press_verdict_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    fit = split == "fit_mine"
    y, A, names = d["y"], d["A"], d["a_names"]

    desc = {}
    for line in open(RUBRICS):
        if line.strip():
            r = json.loads(line)
            if r.get("name"):
                desc.setdefault(r["name"], r.get("description", ""))
    matched = sum(1 for n in names if n in desc)

    keep, med = L.clean_fit(A[fit])
    Af = L.clean_apply(A, keep, med)
    kept_names = [names[j] for j in keep]

    # L3: |r| >= .98 collapse on FIT+MINE values
    Z = Af[fit]
    R = np.corrcoef(Z.T)
    par = list(range(len(kept_names)))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    for i in range(len(kept_names)):
        for j in range(i + 1, len(kept_names)):
            if abs(R[i, j]) >= 0.98:
                par[find(i)] = find(j)
    l3 = len({find(i) for i in range(len(kept_names))})

    per = []
    for k, nm in enumerate(kept_names):
        j = names.index(nm)
        col = A[:, j]
        per.append({"name": nm, "alone_AUC_FITMINE": L.auc(y[fit], Af[fit, k]),
                    "applicability_rate": float(np.isfinite(col).mean()),
                    "mean": float(np.nanmean(col)), "std": float(np.nanstd(col))})
    per.sort(key=lambda r: -abs(r["alone_AUC_FITMINE"] - .5))

    # shortlist pairs for the judges, in-register (rubric text vs rubric text)
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = [f"{n}. {desc.get(n, '')}" for n in kept_names]
    Tf = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(docs)
    S = (Tf @ Tf.T).toarray()
    np.fill_diagonal(S, 0.0)
    pairs = [(i, j, float(S[i, j])) for i in range(len(kept_names))
             for j in range(i + 1, len(kept_names)) if S[i, j] >= TAU_SHORTLIST]
    # also always shortlist the top-40 most similar pairs even below tau, so the
    # judges see the strongest candidates regardless of where the threshold falls
    flat = sorted(((float(S[i, j]), i, j) for i in range(len(kept_names))
                   for j in range(i + 1, len(kept_names))), reverse=True)[:40]
    have = {(i, j) for i, j, _ in pairs}
    for s, i, j in flat:
        if (i, j) not in have:
            pairs.append((i, j, s))

    ANCHOR_SAME = [
        ("Clarity of the release's central news point",
         "How clearly the release states the single main piece of news it is announcing.",
         "How plainly the release identifies its one central news item",
         "Whether a reader can tell, quickly, what the single main announcement is."),
        ("Availability of a named media contact",
         "Whether a named person with a working email or phone number is given for press enquiries.",
         "Press-contact details are supplied and reachable",
         "Whether the release names someone reachable by journalists, with contact details."),
    ]
    ANCHOR_DIFF = [
        ("Evidence and sourcing behind the claims",
         "Whether factual claims are tied to a checkable source, filing, study or audited figure.",
         "Length and verbosity of the release",
         "How long and wordy the release is relative to what it announces."),
        ("Quote quality and authenticity",
         "Whether the quotes say something a person would actually say and add information.",
         "Use of an all-caps city-and-date dateline",
         "Whether the text opens with the conventional wire-service dateline format."),
    ]

    packet = {"items": [], "anchors": []}
    for k, (i, j, s) in enumerate(sorted(pairs, key=lambda p: -p[2])):
        packet["items"].append({
            "pair_id": f"Q{k+1:03d}", "cos": s,
            "X_name": kept_names[i], "X_desc": desc.get(kept_names[i], ""),
            "Y_name": kept_names[j], "Y_desc": desc.get(kept_names[j], "")})
    for k, (xn, xd, yn, yd) in enumerate(ANCHOR_SAME):
        packet["anchors"].append({"pair_id": f"AS{k+1}", "truth": "SAME",
                                  "X_name": xn, "X_desc": xd, "Y_name": yn, "Y_desc": yd})
    for k, (xn, xd, yn, yd) in enumerate(ANCHOR_DIFF):
        packet["anchors"].append({"pair_id": f"AD{k+1}", "truth": "DIFFERENT",
                                  "X_name": xn, "X_desc": xd, "Y_name": yn, "Y_desc": yd})

    out = {"cell": "press_verdict",
           "L0_rubrics_delivered": len(names),
           "L1_distinct_names": len({norm_name(n) for n in names}),
           "L2_after_degeneracy_screen": len(kept_names),
           "L2_dropped": [n for n in names if n not in kept_names],
           "L3_value_clusters_r98": l3,
           "max_abs_column_r": float(np.nanmax(np.abs(R - np.eye(len(kept_names))))),
           "frac_pairs_r_ge_90": float((np.abs(R - np.eye(len(kept_names))) >= .90).sum()
                                       / (len(kept_names) * (len(kept_names) - 1))),
           "rubric_descriptions_matched": matched,
           "per_column": per,
           "alone_AUC_summary": {
               "max": max(p["alone_AUC_FITMINE"] for p in per),
               "min": min(p["alone_AUC_FITMINE"] for p in per),
               "median": float(np.median([p["alone_AUC_FITMINE"] for p in per])),
               "MAD_from_chance": float(np.median([abs(p["alone_AUC_FITMINE"] - .5) for p in per])),
               "n_ge_55": sum(1 for p in per if p["alone_AUC_FITMINE"] >= .55),
               "n_le_45": sum(1 for p in per if p["alone_AUC_FITMINE"] <= .45)},
           "n_pairs_shortlisted": len(pairs),
           "register_statement": (
               "The bank is written in JOURNALISM-ETHICS / PR-PRACTITIONER language "
               "(source transparency, attribution, disclosure, newsroom style, multimedia "
               "assets, CONSORT adherence) and was coverage-selected as 40 k-means medoids "
               "over a 309-rubric pool built for news and science communication generally. "
               "The corpus is CORPORATE AND INSTITUTIONAL PRESS RELEASES and the outcome is "
               "EDITORIAL PICKUP by >=3 tracked outlets. Several rubrics (CONSORT adherence, "
               "primary-outcome effect sizes, apology and acceptance of responsibility, "
               "M&A/SPAC disclosure) are off-register for most items, which is why the bank "
               "is applicability-gated. No cosine threshold decides identity anywhere in "
               "this campaign."),
           }
    (HERE / "census_stage1.json").write_text(json.dumps(out, indent=1, default=float))
    (HERE / "census_blind_packet.json").write_text(json.dumps(packet, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_column"}, indent=1, default=float))
    print(f"\npacket: {len(packet['items'])} real pairs + {len(packet['anchors'])} anchors")
    print("\nTOP/BOTTOM alone-AUC (FIT+MINE only):")
    for p in per[:8]:
        print(f"  {p['alone_AUC_FITMINE']:.3f}  appl={p['applicability_rate']:.2f}  {p['name'][:60]}")


def cmd_finish(a):
    s1 = json.loads((HERE / "census_stage1.json").read_text())
    packet = json.loads((HERE / "census_blind_packet.json").read_text())
    vs = [json.loads(Path(p).read_text()) for p in a.verdicts.split(",")]
    maps = [{v["pair_id"]: v["verdict"].upper() for v in d["verdicts"]} for d in vs]

    anchors = []
    for an in packet["anchors"]:
        got = [m.get(an["pair_id"]) for m in maps]
        anchors.append({"pair_id": an["pair_id"], "truth": an["truth"], "got": got,
                        "pass": [g == an["truth"] for g in got]})
    ids = [it["pair_id"] for it in packet["items"]]
    agree = sum(1 for i in ids if maps[0].get(i) == maps[1].get(i)) / max(1, len(ids))

    name_of = {}
    for it in packet["items"]:
        name_of[it["pair_id"]] = (it["X_name"], it["Y_name"])
    kept = [p["name"] for p in s1["per_column"]]
    idx = {n: k for k, n in enumerate(kept)}

    def union(edges):
        par = list(range(len(kept)))

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        for xn, yn in edges:
            if xn in idx and yn in idx:
                par[find(idx[xn])] = find(idx[yn])
        return len({find(i) for i in range(len(kept))}), par

    strict_edges = [name_of[i] for i in ids if all(m.get(i) == "SAME" for m in maps)]
    loose_edges = [name_of[i] for i in ids if any(m.get(i) == "SAME" for m in maps)]
    l5, par = union(strict_edges)
    l5p, _ = union(loose_edges)

    out = dict(s1)
    out.pop("per_column", None)
    out.update({
        "judges": [d.get("judge", "sealed-blind") for d in vs],
        "judge_raw_agreement": agree,
        "anchor_battery": anchors,
        "anchor_all_pass": all(all(a_["pass"]) for a_ in anchors),
        "n_pairs_adjudicated": len(ids),
        "n_merge_edges_strict": len(strict_edges),
        "n_merge_edges_loose": len(loose_edges),
        "L5_effective_concepts_strict": l5,
        "L5_loose": l5p,
        "merged_pairs_strict": strict_edges,
        "collapse_pct_L0_to_L5": 1 - l5 / s1["L0_rubrics_delivered"],
    })
    out["per_column"] = s1["per_column"]
    (HERE / "census.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "per_column"}, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stage1")
    f = sub.add_parser("finish")
    f.add_argument("--verdicts", required=True)
    a = ap.parse_args()
    {"stage1": cmd_stage1, "finish": cmd_finish}[a.cmd](a)
