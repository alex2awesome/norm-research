#!/usr/bin/env python3
"""ROUND-0 CONCEPT CENSUS of the incoming bank (freeze: "concept census of the
incoming bank at round 0"), AoPS CURATION cell.

Same ladder and same discipline as the press / N&C censuses: the TF-IDF cosine is
used ONLY to shortlist candidate duplicate pairs, and only WITHIN one register
(bank rubric text vs bank rubric text); identity is decided by two sealed blind
judges, never by a cosine threshold.

  L0  criteria delivered
  L1  distinct names (normalised, exact)
  L2  columns surviving the frozen degeneracy screen, fit on FIT+MINE only
  L3  value clusters after collapsing |Pearson r| >= .98 columns
  L5  effective concepts after blind pairwise adjudication (strict: both judges SAME)
  L5' loose rule (either judge says SAME)

THIS CELL ALSO CENSUSES THE V BLOCK.  Unlike press (where V is a reconstructed
88-column hand bank nobody would re-propose), AoPS's V is 24 named surface
features computed on the SOLUTION BODY ONLY -- LaTeX density, display-math count,
word count, type-token ratio, boxed-answer and hide-block markers, proof framing
-- and several of the obvious upstream priors (solution length, LaTeX habit,
forum-markup habit) are ALREADY IN IT.  Any Track-B channel that duplicates a v_* column is a
channel the ARTICULATED instrument already owns, so it cannot be discounted off
Delta without also being discounted off VA.  The V census records exactly which
priors are pre-owned so round 1 cannot double-count them.

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
REPO = HERE.parents[3]
RUBRICS = REPO / "datasets" / "math" / "aops" / "va" / "rubrics.jsonl"
TAU_SHORTLIST = 0.30      # in-register shortlist only; identity decided by judges

# the named upstream priors for AoPS curation, and where they already live
UPSTREAM_PRIORS = {
    "post position in the thread / thread age (arrival order)":
        "NOT in any bank; audited as an OBSERVED covariate in position_line.py using the "
        "TRUE ordinals recovered from forum_solutions.parquet (post_number, sol_rank, "
        "thread_age_days, years_after_contest)",
    "poster standing on the forum (a prolific regular vs a one-post visitor)":
        "PARTIAL -- v_proof_framing, v_first_person, v_meta_doubt, v_heavy_machinery, "
        "v_standard_tech already encode register and machinery habit; standing itself is "
        "unobserved in the text but IS observed in the crawl (poster_n_posts), and is "
        "reported beside the position family without ever entering a bank",
    "LaTeX / display-math habit":
        "ALREADY IN V -- v_latex_density, v_imath_tag_count, v_n_display_math",
    "solution length":
        "ALREADY IN V -- v_log_len, v_word_count, v_sentence_count, v_avg_sentence_words",
    "AoPS forum markup habit (hide blocks, boxed answers, quote blocks)":
        "ALREADY IN V -- v_hide_block, v_boxed, v_quote_block_count, v_list_marker_count",
    "how canonical the problem's approach is (a between-PROBLEM driver)":
        "NOT neutralised by the y-definition on this cell -- unlike the math.SE cells, y "
        "is a match against an external editorial solution, not a within-group split, so "
        "problem-level drivers CAN move the pooled tier. This is exactly why the "
        "within-problem tier is reported beside every pooled number.",
    "contest era / notation drift":
        "NOT in any bank; observed as post_year / contest_year / years_after_contest",
}


V_ALREADY_ARTICULATED_SURFACE = [
    "v_log_len", "v_word_count", "v_sentence_count", "v_avg_sentence_words",
    "v_n_display_math", "v_latex_density", "v_imath_tag_count", "v_numeral_density",
    "v_linebreak_count", "v_list_marker_count", "v_type_token_ratio",
    "v_uppercase_letter_ratio", "v_alpha_share", "v_quote_block_count",
    "v_hide_block", "v_boxed",
]


def norm_name(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _load_desc():
    desc, origin = {}, {}
    for line in open(RUBRICS):
        if not line.strip():
            continue
        r = json.loads(line)
        desc[r["name"]] = r.get("description", "")
        origin[r["name"]] = r.get("origin", [])
    return desc, origin


def _clusters(M, names, thresh=0.98):
    R = np.corrcoef(M.T)
    par = list(range(len(names)))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(R[i, j]) >= thresh:
                par[find(i)] = find(j)
    return len({find(i) for i in range(len(names))}), R


def cmd_stage1(_):
    sk = C.sklearn_guard()
    d = C.load()
    sp = json.loads((HERE / "aops_curation_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    fit = split == "fit_mine"
    y, A, names = d["y"], d["A"], d["a_names"]
    V, v_names = d["V"], d["v_names"]

    desc, origin = _load_desc()
    matched = sum(1 for n in names if n in desc)

    keep, med = L.clean_fit(A[fit])
    Af = L.clean_apply(A, keep, med)
    kept_names = [names[j] for j in keep]
    l3, R = _clusters(Af[fit], kept_names)

    per = []
    for k, nm in enumerate(kept_names):
        j = names.index(nm)
        col = A[:, j]
        per.append({"name": nm, "alone_AUC_FITMINE": L.auc(y[fit], Af[fit, k]),
                    "applicability_rate": float(np.isfinite(col).mean()),
                    "modal_share": float(np.nanmax(np.unique(col[np.isfinite(col)],
                                                             return_counts=True)[1])
                                         / max(1, np.isfinite(col).sum())),
                    "mean": float(np.nanmean(col)), "std": float(np.nanstd(col))})
    per.sort(key=lambda r: -abs(r["alone_AUC_FITMINE"] - .5))

    # V census (this cell's V is a NAMED surface bank; see module docstring)
    vkeep, vmed = L.clean_fit(V[fit])
    Vf = L.clean_apply(V, vkeep, vmed)
    v_kept = [v_names[j] for j in vkeep]
    v_l3, _ = _clusters(Vf[fit], v_kept)
    v_per = sorted(
        [{"name": nm, "alone_AUC_FITMINE": L.auc(y[fit], Vf[fit, k])}
         for k, nm in enumerate(v_kept)],
        key=lambda r: -abs(r["alone_AUC_FITMINE"] - .5))

    # shortlist pairs for the judges, in-register (rubric text vs rubric text)
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = [f"{n}. {desc.get(n, '')}" for n in kept_names]
    Tf = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(docs)
    S = (Tf @ Tf.T).toarray()
    np.fill_diagonal(S, 0.0)
    pairs = [(i, j, float(S[i, j])) for i in range(len(kept_names))
             for j in range(i + 1, len(kept_names)) if S[i, j] >= TAU_SHORTLIST]
    flat = sorted(((float(S[i, j]), i, j) for i in range(len(kept_names))
                   for j in range(i + 1, len(kept_names))), reverse=True)[:40]
    have = {(i, j) for i, j, _ in pairs}
    for s, i, j in flat:
        if (i, j) not in have:
            pairs.append((i, j, s))
    # the build note records five deliberate SPLITS of old axes -- always adjudicate
    # a split's two halves against each other, whatever the cosine says
    for i in range(len(kept_names)):
        for j in range(i + 1, len(kept_names)):
            oi, oj = origin.get(kept_names[i], []), origin.get(kept_names[j], [])
            if set(oi) & set(oj) and (i, j) not in {(a, b) for a, b, _ in pairs}:
                pairs.append((i, j, float(S[i, j])))

    ANCHOR_SAME = [
        ("The route is announced before the algebra",
         "Whether the post says what method it is about to use before it starts computing.",
         "Plan stated up front",
         "Whether the reader is told the intended line of attack before the technical work "
         "begins."),
        ("Notation is introduced before it is used",
         "Whether every symbol the post uses is defined at or before its first appearance.",
         "Symbols are declared, not assumed",
         "Whether the reader is told what each letter stands for before it appears in an "
         "expression."),
    ]
    ANCHOR_DIFF = [
        ("The key obstacle is named",
         "Whether the post says what makes the problem hard and what its central move "
         "exists to defeat.",
         "Amount of displayed LaTeX",
         "How many display-math blocks the post contains."),
        ("Hypotheses are checked, not assumed",
         "Whether the post verifies that the conditions a cited theorem needs actually "
         "hold here.",
         "Length of the post",
         "How many words the post runs to."),
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

    out = {"cell": "aops_curation", "sklearn": sk,
           "L0_criteria_delivered": len(names),
           "L1_distinct_names": len({norm_name(n) for n in names}),
           "L2_after_degeneracy_screen": len(kept_names),
           "L2_dropped": [n for n in names if n not in kept_names],
           "L3_value_clusters_r98": l3,
           "max_abs_column_r": float(np.nanmax(np.abs(R - np.eye(len(kept_names))))),
           "frac_pairs_r_ge_90": float((np.abs(R - np.eye(len(kept_names))) >= .90).sum()
                                       / (len(kept_names) * (len(kept_names) - 1))),
           "rubric_descriptions_matched": matched,
           "gepa_phrased": True,
           "per_column": per,
           "alone_AUC_summary": {
               "max": max(p["alone_AUC_FITMINE"] for p in per),
               "min": min(p["alone_AUC_FITMINE"] for p in per),
               "median": float(np.median([p["alone_AUC_FITMINE"] for p in per])),
               "MAD_from_chance": float(np.median([abs(p["alone_AUC_FITMINE"] - .5)
                                                   for p in per])),
               "n_ge_55": sum(1 for p in per if p["alone_AUC_FITMINE"] >= .55),
               "n_le_45": sum(1 for p in per if p["alone_AUC_FITMINE"] <= .45)},
           "V_census": {
               "L0_features": len(v_names),
               "L2_after_degeneracy_screen": len(v_kept),
               "L3_value_clusters_r98": v_l3,
               "per_column": v_per,
               "already_articulated_surface": V_ALREADY_ARTICULATED_SURFACE},
           "upstream_priors_where_they_already_live": UPSTREAM_PRIORS,
           "n_pairs_shortlisted": len(pairs),
           "register_statement": (
               "The A bank is 44 criteria written in OLYMPIAD WRITE-UP language (plan "
               "before machinery, one line of attack carried to an endpoint, decisive "
               "reason extractable in a sentence, symbols defined before use, care spent "
               "where the difficulty is), GEPA-revised and Gemma-4-31B judged on the "
               "bank's 0.0/0.5/1.0 + NA scale. The corpus is AoPS forum SOLUTIONS shown "
               "with their competition problem, and the outcome is whether the solution "
               "takes the SAME APPROACH as the editorial/wiki write-up. Three register "
               "mismatches are recorded up front: (i) the bank is written about WRITE-UP "
               "QUALITY while y is APPROACH IDENTITY against an unseen reference, so a "
               "beautifully written solution by a different route is a negative -- this "
               "is the cell's whole point; (ii) the judge never sees the editorial, by "
               "design, so no criterion can be about resemblance to it; (iii) an "
               "applicability gate fires on 22.8% of cells, mostly on short solutions "
               "that give a criterion nothing to bite on. No cosine threshold decides "
               "identity anywhere in this campaign."),
           }
    (HERE / "census_stage1.json").write_text(json.dumps(out, indent=1, default=float))
    (HERE / "census_blind_packet.json").write_text(json.dumps(packet, indent=1))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("per_column", "V_census")}, indent=1, default=float))
    print(f"\npacket: {len(packet['items'])} real pairs + {len(packet['anchors'])} anchors")
    print("\nTOP alone-AUC, A bank (FIT+MINE only):")
    for p in per[:10]:
        print(f"  {p['alone_AUC_FITMINE']:.3f}  appl={p['applicability_rate']:.2f}  {p['name'][:60]}")
    print("\nTOP alone-AUC, V bank (FIT+MINE only):")
    for p in v_per[:10]:
        print(f"  {p['alone_AUC_FITMINE']:.3f}  {p['name']}")


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
    name_of = {it["pair_id"]: (it["X_name"], it["Y_name"]) for it in packet["items"]}
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
        return len({find(i) for i in range(len(kept))})

    strict_edges = [name_of[i] for i in ids if all(m.get(i) == "SAME" for m in maps)]
    loose_edges = [name_of[i] for i in ids if any(m.get(i) == "SAME" for m in maps)]
    l5, l5p = union(strict_edges), union(loose_edges)

    out = dict(s1)
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
        "collapse_pct_L0_to_L5": 1 - l5 / s1["L0_criteria_delivered"],
    })
    (HERE / "census.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("per_column", "V_census")}, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stage1")
    f = sub.add_parser("finish")
    f.add_argument("--verdicts", required=True)
    a = ap.parse_args()
    {"stage1": cmd_stage1, "finish": cmd_finish}[a.cmd](a)
