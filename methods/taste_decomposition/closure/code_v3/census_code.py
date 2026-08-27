#!/usr/bin/env python3
"""ROUND-0 CONCEPT CENSUS of the incoming code_v3 A bank (freeze: "concept census of
the incoming bank at round 0").

Levels, cheapest decisive test first, identity NEVER decided by cosine:

  L0  rubrics delivered
  L1  distinct normalised names (exact)
  L2  score columns surviving the frozen degeneracy screen, FIT ON FIT+MINE ONLY
  L3  value clusters after collapsing |Pearson r| >= .98 columns
  L4  embedding SHORTLIST of candidate duplicate pairs (bge-large, tau .79) -- the
      register is uniform (every text is a code-review aspect rubric from one catalog),
      so an in-register cosine shortlist is legitimate; it only decides WHAT GETS READ.
  L5  effective concepts after BLIND PAIRWISE ADJUDICATION by two sealed judges
      (strict rule = both judges SAME), plus an authored anchor battery.

Stage 1 (this file, `shortlist`) emits the blind adjudication packet.
Stage 2 (`finalize`) reads the judges' verdicts back and writes the census.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "maps_hw_si"))
sys.path.insert(0, str(HERE.parent / "robust_mm"))

import cells_code as C                                       # noqa: E402
import closure_core as CC                                    # noqa: E402

AB = HERE / "abank_rescore"
TAU = 0.79


def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def criteria():
    rows = [json.loads(l) for l in open(AB / "criteria_code_abank.jsonl")]
    return {r["aspect_id"]: r for r in rows if r.get("portable")}


def shortlist():
    import embed_lib as E
    d = C.load()
    z = np.load(HERE / "splits.npz", allow_pickle=True)
    fitmask = z["fitmask"]
    crit = criteria()
    a_ids, a_names = d["a_ids"], d["a_names"]
    assert set(a_ids) <= set(crit), "scored aspect not in the criteria file"

    out = {"L0_delivered": len(a_ids)}

    # L1 -- distinct normalised names
    names = [_norm(n) for n in a_names]
    out["L1_distinct_names"] = len(set(names))
    out["L1_exact_name_dupes"] = [n for n in set(names) if names.count(n) > 1]

    # L2 -- frozen degeneracy screen on the 83 SCORE columns, fit on FIT+MINE only
    A = d["A"]
    keep, meds = CC.clean_fit(A[fitmask])
    out["L2_screen_survivors"] = int(len(keep))
    out["L2_dropped"] = [{"aspect_id": a_ids[j], "name": a_names[j],
                          "na_rate_fitmine": float(np.isnan(A[fitmask][:, j]).mean())}
                         for j in range(A.shape[1]) if j not in set(keep.tolist())]
    # the applied-indicator half of the A matrix, screened the same way
    IND = (~np.isnan(A)).astype(float)
    keep_i, _ = CC.clean_fit(IND[fitmask])
    out["L2_applied_indicator_survivors"] = int(len(keep_i))

    # L3 -- value clusters at |Pearson r| >= .98 on the surviving score columns
    Ak = CC.clean_apply(A, keep, meds)
    R = np.corrcoef(Ak[fitmask].T)
    np.fill_diagonal(R, 0.0)
    lab = CC._sl(np.abs(R), 0.98) if hasattr(CC, "_sl") else _single_linkage(np.abs(R), 0.98)
    out["L3_value_clusters"] = int(len(set(lab.tolist())))
    out["L3_max_abs_r"] = float(np.abs(R).max())
    out["L3_frac_pairs_ge_.90"] = float((np.abs(np.triu(R, 1)) >= .90).sum()
                                        / (len(keep) * (len(keep) - 1) / 2))

    # L4 -- in-register embedding shortlist over ALL delivered rubrics
    texts = [f"{crit[a]['name']}: {crit[a]['description']}" for a in a_ids]
    V = E.embed(texts, verbose=True)
    S = V @ V.T
    np.fill_diagonal(S, 0.0)
    iu = np.triu_indices(len(a_ids), 1)
    pairs = [(int(i), int(j), float(S[i, j])) for i, j in zip(*iu) if S[i, j] >= TAU]
    pairs.sort(key=lambda t: -t[2])
    out["L4_tau"] = TAU
    out["L4_max_offdiag_cosine"] = float(S.max())
    out["L4_shortlisted_pairs"] = len(pairs)

    packet = []
    for k, (i, j, c) in enumerate(pairs):
        packet.append({"pair_id": f"P{k:03d}", "cosine": round(c, 4),
                       "a": {"name": crit[a_ids[i]]["name"],
                             "definition": crit[a_ids[i]]["description"]},
                       "b": {"name": crit[a_ids[j]]["name"],
                             "definition": crit[a_ids[j]]["description"]},
                       "_key": [a_ids[i], a_ids[j]]})
    # authored anchor battery: 2 SAME (paraphrase pairs authored here) + 2 DIFFERENT
    anchors = [
        {"pair_id": "ANC_SAME_1", "truth": "SAME",
         "a": {"name": "Tests accompany the change",
               "definition": "The pull request adds or updates automated tests covering the behaviour it changes."},
         "b": {"name": "Change ships with test coverage",
               "definition": "New or modified automated tests are included alongside the code change, covering the modified behaviour."}},
        {"pair_id": "ANC_SAME_2", "truth": "SAME",
         "a": {"name": "Clear rationale in the description",
               "definition": "The change description explains why the change is being made, not only what it does."},
         "b": {"name": "Motivation stated for the change",
               "definition": "The submitted description gives the reason and motivation for the change rather than restating the diff."}},
        {"pair_id": "ANC_DIFF_1", "truth": "DIFFERENT",
         "a": {"name": "Error handling robustness",
               "definition": "Failure paths are handled explicitly and errors are surfaced with enough context to act on."},
         "b": {"name": "Error message spelling and grammar",
               "definition": "Strings emitted on failure are free of typos and grammatical mistakes."}},
        {"pair_id": "ANC_DIFF_2", "truth": "DIFFERENT",
         "a": {"name": "Small, focused, reviewable change",
               "definition": "The change is scoped to one concern and is small enough to review carefully."},
         "b": {"name": "Commit message structure",
               "definition": "Commit messages follow a conventional subject/body structure with an imperative subject line."}},
    ]
    pairs_key = {p["pair_id"]: p["_key"] for p in packet}
    blind = ([{k: v for k, v in p.items() if k != "_key"} for p in packet]
             + [{k: v for k, v in a.items() if k != "truth"} for a in anchors])
    blind.sort(key=lambda b: hashlib.sha256(b["pair_id"].encode()).hexdigest())

    (HERE / "census_stage1.json").write_text(json.dumps(
        {"levels": out, "pairs_key": pairs_key,
         "anchor_key": {a["pair_id"]: a["truth"] for a in anchors}}, indent=1))
    (HERE / "census_blind_packet.json").write_text(json.dumps(blind, indent=1))
    print(json.dumps(out, indent=1))
    print(f"\nblind packet: {len(blind)} pairs ({len(packet)} real + {len(anchors)} anchors)"
          f" -> census_blind_packet.json")


def _single_linkage(S, tau):
    n = S.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] >= tau:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    return np.array([find(i) for i in range(n)])


def finalize(verdict_files):
    st = json.loads((HERE / "census_stage1.json").read_text())
    key, akey = st["pairs_key"], st["anchor_key"]
    d = C.load()
    a_ids = d["a_ids"]
    verds = [json.loads(Path(f).read_text()) for f in verdict_files]
    verds = [{v["pair_id"]: v["verdict"].upper() for v in vv} for vv in verds]

    anchors = []
    for pid, truth in akey.items():
        got = [vv.get(pid) for vv in verds]
        anchors.append({"pair_id": pid, "truth": truth, "judges": got,
                        "correct": [g == truth for g in got]})
    agree = np.mean([verds[0].get(p) == verds[1].get(p) for p in key])

    merge_strict = [key[p] for p in key if all(vv.get(p) == "SAME" for vv in verds)]
    merge_loose = [key[p] for p in key if any(vv.get(p) == "SAME" for vv in verds)]

    def collapse(edges):
        parent = {a: a for a in a_ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for u, v in edges:
            a, b = find(u), find(v)
            if a != b:
                parent[a] = b
        return len({find(a) for a in a_ids}), {a: find(a) for a in a_ids}

    n_strict, mapping = collapse(merge_strict)
    n_loose, _ = collapse(merge_loose)
    out = dict(st["levels"])
    out.update({
        "L5_effective_concepts_strict": n_strict,
        "L5p_effective_concepts_loose": n_loose,
        "merge_edges_strict": len(merge_strict), "merge_edges_loose": len(merge_loose),
        "judge_raw_agreement": float(agree),
        "anchor_battery": anchors,
        "anchor_pass": [int(sum(a["correct"][k] for a in anchors)) for k in range(len(verds))],
        "n_anchors": len(anchors),
        "merged_pairs_strict": [[u, v] for u, v in merge_strict],
        "concept_map": mapping,
    })
    (HERE / "census_code.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("concept_map", "L2_dropped")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["shortlist", "finalize"])
    ap.add_argument("--verdicts", nargs="*", default=[])
    a = ap.parse_args()
    if a.stage == "shortlist":
        shortlist()
    else:
        finalize(a.verdicts)
