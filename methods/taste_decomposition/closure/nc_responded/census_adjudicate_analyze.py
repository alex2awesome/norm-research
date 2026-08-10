#!/usr/bin/env python3
"""Fold the two sealed judges' verdicts into the round-0 concept census.

Primary rule (pilot's): judges must AGREE; disagreement resolves to the STRICT
side (DIFFERENT), so the effective-concept count is an UPPER bound on distinctness
and the duplication estimate is conservative.
Anchor battery reported separately as the instrument check.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(p):
    d = json.loads((HERE / p).read_text())
    return {v["pair_id"]: v for v in d["verdicts"]}


def main():
    key = {k["shown_id"]: k for k in json.loads((HERE / "census_adjudication_key.json").read_text())["key"]}
    j1, j2 = load("census_adjudication_judge1.json"), load("census_adjudication_judge2.json")
    census = json.loads((HERE / "concept_census.json").read_text())
    pairkey = {p["pair_id"]: p for p in json.loads((HERE / "census_pairs_key.json").read_text())["pairs"]}

    rows, anchors = [], []
    agree = 0
    for sid, meta in key.items():
        v1, v2 = j1.get(sid), j2.get(sid)
        if v1 is None or v2 is None:
            continue
        same = (v1["verdict"] == "SAME" and v2["verdict"] == "SAME")
        agree += int(v1["verdict"] == v2["verdict"])
        rec = {"shown_id": sid, "orig_id": meta["orig_id"], "kind": meta["kind"],
               "j1": v1["verdict"], "j2": v2["verdict"], "primary": "SAME" if same else "DIFFERENT",
               "j1_conf": v1.get("confidence"), "j2_conf": v2.get("confidence")}
        if meta["kind"] == "anchor":
            rec["anchor_label"] = meta["anchor_label"]
            rec["j1_correct"] = v1["verdict"] == meta["anchor_label"]
            rec["j2_correct"] = v2["verdict"] == meta["anchor_label"]
            anchors.append(rec)
        else:
            rec["cos"] = pairkey[meta["orig_id"]]["cos"]
            rec["i"] = pairkey[meta["orig_id"]]["i"]
            rec["j"] = pairkey[meta["orig_id"]]["j"]
            rows.append(rec)

    # union-find over the L2-surviving columns using the PRIMARY (strict) SAME edges
    m = census["L2_surviving_columns"]
    parent = list(range(m))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    merged = []
    for r in rows:
        if r["primary"] == "SAME":
            ra, rb = find(r["i"]), find(r["j"])
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
            merged.append(r)
    clus = defaultdict(list)
    for i in range(m):
        clus[find(i)].append(i)

    # a looser variant for the sensitivity range: EITHER judge says SAME
    parent2 = list(range(m))

    def find2(a):
        while parent2[a] != a:
            parent2[a] = parent2[parent2[a]]
            a = parent2[a]
        return a

    for r in rows:
        if r["j1"] == "SAME" or r["j2"] == "SAME":
            ra, rb = find2(r["i"]), find2(r["j"])
            if ra != rb:
                parent2[max(ra, rb)] = min(ra, rb)
    clus2 = defaultdict(list)
    for i in range(m):
        clus2[find2(i)].append(i)

    alone = census["alone_auc_fitmine"]
    names = list(alone.keys())
    out = {
        "n_real_pairs": len(rows),
        "n_anchors": len(anchors),
        "raw_agreement": agree / max(1, len(rows) + len(anchors)),
        "anchor_battery": {
            "detail": anchors,
            "j1_score": f"{sum(a['j1_correct'] for a in anchors)}/{len(anchors)}",
            "j2_score": f"{sum(a['j2_correct'] for a in anchors)}/{len(anchors)}",
            "note": "SAME anchors are AUTHORED paraphrase pairs (missing-mass PART-4 fix 5), "
                    "DIFFERENT anchors are lexical look-alikes with genuinely different targets.",
        },
        "L5_effective_concepts_strict": len(clus),
        "L5_effective_concepts_loose": len(clus2),
        "n_merge_edges_strict": sum(r["primary"] == "SAME" for r in rows),
        "n_merge_edges_loose": sum(r["j1"] == "SAME" or r["j2"] == "SAME" for r in rows),
        "merged_clusters_strict": [[names[i] for i in v] for v in clus.values() if len(v) > 1],
        "pairs": rows,
        "ladder": {
            "L0_delivered": census["L0_delivered"],
            "L1_distinct_names": census["L1_distinct_names"],
            "L2_surviving_columns": census["L2_surviving_columns"],
            "L3_value_clusters_r98": census["L3_value_clusters"],
            "L5_effective_concepts_strict": len(clus),
            "L5_effective_concepts_loose": len(clus2),
        },
    }
    (HERE / "concept_census_final.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("pairs", "merged_clusters_strict", "anchor_battery")}, indent=1))
    print("anchors j1", out["anchor_battery"]["j1_score"], "j2", out["anchor_battery"]["j2_score"])
    for c in out["merged_clusters_strict"]:
        print("  MERGED:", " || ".join(c))


if __name__ == "__main__":
    main()
