#!/usr/bin/env python3
"""BLIND PAIRWISE SPECIES MERGE for a round's proposal pool, then re-selection.

WHY THIS EXISTS, recorded rather than silent.  `species.py` clusters a round's proposals
by bge-large cosine at tau = .79 and then selects the top k species by cross-proposer
support.  Its own docstring justifies the embedding shortcut on the grounds that every
proposal comes from one register.  On mathse_vote round 1 that shortcut demonstrably
UNDER-MERGED the Track-B pool: four proposers across two families independently named the
answer-arrival-order fingerprint under four different names, the embedding put them in
FOUR singleton species, the consensus-first selection rule saw four coin-flips instead of
one four-proposer species, and the channel missed the scored set.

The FREEZE DECLARATION's identity rule is explicit and is not the embedding: "concept
identity by full-recall blind pairwise (NEVER embedding-tau across registers)".  This
script applies that rule: the cosine is used ONLY to shortlist candidate pairs, and
identity is decided by sealed blind judges, exactly as the round-0 concept census does.
Selection then re-runs unchanged on the merged species.

DEVIATION FROM THE mathse_vote REFERENCE, recorded.  There the merge ran on Track B only.
Here it runs on BOTH TRACKS (`--tracks A,B`, the default), because this campaign reports
missing mass on both sides and the identity rule that makes the B-side Good-Turing the
figure of record makes the A-side one the figure of record too.  The tau-only species
table, selection and Good-Turing block are preserved beside the merged ones under
`*_PREMERGE_tau_only` keys for every track touched, so nothing is destroyed.

STRICT = a pair is merged only when EVERY judge says SAME (`all(...)` below).  Running
`apply` with one verdict file gives the single-judge variant; the figure of record is the
two-judge strict merge.

  build   -> <tag>_bmerge_packet.json  (blind judge packet, + planted anchors)
  apply   -> rewrites <tag>_species.json's species and `selected`
             (the pre-merge file is kept as <tag>_species.PREMERGE.json)

CPU only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "robust_mm"))
import embed_lib as E  # noqa: E402

SHORTLIST_COS = 0.55


def _h(s):
    return hashlib.sha256(s.encode()).hexdigest()


# Planted identity anchors, corpus-matched to a joke-sharing forum (the freeze's
# anchor requirement: every blind judging batch carries known-label items).
ANCHOR_SAME = [
    ("Presence of a trailing apology or disclaimer",
     "Score 0-10 for how fully the text ends with an apology, a disclaimer, an edit note "
     "or a plea to the reader. Extent only.",
     "Closing self-deprecating note to the reader",
     "Score 0-10 for whether the writer signs off by apologising for the joke or hedging "
     "about it. Extent only; do not judge quality."),
]
ANCHOR_DIFF = [
    ("How much capitalised shouting the text contains",
     "Score 0-10 for the density of all-caps words and repeated exclamation marks. Count "
     "only; do not judge whether it is effective.",
     "How far the ending reinterprets the setup",
     "Score 0-10 for whether the final move makes the reader re-read the setup with a "
     "second meaning. Judge the mechanism, not the typography."),
]


def cmd_build(a):
    tag = f"{a.cell}_r{a.round}"
    pool = json.loads((HERE / f"{tag}_proposals_fleet.json").read_text())["proposals"]
    packet = {"tag": tag, "shortlist_cos": SHORTLIST_COS, "items": [], "anchors": [],
              "tracks": a.tracks.split(",")}
    for track in a.tracks.split(","):
        P = [p for p in pool if p["track"] == track]
        if not P:
            continue
        texts = [f"{p['name']}. {p['instruction']}" for p in P]
        V = E.embed(texts, verbose=False)
        S = V @ V.T
        np.fill_diagonal(S, 0.0)
        pairs = []
        for i in range(len(P)):
            for j in range(i + 1, len(P)):
                if P[i]["proposer"] == P[j]["proposer"]:
                    continue          # never merge one proposer's own two proposals
                if S[i, j] >= SHORTLIST_COS:
                    pairs.append((i, j, float(S[i, j])))
        pairs.sort(key=lambda t: -t[2])
        pairs = pairs[:a.max_pairs]
        for k, (i, j, s) in enumerate(pairs):
            packet["items"].append({
                "pair_id": f"{track}Q{k+1:03d}", "track": track, "cos": s,
                "X_name": P[i]["name"], "X_desc": P[i]["instruction"],
                "Y_name": P[j]["name"], "Y_desc": P[j]["instruction"],
                "_i": i, "_j": j})
    for k, (xn, xd, yn, yd) in enumerate(ANCHOR_SAME):
        packet["anchors"].append({"pair_id": f"AS{k+1}", "truth": "SAME",
                                  "X_name": xn, "X_desc": xd, "Y_name": yn, "Y_desc": yd})
    for k, (xn, xd, yn, yd) in enumerate(ANCHOR_DIFF):
        packet["anchors"].append({"pair_id": f"AD{k+1}", "truth": "DIFFERENT",
                                  "X_name": xn, "X_desc": xd, "Y_name": yn, "Y_desc": yd})
    blind = {"items": [{k: v for k, v in it.items()
                        if not k.startswith("_") and k != "track"}
                       for it in packet["items"]],
             "anchors": packet["anchors"]}
    (HERE / f"{tag}_bmerge_key.json").write_text(json.dumps(packet, indent=1))
    (HERE / f"{tag}_bmerge_packet.json").write_text(json.dumps(blind, indent=1))
    print(f"{tag}: {len(packet['items'])} cross-proposer pairs shortlisted "
          f"(cos >= {SHORTLIST_COS}) over tracks {a.tracks} + "
          f"{len(packet['anchors'])} anchors -> {tag}_bmerge_packet.json")


def _merge_track(track, P, key_items, maps):
    par = list(range(len(P)))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    edges = 0
    for it in key_items:
        if it.get("track", "B") != track:
            continue
        if all(m.get(it["pair_id"]) == "SAME" for m in maps):
            a_, b_ = find(it["_i"]), find(it["_j"])
            if a_ != b_:
                par[a_] = b_
                edges += 1
    groups = {}
    for i in range(len(P)):
        groups.setdefault(find(i), []).append(i)
    rows = []
    for root, idx in groups.items():
        props = sorted({P[i]["proposer"] for i in idx})
        fams = sorted({P[i]["family"] for i in idx})
        rep = min(idx, key=lambda i: _h(P[i]["pid"]))
        rows.append({"n_members": len(idx), "n_proposers": len(props),
                     "n_families": len(fams), "proposers": props, "families": fams,
                     "rep": rep, "members": idx, "sort_hash": _h(P[rep]["pid"]),
                     "rep_name": P[rep]["name"]})
    rows.sort(key=lambda r: (-r["n_proposers"], -r["n_members"], r["sort_hash"]))
    return rows, edges


def cmd_apply(a):
    tag = f"{a.cell}_r{a.round}"
    key = json.loads((HERE / f"{tag}_bmerge_key.json").read_text())
    sp = json.loads((HERE / f"{tag}_species.json").read_text())
    pool = json.loads((HERE / f"{tag}_proposals_fleet.json").read_text())["proposals"]
    maps = [{v["pair_id"]: v["verdict"].upper()
             for v in json.loads(Path(p).read_text())["verdicts"]}
            for p in a.verdicts.split(",")]

    anchors = []
    for an in key["anchors"]:
        got = [m.get(an["pair_id"]) for m in maps]
        anchors.append({"pair_id": an["pair_id"], "truth": an["truth"], "got": got,
                        "pass": [g == an["truth"] for g in got]})

    (HERE / f"{tag}_species.PREMERGE.json").write_text(json.dumps(sp, indent=1))
    tracks = [t for t in a.tracks.split(",") if t in sp["tracks"]]
    newsel, summary = [], {}
    for track in ("A", "B"):
        if track not in sp["tracks"]:
            continue
        if track not in tracks:
            newsel += [c for c in sp["selected"] if c["track"] == track]
            continue
        P = [p for p in pool if p["track"] == track]
        rows, edges = _merge_track(track, P, key["items"], maps)
        k_t = sum(1 for c in sp["selected"] if c["track"] == track)
        chosen = rows[:k_t]
        for n, r in enumerate(chosen):
            p = P[r["rep"]]
            rec = {"track": track, "blind_id": f"{track}{n+1:02d}", "name": p["name"],
                   "instruction": p["instruction"], "rationale": p.get("rationale", ""),
                   "pid": p["pid"], "proposer": p["proposer"], "family": p["family"],
                   "n_proposers_naming": r["n_proposers"], "n_members": r["n_members"],
                   "member_names": sorted({P[i]["name"] for i in r["members"]})}
            if track == "B":
                rec["upstream_parent"] = p.get("upstream_parent", "surface-only")
                rec["mixed_proposed"] = bool(p.get("mixed", False))
                rec["upstream_parent_all_members"] = sorted(
                    {str(P[i].get("upstream_parent")) for i in r["members"]})
                rec["mixed_any_member"] = bool(any(P[i].get("mixed") for i in r["members"]))
            newsel.append(rec)

        S_obs = len(rows)
        f1 = sum(1 for r in rows if r["n_members"] == 1)
        f2 = sum(1 for r in rows if r["n_members"] == 2)
        merged_gt = {"N_proposals": len(P), "S_obs": S_obs, "f1": f1, "f2": f2,
                     "good_turing_missing_mass": f1 / max(1, len(P)),
                     "cross_proposer_recapture":
                         sum(1 for r in rows if r["n_proposers"] >= 2) / max(1, S_obs),
                     "species_named_by_ge2_families":
                         sum(1 for r in rows if r["n_families"] >= 2),
                     "identity_rule": "blind pairwise, STRICT (all judges must say SAME)",
                     "n_judges": len(maps)}
        blk = sp["tracks"][track]
        blk["species_table_PREMERGE_tau_only"] = blk["species_table"]
        blk["species_table"] = [{k: v for k, v in r.items()
                                 if k not in ("members", "rep")} for r in rows]
        blk["n_species_PREMERGE_tau_only"] = blk["n_species"]
        blk["n_species"] = S_obs
        blk["good_turing_PREMERGE_tau_only"] = blk["good_turing"]
        blk["good_turing"] = merged_gt
        summary[track] = {"merge_edges": edges,
                          "species_before": blk["n_species_PREMERGE_tau_only"],
                          "species_after": S_obs, "merged_good_turing": merged_gt}

    sp["selected"] = newsel
    sp["b_merge"] = {
        "rule": "FREEZE DECLARATION identity rule -- concept identity by full-recall blind "
                "pairwise adjudication, never embedding-tau. The cosine only shortlisted "
                "candidate pairs (cross-proposer pairs at cos >= %.2f)." % SHORTLIST_COS,
        "tracks_merged": tracks,
        "judges": [json.loads(Path(p).read_text()).get("judge", "sealed-blind")
                   for p in a.verdicts.split(",")],
        "strict": True,
        "n_pairs_adjudicated": len(key["items"]),
        "anchor_battery": anchors,
        "anchor_all_pass": all(all(x["pass"]) for x in anchors),
        "per_track": summary,
    }
    (HERE / f"{tag}_species.json").write_text(json.dumps(sp, indent=1))
    print(json.dumps({"per_track": summary, "anchors": anchors}, indent=1))
    print("\nselected after merge:")
    for c in newsel:
        print(f"  {c['blind_id']} P={c['n_proposers_naming']} m={c['n_members']} | "
              f"{c['name'][:60]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--cell", required=True); b.add_argument("--round", required=True)
    b.add_argument("--tracks", default="A,B")
    b.add_argument("--max-pairs", type=int, default=120)
    p = sub.add_parser("apply")
    p.add_argument("--cell", required=True); p.add_argument("--round", required=True)
    p.add_argument("--tracks", default="A,B")
    p.add_argument("--verdicts", required=True)
    a = ap.parse_args()
    {"build": cmd_build, "apply": cmd_apply}[a.cmd](a)
