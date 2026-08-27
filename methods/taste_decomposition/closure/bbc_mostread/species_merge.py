#!/usr/bin/env python3
"""BLIND PAIRWISE SPECIES MERGE for a round's proposal pool, then re-selection.

BOTH TRACKS on this campaign (coordinator brief 2026-08-09: "strict two-judge merge
BOTH tracks, run BEFORE the audit").  The sibling math.SE campaigns merged Track B
only and quoted Track A's mass unmerged at tau; that asymmetry meant the A-side
missing mass was inflated by exactly the f1 mechanism the B-side merge was invented
to remove, so the two tracks' mass figures were not comparable.  Here `--track`
selects the pool and every artifact is suffixed by it (`_bmerge{A,B}_*`), so both
sides get the freeze's identity rule and both mass figures are on one footing.

WHY THIS EXISTS, recorded rather than silent.  `species.py` clusters a round's
proposals by bge-large cosine at tau = .79 and then selects the top k species by
cross-proposer support.  Its own docstring justifies the embedding shortcut on the
grounds that every proposal comes from one register.  On the SIBLING math.SE VOTE cell's round 1 that
shortcut demonstrably UNDER-MERGED the Track-B pool (provenance: that campaign, not
this one -- recorded because a blanket rename would otherwise misattribute it): four proposers across two
families independently named the answer-arrival-order fingerprint --

    claude_opus  "Supplementary framing presupposing an already-populated ..."
    claude_sonnet "Presupposes or names sibling answers on this question"
    codex_luna_a "Answer-Stream Awareness"
    codex_luna_b "Reply-aware framing"

-- and all four tagged their own `upstream_parent` as position in the answer
arrival / entry stream.  The embedding put them in FOUR separate singleton species,
so the consensus-first selection rule saw four coin-flips instead of one
four-proposer species, and the channel missed the scored set.

The FREEZE DECLARATION's identity rule is explicit and is not the embedding:
"concept identity by full-recall blind pairwise (NEVER embedding-tau across
registers)".  This script applies that rule to the Track-B pool: the cosine is
used ONLY to shortlist candidate pairs, and identity is decided by sealed blind
judges, exactly as the round-0 concept census does.  Selection then re-runs
unchanged on the merged species.

  build --track {A,B} -> <tag>_bmerge<T>_packet.json (blind judge packet + 2 anchors)
  apply --track {A,B} -> by DEFAULT writes a NEW <tag>_species_strict<T>.json and leaves
                         <tag>_species.json untouched; pass --inplace for the legacy
                         behaviour (rewrite in place, keeping <tag>_species.PREMERGE.json)

  OUTPUT MODE (2026-08-11, certificate-backfill brief).  The original `apply` rewrote
  <tag>_species.json in place.  That is safe when a round is merged once, as part of its
  own campaign, but the Track-A backfill re-merges ARCHIVED rounds whose tau-era species
  files are already cited, so overwriting them would silently move published numbers.
  The PREMERGE sidecar does NOT protect against this -- it is written only once, by
  whichever track merges first, so a second merge pass overwrites species.json with no
  surviving copy of the tau-era state.  Default is therefore a new file.

  MISSING MASS ON THE MERGED SPECIES also carries the LOO-proposer jackknife, computed
  the same way species.py computes it for the tau-only table (drop one proposer, recount
  species over the survivors, Good-Turing f1/N on the reduced pool), so the strict and
  tau figures are read off the same estimator.

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


ANCHOR_SAME = [
    ("Presence of a trailing signature block",
     "Score 0-10 for how fully the text ends with a personal sign-off: a name, initials, "
     "a handle, or a closing salutation. Extent only.",
     "Author sign-off at the end of the text",
     "Score 0-10 for whether the writer closes by naming or initialling themselves. "
     "Extent only; do not judge quality.")]
ANCHOR_DIFF = [
    ("How much displayed mathematics the text contains",
     "Score 0-10 for the density of set-off equation blocks. Count only; do not judge "
     "whether the mathematics is correct.",
     "How much the text hedges its own claims",
     "Score 0-10 for the frequency of 'I think', 'perhaps', 'it seems' and similar. "
     "Count only; do not judge whether the hedging is warranted.")]


def cmd_build(a):
    tag = f"{a.cell}_r{a.round}"
    T = a.track
    sp = json.loads((HERE / f"{tag}_species.json").read_text())
    pool = json.loads((HERE / f"{tag}_proposals_fleet.json").read_text())["proposals"]
    B = [p for p in pool if p["track"] == T]
    texts = [f"{p['name']}. {p['instruction']}" for p in B]
    V = E.embed(texts, verbose=False)
    S = V @ V.T
    np.fill_diagonal(S, 0.0)

    pairs = []
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            if B[i]["proposer"] == B[j]["proposer"]:
                continue          # never merge one proposer's own two channels
            if S[i, j] >= SHORTLIST_COS:
                pairs.append((i, j, float(S[i, j])))
    pairs.sort(key=lambda t: -t[2])
    pairs = pairs[:a.max_pairs]

    packet = {"tag": tag, "shortlist_cos": SHORTLIST_COS, "items": [], "anchors": []}
    for k, (i, j, s) in enumerate(pairs):
        packet["items"].append({
            "pair_id": f"Q{k+1:03d}", "cos": s,
            "X_name": B[i]["name"], "X_desc": B[i]["instruction"],
            "Y_name": B[j]["name"], "Y_desc": B[j]["instruction"],
            "_i": i, "_j": j})
    for k, (xn, xd, yn, yd) in enumerate(ANCHOR_SAME):
        packet["anchors"].append({"pair_id": f"AS{k+1}", "truth": "SAME",
                                  "X_name": xn, "X_desc": xd, "Y_name": yn, "Y_desc": yd})
    for k, (xn, xd, yn, yd) in enumerate(ANCHOR_DIFF):
        packet["anchors"].append({"pair_id": f"AD{k+1}", "truth": "DIFFERENT",
                                  "X_name": xn, "X_desc": xd, "Y_name": yn, "Y_desc": yd})
    blind = {"items": [{k: v for k, v in it.items() if not k.startswith("_")}
                       for it in packet["items"]],
             "anchors": packet["anchors"]}
    (HERE / f"{tag}_bmerge{T}_key.json").write_text(json.dumps(packet, indent=1))
    (HERE / f"{tag}_bmerge{T}_packet.json").write_text(json.dumps(blind, indent=1))
    print(f"{tag}: {len(pairs)} Track-{T} pairs shortlisted (cos >= {SHORTLIST_COS}, "
          f"cross-proposer only) + {len(packet['anchors'])} anchors "
          f"-> {tag}_bmerge{T}_packet.json")


def cmd_apply(a):
    tag = f"{a.cell}_r{a.round}"
    T = a.track
    key = json.loads((HERE / f"{tag}_bmerge{T}_key.json").read_text())
    sp = json.loads((HERE / f"{tag}_species.json").read_text())
    pool = json.loads((HERE / f"{tag}_proposals_fleet.json").read_text())["proposals"]
    B = [p for p in pool if p["track"] == T]
    maps = [{v["pair_id"]: v["verdict"].upper()
             for v in json.loads(Path(p).read_text())["verdicts"]}
            for p in a.verdicts.split(",")]

    anchors = []
    for an in key["anchors"]:
        got = [m.get(an["pair_id"]) for m in maps]
        anchors.append({"pair_id": an["pair_id"], "truth": an["truth"], "got": got,
                        "pass": [g == an["truth"] for g in got]})

    par = list(range(len(B)))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    edges = 0
    for it in key["items"]:
        if all(m.get(it["pair_id"]) == "SAME" for m in maps):
            a_, b_ = find(it["_i"]), find(it["_j"])
            if a_ != b_:
                par[a_] = b_
                edges += 1

    groups = {}
    for i in range(len(B)):
        groups.setdefault(find(i), []).append(i)

    rows = []
    for root, idx in groups.items():
        props = sorted({B[i]["proposer"] for i in idx})
        fams = sorted({B[i]["family"] for i in idx})
        rep = min(idx, key=lambda i: _h(B[i]["pid"]))
        rows.append({"n_members": len(idx), "n_proposers": len(props),
                     "n_families": len(fams), "proposers": props, "families": fams,
                     "rep": rep, "members": idx,
                     "sort_hash": _h(B[rep]["pid"]),
                     "rep_name": B[rep]["name"]})
    rows.sort(key=lambda r: (-r["n_proposers"], -r["n_members"], r["sort_hash"]))

    k_b = sum(1 for c in sp["selected"] if c["track"] == T)
    chosen = rows[:k_b]
    newsel = [c for c in sp["selected"] if c["track"] != T]
    for n, r in enumerate(chosen):
        p = B[r["rep"]]
        newsel.append({"track": T, "blind_id": f"{T}{n+1:02d}", "name": p["name"],
                       "instruction": p["instruction"], "rationale": p.get("rationale", ""),
                       "upstream_parent": p.get("upstream_parent", "surface-only"),
                       "mixed": bool(p.get("mixed", False)),
                       "proposer": p["proposer"], "family": p["family"],
                       "n_proposers_naming": r["n_proposers"],
                       "n_members": r["n_members"],
                       "member_names": sorted({B[i]["name"] for i in r["members"]}),
                       "upstream_parent_all_members": sorted(
                           {str(B[i].get("upstream_parent")) for i in r["members"]}),
                       "mixed_any_member": bool(any(B[i].get("mixed") for i in r["members"]))})

    # Good-Turing on the MERGED species (the freeze's identity rule, so this is the
    # figure of record for the B side this round; the tau-only one is kept beside it)
    S_obs = len(rows)
    f1 = sum(1 for r in rows if r["n_members"] == 1)
    f2 = sum(1 for r in rows if r["n_members"] == 2)
    merged_gt = {"N_proposals": len(B), "S_obs": S_obs, "f1": f1, "f2": f2,
                 "good_turing_missing_mass": f1 / max(1, len(B)),
                 "cross_proposer_recapture": sum(1 for r in rows if r["n_proposers"] >= 2)
                 / max(1, S_obs),
                 "species_named_by_ge2_families": sum(1 for r in rows if r["n_families"] >= 2)}

    # LOO-proposer jackknife, same estimator species.py uses for the tau-only table:
    # drop one proposer, recount species over the survivors, Good-Turing f1/N.
    prop_ids = sorted({p["proposer"] for p in B})
    jack = []
    for drop in prop_ids:
        sizes = [sum(1 for i in r["members"] if B[i]["proposer"] != drop) for r in rows]
        sizes = [s for s in sizes if s > 0]
        n_keep = sum(sizes)
        if n_keep < 2:
            continue
        jack.append(sum(1 for s in sizes if s == 1) / n_keep)
    merged_gt["P"] = len(prop_ids)
    merged_gt["proposers"] = prop_ids
    merged_gt["jackknife_LOPO_missing_mass"] = {
        "values": [round(v, 4) for v in jack],
        "min": float(min(jack)) if jack else None,
        "max": float(max(jack)) if jack else None,
        "mean": float(np.mean(jack)) if jack else None,
    }

    if a.inplace:
        pre = HERE / f"{tag}_species.PREMERGE.json"
        if not pre.exists():                  # written once, by the first track merged
            pre.write_text(json.dumps(sp, indent=1))
    newsel.sort(key=lambda c: (c["track"], c["blind_id"]))
    sp["selected"] = newsel
    sp["tracks"][T]["species_table_PREMERGE_tau_only"] = sp["tracks"][T]["species_table"]
    sp["tracks"][T]["species_table"] = [
        {k: v for k, v in r.items() if k not in ("members", "rep")} for r in rows]
    sp["tracks"][T]["n_species_PREMERGE_tau_only"] = sp["tracks"][T]["n_species"]
    sp["tracks"][T]["n_species"] = S_obs
    sp["tracks"][T]["good_turing_PREMERGE_tau_only"] = sp["tracks"][T]["good_turing"]
    sp["tracks"][T]["good_turing"] = merged_gt
    sp.setdefault("blind_merge", {})[T] = {
        "rule": "FREEZE DECLARATION identity rule -- concept identity by full-recall blind "
                "pairwise adjudication, never embedding-tau. The cosine only shortlisted "
                "candidate pairs (cross-proposer pairs at cos >= %.2f)." % SHORTLIST_COS,
        "judges": [json.loads(Path(p).read_text()).get("judge", "sealed-blind")
                   for p in a.verdicts.split(",")],
        "n_pairs_adjudicated": len(key["items"]),
        "n_merge_edges_strict": edges,
        "anchor_battery": anchors,
        "anchor_all_pass": all(all(x["pass"]) for x in anchors),
        "n_species_before": sp["tracks"][T]["n_species_PREMERGE_tau_only"],
        "n_species_after": S_obs,
        "track": T,
    }
    if a.inplace:
        out_path = HERE / f"{tag}_species.json"
    else:
        out_path = HERE / f"{tag}_species_strict{T}.json"
        sp["strict_merge_output"] = {
            "source_species_file": f"{tag}_species.json",
            "source_left_untouched": True,
            "note": "Track-%s strict two-judge merge written as a NEW file so the cited "
                    "tau-era species file is not moved. The tau-only figures are preserved "
                    "here under the *_PREMERGE_tau_only keys." % T,
        }
    out_path.write_text(json.dumps(sp, indent=1))
    print(f"wrote {out_path.name} (inplace={a.inplace})")
    print(json.dumps({"track": T, "merge_edges": edges, "species_before":
                      sp["tracks"][T]["n_species_PREMERGE_tau_only"],
                      "species_after": S_obs, "anchors": anchors,
                      "merged_good_turing": merged_gt}, indent=1))
    print(f"\nselected Track-{T} after merge:")
    for c in newsel:
        if c["track"] == T:
            print(f"  {c['blind_id']} P={c['n_proposers_naming']} m={c['n_members']} | "
                  f"{c['name'][:60]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--cell", required=True); b.add_argument("--round", required=True)
    b.add_argument("--max-pairs", type=int, default=120)
    b.add_argument("--track", choices=["A", "B"], default="B")
    p = sub.add_parser("apply")
    p.add_argument("--cell", required=True); p.add_argument("--round", required=True)
    p.add_argument("--verdicts", required=True)
    p.add_argument("--track", choices=["A", "B"], default="B")
    p.add_argument("--inplace", action="store_true",
                   help="legacy behaviour: rewrite <tag>_species.json in place (keeping a "
                        "one-time PREMERGE sidecar). Default writes a NEW "
                        "<tag>_species_strict<T>.json and never touches the original.")
    a = ap.parse_args()
    {"build": cmd_build, "apply": cmd_apply}[a.cmd](a)
