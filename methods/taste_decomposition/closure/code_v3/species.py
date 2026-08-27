#!/usr/bin/env python3
"""Species clustering, per-round Good-Turing missing mass (both tracks), and the
scored-set selection rule for the map-focused batch.

WHY A SELECTION RULE EXISTS, stated explicitly because it is an operational
decision and not in the frozen text.  The freeze fixes k_A=15 / k_B=10 criteria
per round AND a sealed fleet of P proposers.  A fleet of P=4-6 emits 60-90 A
proposals and 40-60 B proposals per round; scoring all of them corpus-wide would
multiply the frozen judge budget by P.  So each round:

  1. the fleet's proposals are clustered into SPECIES (single linkage on
     bge-large cosine at tau=.79 -- the pilot's calibrated band, and legitimate
     here because every proposal comes from the same fleet reading the same
     slice, i.e. ONE register; the cross-register prohibition in
     notes/2026-08-06__missing-mass-robustification.md 2.3 does not apply);
  2. Good-Turing missing mass + leave-one-proposer-out jackknife are computed on
     the FULL species pool, per track (FREEZE ADDENDUM: B-side missing mass);
  3. the SCORED set is the top k species by cross-proposer support
     (n_distinct_proposers desc, then stable sha256 of the species
     representative), one representative per species.  Selection is label-blind
     and provenance-neutral (the representative is the member whose sha256 sorts
     first, so no family is favoured).

Consensus-first selection is the right bias for a MAP: a channel named
independently by several sealed proposers is the one whose alone-AUC most
deserves a corpus-wide measurement.  Singleton species enter once the
multi-proposer species are exhausted, so diversity is not lost.

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

TAU = E.TAU


def _h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def cluster(props):
    texts = [E.crit_text(p["name"], p["instruction"]) for p in props]
    V = E.embed(texts, verbose=False)
    S = V @ V.T
    lab = E.single_linkage(S, TAU)
    return lab, S


def gt_block(props, lab):
    """Good-Turing mass + jackknife over proposers for one track's pool."""
    n = len(props)
    sizes = np.bincount(lab)
    stats = E.chao1(sizes)
    prop_ids = sorted({p["proposer"] for p in props})
    # cross-proposer recapture: fraction of species named by >1 proposer
    spec_props = {}
    for p, l in zip(props, lab):
        spec_props.setdefault(int(l), set()).add(p["proposer"])
    multi = sum(1 for v in spec_props.values() if len(v) > 1)
    fams = {}
    for p, l in zip(props, lab):
        fams.setdefault(int(l), set()).add(p["family"])
    jack = []
    for drop in prop_ids:
        keep = [i for i, p in enumerate(props) if p["proposer"] != drop]
        if len(keep) < 2:
            continue
        s2 = np.bincount(lab[keep])
        s2 = s2[s2 > 0]
        jack.append(E.chao1(s2)["good_turing_missing_mass"])
    return {
        "N_proposals": int(n), "P": len(prop_ids), "proposers": prop_ids,
        "n_families": len({p["family"] for p in props}),
        "S_obs": stats["S_obs"], "f1": stats["f1"], "f2": stats["f2"],
        "good_turing_missing_mass": stats["good_turing_missing_mass"],
        "chao1_bias_corrected_NOT_QUOTED": stats["chao1_bias_corrected"],
        "cross_proposer_recapture": float(multi / max(1, stats["S_obs"])),
        "species_named_by_ge2_families": int(sum(1 for v in fams.values() if len(v) > 1)),
        "jackknife_LOPO_missing_mass": {
            "values": [round(v, 4) for v in jack],
            "min": float(min(jack)) if jack else None,
            "max": float(max(jack)) if jack else None,
            "mean": float(np.mean(jack)) if jack else None,
        },
        "tau": TAU,
    }


def select(props, lab, k):
    spec = {}
    for i, (p, l) in enumerate(zip(props, lab)):
        spec.setdefault(int(l), []).append(i)
    rows = []
    for l, idx in spec.items():
        proposers = {props[i]["proposer"] for i in idx}
        fams = {props[i]["family"] for i in idx}
        rep = sorted(idx, key=lambda i: _h(props[i]["pid"]))[0]
        rows.append({"species": l, "n_members": len(idx),
                     "n_proposers": len(proposers), "n_families": len(fams),
                     "rep": rep, "members": idx,
                     "sort_hash": _h(props[rep]["name"] + props[rep]["instruction"])})
    rows.sort(key=lambda r: (-r["n_proposers"], -r["n_members"], r["sort_hash"]))
    return rows[:k], rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--force", action="store_true", help="rebuild a merged/scored round (destructive)")
    ap.add_argument("--round", type=int, default=1)
    a = ap.parse_args()
    tag = f"{a.cell}_r{a.round}"

    # LANDMINE GUARD (2026-08-08, ported from mathse_vote after a near-miss there).
    # species.py is not idempotent once a round is merged/scored: re-running it rebuilds
    # the tau-only clustering and SILENTLY overwrites a species.json that audits, scores
    # and routing are keyed to. Refuse to clobber such a round unless --force is given.
    spf = HERE / f"{tag}_species.json"
    if spf.exists() and not getattr(a, "force", False):
        import json as _json
        prev = _json.loads(spf.read_text())
        reasons = []
        if "b_merge" in prev:
            reasons.append("it carries a blind pairwise merge (b_merge)")
        if (HERE / f"{tag}_scores.npz").exists():
            reasons.append("the round is already scored")
        if (HERE / f"{tag}_routing_final.json").exists():
            reasons.append("the round is already routed/audited")
        if reasons:
            raise SystemExit(
                f"REFUSING to overwrite {spf.name}: " + "; ".join(reasons) + ".\n"
                "Re-running species.py would rebuild the tau-only selection and break the "
                "id<->criterion mapping that the audit and the scores depend on.\n"
                "Pass --force only if you intend to rebuild the round from scratch.")
    d = json.loads((HERE / f"{tag}_proposals_fleet.json").read_text())
    props = d["proposals"]

    out = {"tag": tag, "cell": a.cell, "round": a.round, "tau": TAU,
           "selection_rule": "top-k species by (n_distinct_proposers, n_members, "
                             "stable sha256); representative = member with smallest "
                             "sha256(pid)",
           "tracks": {}, "selected": []}
    for track, k in (("A", d["k_A"]), ("B", d["k_B"])):
        sub = [p for p in props if p["track"] == track]
        if not sub:
            continue
        lab, _ = cluster(sub)
        gt = gt_block(sub, lab)
        top, allspec = select(sub, lab, k)
        out["tracks"][track] = {
            "good_turing": gt,
            "n_species": len(allspec),
            "species_table": [
                {"n_proposers": r["n_proposers"], "n_members": r["n_members"],
                 "n_families": r["n_families"], "rep_name": sub[r["rep"]]["name"],
                 "selected": r in top}
                for r in allspec],
        }
        for j, r in enumerate(top):
            p = sub[r["rep"]]
            rec = {"blind_id": f"{track}{j+1:02d}", "track": track,
                   "name": p["name"], "instruction": p["instruction"],
                   "rationale": p["rationale"], "pid": p["pid"],
                   "proposer": p["proposer"], "family": p["family"],
                   "n_proposers_naming": r["n_proposers"], "n_members": r["n_members"],
                   "member_names": sorted({sub[i]["name"] for i in r["members"]})}
            if track == "B":
                parents = sorted({sub[i].get("upstream_parent", "surface-only")
                                  for i in r["members"]})
                rec["upstream_parent"] = p.get("upstream_parent", "surface-only")
                rec["upstream_parent_all_members"] = parents
                rec["mixed_proposed"] = bool(p.get("mixed", False))
                rec["mixed_any_member"] = bool(any(sub[i].get("mixed") for i in r["members"]))
            out["selected"].append(rec)

    # ---- BOTH-TRACK BLIND MERGE BEFORE AUDIT (accumulated ruling) -----------
    # The per-track clustering above cannot see a concept that BOTH tracks named -- one
    # proposer calling it quality, another calling it nuisance. Scoring it twice under
    # two blind ids would double-count it in the bank and in the nuisance set, and the
    # audit would route the two copies independently and possibly inconsistently. So the
    # SELECTED sets from both tracks are re-embedded together, in one space, BEFORE the
    # audit; cross-track pairs at or above TAU are recorded, and the B-side copy is
    # dropped (the A side is kept because the audit can still re-route it to B, whereas a
    # dropped A copy could never be recovered).
    sel = out["selected"]
    if sel:
        import embed_lib as E2
        V = E2.embed([E2.crit_text(s["name"], s["instruction"]) for s in sel], verbose=False)
        S = V @ V.T
        cross, drop = [], set()
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                if sel[i]["track"] == sel[j]["track"]:
                    continue
                if S[i, j] >= TAU:
                    a, b = (i, j) if sel[i]["track"] == "A" else (j, i)
                    cross.append({"cosine": float(S[i, j]),
                                  "A_id": sel[a]["blind_id"], "A_name": sel[a]["name"],
                                  "B_id": sel[b]["blind_id"], "B_name": sel[b]["name"],
                                  "dropped": sel[b]["blind_id"]})
                    drop.add(sel[b]["blind_id"])
        out["cross_track_merge"] = {
            "tau": TAU, "n_cross_track_duplicates": len(cross), "pairs": cross,
            "dropped_ids": sorted(drop),
            "rule": "selected A and B sets re-embedded in ONE space before the audit; "
                    "cross-track pairs >= tau have the B copy dropped (A is kept because "
                    "the audit can still re-route it to B)"}
        if drop:
            out["selected"] = [s for s in sel if s["blind_id"] not in drop]
        print(f"[{tag}] cross-track blind merge: {len(cross)} duplicate(s), "
              f"dropped {sorted(drop)}")

    (HERE / f"{tag}_species.json").write_text(json.dumps(out, indent=1))
    for t, blk in out["tracks"].items():
        g = blk["good_turing"]
        print(f"[{tag}] track {t}: N={g['N_proposals']} P={g['P']} fam={g['n_families']} "
              f"S_obs={g['S_obs']} f1={g['f1']} f2={g['f2']} "
              f"M_hat={g['good_turing_missing_mass']:.3f} "
              f"recapture={g['cross_proposer_recapture']:.2f}")
    print(f"  selected {len(out['selected'])} criteria -> {tag}_species.json")


if __name__ == "__main__":
    main()
