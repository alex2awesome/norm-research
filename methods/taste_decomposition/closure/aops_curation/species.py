#!/usr/bin/env python3
"""Species clustering, per-round Good-Turing missing mass (both tracks), and the

TWO-TIER RULE (design note S8 addendum; coordinator 2026-08-08, registered pre-Delta_2).
Only TIER S (sealed) proposals may enter the Good-Turing / Chao1 machinery. A
taxonomy-DIRECTED coverage sweep is TIER D: its criteria may be scored and may join the
bank through the ordinary blind audit, but they are excluded from species counts, f1/f2,
missing mass, the LOPO jackknife and cross-proposer recapture, because directed prompting
breaks the proposal-independence those estimators assume (measured: +.10 mean target
cosine, with category-level bank visibility = weak unsealing). Any proposal carrying
tier == "D" is dropped here with a recorded count; any table quoting mass must name the
tier it counted.

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


def _drop_directed(props, label):
    keep = [c for c in props if str(c.get("tier", "S")).upper() != "D"]
    n = len(props) - len(keep)
    if n:
        print(f"[two-tier] {label}: dropped {n} TIER-D (directed) proposals from the "
              f"Good-Turing estimator; {len(keep)} sealed remain", flush=True)
    return keep, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--round", default="1")
    ap.add_argument("--force", action="store_true",
                    help="rebuild a merged/scored round from scratch (destructive)")
    a = ap.parse_args()
    tag = f"{a.cell}_r{a.round}"

    # LANDMINE GUARD (2026-08-08, after a near-miss).  species.py is not idempotent once a
    # round has been merged: re-running it rebuilds the tau-only clustering and SILENTLY
    # overwrites a species.json that a blind audit, an arbiter and a corpus-wide Gemma pass
    # were all keyed to.  It happened here on a routine regression check of round 1 and was
    # caught only by diffing against the backup.  Refuse to clobber a merged or already
    # scored round unless --force is given.
    spf = HERE / f"{tag}_species.json"
    if spf.exists() and not a.force:
        prev = json.loads(spf.read_text())
        reasons = []
        if "b_merge" in prev or "blind_merge" in prev:
            # key renamed to `blind_merge` when the merge was generalised to BOTH
            # tracks on this campaign; the guard must recognise both spellings or it
            # silently stops protecting merged rounds.
            reasons.append("it carries a blind pairwise merge "
                           f"({sorted(set(prev.get('blind_merge', {})) )or 'b_merge'})")
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
        # TWO-TIER RULE: selection may see every proposal, but the Good-Turing estimator
        # sees TIER S only.  Clustering is re-run on the sealed subset so species counts,
        # f1/f2 and recapture are computed on an independence-preserving pool.
        sealed, n_dropped = _drop_directed(sub, f"{tag} track {track}")
        if n_dropped:
            lab_sealed, _ = cluster(sealed)
            gt = gt_block(sealed, lab_sealed)
            gt["tier"] = "S"
            gt["n_directed_excluded"] = n_dropped
        else:
            gt = gt_block(sub, lab)
            gt["tier"] = "S"
            gt["n_directed_excluded"] = 0
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
