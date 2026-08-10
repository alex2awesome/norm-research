#!/usr/bin/env python3
"""Round pipeline step 2: species-dedup the fleet pool, select the round's
k_A=15 / k_B=10 scored criteria, and build the BLIND audit pool.

Selection rule (declared before any round runs; label-blind and MONITOR-blind):
  1. A blind judge partitions each track's fleet pool into distinct concept
     SPECIES by full recall over the pool (never an embedding threshold).
  2. Species are ordered by  (#distinct proposers naming it) desc,
     then (#distinct families) desc, then stable sha256 of the species key.
  3. Species are taken in that order; the representative phrasing is drawn from
     the proposer that has contributed the FEWEST representatives so far
     (family-balance), ties broken by stable sha256.
  4. Track A takes 15 species. Track B takes 8 species; the remaining 2 B slots
     are COORDINATOR-PLANTED PROBES -- shallow keyword counterparts of two of the
     selected A criteria (the freeze's "2 planted probe pairs").

The audit pool is provenance-stripped and hash-ordered, so the auditor cannot see
which track proposed an item nor which items are probes.

Usage:
  python select_and_route.py --round 1 --species round1_species.json \
      --probes round1_probes.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
K_A, K_B_FLEET, N_PROBES = 15, 8, 2

# FREEZE ADDENDUM 3: decomposed components of a retired MIXED parent count toward the
# round's k budgets, so the fleet's share shrinks by however many components are injected.
# Components are split evenly off the two tracks (candidate-real side from k_A, surface
# side from k_B) and are routed independently by the blind audit like any other item.


def h(s):
    return hashlib.sha256(str(s).encode()).hexdigest()


def select(pool, species, k):
    by_pid = {p["pid"]: p for p in pool}
    sp = []
    for key, pids in species.items():
        pids = [q for q in pids if q in by_pid]
        if not pids:
            continue
        props = {by_pid[q]["proposer"] for q in pids}
        fams = {by_pid[q]["family"] for q in pids}
        sp.append({"key": key, "pids": pids, "n_proposers": len(props),
                   "n_families": len(fams), "proposers": sorted(props)})
    sp.sort(key=lambda s: (-s["n_proposers"], -s["n_families"], h(s["key"])))
    used = Counter()
    out = []
    for s in sp[:k]:
        cands = sorted(s["pids"], key=lambda q: (used[by_pid[q]["proposer"]], h(q)))
        rep = by_pid[cands[0]]
        used[rep["proposer"]] += 1
        out.append({**rep, "species_key": s["key"],
                    "species_support_proposers": s["n_proposers"],
                    "species_support_families": s["n_families"],
                    "species_members": s["pids"]})
    return out, sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--species", default=None)
    ap.add_argument("--probes", default=None)
    ap.add_argument("--decompositions", default=None)
    a = ap.parse_args()
    r = a.round
    dpath = Path(a.decompositions or HERE / f"round{r}_decompositions.json")
    decomps = json.loads(dpath.read_text()) if dpath.exists() else []
    n_real = sum(1 for d in decomps if d["role"] == "candidate_real")
    n_surf = len(decomps) - n_real
    k_a, k_b = K_A - n_real, K_B_FLEET - n_surf
    if decomps:
        print(f"[r{r}] ADDENDUM 3: {len(decomps)} decomposition components injected "
              f"({n_real} candidate-real, {n_surf} surface); fleet budget "
              f"{K_A}/{K_B_FLEET} -> {k_a}/{k_b}")
    spec = json.loads(Path(a.species or HERE / f"round{r}_species.json").read_text())

    selected, sp_stats = {}, {}
    for track, k in (("A", k_a), ("B", k_b)):
        pool = json.loads((HERE / f"round{r}_fleet_{track}.json").read_text())["proposals"]
        sel, sps = select(pool, spec[track], k)
        selected[track] = sel
        sp_stats[track] = {"n_proposals": len(pool), "n_species": len(sps),
                           "species_by_support": Counter(
                               s["n_proposers"] for s in sps)}
        print(f"[r{r}/{track}] pool={len(pool)} species={len(sps)} selected={len(sel)}")

    probes = json.loads(Path(a.probes or HERE / f"round{r}_probes.json").read_text())
    assert len(probes) == N_PROBES, f"expect {N_PROBES} planted probes"

    crits = []
    for i, s in enumerate(selected["A"]):
        crits.append({"cid": f"R{r}A{i+1:02d}", "proposed_track": "A",
                      "name": s["name"], "instruction": s["instruction"],
                      "provenance": {"proposer": s["proposer"], "family": s["family"],
                                     "model": s["model"], "pid": s["pid"],
                                     "species_support_proposers":
                                         s["species_support_proposers"],
                                     "species_members": s["species_members"]},
                      "rationale": s["rationale"]})
    for i, s in enumerate(selected["B"]):
        # FREEZE ADDENDUM 2: carry the conjectured upstream parent + MIXED flag
        crits.append({"cid": f"R{r}B{i+1:02d}", "proposed_track": "B",
                      "name": s["name"], "instruction": s["instruction"],
                      "upstream_parent": s.get("upstream_parent"),
                      "mixed": bool(s.get("mixed")),
                      "provenance": {"proposer": s["proposer"], "family": s["family"],
                                     "model": s["model"], "pid": s["pid"],
                                     "species_support_proposers":
                                         s["species_support_proposers"],
                                     "species_members": s["species_members"]},
                      "rationale": s["rationale"]})
    for d in decomps:
        crits.append({"cid": d["cid"],
                      "proposed_track": "A" if d["role"] == "candidate_real" else "B",
                      "name": d["name"], "instruction": d["instruction"],
                      "upstream_parent": (None if d["role"] == "candidate_real"
                                          else "decomposed component (surface half)"),
                      "mixed": False,
                      "provenance": {"proposer": "addendum3_decomposition",
                                     "family": "decomposition", "model": d.get("authored_by", "decomposition"),
                                     "pid": d["cid"], "decomposed_family": d["family"],
                                     "role": d["role"]},
                      "rationale": d["rationale"], "DECOMPOSITION_COMPONENT": True})

    for i, p in enumerate(probes):
        crits.append({"cid": f"R{r}P{i+1:02d}", "proposed_track": "B",
                      "name": p["name"], "instruction": p["instruction"],
                      "upstream_parent": "surface-only", "mixed": False,
                      "provenance": {"proposer": "coordinator_planted_probe",
                                     "family": "planted", "model": "planted",
                                     "pid": f"probe{i+1}",
                                     "counterpart_cid": p["counterpart_cid"]},
                      "rationale": p["rationale"], "PLANTED_PROBE": True})

    (HERE / f"round{r}_proposals_provenance.json").write_text(
        json.dumps({"round": r, "species_stats": {k: {"n_proposals": v["n_proposals"],
                                                      "n_species": v["n_species"]}
                                                  for k, v in sp_stats.items()},
                    "criteria": crits}, indent=1))

    blind, key = [], {}
    for c in crits:
        aid = "X" + h(f"{r}|{c['cid']}")[:8]
        blind.append({"aid": aid, "name": c["name"], "instruction": c["instruction"]})
        key[aid] = c["cid"]
    blind.sort(key=lambda d: d["aid"])
    (HERE / f"round{r}_audit_blind.json").write_text(json.dumps(blind, indent=1))
    (HERE / f"round{r}_audit_key.json").write_text(json.dumps(key, indent=1))
    print(f"[r{r}] audit pool: {len(blind)} items (provenance stripped, hash-ordered)")
    print(f"      {len(selected['A'])} A + {len(selected['B'])} B + {N_PROBES} probes")


if __name__ == "__main__":
    main()
